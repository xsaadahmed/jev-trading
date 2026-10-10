"""Resumable bulk historical Jev inference (MASTER_SPEC Phase 7).

Enumerates (symbol, date) states for:
- 2016-01-04 .. 2023-12-29 (dev), and
- 2026-01-02 .. day before FORWARD_START (interim; or latest completed session
  if ``forward_registry`` is empty).

**Never** includes sealed 2024-01-02 .. 2025-12-31. Phase 7 never uses a seal
token. Missing responses are logged and never imputed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from jevlab.cv.walkforward import DEV_MAX_DATE
from jevlab.jev.audit import (
    DEFAULT_BUDGET_USD,
    DEFAULT_TOKENS_PER_CALL,
    INPUT_USD_PER_MILLION_TOKENS,
    BudgetExceededError,
    BudgetGuard,
)
from jevlab.jev.cache import JevCache, request_hash
from jevlab.jev.parse import JEV_FEATURE_COLUMNS, parse_response
from jevlab.jev.privacy import assert_state_private
from jevlab.jev.qset import QuestionSet
from jevlab.jev.schemas import SystemOneResponse
from jevlab.jev.state_builder import build_states_frame
from jevlab.jev.vocabulary import CLOSED_VOCABULARY
from jevlab.utils.calendar import nyse_session_dates, previous_nyse_session
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

DEV_START: date = date(2016, 1, 4)
SEALED_START: date = date(2024, 1, 2)
SEALED_END: date = date(2025, 12, 31)
INTERIM_START: date = date(2026, 1, 2)
UPPER_BOUND_STATES: int = 28_600
CONTAMINATION_TAG: str = "potentially_training_contaminated"
DEFAULT_CONCURRENCY: int = 4


class SealedBulkError(PermissionError):
    """Raised if a bulk job would include sealed-period dates."""


class ModelResolvedMismatchError(RuntimeError):
    """Resolved model id differs from the pinned research id."""


class InferClient(Protocol):
    def infer(
        self,
        state: dict[str, str],
        questions: Any,
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...


@dataclass
class WorkItem:
    symbol: str
    date: date
    state: dict[str, str]
    request_hash: str


@dataclass
class BatchPlan:
    n_total: int
    n_dev: int
    n_interim: int
    n_unique_hashes: int
    n_cache_hits: int
    n_to_call: int
    estimated_cost_usd: float
    upper_bound_states: int = UPPER_BOUND_STATES
    date_min: date | None = None
    date_max: date | None = None
    sealed_excluded: tuple[date, date] = (SEALED_START, SEALED_END)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_dev": self.n_dev,
            "n_interim": self.n_interim,
            "n_unique_hashes": self.n_unique_hashes,
            "n_cache_hits": self.n_cache_hits,
            "n_to_call": self.n_to_call,
            "estimated_cost_usd": self.estimated_cost_usd,
            "upper_bound_states": self.upper_bound_states,
            "date_min": str(self.date_min) if self.date_min else None,
            "date_max": str(self.date_max) if self.date_max else None,
            "sealed_excluded": [str(self.sealed_excluded[0]), str(self.sealed_excluded[1])],
            "price_usd_per_million_input_tokens": INPUT_USD_PER_MILLION_TOKENS,
            "contamination_tag": CONTAMINATION_TAG,
        }


@dataclass
class BatchResult:
    plan: BatchPlan
    n_ok: int = 0
    n_error: int = 0
    n_skipped_cache: int = 0
    n_missing: int = 0
    features_path: Path | None = None
    checkpoint_path: Path | None = None
    dry_run: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


def read_forward_start_date(registry_path: Path) -> date | None:
    """Return ``forward_start_date`` from the immutable forward_registry, if any."""
    from jevlab.experiments.forward_registry import (
        read_forward_start_date as _read_fwd,
    )

    return _read_fwd(registry_path)


def interim_end_date(
    *,
    forward_start: date | None,
    latest_completed_session: date | None,
) -> date | None:
    """Last interim session: day before FORWARD_START, else latest completed."""
    if forward_start is not None:
        try:
            return previous_nyse_session(forward_start)
        except Exception:
            sessions = nyse_session_dates(INTERIM_START, forward_start)
            before = [d for d in sessions if d < forward_start]
            return before[-1] if before else None
    return latest_completed_session


def assert_no_sealed_dates(dates: list[date]) -> None:
    sealed = [d for d in dates if SEALED_START <= d <= SEALED_END]
    if sealed:
        raise SealedBulkError(
            f"bulk refused {len(sealed)} sealed dates "
            f"[{SEALED_START}..{SEALED_END}]; Phase 7 never uses a seal token. "
            f"e.g. {sealed[:3]}"
        )


def enumerate_feature_rows(
    features: pd.DataFrame,
    *,
    forward_start: date | None,
    latest_completed_session: date | None,
) -> pd.DataFrame:
    """Filter feature rows to allowed Phase 7 date windows (sealed excluded)."""
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    end_interim = interim_end_date(
        forward_start=forward_start,
        latest_completed_session=latest_completed_session,
    )
    mask_dev = (work["date"] >= DEV_START) & (work["date"] <= DEV_MAX_DATE)
    if end_interim is not None and end_interim >= INTERIM_START:
        mask_interim = (work["date"] >= INTERIM_START) & (work["date"] <= end_interim)
    else:
        mask_interim = pd.Series(False, index=work.index)
    out = work.loc[mask_dev | mask_interim].reset_index(drop=True)
    assert_no_sealed_dates(out["date"].tolist())
    sealed_mask = (out["date"] >= SEALED_START) & (out["date"] <= SEALED_END)
    if bool(sealed_mask.any()):
        raise SealedBulkError("sealed rows present after filter")
    return out


def build_work_items(
    features: pd.DataFrame,
    qset: QuestionSet,
    *,
    route: str,
    model_id: str,
    bucket_edges: list[float] | tuple[float, ...],
) -> list[WorkItem]:
    """Build privacy-validated states and request hashes for each row."""
    states = build_states_frame(features, edges=tuple(bucket_edges))
    if states.empty:
        return []
    questions = qset.to_cacheable_dict()
    items: list[WorkItem] = []
    for row in states.itertuples(index=False):
        raw_state = row.state
        if not isinstance(raw_state, dict):
            raise TypeError(f"state must be dict, got {type(raw_state)!r}")
        state: dict[str, str] = {str(k): str(v) for k, v in raw_state.items()}
        assert_state_private(state, CLOSED_VOCABULARY)
        rh = request_hash(
            route=route,
            model_requested=model_id,
            state=state,
            questions=questions,
        )
        items.append(
            WorkItem(
                symbol=str(row.symbol),
                date=pd.Timestamp(str(row.date)).date(),
                state=state,
                request_hash=rh,
            )
        )
    assert_no_sealed_dates([it.date for it in items])
    return items


def dedupe_work_items(items: list[WorkItem]) -> list[WorkItem]:
    """Keep first occurrence of each request_hash (identical states share a call)."""
    seen: set[str] = set()
    out: list[WorkItem] = []
    for it in items:
        if it.request_hash in seen:
            continue
        seen.add(it.request_hash)
        out.append(it)
    return out


class BatchCheckpoint:
    """SQLite checkpoint for kill-and-resume."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    request_hash TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INT,
                    latency_ms INT,
                    input_tokens INT,
                    http_status INT,
                    model_resolved TEXT,
                    error TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def get_status(self, request_hash: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM batch_jobs WHERE request_hash = ?",
                (request_hash,),
            ).fetchone()
        return None if row is None else str(row[0])

    def completed_hashes(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT request_hash FROM batch_jobs WHERE status IN ('ok', 'skipped_cache')"
            ).fetchall()
        return {str(r[0]) for r in rows}

    def record(
        self,
        *,
        request_hash: str,
        symbol: str,
        date_value: date,
        status: str,
        attempt: int | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        http_status: int | None = None,
        model_resolved: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_jobs (
                    request_hash, symbol, date, status, attempt, latency_ms,
                    input_tokens, http_status, model_resolved, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(request_hash) DO UPDATE SET
                    status=excluded.status,
                    attempt=excluded.attempt,
                    latency_ms=excluded.latency_ms,
                    input_tokens=excluded.input_tokens,
                    http_status=excluded.http_status,
                    model_resolved=excluded.model_resolved,
                    error=excluded.error,
                    updated_at=datetime('now')
                """,
                (
                    request_hash,
                    symbol,
                    date_value.isoformat(),
                    status,
                    attempt,
                    latency_ms,
                    input_tokens,
                    http_status,
                    model_resolved,
                    error,
                ),
            )
            conn.commit()

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM batch_jobs GROUP BY status"
            ).fetchall()
        return {str(s): int(c) for s, c in rows}


class ThreadSafeBudget:
    """Serialize BudgetGuard plan/record under concurrent workers."""

    def __init__(self, guard: BudgetGuard) -> None:
        self.guard = guard
        self._lock = threading.Lock()

    def plan(self, n_calls: int) -> float:
        with self._lock:
            return self.guard.plan(n_calls)

    def record_usage(self, input_tokens: int) -> float:
        with self._lock:
            return self.guard.record_usage(input_tokens)


def plan_batch(
    items: list[WorkItem],
    cache: JevCache,
    *,
    budget: BudgetGuard | None = None,
    tokens_per_call: int = DEFAULT_TOKENS_PER_CALL,
) -> BatchPlan:
    """Count unique hashes, cache hits, and projected spend.

    Raises ``BudgetExceededError`` if projected spend for uncached calls exceeds
    the budget cap (abort before any live spend).
    """
    unique = dedupe_work_items(items)
    hits = 0
    to_call = 0
    for it in unique:
        if cache.get(it.request_hash) is not None:
            hits += 1
        else:
            to_call += 1
    guard = budget or BudgetGuard(tokens_per_call=tokens_per_call)
    est = guard.estimate_usd(to_call)
    if est > guard.budget_usd + 1e-12:
        raise BudgetExceededError(
            f"projected spend ${est:.4f} exceeds budget ${guard.budget_usd:.2f} "
            f"({to_call} calls x {tokens_per_call} tokens)"
        )
    dates = [it.date for it in items]
    n_dev = sum(1 for d in dates if d <= DEV_MAX_DATE)
    n_interim = sum(1 for d in dates if d >= INTERIM_START)
    return BatchPlan(
        n_total=len(items),
        n_dev=n_dev,
        n_interim=n_interim,
        n_unique_hashes=len(unique),
        n_cache_hits=hits,
        n_to_call=to_call,
        estimated_cost_usd=est,
        date_min=min(dates) if dates else None,
        date_max=max(dates) if dates else None,
    )


def _parse_usage_tokens(response: SystemOneResponse) -> int:
    if response.usage is None or response.usage.input_tokens is None:
        return DEFAULT_TOKENS_PER_CALL
    return int(response.usage.input_tokens)


def run_one(
    item: WorkItem,
    *,
    client: InferClient,
    cache: JevCache,
    cache_lock: threading.Lock,
    qset: QuestionSet,
    route: str,
    model_id: str,
    state_schema_version: str,
    checkpoint: BatchCheckpoint,
    budget: ThreadSafeBudget,
    pin_model: bool = True,
) -> dict[str, Any]:
    """Execute or skip one unique request_hash; never impute on failure."""
    with cache_lock:
        existing = cache.get(item.request_hash)
    if existing is not None:
        checkpoint.record(
            request_hash=item.request_hash,
            symbol=item.symbol,
            date_value=item.date,
            status="skipped_cache",
            attempt=existing.attempt,
            model_resolved=existing.model_resolved,
        )
        return {
            "status": "skipped_cache",
            "request_hash": item.request_hash,
            "response": existing.response,
            "model_resolved": existing.model_resolved,
        }

    budget.plan(1)
    t0 = time.perf_counter()
    try:
        response = client.infer(item.state, qset.as_request_questions(), model=model_id)
    except Exception as exc:
        checkpoint.record(
            request_hash=item.request_hash,
            symbol=item.symbol,
            date_value=item.date,
            status="error",
            error=str(exc),
        )
        return {"status": "error", "request_hash": item.request_hash, "error": str(exc)}

    latency_ms = int((time.perf_counter() - t0) * 1000)
    resolved = response.model
    if pin_model and resolved != model_id:
        checkpoint.record(
            request_hash=item.request_hash,
            symbol=item.symbol,
            date_value=item.date,
            status="error",
            latency_ms=latency_ms,
            model_resolved=resolved,
            error=f"model_resolved {resolved} != pinned {model_id}",
        )
        raise ModelResolvedMismatchError(f"model_resolved={resolved!r} != pinned={model_id!r}")

    tokens = _parse_usage_tokens(response)
    budget.record_usage(tokens)
    payload = response.model_dump(mode="json")
    with cache_lock:
        cache.put(
            req_hash=item.request_hash,
            route=route,
            model_requested=model_id,
            model_resolved=resolved,
            purpose="research",
            qset_version=qset.qset_version,
            state_schema_version=state_schema_version,
            response=payload,
        )
    checkpoint.record(
        request_hash=item.request_hash,
        symbol=item.symbol,
        date_value=item.date,
        status="ok",
        attempt=1,
        latency_ms=latency_ms,
        input_tokens=tokens,
        http_status=200,
        model_resolved=resolved,
    )
    return {
        "status": "ok",
        "request_hash": item.request_hash,
        "response": payload,
        "model_resolved": resolved,
        "latency_ms": latency_ms,
        "input_tokens": tokens,
    }


def build_jev_features_table(
    items: list[WorkItem],
    *,
    cache: JevCache,
    qset_version: str,
    state_schema_version: str,
    pinned_model: str,
) -> pd.DataFrame:
    """Assemble jev_features from cache; missing responses stay missing (no impute)."""
    by_hash: dict[str, list[WorkItem]] = {}
    for it in items:
        by_hash.setdefault(it.request_hash, []).append(it)

    rows: list[dict[str, Any]] = []
    for rh, group in by_hash.items():
        hit = cache.get(rh)
        if hit is None:
            continue  # never impute
        if hit.model_resolved != pinned_model:
            raise ModelResolvedMismatchError(
                f"cached model_resolved={hit.model_resolved!r} != pinned={pinned_model!r}"
            )
        try:
            feats = parse_response(hit.response)
        except Exception:
            continue  # malformed stays missing
        for it in group:
            row: dict[str, Any] = {
                "symbol": it.symbol,
                "date": it.date,
                "qset_version": qset_version,
                "state_schema_version": state_schema_version,
                "model_resolved": hit.model_resolved,
                "request_hash": rh,
                "inference_ids": rh,
            }
            for col in JEV_FEATURE_COLUMNS:
                row[col] = feats.get(col)
            row["j_order_disagree"] = feats.get("j_order_disagree")
            rows.append(row)
    cols = [
        "symbol",
        "date",
        "qset_version",
        "state_schema_version",
        "model_resolved",
        "request_hash",
        "inference_ids",
        *JEV_FEATURE_COLUMNS,
        "j_order_disagree",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def write_jev_features(frame: pd.DataFrame, out_dir: Path) -> Path:
    """Write parquet under ``data/features/jev/qset_v1/`` with contamination metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "jev_features.parquet"
    meta_path = out_dir / "metadata.json"
    frame.to_parquet(path, index=False)
    meta = {
        "contamination_tag": CONTAMINATION_TAG,
        "potentially_training_contaminated": True,
        "n_rows": len(frame),
        "columns": list(frame.columns),
        "written_at": pd.Timestamp.now("UTC").isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def reconcile_row_counts(
    enumerated: int,
    features_rows: int,
    *,
    n_missing: int,
) -> dict[str, Any]:
    """Row-count reconciliation: features + missing == enumerated (no impute)."""
    ok = features_rows + n_missing == enumerated
    return {
        "enumerated": enumerated,
        "features_rows": features_rows,
        "missing": n_missing,
        "reconciled": ok,
    }


def run_bulk_inference(
    features: pd.DataFrame,
    *,
    qset: QuestionSet,
    cache: JevCache,
    client: InferClient | None,
    route: str,
    model_id: str,
    state_schema_version: str,
    bucket_edges: list[float] | tuple[float, ...],
    out_dir: Path,
    checkpoint_path: Path,
    forward_start: date | None = None,
    latest_completed_session: date | None = None,
    dry_run: bool = True,
    budget_usd: float = DEFAULT_BUDGET_USD,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_items: int | None = None,
    work_items: list[WorkItem] | None = None,
) -> BatchResult:
    """Plan (and optionally execute) bulk inference with resume support.

    ``work_items`` may be supplied by tests to bypass feature→state construction.
    """
    if work_items is None:
        filtered = enumerate_feature_rows(
            features,
            forward_start=forward_start,
            latest_completed_session=latest_completed_session,
        )
        if max_items is not None:
            filtered = filtered.iloc[:max_items].copy()
        items = build_work_items(
            filtered,
            qset,
            route=route,
            model_id=model_id,
            bucket_edges=bucket_edges,
        )
    else:
        items = list(work_items)
        assert_no_sealed_dates([it.date for it in items])
        if max_items is not None:
            items = items[:max_items]

    budget = BudgetGuard(budget_usd=budget_usd)
    plan = plan_batch(items, cache, budget=budget)
    result = BatchResult(plan=plan, dry_run=dry_run, checkpoint_path=checkpoint_path)
    result.meta["contamination_tag"] = CONTAMINATION_TAG
    result.meta["api_calls"] = 0

    if dry_run:
        return result

    if client is None:
        raise ValueError("client is required when dry_run=False")

    checkpoint = BatchCheckpoint(checkpoint_path)
    unique = dedupe_work_items(items)
    done = checkpoint.completed_hashes()
    cache_lock = threading.Lock()
    safe_budget = ThreadSafeBudget(budget)

    def _needs_work(it: WorkItem) -> bool:
        if it.request_hash in done:
            return False
        status = checkpoint.get_status(it.request_hash)
        return status not in {"ok", "skipped_cache"}

    pending = [it for it in unique if _needs_work(it)]

    def _job(it: WorkItem) -> dict[str, Any]:
        return run_one(
            it,
            client=client,
            cache=cache,
            cache_lock=cache_lock,
            qset=qset,
            route=route,
            model_id=model_id,
            state_schema_version=state_schema_version,
            checkpoint=checkpoint,
            budget=safe_budget,
        )

    n_ok = 0
    n_err = 0
    n_skip = 0
    workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, it): it for it in pending}
        for fut in as_completed(futures):
            try:
                out = fut.result()
            except ModelResolvedMismatchError:
                raise
            except Exception as exc:
                it = futures[fut]
                checkpoint.record(
                    request_hash=it.request_hash,
                    symbol=it.symbol,
                    date_value=it.date,
                    status="error",
                    error=str(exc),
                )
                n_err += 1
                continue
            st = str(out.get("status"))
            if st == "ok":
                n_ok += 1
            elif st == "skipped_cache":
                n_skip += 1
            else:
                n_err += 1

    # Already-completed from prior run
    already = len(done)
    result.n_ok = n_ok
    result.n_error = n_err
    result.n_skipped_cache = n_skip + already

    feat = build_jev_features_table(
        items,
        cache=cache,
        qset_version=qset.qset_version,
        state_schema_version=state_schema_version,
        pinned_model=model_id,
    )
    keyed: set[tuple[str, date]] = set()
    if len(feat):
        for r in feat.itertuples(index=False):
            keyed.add((str(r.symbol), pd.Timestamp(str(r.date)).date()))
    missing = sum(1 for it in items if (it.symbol, it.date) not in keyed)
    result.n_missing = missing
    result.meta["reconciliation"] = reconcile_row_counts(len(items), len(feat), n_missing=missing)
    path = write_jev_features(feat, out_dir)
    result.features_path = path
    result.meta["api_calls"] = n_ok
    result.meta["checkpoint_counts"] = checkpoint.counts()
    return result
