"""Live Phase 6 contamination diagnostics (MASTER_SPEC §11.4).

Real Jev calls only. Never touches ``data/sealed/``. Never freezes qset.
Never auto-fills the human disposition decision.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from jevlab.features.columns import MODEL_A_COLUMNS, STATE_V1_SOURCE_COLUMNS
from jevlab.jev.audit import (
    CONTAMINATION_LABEL,
    DEFAULT_TOKENS_PER_CALL,
    WARNING_BANNER,
    AuditReport,
    AuditSection,
    BudgetGuard,
    EraId,
    assert_audit_dates_allowed,
    assign_era,
    build_state_v1c_from_percentiles,
    class_probability_concentration,
    d1_from_predictions,
    d2_from_scores,
    d3_rank_ic_by_era,
    d5_fine_vs_coarse,
    degeneracy_check,
    determinism_stats,
    plan_audit_calls,
    sample_dev_states,
    sample_era_balanced,
    sample_rows_with_spacing,
    summarize_numeric_series,
    write_audit_report,
)
from jevlab.jev.cache import JevCache, request_hash
from jevlab.jev.parse import JEV_FEATURE_COLUMNS, parse_response
from jevlab.jev.privacy import assert_state_private
from jevlab.jev.qset import QuestionSet
from jevlab.jev.schemas import ChoiceAnswer, NoulAnswer, SystemOneResponse
from jevlab.jev.state_builder import build_state_from_percentiles, compute_trailing_percentiles
from jevlab.jev.vocabulary import CLOSED_VOCABULARY, DEFAULT_BUCKET_EDGES
from jevlab.utils.logging import get_logger

log = get_logger(__name__)


class InferClient(Protocol):
    def infer(
        self,
        state: dict[str, str],
        questions: dict[str, Any],
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...


class ModelResolvedMismatchError(RuntimeError):
    """Pinned model id does not match resolved response model."""


@dataclass
class CallStats:
    attempted: int = 0
    successful: int = 0
    retries_observed: int = 0
    failures: int = 0
    cache_reused: int = 0
    input_tokens: int = 0
    spent_usd: float = 0.0
    resolved_models: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "successful": self.successful,
            "retries_observed": self.retries_observed,
            "failures": self.failures,
            "cache_reused": self.cache_reused,
            "input_tokens": self.input_tokens,
            "spent_usd": self.spent_usd,
            "resolved_models": sorted(self.resolved_models),
        }


class AuditCheckpoint:
    """Resumable per-job status for live audit sections."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_jobs (
                    job_key TEXT PRIMARY KEY,
                    section TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_hash TEXT,
                    attempt INT,
                    model_resolved TEXT,
                    input_tokens INT,
                    error TEXT,
                    payload_json TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, job_key: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM audit_jobs WHERE job_key = ?", (job_key,)).fetchone()
            return row if isinstance(row, sqlite3.Row) else None

    def record(
        self,
        *,
        job_key: str,
        section: str,
        status: str,
        request_hash: str | None = None,
        attempt: int | None = None,
        model_resolved: str | None = None,
        input_tokens: int | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_jobs (
                    job_key, section, status, request_hash, attempt, model_resolved,
                    input_tokens, error, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    status=excluded.status,
                    request_hash=excluded.request_hash,
                    attempt=excluded.attempt,
                    model_resolved=excluded.model_resolved,
                    input_tokens=excluded.input_tokens,
                    error=excluded.error,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    job_key,
                    section,
                    status,
                    request_hash,
                    attempt,
                    model_resolved,
                    input_tokens,
                    error,
                    json.dumps(payload) if payload is not None else None,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()


def _usage_tokens(response: SystemOneResponse) -> int:
    if response.usage is None or response.usage.input_tokens is None:
        return DEFAULT_TOKENS_PER_CALL
    return int(response.usage.input_tokens)


def _row_key(symbol: str, d: date) -> str:
    return f"{symbol}|{d.isoformat()}"


def attach_percentiles_and_states(
    features: pd.DataFrame,
    *,
    edges: tuple[float, ...] = DEFAULT_BUCKET_EDGES,
) -> pd.DataFrame:
    """Return feature rows with fine/coarse states and MODEL_A columns preserved."""
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    assert_audit_dates_allowed(work["date"].tolist())
    pct = compute_trailing_percentiles(work)
    pct_cols = [c for c in pct.columns if c not in {"symbol", "date"}]
    merged = work.merge(
        pct.rename(columns={c: f"{c}__pct" for c in pct_cols}),
        on=["symbol", "date"],
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        percentiles: dict[str, float] = {}
        skip = False
        for col in pct_cols:
            val = r[f"{col}__pct"]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                skip = True
                break
            percentiles[col] = float(val)
        if skip:
            continue
        fine = build_state_from_percentiles(
            percentiles,
            down_streak=float(r["down_streak"]),
            edges=edges,
        )
        coarse = build_state_v1c_from_percentiles(
            percentiles,
            down_streak=float(r["down_streak"]),
        )
        assert_state_private(fine, CLOSED_VOCABULARY)
        assert_state_private(coarse, CLOSED_VOCABULARY)
        row = {c: r[c] for c in merged.columns if not str(c).endswith("__pct")}
        row["state_fine"] = fine
        row["state_coarse"] = coarse
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    assert_audit_dates_allowed(out["date"].tolist())
    return out


def _load_or_infer(
    *,
    client: InferClient,
    cache: JevCache,
    qset: QuestionSet,
    state: dict[str, str],
    route: str,
    model_id: str,
    purpose: str,
    state_schema_version: str,
    budget: BudgetGuard,
    stats: CallStats,
    section: str,
    job_key: str,
    checkpoint: AuditCheckpoint,
    attempt_index: int,
) -> dict[str, Any]:
    """Ensure attempt_index exists (1-based); call API only when missing.

    Determinism repeats are stored as separate cache attempts and never overwrite
    prior historical entries.
    """
    if attempt_index < 1:
        raise ValueError("attempt_index must be >= 1")
    questions = qset.to_cacheable_dict()
    req_h = request_hash(
        route=route,
        model_requested=model_id,
        state=state,
        questions=questions,
    )

    existing = checkpoint.get(job_key)
    if (
        existing is not None
        and existing["status"] in {"ok", "cache_hit"}
        and existing["payload_json"]
    ):
        payload = json.loads(str(existing["payload_json"]))
        resp = payload.get("response")
        if isinstance(resp, dict):
            stats.cache_reused += 1
            mr = existing["model_resolved"] or resp.get("model")
            if mr:
                stats.resolved_models.add(str(mr))
                if str(mr) != model_id:
                    raise ModelResolvedMismatchError(
                        f"checkpoint model_resolved={mr!r} != pinned={model_id!r}"
                    )
            return {
                "status": str(existing["status"]),
                "request_hash": req_h,
                "attempt": int(existing["attempt"] or attempt_index),
                "response": resp,
                "model_resolved": mr,
            }

    rec = cache.get(req_h, attempt=attempt_index)
    if rec is not None:
        stats.cache_reused += 1
        if rec.model_resolved:
            stats.resolved_models.add(str(rec.model_resolved))
            if rec.model_resolved != model_id:
                raise ModelResolvedMismatchError(
                    f"cached model_resolved={rec.model_resolved!r} != pinned={model_id!r}"
                )
        checkpoint.record(
            job_key=job_key,
            section=section,
            status="cache_hit",
            request_hash=req_h,
            attempt=rec.attempt,
            model_resolved=rec.model_resolved,
            payload={"response": rec.response},
        )
        return {
            "status": "cache_hit",
            "request_hash": req_h,
            "attempt": rec.attempt,
            "response": rec.response,
            "model_resolved": rec.model_resolved,
        }

    # Fill missing attempts in order so attempt numbers stay contiguous.
    have = cache.next_attempt(req_h) - 1
    last: dict[str, Any] | None = None
    max_transient_retries = 12
    while have < attempt_index:
        budget.plan(1)
        stats.attempted += 1
        t0 = time.perf_counter()
        response: SystemOneResponse | None = None
        last_exc: Exception | None = None
        for trial in range(max_transient_retries):
            try:
                response = client.infer(state, qset.as_request_questions(), model=model_id)
                last_exc = None
                break
            except ModelResolvedMismatchError:
                raise
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = any(
                    tok in msg
                    for tok in (
                        "getaddrinfo",
                        "connecterror",
                        "connection reset",
                        "disconnected",
                        "temporarily unavailable",
                        "timeout",
                        "10054",
                        "11001",
                        "network",
                        "remoteprotocol",
                    )
                )
                if not transient or trial + 1 >= max_transient_retries:
                    break
                stats.retries_observed += 1
                sleep_s = min(90.0, 2.0**trial)
                log.warning(
                    "audit_transient_retry",
                    job_key=job_key,
                    trial=trial + 1,
                    sleep_s=sleep_s,
                    error=str(exc),
                )
                time.sleep(sleep_s)
        if response is None:
            stats.failures += 1
            checkpoint.record(
                job_key=job_key,
                section=section,
                status="error",
                request_hash=req_h,
                error=str(last_exc),
            )
            assert last_exc is not None
            raise last_exc

        latency_ms = int((time.perf_counter() - t0) * 1000)
        resolved = response.model
        stats.resolved_models.add(resolved)
        if resolved != model_id:
            checkpoint.record(
                job_key=job_key,
                section=section,
                status="model_mismatch",
                request_hash=req_h,
                model_resolved=resolved,
                error=f"resolved {resolved} != pinned {model_id}",
            )
            raise ModelResolvedMismatchError(f"model_resolved={resolved!r} != pinned={model_id!r}")

        tokens = _usage_tokens(response)
        cost = budget.record_usage(tokens)
        stats.input_tokens += tokens
        stats.spent_usd += cost
        stats.successful += 1
        payload = response.model_dump(mode="json")
        purpose_lit = "audit" if purpose == "audit" else "research"
        put_rec = cache.put(
            req_hash=req_h,
            route=route,
            model_requested=model_id,
            model_resolved=resolved,
            purpose=purpose_lit,  # type: ignore[arg-type]
            qset_version=qset.qset_version,
            state_schema_version=state_schema_version,
            response=payload,
        )
        last = {
            "status": "ok",
            "request_hash": req_h,
            "attempt": put_rec.attempt,
            "response": payload,
            "model_resolved": resolved,
            "input_tokens": tokens,
            "latency_ms": latency_ms,
        }
        have = put_rec.attempt

    assert last is not None
    checkpoint.record(
        job_key=job_key,
        section=section,
        status="ok",
        request_hash=req_h,
        attempt=int(last["attempt"]),
        model_resolved=str(last["model_resolved"]),
        input_tokens=int(last.get("input_tokens") or 0),
        payload={"response": last["response"], "latency_ms": last.get("latency_ms")},
    )
    return last


def _model_a_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = [c for c in MODEL_A_COLUMNS if c in frame.columns]
    missing = [c for c in MODEL_A_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"features missing MODEL_A columns: {missing}")
    return frame.loc[:, cols].to_numpy(dtype=float)


def _parse_audit_choice(response: dict[str, Any], qid: str = "era_class") -> str:
    ans = ChoiceAnswer.model_validate(SystemOneResponse.model_validate(response).answers[qid])
    return str(ans.choice)


def _parse_audit_noul(response: dict[str, Any], qid: str = "outcome_outperform") -> float:
    ans = NoulAnswer.model_validate(SystemOneResponse.model_validate(response).answers[qid])
    return float(ans.noul)


def _extract_qset_outputs(response: dict[str, Any]) -> dict[str, Any]:
    """Raw + parsed outputs for degeneracy/determinism on research qset."""
    resp = SystemOneResponse.model_validate(response)
    feats = parse_response(resp)
    dislocation = NoulAnswer.model_validate(resp.answers["dislocation"])
    regime = NoulAnswer.model_validate(resp.answers["regime_supportive"])
    setup_a = ChoiceAnswer.model_validate(resp.answers["setup_type_a"])
    setup_b = ChoiceAnswer.model_validate(resp.answers["setup_type_b"])
    return {
        "features": feats,
        "noul": {
            "dislocation": float(dislocation.noul),
            "regime_supportive": float(regime.noul),
        },
        "choice": {
            "setup_type_a": dict(setup_a.probabilities),
            "setup_type_b": dict(setup_b.probabilities),
        },
    }


def sample_d2_states(
    features: pd.DataFrame,
    *,
    n: int = 2000,
    seed: int = 99,
    exclude: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """2,000 train-period states disjoint from the 300-state dev sample."""
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    work = work.loc[(work["date"] >= date(2016, 1, 4)) & (work["date"] <= date(2018, 12, 31))]
    if exclude is not None and not exclude.empty:
        ex = exclude.copy()
        ex["date"] = pd.to_datetime(ex["date"]).dt.date
        keys = set(zip(ex["symbol"].astype(str), ex["date"], strict=False))
        mask = [
            (str(s), d) not in keys
            for s, d in zip(work["symbol"].astype(str), work["date"], strict=True)
        ]
        work = work.loc[mask]
    assert_audit_dates_allowed(work["date"].tolist())
    return sample_rows_with_spacing(work, n=n, seed=seed, min_session_gap=1)


def sample_d5_states(
    features: pd.DataFrame,
    *,
    n: int = 3000,
    seed: int = 123,
) -> pd.DataFrame:
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    work = work.loc[(work["date"] >= date(2016, 1, 4)) & (work["date"] <= date(2023, 12, 29))]
    assert_audit_dates_allowed(work["date"].tolist())
    return sample_rows_with_spacing(work, n=n, seed=seed, min_session_gap=1)


def _privacy_scan(states: list[dict[str, str]]) -> dict[str, Any]:
    bad = 0
    for st in states:
        try:
            assert_state_private(st, CLOSED_VOCABULARY)
        except Exception:
            bad += 1
    return {
        "n_states": len(states),
        "n_failures": bad,
        "all_passed": bad == 0,
        "note": "no ticker/date/year/raw market measurement permitted in research states",
    }


def run_audit_live(
    features: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    research_qset: QuestionSet,
    audit_qset: QuestionSet,
    client: InferClient,
    cache: JevCache,
    route: str,
    model_id: str,
    state_schema_version: str = "state_v1",
    bucket_edges: list[float] | tuple[float, ...] | None = None,
    sealed_root: Path | None = None,
    out_dir: Path,
    checkpoint_path: Path,
    budget_usd: float = 25.0,
    n_dev: int = 300,
    n_determinism_repeats: int = 3,
    n_d1: int = 600,
    n_d2: int = 2000,
    n_d5: int = 3000,
    jev_cutoff_path: Path | None = None,
    registry_path: Path | None = None,
) -> AuditReport:
    """Execute real §11.4 diagnostics; write AUDIT_REPORT; do not freeze qset."""
    # sealed_root is recorded for operator context; audit loaders never open it.
    if sealed_root is not None:
        log.info("audit_live_sealed_root_ignored", sealed_root=str(sealed_root))

    edges = tuple(bucket_edges) if bucket_edges else DEFAULT_BUCKET_EDGES
    prepared = attach_percentiles_and_states(features, edges=edges)
    if prepared.empty:
        raise RuntimeError("no eligible feature rows with warm percentiles for audit")

    plan = plan_audit_calls(
        n_dev=n_dev,
        n_determinism_repeats=n_determinism_repeats,
        n_d1=n_d1,
        n_d2=n_d2,
        n_d5=n_d5,
        include_d5_fine=True,
    )
    budget = BudgetGuard(budget_usd=budget_usd)
    budget.plan(int(plan["total_calls"]))
    stats = CallStats()
    checkpoint = AuditCheckpoint(checkpoint_path)
    warnings: list[str] = []
    privacy_states: list[dict[str, str]] = []

    report = AuditReport(
        meta={
            "generated_at": datetime.now(UTC).isoformat(),
            "live": True,
            "api_calls": 0,
            "plan": plan,
            "model_requested": model_id,
            "research_qset_version": research_qset.qset_version,
            "research_qset_hash": research_qset.content_hash,
            "audit_qset_version": audit_qset.qset_version,
            "audit_qset_hash": audit_qset.content_hash,
            "budget_usd": budget_usd,
            "warning_banner": WARNING_BANNER,
            "contamination_label": CONTAMINATION_LABEL,
            "qset_frozen": False,
            "phase7_not_run": True,
        }
    )
    report.sections.append(AuditSection(name="Plan / cost estimate", body=plan))

    # --- Dev + determinism (qset_v1, fine states) ---
    dev = sample_dev_states(prepared, n=n_dev, seed=42)
    assert_audit_dates_allowed(dev["date"].tolist())
    report.sections.append(
        AuditSection(
            name="Dev sample (no labels)",
            body={
                "n": len(dev),
                "date_min": str(dev["date"].min()) if len(dev) else None,
                "date_max": str(dev["date"].max()) if len(dev) else None,
                "labels_joined": False,
                "warning_banner": WARNING_BANNER,
            },
        )
    )

    attempt_frames: list[pd.DataFrame] = []
    noul_by_name: dict[str, list[float]] = {"dislocation": [], "regime_supportive": []}
    choice_by_name: dict[str, list[dict[str, float]]] = {
        "setup_type_a": [],
        "setup_type_b": [],
    }
    missing_errors = 0

    for attempt_i in range(1, n_determinism_repeats + 1):
        feat_rows: list[dict[str, Any]] = []
        for _, row in dev.iterrows():
            state = dict(row["state_fine"])
            privacy_states.append(state)
            job_key = f"det|{_row_key(str(row['symbol']), row['date'])}|a{attempt_i}"
            try:
                result = _load_or_infer(
                    client=client,
                    cache=cache,
                    qset=research_qset,
                    state=state,
                    route=route,
                    model_id=model_id,
                    purpose="research",
                    state_schema_version=state_schema_version,
                    budget=budget,
                    stats=stats,
                    section="determinism",
                    job_key=job_key,
                    checkpoint=checkpoint,
                    attempt_index=attempt_i,
                )
                extracted = _extract_qset_outputs(result["response"])
            except ModelResolvedMismatchError:
                raise
            except Exception as exc:
                missing_errors += 1
                log.warning("determinism_call_failed", error=str(exc), job_key=job_key)
                extracted = {
                    "features": {c: float("nan") for c in JEV_FEATURE_COLUMNS},
                    "noul": {"dislocation": float("nan"), "regime_supportive": float("nan")},
                    "choice": {"setup_type_a": {}, "setup_type_b": {}},
                }
            feat_rows.append(
                {
                    "symbol": row["symbol"],
                    "date": row["date"],
                    **extracted["features"],
                }
            )
            if attempt_i == 1:
                for k, v in extracted["noul"].items():
                    noul_by_name[k].append(float(v))
                for k, v in extracted["choice"].items():
                    choice_by_name[k].append(dict(v))
        attempt_frames.append(pd.DataFrame(feat_rows))

    deg_flags = degeneracy_check(noul_by_name, choice_by_name)
    deg_body: dict[str, Any] = {
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
        "missing_error_count": missing_errors,
        "outputs": {},
        "degeneracy": deg_flags,
        "rule_triggered": bool(deg_flags["flags"]),
        "qset_wording_unchanged": not bool(deg_flags["flags"]),
    }
    for name, vals in noul_by_name.items():
        finite = [v for v in vals if np.isfinite(v)]
        deg_body["outputs"][name] = {
            **summarize_numeric_series(finite),
            "class_probability_concentration": {
                "mean_abs_2p_minus_1": float(
                    np.mean([abs(2 * v - 1) for v in finite]) if finite else float("nan")
                )
            },
            "missing_error_count": int(sum(1 for v in vals if not np.isfinite(v))),
        }
    for name, rows in choice_by_name.items():
        deg_body["outputs"][name] = {
            "concentration": class_probability_concentration(rows),
            "missing_error_count": int(sum(1 for r in rows if not r)),
        }
    if deg_flags["flags"]:
        warnings.append(f"degeneracy:{','.join(deg_flags['flags'])}")
        deg_body["stop_before_qset_revision"] = True
        deg_body["note"] = (
            "Degeneracy rule triggered. STOP before modifying qset wording. "
            "Do not silently revise questions."
        )
    report.sections.append(AuditSection(name="Degeneracy check (qset_v1)", body=deg_body))

    det_cols = list(JEV_FEATURE_COLUMNS)
    det_body = determinism_stats(
        attempt_frames,
        det_cols,
        noul_cols=["j_dislocation", "j_regime"],
    )
    det_body["n_states"] = len(dev)
    det_body["n_repeats"] = n_determinism_repeats
    det_body["attempts_stored_separately"] = True
    if det_body.get("mitigation_triggered"):
        warnings.append("determinism_mitigation_triggered")
    report.sections.append(AuditSection(name="Determinism audit (qset_v1)", body=det_body))

    # If degeneracy triggered, still finish diagnostics but mark for human stop-before-revision.
    # User asked to STOP before modifying qset — we never modify qset here.

    # --- D1 ---
    d1_sample = sample_era_balanced(prepared, per_era=n_d1 // 3, seed=7, min_session_gap=5)
    eras: list[EraId] = []
    preds: list[str] = []
    d1_dates: list[date] = []
    d1_kept: list[dict[str, Any]] = []
    for _, row in d1_sample.iterrows():
        era = assign_era(row["date"])
        if era is None:
            continue
        state = dict(row["state_fine"])
        privacy_states.append(state)
        job_key = f"d1|{_row_key(str(row['symbol']), row['date'])}"
        result = _load_or_infer(
            client=client,
            cache=cache,
            qset=audit_qset,
            state=state,
            route=route,
            model_id=model_id,
            purpose="audit",
            state_schema_version=state_schema_version,
            budget=budget,
            stats=stats,
            section="d1",
            job_key=job_key,
            checkpoint=checkpoint,
            attempt_index=1,
        )
        eras.append(era)
        preds.append(_parse_audit_choice(result["response"]))
        d1_dates.append(row["date"])
        d1_kept.append({c: row[c] for c in MODEL_A_COLUMNS})
    X1 = pd.DataFrame(d1_kept).to_numpy(dtype=float) if d1_kept else np.zeros((0, 17))
    d1_body = d1_from_predictions(eras, preds, X_benchmark=X1, dates=np.asarray(d1_dates))
    if d1_body.get("interpretation") == "WARNING SIGNAL":
        warnings.append("D1_WARNING_SIGNAL")
    report.sections.append(AuditSection(name="D1 era diagnostic", body=d1_body))

    # --- D2 ---
    d2_sample = sample_d2_states(prepared, n=n_d2, seed=99, exclude=dev)
    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"]).dt.date
    d2_join = d2_sample.merge(
        lab.loc[:, ["symbol", "date", "y", "X"]],
        on=["symbol", "date"],
        how="left",
    )
    d2_scores: list[float] = []
    d2_y: list[float] = []
    d2_dates: list[date] = []
    d2_kept: list[dict[str, Any]] = []
    for _, row in d2_join.iterrows():
        if not np.isfinite(float(row.get("y", np.nan))):
            continue
        state = dict(row["state_fine"])
        privacy_states.append(state)
        job_key = f"d2|{_row_key(str(row['symbol']), row['date'])}"
        result = _load_or_infer(
            client=client,
            cache=cache,
            qset=audit_qset,
            state=state,
            route=route,
            model_id=model_id,
            purpose="audit",
            state_schema_version=state_schema_version,
            budget=budget,
            stats=stats,
            section="d2",
            job_key=job_key,
            checkpoint=checkpoint,
            attempt_index=1,
        )
        d2_scores.append(_parse_audit_noul(result["response"]))
        d2_y.append(float(row["y"]))
        d2_dates.append(row["date"])
        d2_kept.append({c: row[c] for c in MODEL_A_COLUMNS})
    X2 = pd.DataFrame(d2_kept).to_numpy(dtype=float) if d2_kept else np.zeros((0, 17))
    d2_body = d2_from_scores(
        np.asarray(d2_y),
        np.asarray(d2_scores),
        np.asarray(d2_dates),
        X_benchmark=X2 if len(d2_y) else None,
    )
    if "WARNING SIGNAL" in str(d2_body.get("interpretation", "")):
        warnings.append("D2_WARNING_SIGNAL")
    report.sections.append(AuditSection(name="D2 outcome diagnostic", body=d2_body))

    # --- D3 deferred ---
    report.sections.append(
        AuditSection(
            name="D3 era-stability",
            body={
                "warning_banner": WARNING_BANNER,
                "contamination_label": CONTAMINATION_LABEL,
                "status": "DEFERRED UNTIL AFTER PHASE 7",
                "reason": (
                    "D3 requires frozen-qset features over 2016-2023 from Phase 7 "
                    "bulk inference; those artifacts do not exist yet."
                ),
                "scaffold": d3_rank_ic_by_era(
                    pd.DataFrame(
                        {
                            "date": [date(2017, 1, 3)],
                            "j_dislocation": [float("nan")],
                            "X": [float("nan")],
                        }
                    ),
                    score_cols=["j_dislocation"],
                ),
            },
        )
    )

    # --- D5 fine vs coarse (qset_v1) ---
    d5_sample = sample_d5_states(prepared, n=n_d5, seed=123)
    d5_join = d5_sample.merge(
        lab.loc[:, ["symbol", "date", "y", "X"]],
        on=["symbol", "date"],
        how="left",
    )
    fine_scores: list[float] = []
    coarse_scores: list[float] = []
    d5_y: list[float] = []
    for _, row in d5_join.iterrows():
        if not np.isfinite(float(row.get("y", np.nan))):
            continue
        y_val = float(row["y"])
        for kind, state_col in (("fine", "state_fine"), ("coarse", "state_coarse")):
            state = dict(row[state_col])
            privacy_states.append(state)
            job_key = f"d5|{kind}|{_row_key(str(row['symbol']), row['date'])}"
            result = _load_or_infer(
                client=client,
                cache=cache,
                qset=research_qset,
                state=state,
                route=route,
                model_id=model_id,
                purpose="research",
                state_schema_version=(state_schema_version if kind == "fine" else "state_v1c"),
                budget=budget,
                stats=stats,
                section=f"d5_{kind}",
                job_key=job_key,
                checkpoint=checkpoint,
                attempt_index=1,
            )
            feats = parse_response(SystemOneResponse.model_validate(result["response"]))
            score = float(feats["j_dislocation"])
            if kind == "fine":
                fine_scores.append(score)
            else:
                coarse_scores.append(score)
        d5_y.append(y_val)
    d5_body = d5_fine_vs_coarse(
        np.asarray(fine_scores),
        np.asarray(coarse_scores),
        np.asarray(d5_y, dtype=int),
    )
    # Warning if fine has relationship and coarse does not (fingerprint dependence).
    fine_ic = d5_body.get("fine_rank_ic", float("nan"))
    coarse_ic = d5_body.get("coarse_rank_ic", float("nan"))
    delta = d5_body.get("delta_ic_fine_minus_coarse", float("nan"))
    d5_warn = (
        np.isfinite(fine_ic)
        and np.isfinite(coarse_ic)
        and abs(float(fine_ic)) > 0.05
        and abs(float(coarse_ic)) < 0.02
        and float(delta) > 0.05
    )
    d5_body["interpretation"] = "WARNING SIGNAL" if d5_warn else "NOT A WARNING SIGNAL"
    d5_body["n"] = len(d5_y)
    if d5_warn:
        warnings.append("D5_WARNING_SIGNAL")
    report.sections.append(AuditSection(name="D5 granularity diagnostic", body=d5_body))

    # --- D6 ---
    cutoff_status = {
        "warning_banner": WARNING_BANNER,
        "path": str(jev_cutoff_path) if jev_cutoff_path else "docs/jev_cutoff.md",
        "cutoff_known": False,
        "market_history_included": "unknown",
        "status": "cutoff remains undocumented",
        "contamination_label": CONTAMINATION_LABEL,
    }
    if jev_cutoff_path is not None and Path(jev_cutoff_path).is_file():
        text = Path(jev_cutoff_path).read_text(encoding="utf-8")
        cutoff_status["file_excerpt_status_lines"] = [
            ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")
        ]
        if "Cutoff known?: unknown" in text or "Cutoff known?:unknown" in text.replace(" ", ""):
            cutoff_status["cutoff_known"] = False
        cutoff_status["query_log_empty"] = "| | | | |" in text or text.count("| ----") >= 1
    report.sections.append(AuditSection(name="D6 vendor cutoff", body=cutoff_status))

    # --- Privacy / sealed ---
    privacy = _privacy_scan(privacy_states)
    sealed_access_count = 0
    if registry_path is not None and Path(registry_path).is_file():
        try:
            with sqlite3.connect(registry_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM sealed_access_log").fetchone()
                sealed_access_count = int(row[0]) if row else 0
        except sqlite3.Error:
            sealed_access_count = 0
    hist_acceptance_exists = (
        out_dir.parent / "acceptance" / "HISTORICAL_ACCEPTANCE.json"
    ).is_file()
    privacy_body = {
        "warning_banner": WARNING_BANNER,
        "privacy": privacy,
        "sealed_access_log_count": sealed_access_count,
        "historical_acceptance_exists": hist_acceptance_exists,
        "max_feature_date": str(prepared["date"].max()),
        "sealed_untouched": sealed_access_count == 0 and not hist_acceptance_exists,
    }
    if not privacy["all_passed"]:
        warnings.append("privacy_validation_failures")
    report.sections.append(AuditSection(name="Privacy / sealed check", body=privacy_body))

    report.sections.append(
        AuditSection(
            name="API usage / cost",
            body={
                "warning_banner": WARNING_BANNER,
                **stats.to_dict(),
                "budget_spent_usd": budget.spent_usd,
                "budget_cap_usd": budget_usd,
                "model_requested": model_id,
                "resolved_model_consistent": stats.resolved_models == {model_id}
                or (not stats.resolved_models and stats.successful == 0),
            },
        )
    )
    report.sections.append(
        AuditSection(
            name="Warnings observed",
            body={
                "warning_banner": WARNING_BANNER,
                "warnings": warnings,
                "degeneracy_triggered": bool(deg_flags["flags"]),
                "determinism_mitigation_triggered": bool(det_body.get("mitigation_triggered")),
            },
        )
    )

    report.meta["api_calls"] = stats.successful
    report.meta["call_stats"] = stats.to_dict()
    report.meta["warnings"] = warnings
    report.meta["spent_usd"] = budget.spent_usd

    # Silence unused import lint for STATE map (documentation of parity).
    _ = STATE_V1_SOURCE_COLUMNS

    write_audit_report(report, out_dir)
    # Also dump raw attempt frames for determinism (attempts stored separately).
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, df in enumerate(attempt_frames, start=1):
        df.to_parquet(raw_dir / f"determinism_attempt_{i}.parquet", index=False)
    (raw_dir / "call_stats.json").write_text(
        json.dumps(stats.to_dict(), indent=2), encoding="utf-8"
    )
    return report


def update_disposition_factual(
    path: Path,
    *,
    report_hash: str,
    audit_report_path: str,
    audit_qset_version: str,
    audit_qset_hash: str,
    research_qset_hash: str,
    live_calls: bool,
    warnings: list[str],
    diagnostics: list[str],
) -> None:
    """Fill only factual template fields; never write the human decision."""
    text = path.read_text(encoding="utf-8")
    # Keep decision checkboxes untouched.
    if (
        "- [x] `proceed`" in text
        or "- [x] `stop`" in text
        or "- [x] `proceed_with_caveats`" in text
    ):
        raise RuntimeError("refusing to modify a disposition that already has a checked decision")

    factual = f"""# Contamination disposition (HUMAN COMPLETE — do not auto-fill)

Diagnostics under MASTER_SPEC §11.4 produce **warning signals, not proof of
memorization or of its absence**. Every historical Jev result remains
**potentially training-contaminated** regardless of this disposition.

A `stop` decision blocks the project. A `proceed` / `proceed_with_caveats`
decision means only that work may continue; it does **not** clear contamination.

---

## Diagnostics run

| Item | Value (fill in) |
| ---- | --------------- |
| AUDIT_REPORT path | `{audit_report_path}` |
| AUDIT_REPORT hash | `{report_hash}` |
| audit_qset version/hash | `{audit_qset_version}` / `{audit_qset_hash}` |
| qset_v1 version/hash (if frozen) | not frozen; content hash `{research_qset_hash}` |
| Date range used | sessions ≤ 2023-12-29 only (sealed untouched) |
| Live calls made? | {"yes" if live_calls else "no"} |
| Diagnostics executed | {", ".join(diagnostics)} |

## Warnings observed

_List any D1/D2/D5 (and other) warnings. Free text._

{chr(10).join(f"- {w}" for w in warnings) if warnings else "- (none auto-detected; review report)"}

## Decision

Choose **exactly one** (human only; tooling must never write this field):

- [ ] `proceed`
- [ ] `proceed_with_caveats`
- [ ] `stop`

## Reviewer

- Name:
- Date (YYYY-MM-DD):
- Notes:
"""
    path.write_text(factual, encoding="utf-8")
