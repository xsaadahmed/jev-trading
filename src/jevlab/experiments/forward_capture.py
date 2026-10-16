"""Forward INPUT capture — shadow logging, no orders (MASTER_SPEC Phase 13A).

After each completed NYSE session at ≥ 18:30 ET: download bars/VIX (with
``available_at`` / ``retrieved_at``), build features via ``as_of(decision_time)``,
build ``state_v1``, validate privacy, call Jev with the pinned model id, and
store inputs under ``data/forward/``.

This module:
- places **no** orders
- imports **no** ``jevlab.brokers`` code
- does **not** score production/baseline models
- does **not** join labels

Jev failures and model-id mismatches mark the day ``incomplete`` (never imputed).
The first successful complete capture writes the immutable ``forward_registry`` row.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd

from jevlab.config.loader import ResearchConfigs, load_all_configs
from jevlab.data.availability import ET, decision_time
from jevlab.data.downloader import MarketDataDownloader, default_store
from jevlab.experiments.forward_registry import (
    ForwardRegistryRow,
    read_deployment,
    register_deployment,
)
from jevlab.features.columns import FEATURE_VERSION
from jevlab.features.pipeline import (
    build_features_for_session,
    read_feature_store,
    write_feature_store,
)
from jevlab.jev.cache import request_hash
from jevlab.jev.client import JevClient
from jevlab.jev.privacy import assert_state_private, validate_state_privacy
from jevlab.jev.qset import QuestionSet, load_qset
from jevlab.jev.schemas import SystemOneResponse
from jevlab.jev.state_builder import build_states_frame
from jevlab.jev.vocabulary import CLOSED_VOCABULARY, SCHEMA_VALUE
from jevlab.utils.calendar import is_nyse_session, previous_nyse_session
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger
from jevlab.utils.paths import ProjectPaths, get_project_root

log = get_logger(__name__)

CaptureStatus = Literal[
    "complete",
    "incomplete",
    "skipped_idempotent",
    "skipped_holiday",
    "too_early",
]

PURPOSE_FORWARD: str = "forward"


class InferClient(Protocol):
    """Minimal Jev client surface used by forward capture (no broker types)."""

    def infer(
        self,
        state: Mapping[str, Any] | str | list[Any],
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...


class ModelResolvedMismatchError(RuntimeError):
    """Resolved Jev model id differs from the pinned research id."""


@dataclass
class CaptureResult:
    """Outcome of one ``forward-capture`` invocation."""

    status: CaptureStatus
    session_date: date | None
    decision_time: str | None = None
    message: str = ""
    n_symbols: int = 0
    n_jev_ok: int = 0
    n_jev_failed: int = 0
    model_requested: str | None = None
    model_resolved: str | None = None
    registry_written: bool = False
    forward_start_date: date | None = None
    deployment_timestamp: str | None = None
    output_dir: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.session_date is not None:
            payload["session_date"] = self.session_date.isoformat()
        if self.forward_start_date is not None:
            payload["forward_start_date"] = self.forward_start_date.isoformat()
        return payload


def session_dir(forward_root: Path, session: date) -> Path:
    """``data/forward/YYYY-MM-DD/``."""
    return Path(forward_root) / session.isoformat()


def read_session_status(forward_root: Path, session: date) -> dict[str, Any] | None:
    path = session_dir(forward_root, session) / "status.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    return raw


def is_session_complete(forward_root: Path, session: date) -> bool:
    status = read_session_status(forward_root, session)
    return status is not None and status.get("status") == "complete"


def write_session_status(forward_root: Path, session: date, payload: Mapping[str, Any]) -> Path:
    out = session_dir(forward_root, session)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "status.json"
    path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    return path


def resolve_capture_session(
    now: datetime,
    *,
    session: date | None = None,
) -> tuple[CaptureStatus | None, date | None, str]:
    """Resolve the NYSE session to capture given wall-clock ``now``.

    With an explicit ``session``, refuse holidays and refuse before that
    session's 18:30 ET decision time.

    With ``session=None`` (scheduler / default ops), return the **latest
    completed** NYSE session — the most recent session whose decision time
    is ≤ ``now``. After midnight ET this is yesterday's session, not the
    incomplete new calendar date. Holidays/weekends fall back to the prior
    completed session rather than no-op skipping.

    Returns ``(early_status, session, message)``. If ``early_status`` is set,
    the caller should return immediately without downloading.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_et = now.astimezone(ET)

    if session is not None:
        if not is_nyse_session(session):
            return "skipped_holiday", session, f"{session.isoformat()} is not an NYSE session"
        dt = decision_time(session)
        if now_et < dt.to_pydatetime():
            return "too_early", session, f"before decision_time {dt.isoformat()}"
        return None, session, "ok"

    # Latest completed: walk back from today until decision_time <= now.
    today = now_et.date()
    candidate = today if is_nyse_session(today) else previous_nyse_session(today)
    for _ in range(10):
        dt = decision_time(candidate)
        if now_et >= dt.to_pydatetime():
            return None, candidate, "ok"
        candidate = previous_nyse_session(candidate)
    return (
        "too_early",
        today,
        f"no completed NYSE session with decision_time ≤ {now_et.isoformat()}",
    )


def _state_hash(state: Mapping[str, str]) -> str:
    return sha256_of(canonical_json(dict(state)))


def _call_jev_for_states(
    states: pd.DataFrame,
    *,
    client: InferClient,
    qset: QuestionSet,
    route: str,
    model_id: str,
    state_schema_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, bool]:
    """Call Jev per state; never impute.

    Returns ``(ok_rows, fail_rows, resolved_or_none, model_mismatch)``.

    On model-id mismatch the Jev part aborts after recording the mismatch for
    that symbol; remaining symbols are not called (never imputed).
    """
    questions = qset.as_request_questions()
    cacheable = qset.to_cacheable_dict()
    ok_rows: list[dict[str, Any]] = []
    fail_rows: list[dict[str, Any]] = []
    resolved_seen: str | None = None

    for row in states.itertuples(index=False):
        raw_state = row.state
        if not isinstance(raw_state, dict):
            fail_rows.append(
                {
                    "symbol": str(row.symbol),
                    "date": str(row.date),
                    "error": f"state must be dict, got {type(raw_state)!r}",
                }
            )
            continue
        state = {str(k): str(v) for k, v in raw_state.items()}
        violations = validate_state_privacy(state, CLOSED_VOCABULARY)
        if violations:
            fail_rows.append(
                {
                    "symbol": str(row.symbol),
                    "date": str(row.date),
                    "error": f"privacy violations: {[v.code for v in violations]}",
                }
            )
            continue
        assert_state_private(state, CLOSED_VOCABULARY)
        rh = request_hash(
            route=route,
            model_requested=model_id,
            state=state,
            questions=cacheable,
        )
        try:
            response = client.infer(state, questions, model=model_id)
        except Exception as exc:
            fail_rows.append(
                {
                    "symbol": str(row.symbol),
                    "date": str(row.date),
                    "request_hash": rh,
                    "error": str(exc),
                }
            )
            continue

        resolved = str(response.model)
        if resolved != model_id:
            fail_rows.append(
                {
                    "symbol": str(row.symbol),
                    "date": str(row.date),
                    "request_hash": rh,
                    "model_resolved": resolved,
                    "error": f"model_resolved {resolved!r} != pinned {model_id!r}",
                }
            )
            # Abort the Jev part for the remainder of the day.
            return ok_rows, fail_rows, resolved, True

        resolved_seen = resolved
        payload = response.model_dump(mode="json")
        ok_rows.append(
            {
                "symbol": str(row.symbol),
                "date": str(row.date),
                "request_hash": rh,
                "state_hash": _state_hash(state),
                "model_requested": model_id,
                "model_resolved": resolved,
                "state_schema_version": state_schema_version,
                "qset_version": qset.qset_version,
                "purpose": PURPOSE_FORWARD,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "response": payload,
            }
        )
    return ok_rows, fail_rows, resolved_seen, False


def _write_capture_artifacts(
    out_dir: Path,
    *,
    features: pd.DataFrame,
    states: pd.DataFrame,
    jev_ok: Sequence[Mapping[str, Any]],
    jev_fail: Sequence[Mapping[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "features.parquet", index=False)

    state_rows: list[dict[str, Any]] = []
    for row in states.itertuples(index=False):
        st = row.state
        if not isinstance(st, dict):
            continue
        state_rows.append(
            {
                "symbol": str(row.symbol),
                "date": str(row.date),
                "state_schema_version": SCHEMA_VALUE,
                "state_hash": _state_hash({str(k): str(v) for k, v in st.items()}),
                "state": st,
            }
        )
    (out_dir / "states.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in state_rows)
        + ("\n" if state_rows else ""),
        encoding="utf-8",
    )

    raw_path = out_dir / "jev_responses.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for rec in jev_ok:
            fh.write(json.dumps(dict(rec), ensure_ascii=False, separators=(",", ":")) + "\n")

    fail_path = out_dir / "jev_failures.jsonl"
    with fail_path.open("w", encoding="utf-8") as fh:
        for rec in jev_fail:
            fh.write(json.dumps(dict(rec), ensure_ascii=False, separators=(",", ":")) + "\n")


def run_forward_capture(
    *,
    paths: ProjectPaths | None = None,
    configs: ResearchConfigs | None = None,
    now: datetime | None = None,
    session: date | None = None,
    client: InferClient | None = None,
    downloader: MarketDataDownloader | None = None,
    download: bool = True,
    bars: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    feature_history: pd.DataFrame | None = None,
    session_features: pd.DataFrame | None = None,
    qset: QuestionSet | None = None,
    register: bool = True,
    git_commit: str | None = None,
    deployment_timestamp: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CaptureResult:
    """Run one forward INPUT capture for the latest (or given) completed session.

    Inject ``bars``/``vix``/``client``/``feature_history`` for offline tests.
    When ``download`` is True and frames are omitted, Tiingo+FRED are used
    (Alpaca cross-check disabled; ``jevlab.brokers`` is never imported).
    """
    root_paths = paths or ProjectPaths.from_root(get_project_root())
    root_paths.ensure_runtime_dirs()
    cfgs = configs or load_all_configs(root_paths.config)
    forward_root = root_paths.data_forward
    registry_path = root_paths.registry / "experiments.sqlite"

    wall = now or (clock() if clock is not None else datetime.now(UTC))
    if wall.tzinfo is None:
        wall = wall.replace(tzinfo=UTC)

    early, target, msg = resolve_capture_session(wall, session=session)
    if early is not None:
        return CaptureResult(status=early, session_date=target, message=msg)

    assert target is not None
    dt = decision_time(target)
    model_id = cfgs.jev.model_id

    if is_session_complete(forward_root, target):
        return CaptureResult(
            status="skipped_idempotent",
            session_date=target,
            decision_time=dt.isoformat(),
            message=f"capture already complete for {target.isoformat()}",
            model_requested=model_id,
            output_dir=str(session_dir(forward_root, target)),
        )

    # --- market data ---
    store = default_store(root_paths, cfgs.data)
    if download and (bars is None or vix is None):
        dl = downloader or MarketDataDownloader.from_config(
            paths=root_paths,
            store=store,
            include_cross_check=False,
            require_secrets=True,
        )
        # Refresh a warm lookback ending at the capture session (no future).
        start = date.fromisoformat(cfgs.data.start_date)
        for sym in cfgs.universe.symbols:
            dl.download_symbol_bars(dl.primary, sym, start, target, force=False)
        dl.download_vix(start, target, force=False)
        bars = store.read_bars(vendor=cfgs.data.primary_vendor)
        vix = store.read_external(series_id=cfgs.data.vix_series_id)

    symbols = list(cfgs.universe.symbols)
    if session_features is None:
        if bars is None or vix is None:
            return CaptureResult(
                status="incomplete",
                session_date=target,
                decision_time=dt.isoformat(),
                message="bars/vix unavailable",
                model_requested=model_id,
            )
        session_features = build_features_for_session(
            bars,
            vix,
            target,
            symbols=symbols,
            spy_symbol="SPY",
            feature_version=FEATURE_VERSION,
        )
    if session_features.empty:
        out = session_dir(forward_root, target)
        write_session_status(
            forward_root,
            target,
            {
                "status": "incomplete",
                "session_date": target.isoformat(),
                "decision_time": dt.isoformat(),
                "message": "no features built for session",
                "model_requested": model_id,
            },
        )
        return CaptureResult(
            status="incomplete",
            session_date=target,
            decision_time=dt.isoformat(),
            message="no features built for session",
            model_requested=model_id,
            output_dir=str(out),
        )

    # Trailing history for state percentiles (prefer injected / on-disk store).
    if feature_history is None:
        hist = read_feature_store(root_paths.data_features / FEATURE_VERSION)
    else:
        hist = feature_history.copy()
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
        hist = hist.loc[hist["date"] < target]
    combined = (
        pd.concat([hist, session_features], ignore_index=True)
        if not hist.empty
        else session_features.copy()
    )
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Persist session features into the main feature store (idempotent overwrite of partition).
    write_feature_store(session_features, root_paths.data_features / FEATURE_VERSION)

    edges = tuple(float(x) for x in cfgs.jev.bucket_edges)
    states_all = build_states_frame(combined, edges=edges)
    if states_all.empty or "date" not in states_all.columns:
        states = pd.DataFrame(columns=["symbol", "date", "state"])
    else:
        states = states_all.loc[pd.to_datetime(states_all["date"]).dt.date == target].reset_index(
            drop=True
        )
    if states.empty:
        out = session_dir(forward_root, target)
        _write_capture_artifacts(
            out, features=session_features, states=states, jev_ok=[], jev_fail=[]
        )
        write_session_status(
            forward_root,
            target,
            {
                "status": "incomplete",
                "session_date": target.isoformat(),
                "decision_time": dt.isoformat(),
                "message": "no states (warm-up / NaN percentiles)",
                "model_requested": model_id,
                "n_symbols": len(session_features),
            },
        )
        return CaptureResult(
            status="incomplete",
            session_date=target,
            decision_time=dt.isoformat(),
            message="no states (warm-up / NaN percentiles)",
            model_requested=model_id,
            n_symbols=len(session_features),
            output_dir=str(out),
        )

    if qset is None:
        qpath = Path(cfgs.jev.qset_path)
        if not qpath.is_file():
            qpath = root_paths.root / cfgs.jev.qset_path
        qset_obj = load_qset(
            qpath,
            expected_hash=cfgs.jev.qset_hash,
            expected_purpose="research",
        )
    else:
        qset_obj = qset

    jev_client: InferClient
    jev_client = client if client is not None else JevClient(config=cfgs.jev)

    jev_ok, jev_fail, resolved, mismatch = _call_jev_for_states(
        states,
        client=jev_client,
        qset=qset_obj,
        route=cfgs.jev.route,
        model_id=model_id,
        state_schema_version=cfgs.jev.state_schema_version,
    )
    if mismatch:
        log.error(
            "forward_capture.model_mismatch",
            model_resolved=resolved,
            pinned=model_id,
            session=target.isoformat(),
        )

    out = session_dir(forward_root, target)
    _write_capture_artifacts(
        out,
        features=session_features,
        states=states,
        jev_ok=jev_ok,
        jev_fail=jev_fail,
    )

    n_ok = len(jev_ok)
    n_fail = len(jev_fail)
    n_sym = len(states)
    complete = (not mismatch) and n_fail == 0 and n_ok == n_sym and n_sym > 0

    status_payload: dict[str, Any] = {
        "status": "complete" if complete else "incomplete",
        "session_date": target.isoformat(),
        "decision_time": dt.isoformat(),
        "captured_at": datetime.now(UTC).isoformat(),
        "model_requested": model_id,
        "model_resolved": resolved,
        "feature_version": FEATURE_VERSION,
        "state_schema_version": cfgs.jev.state_schema_version,
        "qset_version": qset_obj.qset_version,
        "qset_hash": qset_obj.content_hash,
        "n_symbols": n_sym,
        "n_jev_ok": n_ok,
        "n_jev_failed": n_fail,
        "model_mismatch": mismatch,
        "purpose": PURPOSE_FORWARD,
        "notes": (
            "Forward INPUT capture only. No orders. No model scoring. No labels. "
            "Forward evidence is valid only for the pinned Jev version that produced it. "
            "Historical Jev results remain potentially training-contaminated."
        ),
    }
    write_session_status(forward_root, target, status_payload)

    registry_row: ForwardRegistryRow | None = None
    registry_written = False
    if complete and register:
        existing = read_deployment(registry_path)
        if existing is None:
            registry_row = register_deployment(
                registry_path=registry_path,
                jev_model_id=model_id,
                qset_hash=qset_obj.content_hash,
                feature_version=FEATURE_VERSION,
                state_schema_version=cfgs.jev.state_schema_version,
                git_commit=git_commit,
                deployment_timestamp=deployment_timestamp,
                project_root=root_paths.root,
            )
            registry_written = True
            log.info(
                "forward_registry.written",
                registry_id=registry_row.registry_id,
                forward_start_date=registry_row.forward_start_date.isoformat(),
                deployment_timestamp=registry_row.deployment_timestamp.isoformat(),
            )
        else:
            registry_row = existing

    message = (
        "ok"
        if complete
        else (
            "model_resolved mismatch; Jev part aborted"
            if mismatch
            else f"Jev incomplete: ok={n_ok} failed={n_fail} of {n_sym} (never imputed)"
        )
    )
    return CaptureResult(
        status="complete" if complete else "incomplete",
        session_date=target,
        decision_time=dt.isoformat(),
        message=message,
        n_symbols=n_sym,
        n_jev_ok=n_ok,
        n_jev_failed=n_fail,
        model_requested=model_id,
        model_resolved=resolved,
        registry_written=registry_written,
        forward_start_date=None if registry_row is None else registry_row.forward_start_date,
        deployment_timestamp=(
            None if registry_row is None else registry_row.deployment_timestamp.isoformat()
        ),
        output_dir=str(out),
        meta={"model_mismatch": mismatch, "qset_hash": qset_obj.content_hash},
    )


# Re-export for callers that previously used batch_runner's helper via capture.
latest_completed_session_before = previous_nyse_session

__all__ = [
    "CaptureResult",
    "InferClient",
    "ModelResolvedMismatchError",
    "is_session_complete",
    "read_session_status",
    "resolve_capture_session",
    "run_forward_capture",
    "session_dir",
]
