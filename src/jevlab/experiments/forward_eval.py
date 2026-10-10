"""Forward scoring, label join, reports, and SCALE evaluator (Phase 13B).

No orders. No ``jevlab.brokers`` imports. ``SCALE_ACCEPTANCE.json`` does not
raise ``TINY_LIVE_CEILING_USD``; V1 live capital remains capped at $10.00.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from jevlab.constants import TINY_LIVE_CEILING_USD
from jevlab.evaluation.metrics import max_drawdown_dec
from jevlab.evaluation.predictive import (
    binary_log_loss,
    daily_spearman_ic_series,
)
from jevlab.evaluation.tests import forward_consistency_z
from jevlab.experiments.artifacts import (
    AcceptanceArtifact,
    ArtifactError,
    verify_acceptance,
    write_acceptance,
)
from jevlab.experiments.forward_capture import (
    is_session_complete,
    session_dir,
)
from jevlab.experiments.production_models import (
    list_model_artifacts,
    load_production_pipeline,
    production_hashes_path,
    read_production_hashes,
)
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import MODEL_B_COLUMNS
from jevlab.signals.selection import select_top_k
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger
from jevlab.utils.registry import init_registry

log = get_logger(__name__)

FIRST_LABEL_JOIN_FILENAME: str = "first_forward_label_join.json"
CHECKPOINT_60_FILENAME: str = "forward_checkpoint_60.json"
SCALE_GATE_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
SCALE_POWER_STATEMENT: str = (
    "120 sessions is weak statistical evidence; SCALE_READY means "
    "'nothing contradicts the historical result and nothing broke,' "
    "not 'the edge is proven.'"
)
# Documented invariant: SCALE_READY never raises the V1 capital ceiling.
SCALE_DOES_NOT_RAISE_CEILING: str = (
    f"SCALE_ACCEPTANCE does not modify TINY_LIVE_CEILING_USD "
    f"(remains {TINY_LIVE_CEILING_USD:.2f}); V1 live code cannot exceed $10.00."
)

COMPLETED_LEDGER_STATUSES: frozenset[str] = frozenset(
    {"completed", "filled", "closed", "completed_round_trip"}
)


class ForwardLabelJoinError(ArtifactError):
    """Raised when forward-label join is refused by the §10.7 hash guard."""


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    detail: str


@dataclass
class ScaleGateMetrics:
    """Fixture-friendly inputs for S1-S8 (Sec 16.3)."""

    tiny_live_granted: bool = True
    n_completed_live_round_trips: int = 20
    n_forward_sessions_labeled: int = 120
    delta_ic: float = 0.02
    delta_log_loss: float = -0.01
    ic_hist: float = 0.03
    se_fwd: float = 0.02
    ks_jev_max: float = 0.10
    model_id_constant: bool = True
    fraction_missing_jev: float = 0.01
    cum_net_return_b: float = 0.05
    cum_net_return_a: float = 0.04
    max_drawdown_b: float = 0.08
    mean_realized_cost_bps: float = 4.0
    normal_cost_bps_assumption: float = 2.5
    unreconciled_orders: int = 0
    risk_limit_breaches: int = 0
    incidents: int = 0
    human_signoff: bool = True
    jev_version_changed: bool = False

    def with_overrides(self, **kwargs: Any) -> ScaleGateMetrics:
        return replace(self, **kwargs)


@dataclass
class ScaleEvalResult:
    gate_report: dict[str, Any]
    acceptance: AcceptanceArtifact | None
    gate_report_path: Path
    rejection_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.acceptance is not None


@dataclass
class ForwardScoreResult:
    session_date: date
    n_scored: int
    artifact_ids: dict[str, str]
    scores_path: Path
    selected_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ForwardReportResult:
    n_sessions_scored: int
    n_sessions_labeled: int
    rolling: dict[str, Any]
    ks_drift: dict[str, float]
    checkpoint_60_path: Path | None
    report_path: Path
    labels_joined: bool


def first_label_join_path(forward_root: Path) -> Path:
    return Path(forward_root) / FIRST_LABEL_JOIN_FILENAME


def read_first_label_join(forward_root: Path) -> dict[str, Any] | None:
    path = first_label_join_path(forward_root)
    if not path.is_file():
        return None
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else None


def assert_can_join_forward_labels(forward_root: Path) -> dict[str, Any]:
    """Refuse label join if production hashes are absent or newer than first join.

    MASTER_SPEC §10.7 / Phase 13B guard.
    """
    hashes = read_production_hashes(production_hashes_path(forward_root))
    if hashes is None:
        raise ForwardLabelJoinError("production model hashes absent; refuse forward-label join")
    first = read_first_label_join(forward_root)
    if first is not None:
        hashes_at = datetime.fromisoformat(str(hashes["recorded_at"]))
        if hashes_at.tzinfo is None:
            hashes_at = hashes_at.replace(tzinfo=UTC)
        first_at = datetime.fromisoformat(str(first["joined_at"]))
        if first_at.tzinfo is None:
            first_at = first_at.replace(tzinfo=UTC)
        if hashes_at > first_at:
            raise ForwardLabelJoinError("production hashes newer than the first label join; refuse")
    return hashes


def mark_first_label_join(
    forward_root: Path,
    *,
    hashes: Mapping[str, Any],
    joined_at: datetime | None = None,
) -> Path:
    """Record the first successful label-join timestamp (idempotent)."""
    path = first_label_join_path(forward_root)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joined_at": (joined_at or datetime.now(UTC)).isoformat(),
        "production_artifact_ids": dict(hashes.get("artifact_ids", {})),
        "production_content_digests": dict(hashes.get("content_digests", {})),
        "hashes_recorded_at": hashes.get("recorded_at"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _upsert_forward_score(
    registry_path: Path,
    *,
    session_date: date,
    symbol: str,
    model_artifact_id: str,
    score: float,
    selected: int,
    jev_inference_ids: str | None = None,
) -> None:
    engine = init_registry(registry_path)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO forward_scores (
                    session_date, symbol, model_artifact_id, score, selected,
                    jev_inference_ids
                ) VALUES (
                    :session_date, :symbol, :model_artifact_id, :score, :selected,
                    :jev_inference_ids
                )
                ON CONFLICT(session_date, symbol, model_artifact_id) DO UPDATE SET
                    score = excluded.score,
                    selected = excluded.selected,
                    jev_inference_ids = excluded.jev_inference_ids
                """
            ),
            {
                "session_date": session_date.isoformat(),
                "symbol": symbol,
                "model_artifact_id": model_artifact_id,
                "score": float(score),
                "selected": int(selected),
                "jev_inference_ids": jev_inference_ids,
            },
        )


def score_forward_frame(
    frame: pd.DataFrame,
    *,
    registry_path: Path,
    artifact_dir: Path,
    forward_root: Path,
    session_date: date | None = None,
    top_k: int = 3,
    cost_bps_per_side: float = 2.5,
    sell_fee_bps: float = 0.3,
) -> ForwardScoreResult:
    """Score one session (or multi-date frame) with production A and B.

    Stores scores + top-K selections in ``forward_scores`` and under
    ``data/forward/scores/``. Hypothetical trade notionals use normal costs.
    """
    prod = list_model_artifacts(registry_path, kind="production_forward")
    if set(prod) != {"A", "B"}:
        raise ArtifactError("production_forward artifacts A and B required for scoring")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    if session_date is not None:
        work = work.loc[work["date"] == session_date].reset_index(drop=True)
    if work.empty:
        raise ArtifactError("no rows to score")
    sessions = sorted(work["date"].unique().tolist())
    if len(sessions) != 1 and session_date is None:
        # Score each session separately for clarity
        last: ForwardScoreResult | None = None
        for d in sessions:
            last = score_forward_frame(
                work,
                registry_path=registry_path,
                artifact_dir=artifact_dir,
                forward_root=forward_root,
                session_date=d,
                top_k=top_k,
                cost_bps_per_side=cost_bps_per_side,
                sell_fee_bps=sell_fee_bps,
            )
        assert last is not None
        return last

    day = sessions[0]
    if "eligible" not in work.columns:
        work["eligible"] = 1

    artifact_ids = {m: prod[m]["model_artifact_id"] for m in ("A", "B")}
    pipes = {m: load_production_pipeline(artifact_dir, m, artifact_ids[m]) for m in ("A", "B")}
    colmap = {"A": list(MODEL_A_COLUMNS), "B": list(MODEL_B_COLUMNS)}
    scored_parts: list[pd.DataFrame] = []
    selected_counts: dict[str, int] = {}

    for mid in ("A", "B"):
        cols = colmap[mid]
        missing = [c for c in cols if c not in work.columns]
        if missing:
            raise ArtifactError(f"score frame missing columns for {mid}: {missing}")
        X = work.loc[:, cols].to_numpy(dtype=float)
        prob = pipes[mid].predict_proba(X)[:, 1]
        part = work.loc[:, ["symbol", "date", "eligible"]].copy()
        part["score"] = prob
        part["model_id"] = mid
        part["model_artifact_id"] = artifact_ids[mid]
        selected = select_top_k(part, k=top_k, score_col="score")
        selected_counts[mid] = int(selected["selected"].sum())
        for rec in selected.to_dict(orient="records"):
            _upsert_forward_score(
                registry_path,
                session_date=day,
                symbol=str(rec["symbol"]),
                model_artifact_id=artifact_ids[mid],
                score=float(rec["score"]),
                selected=int(rec["selected"]),
            )
        scored_parts.append(selected)

    all_scores = pd.concat(scored_parts, ignore_index=True)
    # Hypothetical trades under normal costs (shadow only).
    rt_cost = 2.0 * cost_bps_per_side / 1e4 + sell_fee_bps / 1e4
    trades = all_scores.loc[all_scores["selected"] == 1].copy()
    trades["hypothetical_cost_dec"] = rt_cost
    trades["cost_scenario"] = "normal"

    scores_dir = Path(forward_root) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_path = scores_dir / f"{day.isoformat()}.parquet"
    all_scores.to_parquet(scores_path, index=False)
    trades_path = scores_dir / f"{day.isoformat()}_trades.parquet"
    trades.to_parquet(trades_path, index=False)

    return ForwardScoreResult(
        session_date=day,
        n_scored=len(work),
        artifact_ids=artifact_ids,
        scores_path=scores_path,
        selected_counts=selected_counts,
    )


def join_forward_labels(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    forward_root: Path,
) -> pd.DataFrame:
    """Join realized labels onto forward scores; enforces hash-before-join guard."""
    hashes = assert_can_join_forward_labels(forward_root)
    mark_first_label_join(forward_root, hashes=hashes)

    sc = scores.copy()
    sc["date"] = pd.to_datetime(sc["date"]).dt.date
    lb = labels.copy()
    lb["date"] = pd.to_datetime(lb["date"]).dt.date
    # Only join rows whose exit window is complete (label non-null).
    lb = lb.dropna(subset=["y"])
    merged = sc.merge(
        lb.loc[:, ["symbol", "date", "y", "R", "Rbar", "y_cost", "exit_date"]],
        on=["symbol", "date"],
        how="inner",
    )
    return merged


def ks_jev_drift(
    baseline: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    columns: Sequence[str] = JEV_FEATURE_COLUMNS,
) -> dict[str, float]:
    """KS statistic per Jev feature: forward vs 2019-2023 baseline."""
    out: dict[str, float] = {}
    for col in columns:
        if col not in baseline.columns or col not in forward.columns:
            out[col] = float("nan")
            continue
        a = baseline[col].dropna().to_numpy(dtype=float)
        b = forward[col].dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            out[col] = float("nan")
            continue
        out[col] = float(stats.ks_2samp(a, b).statistic)
    return out


def rolling_ab_metrics(labeled: pd.DataFrame) -> dict[str, Any]:
    """Rolling A-vs-B IC / log-loss / hypothetical net returns from labeled scores."""
    if labeled.empty:
        return {
            "delta_ic": float("nan"),
            "delta_log_loss": float("nan"),
            "cum_net_return_a": float("nan"),
            "cum_net_return_b": float("nan"),
            "max_drawdown_b": float("nan"),
            "n_dates": 0,
        }
    work = labeled.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date

    def _side(mid: str) -> pd.DataFrame:
        sub = work.loc[work["model_id"] == mid].copy()
        if "prob" not in sub.columns and "score" in sub.columns:
            sub = sub.rename(columns={"score": "prob"})
        return sub

    a = _side("A")
    b = _side("B")
    if "eligible" not in a.columns and not a.empty:
        a = a.assign(eligible=1)
    if "eligible" not in b.columns and not b.empty:
        b = b.assign(eligible=1)
    ic_a = daily_spearman_ic_series(a, score_col="prob", target_col="y")
    ic_b = daily_spearman_ic_series(b, score_col="prob", target_col="y")
    mean_ic_a = float(ic_a.mean()) if len(ic_a) else float("nan")
    mean_ic_b = float(ic_b.mean()) if len(ic_b) else float("nan")
    ll_a = (
        binary_log_loss(a["y"].to_numpy(dtype=float), a["prob"].to_numpy(dtype=float))
        if not a.empty
        else float("nan")
    )
    ll_b = (
        binary_log_loss(b["y"].to_numpy(dtype=float), b["prob"].to_numpy(dtype=float))
        if not b.empty
        else float("nan")
    )

    def _hyp_ret(side: pd.DataFrame) -> tuple[float, float]:
        if side.empty or "R" not in side.columns:
            return float("nan"), float("nan")
        sel = side.loc[side.get("selected", 1) == 1] if "selected" in side.columns else side
        if sel.empty:
            return 0.0, 0.0
        # Equal-weight daily mean of selected R minus cost if present
        daily = sel.groupby("date")["R"].mean()
        if "y_cost" in sel.columns:
            # Approximate: use mean Rbar-style when available
            pass
        if "hypothetical_cost_dec" in sel.columns:
            cost = float(sel["hypothetical_cost_dec"].iloc[0])
            daily = daily - cost
        elif "roundtrip_cost_dec" in sel.columns:
            cost = float(sel["roundtrip_cost_dec"].mean())
            daily = daily - cost
        equity = (1.0 + daily.fillna(0.0)).cumprod()
        cum = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
        dd = abs(max_drawdown_dec(equity)) if len(equity) else 0.0
        return cum, dd

    cum_a, _ = _hyp_ret(a)
    cum_b, dd_b = _hyp_ret(b)
    return {
        "delta_ic": mean_ic_b - mean_ic_a,
        "delta_log_loss": ll_b - ll_a,
        "mean_ic_a": mean_ic_a,
        "mean_ic_b": mean_ic_b,
        "log_loss_a": ll_a,
        "log_loss_b": ll_b,
        "cum_net_return_a": cum_a,
        "cum_net_return_b": cum_b,
        "max_drawdown_b": dd_b,
        "n_dates": int(work["date"].nunique()),
    }


def write_60_session_checkpoint(
    out_dir: Path,
    *,
    metrics: Mapping[str, Any],
    n_sessions: int,
) -> Path:
    """Informational 60-session checkpoint - cannot grant SCALE_READY."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CHECKPOINT_60_FILENAME
    delta_ic = metrics.get("delta_ic")
    human_review = (
        delta_ic is not None
        and isinstance(delta_ic, (int, float))
        and np.isfinite(delta_ic)
        and float(delta_ic) < 0
    )
    payload = {
        "kind": "forward_checkpoint_60",
        "informational_only": True,
        "cannot_grant_status": True,
        "grants_scale_ready": False,
        "n_sessions": n_sessions,
        "metrics": dict(metrics),
        "mandatory_human_review_if_delta_ic_negative": human_review,
        "note": (
            "This checkpoint is informational only and cannot grant SCALE_READY "
            "or any acceptance status (§16.3 S2)."
        ),
        "power_statement": SCALE_POWER_STATEMENT,
        "created_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def run_forward_report(
    *,
    forward_root: Path,
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    baseline_jev: pd.DataFrame,
    forward_jev: pd.DataFrame,
    report_dir: Path,
    write_checkpoint_at: int = 60,
) -> ForwardReportResult:
    """Fill realized labels, rolling A-vs-B tables, KS drift, optional 60-session report."""
    labeled = join_forward_labels(scores, labels, forward_root=forward_root)
    rolling = rolling_ab_metrics(labeled)
    ks = ks_jev_drift(baseline_jev, forward_jev)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    n_labeled = int(labeled["date"].nunique()) if not labeled.empty else 0
    n_scored = int(scores["date"].nunique()) if not scores.empty and "date" in scores.columns else 0

    checkpoint_path: Path | None = None
    if n_labeled >= write_checkpoint_at:
        checkpoint_path = write_60_session_checkpoint(
            report_dir, metrics=rolling, n_sessions=n_labeled
        )

    report = {
        "n_sessions_scored": n_scored,
        "n_sessions_labeled": n_labeled,
        "rolling": rolling,
        "ks_drift": ks,
        "ks_max": max((v for v in ks.values() if np.isfinite(v)), default=float("nan")),
        "checkpoint_60": str(checkpoint_path) if checkpoint_path else None,
        "contamination_note": (
            "Forward sessions post-date deployment; historical Jev results remain "
            "potentially training-contaminated."
        ),
        "created_at": datetime.now(UTC).isoformat(),
    }
    report_path = report_dir / "forward_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not labeled.empty:
        labeled.to_parquet(report_dir / "forward_labeled.parquet", index=False)

    return ForwardReportResult(
        n_sessions_scored=n_scored,
        n_sessions_labeled=n_labeled,
        rolling=rolling,
        ks_drift=ks,
        checkpoint_60_path=checkpoint_path,
        report_path=report_path,
        labels_joined=True,
    )


def count_completed_live_round_trips(registry_path: Path) -> int:
    """Count completed live round trips in ``strategy_ledger`` (S1)."""
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT status, mode FROM strategy_ledger")).fetchall()
    n = 0
    for status, mode in rows:
        st = str(status or "").lower()
        md = str(mode or "").lower()
        if st not in COMPLETED_LEDGER_STATUSES:
            continue
        # Live / tiny_live / unspecified mode (fixtures) count as live trips.
        if md in {"", "live", "tiny_live"}:
            n += 1
    return n


def evaluate_scale_gates(metrics: ScaleGateMetrics) -> list[GateResult]:
    """Evaluate S1-S8 independently; each can block alone."""
    results: list[GateResult] = []

    # S1
    s1_ok = metrics.tiny_live_granted and metrics.n_completed_live_round_trips >= 20
    results.append(
        GateResult(
            "S1",
            s1_ok,
            f"tiny_live={metrics.tiny_live_granted} "
            f"round_trips={metrics.n_completed_live_round_trips}",
        )
    )

    # S2
    s2_ok = metrics.n_forward_sessions_labeled >= 120
    results.append(
        GateResult(
            "S2",
            s2_ok,
            f"labeled_sessions={metrics.n_forward_sessions_labeled} (need >=120); "
            "60-session checkpoint is informational only",
        )
    )

    # S3
    s3_ok = metrics.delta_ic > 0 and metrics.delta_log_loss < 0
    results.append(
        GateResult(
            "S3",
            s3_ok,
            f"delta_ic={metrics.delta_ic} delta_log_loss={metrics.delta_log_loss}",
        )
    )

    # S4
    zinfo = forward_consistency_z(metrics.ic_hist, metrics.delta_ic, metrics.se_fwd)
    s4_ok = bool(zinfo["passes_s4"])
    results.append(GateResult("S4", s4_ok, f"z={zinfo['z']} threshold={zinfo['threshold']}"))

    # S5
    s5_ok = (
        metrics.ks_jev_max < 0.15
        and metrics.model_id_constant
        and metrics.fraction_missing_jev <= 0.02
    )
    results.append(
        GateResult(
            "S5",
            s5_ok,
            f"ks_max={metrics.ks_jev_max} model_constant={metrics.model_id_constant} "
            f"missing_frac={metrics.fraction_missing_jev}",
        )
    )

    # S6
    s6_ok = (
        metrics.cum_net_return_b > -0.05
        and metrics.max_drawdown_b <= 0.15
        and metrics.cum_net_return_b >= metrics.cum_net_return_a - 0.025
    )
    results.append(
        GateResult(
            "S6",
            s6_ok,
            f"cum_b={metrics.cum_net_return_b} dd_b={metrics.max_drawdown_b} "
            f"cum_a={metrics.cum_net_return_a}",
        )
    )

    # S7
    cost_cap = 2.0 * metrics.normal_cost_bps_assumption
    s7_ok = (
        metrics.mean_realized_cost_bps <= cost_cap
        and metrics.mean_realized_cost_bps <= 5.0
        and metrics.unreconciled_orders == 0
        and metrics.risk_limit_breaches == 0
        and metrics.incidents == 0
    )
    results.append(
        GateResult(
            "S7",
            s7_ok,
            f"cost_bps={metrics.mean_realized_cost_bps} cap={cost_cap} "
            f"unrec={metrics.unreconciled_orders} breaches={metrics.risk_limit_breaches} "
            f"incidents={metrics.incidents}",
        )
    )

    # S8
    s8_ok = metrics.human_signoff and not metrics.jev_version_changed
    results.append(
        GateResult(
            "S8",
            s8_ok,
            f"signoff={metrics.human_signoff} jev_changed={metrics.jev_version_changed}",
        )
    )
    return results


def evaluate_scale(
    *,
    metrics: ScaleGateMetrics,
    acceptance_dir: Path,
    registry_path: Path,
    tiny_live_path: Path,
    git_commit: str,
    research_config_hash: str,
    qset_hash: str,
    model_resolved: str,
) -> ScaleEvalResult:
    """Write SCALE_ACCEPTANCE.json only when all S1-S8 pass; parent = TINY_LIVE digest.

    Does not modify ``TINY_LIVE_CEILING_USD`` or enable capital above $10.
    """
    # Parent must be a valid TINY_LIVE artifact.
    tiny_body = verify_acceptance(tiny_live_path, research_config_hash=research_config_hash)
    if tiny_body.get("status") != "TINY_LIVE_READY":
        raise ArtifactError(f"parent must be TINY_LIVE_READY, got {tiny_body.get('status')}")
    parent_digest = str(tiny_body["integrity_digest"])

    results = evaluate_scale_gates(metrics)
    all_passed = all(r.passed for r in results)
    gate_report = {
        "gates": [asdict(r) for r in results],
        "all_passed": all_passed,
        "power_statement": SCALE_POWER_STATEMENT,
        "ceiling_note": SCALE_DOES_NOT_RAISE_CEILING,
        "tiny_live_ceiling_usd": TINY_LIVE_CEILING_USD,
        "created_at": datetime.now(UTC).isoformat(),
    }
    acceptance_dir = Path(acceptance_dir)
    acceptance_dir.mkdir(parents=True, exist_ok=True)
    gate_path = acceptance_dir / "SCALE_GATE_REPORT.json"
    gate_path.write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
    gate_sha = sha256_of(canonical_json(gate_report))

    if not all_passed:
        rej = acceptance_dir / "SCALE_REJECTION.json"
        rej.write_text(
            json.dumps(
                {
                    "status": "SCALE_REJECTED",
                    "gates": gate_report["gates"],
                    "ceiling_note": SCALE_DOES_NOT_RAISE_CEILING,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ScaleEvalResult(
            gate_report=gate_report,
            acceptance=None,
            gate_report_path=gate_path,
            rejection_path=rej,
        )

    art = write_acceptance(
        "scale",
        {
            "gates": gate_report["gates"],
            "power_statement": SCALE_POWER_STATEMENT,
            "ceiling_note": SCALE_DOES_NOT_RAISE_CEILING,
            "tiny_live_ceiling_usd": TINY_LIVE_CEILING_USD,
            "note": "integrity_digest is an integrity digest, not a cryptographic signature",
        },
        out_dir=acceptance_dir,
        git_commit=git_commit,
        research_config_hash=research_config_hash,
        qset_hash=qset_hash,
        model_resolved=model_resolved,
        gate_report_sha256=gate_sha,
        parent_artifact_digest=parent_digest,
        registry_path=registry_path,
    )
    # Hard invariant: this code path must not raise the ceiling.
    if TINY_LIVE_CEILING_USD > 10.00 + 1e-12:
        raise ArtifactError("invariant broken: TINY_LIVE_CEILING_USD exceeded 10")
    return ScaleEvalResult(
        gate_report=gate_report,
        acceptance=art,
        gate_report_path=gate_path,
    )


def list_complete_forward_sessions(forward_root: Path) -> list[date]:
    root = Path(forward_root)
    if not root.is_dir():
        return []
    out: list[date] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            d = date.fromisoformat(child.name)
        except ValueError:
            continue
        if is_session_complete(root, d):
            out.append(d)
    return out


def load_session_score_frame(forward_root: Path, session: date) -> pd.DataFrame:
    """Load features (+ optional jev) from a captured forward session directory."""
    sdir = session_dir(forward_root, session)
    feat_path = sdir / "features.parquet"
    if not feat_path.is_file():
        raise ArtifactError(f"missing features for session {session}")
    feats = pd.read_parquet(feat_path)
    jev_path = sdir / "jev_features.parquet"
    if jev_path.is_file():
        jev = pd.read_parquet(jev_path)
        feats = feats.merge(jev, on=["symbol", "date"], how="left", suffixes=("", "_jev"))
    feats["date"] = pd.to_datetime(feats["date"]).dt.date
    return feats
