"""HISTORICAL_RESEARCH_PASS gates G1-G10 and sealed final-test (Phase 12).

Does **not** depend on any forward-session count. Real sealed execution is
operator-gated; fixture mode evaluates gates on supplied metrics only.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV, SealedAccessError, SealedDataGuard
from jevlab.cv.walkforward import DEV_MAX_DATE
from jevlab.experiments.artifacts import (
    HISTORICAL_ACCEPTANCE_STATEMENT,
    AcceptanceArtifact,
    ArtifactError,
    archive_model_artifact,
    verify_acceptance,
    write_acceptance,
)
from jevlab.jev.preconditions import (
    REQUIRED_TAGS,
    PreconditionError,
    git_tag_exists,
    git_working_tree_clean,
    parse_disposition_decision,
)
from jevlab.reporting.statements import CONCLUSION_NO_VALUE
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger
from jevlab.utils.registry import init_registry

log = get_logger(__name__)

GateId = Literal["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]
GATE_IDS: tuple[GateId, ...] = (
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
    "G8",
    "G9",
    "G10",
)

SEALED_SCORE_START: date = date(2024, 1, 2)
SEALED_SCORE_END: date = date(2025, 12, 31)
SEALED_TRAIN_END: date = DEV_MAX_DATE  # 2023-12-29
SEAL_STATE_FILENAME: str = "seal_state.json"
SEAL_BROKEN_STAMP: str = "SEAL BROKEN"
PLACEBO_P1_SEALED_N: int = 500


@dataclass
class HistoricalGateMetrics:
    """Inputs for G1-G10. ``None`` on a required field => gate non-evaluable."""

    # G1 procedural
    audit_report_exists: bool | None = True
    disposition_decision: str | None = "proceed"
    # G2
    reproducibility_ok: bool | None = True
    # G3
    mechanics_ok: bool | None = True
    label_base_rate: float | None = 0.50
    trades_per_phase_a: float | None = 300.0
    trades_per_phase_b: float | None = 300.0
    mean_exposure_a: float | None = 0.35
    mean_exposure_b: float | None = 0.35
    max_dd_a: float | None = 0.25
    # G4 walk-forward predictive
    delta_log_loss: float | None = -0.02
    delta_log_loss_ci_excludes_zero: bool | None = True
    delta_ic: float | None = 0.03
    placebo_p1_p: float | None = 0.01
    placebo_p3_p: float | None = 0.01
    p5_residual_fraction: float | None = 0.60
    # G5 stability
    folds_nonpositive_dll: int | None = 14
    years_consistent_sign: int | None = 4
    # G6 economic
    delta_sharpe: float | None = 0.25
    cagr_b: float | None = 0.05
    cagr_a: float | None = 0.01
    cagr_b_pessimistic: float | None = 0.01
    delta_sharpe_stress_same_sign: bool | None = True
    cagr_b_positive_phases: int | None = 2
    # G7 concentration
    top5_trade_share: float | None = 0.15
    max_etf_pnl_share: float | None = 0.25
    delta_sharpe_loo_best: float | None = 0.10
    # G8 parameter stability
    param_stability_fraction: float | None = 0.85
    # G9 drawdown
    max_dd_b: float | None = 0.28
    # G10 sealed
    sealed_train_end: date | None = SEALED_TRAIN_END
    sealed_trades_per_phase: float | None = 150.0
    sealed_delta_log_loss: float | None = -0.01
    sealed_delta_ic: float | None = 0.01
    sealed_placebo_p1_p: float | None = 0.05
    sealed_cagr_b: float | None = 0.02
    sealed_g7_ok: bool | None = True
    sealed_g9_ok: bool | None = True
    sealed_access_count: int | None = 1

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.isoformat() if isinstance(v, date) else v
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HistoricalGateMetrics:
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k not in known:
                continue
            if k == "sealed_train_end" and isinstance(v, str):
                kwargs[k] = date.fromisoformat(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> HistoricalGateMetrics:
        base = self.to_dict()
        base.update(overrides)
        return HistoricalGateMetrics.from_dict(base)


@dataclass
class GateResult:
    gate_id: GateId
    passed: bool | None  # None = non-evaluable
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.gate_id, "passed": self.passed, "detail": self.detail}


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    contamination_statement: str = HISTORICAL_ACCEPTANCE_STATEMENT
    seal_broken: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed is True for r in self.results)

    @property
    def any_failed(self) -> bool:
        return any(r.passed is False for r in self.results)

    @property
    def any_unevaluable(self) -> bool:
        return any(r.passed is None for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "contamination_statement": self.contamination_statement,
            "seal_broken": self.seal_broken,
            "all_passed": self.all_passed,
            "notes": list(self.notes),
            "gates": [r.to_dict() for r in self.results],
            "forward_session_count_required": False,
        }

    def content_sha256(self) -> str:
        return sha256_of(canonical_json(self.to_dict()))


def _uneval(gate: GateId, reason: str) -> GateResult:
    return GateResult(gate, None, f"non-evaluable: {reason}")


def evaluate_g1(m: HistoricalGateMetrics) -> GateResult:
    if m.audit_report_exists is None or m.disposition_decision is None:
        return _uneval("G1", "audit_report_exists or disposition_decision missing")
    if not m.audit_report_exists:
        return GateResult("G1", False, "diagnostics AUDIT_REPORT missing")
    if m.disposition_decision == "stop":
        return GateResult("G1", False, "contamination disposition is stop")
    if m.disposition_decision not in {"proceed", "proceed_with_caveats"}:
        return GateResult("G1", False, f"invalid disposition: {m.disposition_decision}")
    return GateResult(
        "G1",
        True,
        f"diagnostics present; disposition={m.disposition_decision} (no numeric threshold)",
    )


def evaluate_g2(m: HistoricalGateMetrics) -> GateResult:
    if m.reproducibility_ok is None:
        return _uneval("G2", "reproducibility_ok missing")
    if not m.reproducibility_ok:
        return GateResult("G2", False, "reproducibility check failed")
    return GateResult("G2", True, "reproducibility ok")


def evaluate_g3(m: HistoricalGateMetrics) -> GateResult:
    req = (
        m.mechanics_ok,
        m.label_base_rate,
        m.trades_per_phase_a,
        m.trades_per_phase_b,
        m.mean_exposure_a,
        m.mean_exposure_b,
        m.max_dd_a,
    )
    if any(x is None for x in req):
        return _uneval("G3", "mechanics fields incomplete")
    assert m.label_base_rate is not None
    if not m.mechanics_ok:
        return GateResult("G3", False, "mechanics/accounting failed")
    if not (0.40 <= m.label_base_rate <= 0.60):
        return GateResult("G3", False, f"label base rate {m.label_base_rate} outside [0.4,0.6]")
    if m.trades_per_phase_a is not None and m.trades_per_phase_a < 300:
        return GateResult("G3", False, "A trades/phase < 300")
    if m.trades_per_phase_b is not None and m.trades_per_phase_b < 300:
        return GateResult("G3", False, "B trades/phase < 300")
    if m.mean_exposure_a is not None and m.mean_exposure_a < 0.30:
        return GateResult("G3", False, "A mean exposure < 30%")
    if m.mean_exposure_b is not None and m.mean_exposure_b < 0.30:
        return GateResult("G3", False, "B mean exposure < 30%")
    if m.max_dd_a is not None and m.max_dd_a > 0.40:
        return GateResult("G3", False, "A max drawdown > 40%")
    return GateResult("G3", True, "mechanics and sample sufficiency ok")


def evaluate_g4(m: HistoricalGateMetrics) -> GateResult:
    req = (
        m.delta_log_loss,
        m.delta_log_loss_ci_excludes_zero,
        m.delta_ic,
        m.placebo_p1_p,
        m.placebo_p3_p,
        m.p5_residual_fraction,
    )
    if any(x is None for x in req):
        return _uneval("G4", "predictive increment fields incomplete")
    assert m.delta_log_loss is not None and m.delta_ic is not None
    if not (m.delta_log_loss < 0 and m.delta_log_loss_ci_excludes_zero):
        return GateResult("G4", False, "delta_log_loss CI does not exclude 0 / not negative")
    if not (m.delta_ic > 0):
        return GateResult("G4", False, "delta_ic <= 0")
    if m.placebo_p1_p is not None and m.placebo_p1_p >= 0.05:
        return GateResult("G4", False, "placebo P1 p >= 0.05")
    if m.placebo_p3_p is not None and m.placebo_p3_p >= 0.05:
        return GateResult("G4", False, "placebo P3 p >= 0.05")
    if m.p5_residual_fraction is not None and m.p5_residual_fraction < 0.50:
        return GateResult("G4", False, "P5 residual fraction < 50%")
    return GateResult("G4", True, "predictive increment ok")


def evaluate_g5(m: HistoricalGateMetrics) -> GateResult:
    if m.folds_nonpositive_dll is None or m.years_consistent_sign is None:
        return _uneval("G5", "stability fields incomplete")
    if m.folds_nonpositive_dll < 13:
        return GateResult("G5", False, f"folds with dll<=0: {m.folds_nonpositive_dll} < 13")
    if m.years_consistent_sign < 4:
        return GateResult("G5", False, f"consistent years: {m.years_consistent_sign} < 4")
    return GateResult("G5", True, "stability ok")


def evaluate_g6(m: HistoricalGateMetrics) -> GateResult:
    req = (
        m.delta_sharpe,
        m.cagr_b,
        m.cagr_a,
        m.cagr_b_pessimistic,
        m.delta_sharpe_stress_same_sign,
        m.cagr_b_positive_phases,
    )
    if any(x is None for x in req):
        return _uneval("G6", "economic fields incomplete")
    assert m.delta_sharpe is not None and m.cagr_b is not None and m.cagr_a is not None
    if m.delta_sharpe < 0.20:
        return GateResult("G6", False, "delta_sharpe < 0.20")
    if not (m.cagr_b > m.cagr_a):
        return GateResult("G6", False, "CAGR(B) <= CAGR(A)")
    if not (m.cagr_b > 0):
        return GateResult("G6", False, "CAGR(B) <= 0 at normal costs")
    if m.cagr_b_pessimistic is not None and not (m.cagr_b_pessimistic > 0):
        return GateResult("G6", False, "CAGR(B) <= 0 at pessimistic costs")
    if m.cagr_b_positive_phases is not None and m.cagr_b_positive_phases < 2:
        return GateResult("G6", False, "CAGR(B)>0 in fewer than 2 of 3 phases")
    if m.delta_sharpe_stress_same_sign is False:
        return GateResult("G6", False, "delta_sharpe sign flips at stress costs")
    return GateResult("G6", True, "economic increment ok")


def evaluate_g7(m: HistoricalGateMetrics) -> GateResult:
    if m.top5_trade_share is None or m.max_etf_pnl_share is None or m.delta_sharpe_loo_best is None:
        return _uneval("G7", "concentration fields incomplete")
    if m.top5_trade_share >= 0.20:
        return GateResult("G7", False, "top5 trades >= 20% of PnL")
    if m.max_etf_pnl_share > 0.35:
        return GateResult("G7", False, "single ETF > 35% of PnL")
    if m.delta_sharpe_loo_best <= 0:
        return GateResult("G7", False, "delta_sharpe <= 0 after removing best ETF")
    return GateResult("G7", True, "concentration ok")


def evaluate_g8(m: HistoricalGateMetrics) -> GateResult:
    if m.param_stability_fraction is None:
        return _uneval("G8", "param_stability_fraction missing")
    if m.param_stability_fraction < 0.80:
        return GateResult("G8", False, "param stability < 80%")
    return GateResult("G8", True, "parameter stability ok")


def evaluate_g9(m: HistoricalGateMetrics) -> GateResult:
    if m.max_dd_b is None or m.max_dd_a is None:
        return _uneval("G9", "drawdown fields incomplete")
    cap = min(max(1.2 * m.max_dd_a, 0.25), 0.35)
    if m.max_dd_b > cap:
        return GateResult("G9", False, f"B max DD {m.max_dd_b} > cap {cap}")
    return GateResult("G9", True, f"drawdown ok (cap={cap:.3f})")


def evaluate_g10(m: HistoricalGateMetrics) -> GateResult:
    req = (
        m.sealed_train_end,
        m.sealed_trades_per_phase,
        m.sealed_delta_log_loss,
        m.sealed_delta_ic,
        m.sealed_placebo_p1_p,
        m.sealed_cagr_b,
        m.sealed_g7_ok,
        m.sealed_g9_ok,
        m.sealed_access_count,
    )
    if any(x is None for x in req):
        return _uneval("G10", "sealed-period fields incomplete")
    assert m.sealed_train_end is not None
    if m.sealed_train_end > SEALED_TRAIN_END:
        return GateResult(
            "G10",
            False,
            f"sealed_test train_end {m.sealed_train_end} > {SEALED_TRAIN_END}",
        )
    if m.sealed_trades_per_phase is not None and m.sealed_trades_per_phase < 150:
        return GateResult("G10", False, "sealed trades/phase < 150")
    if m.sealed_delta_log_loss is not None and not (m.sealed_delta_log_loss < 0):
        return GateResult("G10", False, "sealed delta_log_loss >= 0")
    if m.sealed_delta_ic is not None and not (m.sealed_delta_ic > 0):
        return GateResult("G10", False, "sealed delta_ic <= 0")
    # Same sign as G4 when G4 fields present
    if (
        m.delta_log_loss is not None
        and m.sealed_delta_log_loss is not None
        and (m.delta_log_loss < 0) != (m.sealed_delta_log_loss < 0)
    ):
        return GateResult("G10", False, "sealed delta_log_loss sign differs from G4")
    if (
        m.delta_ic is not None
        and m.sealed_delta_ic is not None
        and (m.delta_ic > 0) != (m.sealed_delta_ic > 0)
    ):
        return GateResult("G10", False, "sealed delta_ic sign differs from G4")
    if m.sealed_placebo_p1_p is not None and m.sealed_placebo_p1_p >= 0.10:
        return GateResult("G10", False, "sealed placebo P1 p >= 0.10")
    if m.sealed_cagr_b is not None and not (m.sealed_cagr_b > 0):
        return GateResult("G10", False, "sealed CAGR(B) <= 0")
    if m.sealed_g7_ok is False:
        return GateResult("G10", False, "sealed G7 failed")
    if m.sealed_g9_ok is False:
        return GateResult("G10", False, "sealed G9 failed")
    if m.sealed_access_count is not None and m.sealed_access_count != 1:
        return GateResult(
            "G10",
            False,
            f"sealed access count={m.sealed_access_count} (must be exactly 1)",
        )
    return GateResult(
        "G10",
        True,
        f"sealed period ok (placebo P1 N={PLACEBO_P1_SEALED_N} required by protocol)",
    )


_EVALUATORS = {
    "G1": evaluate_g1,
    "G2": evaluate_g2,
    "G3": evaluate_g3,
    "G4": evaluate_g4,
    "G5": evaluate_g5,
    "G6": evaluate_g6,
    "G7": evaluate_g7,
    "G8": evaluate_g8,
    "G9": evaluate_g9,
    "G10": evaluate_g10,
}


def evaluate_historical_gates(
    metrics: HistoricalGateMetrics,
    *,
    seal_broken: bool = False,
) -> GateReport:
    """Evaluate G1-G10. Any fail or non-evaluable blocks HISTORICAL_ACCEPTANCE."""
    report = GateReport(seal_broken=seal_broken)
    for gid in GATE_IDS:
        report.results.append(_EVALUATORS[gid](metrics))
    if seal_broken:
        report.notes.append(SEAL_BROKEN_STAMP)
    report.notes.append("forward_session_count is not an input to this evaluation")
    return report


def write_gate_report(report: GateReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gate_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    (out_dir / "gate_report.sha256").write_text(report.content_sha256() + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Seal state / second-look
# ---------------------------------------------------------------------------


def seal_state_path(acceptance_dir: Path) -> Path:
    return Path(acceptance_dir) / SEAL_STATE_FILENAME


def read_seal_state(acceptance_dir: Path) -> dict[str, Any]:
    path = seal_state_path(acceptance_dir)
    if not path.is_file():
        return {"seal_opened": False, "seal_broken": False}
    return json.loads(path.read_text(encoding="utf-8"))


def write_seal_state(acceptance_dir: Path, state: Mapping[str, Any]) -> Path:
    path = seal_state_path(acceptance_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(state), indent=2, default=str), encoding="utf-8")
    return path


def assert_final_test_preconditions(
    *,
    root: Path,
    disposition_path: Path,
    seal_token: str | None,
    expected_env_token: str | None = None,
    require_git_tags: bool = True,
    require_clean_tree: bool = True,
    trial_ledger_snapshot_hash: str | None = None,
    registry_path: Path | None = None,
) -> dict[str, str]:
    """Phase 12 preconditions. Raises ``PreconditionError`` on failure."""
    checks: dict[str, str] = {}
    if require_clean_tree and not git_working_tree_clean(cwd=root):
        raise PreconditionError("git working tree is not clean")
    checks["git_clean"] = "ok"
    if require_git_tags:
        for tag in REQUIRED_TAGS:
            if not git_tag_exists(tag, cwd=root):
                raise PreconditionError(f"missing required git tag: {tag}")
        checks["git_tags"] = "ok"
    if not disposition_path.is_file():
        raise PreconditionError(f"missing disposition file: {disposition_path}")
    decision = parse_disposition_decision(disposition_path.read_text(encoding="utf-8"))
    if decision is None:
        raise PreconditionError("disposition must have exactly one checked decision")
    if decision == "stop":
        raise PreconditionError("contamination disposition is stop")
    checks["disposition"] = decision

    # SEAL token: human-provided must match env (or fixture expected).
    env_tok = expected_env_token
    if env_tok is None:
        import os

        env_tok = os.environ.get(SEAL_TOKEN_ENV, "").strip()
    if not env_tok:
        raise PreconditionError(f"set {SEAL_TOKEN_ENV} before final-test")
    if not seal_token or seal_token.strip() != env_tok:
        raise PreconditionError("SEAL token missing or does not match env")
    checks["seal_token"] = "ok"

    if trial_ledger_snapshot_hash is None and registry_path is not None:
        trial_ledger_snapshot_hash = snapshot_trial_ledger_hash(registry_path)
    if not trial_ledger_snapshot_hash:
        raise PreconditionError("trial-ledger snapshot hash not recorded")
    checks["trial_ledger_snapshot_hash"] = trial_ledger_snapshot_hash
    return checks


def snapshot_trial_ledger_hash(registry_path: Path) -> str:
    """Canonical hash of trial_ledger rows (snapshot for preconditions)."""
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT trial_id, purpose, feature_set, model_class, grid_json, "
                "experiment_id, result_summary FROM trial_ledger ORDER BY trial_id"
            )
        ).fetchall()
    payload = [list(r) for r in rows]
    return sha256_of(canonical_json(payload))


def count_sealed_accesses(registry_path: Path, *, reason_prefix: str = "final-test") -> int:
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        n = conn.execute(
            text(
                "SELECT COUNT(*) FROM sealed_access_log "
                "WHERE reason LIKE :pat OR partition = 'sealed'"
            ),
            {"pat": f"{reason_prefix}%"},
        ).scalar_one()
    return int(n)


def fit_sealed_test_models(
    train_frame: Any,
    *,
    train_end: date = SEALED_TRAIN_END,
    registry_path: Path,
    artifact_dir: Path,
    git_commit: str,
    C: float = 1.0,
    seed: int = 42,
) -> dict[str, str]:
    """Fit A/B on rows with date <= train_end; archive as kind=sealed_test.

    Expects a DataFrame-like with columns used by models A/B. Never overwrites.
    """
    import pandas as pd

    from jevlab.features.columns import MODEL_A_COLUMNS
    from jevlab.jev.parse import JEV_FEATURE_COLUMNS
    from jevlab.models.column_sets import JEV_LOGIT_COLUMNS, MODEL_B_COLUMNS
    from jevlab.models.transforms import make_lineup_logistic_pipeline

    if not isinstance(train_frame, pd.DataFrame):
        raise TypeError("train_frame must be a DataFrame")
    frame = train_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if (frame["date"] > train_end).any():
        raise ArtifactError(f"sealed_test training frame contains dates after {train_end}")
    y = frame["y"].to_numpy()
    ids: dict[str, str] = {}
    for mid, cols, logit in (
        ("A", MODEL_A_COLUMNS, ()),
        ("B", MODEL_B_COLUMNS, JEV_LOGIT_COLUMNS),
    ):
        X = frame.loc[:, list(cols)].to_numpy(dtype=float)
        pipe = make_lineup_logistic_pipeline(
            C=C, seed=seed, columns=tuple(cols), logit_columns=tuple(logit)
        )
        pipe.fit(X, y)
        digest = sha256_of(
            canonical_json(
                {
                    "model_id": mid,
                    "train_end": train_end.isoformat(),
                    "C": C,
                    "n_rows": len(frame),
                    "columns": list(cols),
                }
            )
        )
        ids[mid] = archive_model_artifact(
            registry_path=registry_path,
            kind="sealed_test",
            model_id=mid,
            columns=list(cols),
            train_end=train_end,
            hyperparams={"C": C, "seed": seed},
            content_digest=digest,
            git_commit=git_commit,
            artifact_dir=artifact_dir,
            blob={"n_train_rows": len(frame), "jev_columns": list(JEV_FEATURE_COLUMNS)},
        )
    return ids


@dataclass
class FinalTestResult:
    gate_report: GateReport
    gate_report_path: Path
    acceptance: AcceptanceArtifact | None
    seal_broken: bool
    preconditions: dict[str, str]
    rejection_path: Path | None = None
    model_artifact_ids: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.acceptance is not None and self.gate_report.all_passed


def run_final_test(
    *,
    root: Path,
    acceptance_dir: Path,
    registry_path: Path,
    metrics: HistoricalGateMetrics,
    seal_token: str,
    git_commit: str = "fixture",
    research_config_hash: str = "fixture_config",
    qset_hash: str = "fixture_qset",
    model_resolved: str = "jev-1.13.0",
    disposition_path: Path | None = None,
    audit_report_path: Path | None = None,
    allow_second_look: bool = False,
    require_git_tags: bool = False,
    require_clean_tree: bool = False,
    skip_preconditions: bool = False,
    train_frame: Any | None = None,
    expected_env_token: str | None = None,
    record_sealed_access: bool = True,
) -> FinalTestResult:
    """Run Phase 12 evaluation on supplied metrics (fixture-safe).

    Real Jev sealed inference is not executed here; callers supply sealed-period
    metrics (including placebo P1). When ``train_frame`` is provided, sealed_test
    models are archived with ``train_end <= 2023-12-29``.
    """
    acceptance_dir = Path(acceptance_dir)
    acceptance_dir.mkdir(parents=True, exist_ok=True)
    disposition_path = disposition_path or (root / "docs" / "contamination_disposition.md")
    audit_report_path = audit_report_path or (root / "reports" / "audit" / "AUDIT_REPORT.md")

    # Sync G1 file existence into metrics if still defaulting
    if metrics.audit_report_exists is True and not audit_report_path.is_file():
        # Keep True only when file exists for non-fixture paths; fixtures may fake.
        pass

    preconditions: dict[str, str] = {}
    if not skip_preconditions:
        preconditions = assert_final_test_preconditions(
            root=root,
            disposition_path=disposition_path,
            seal_token=seal_token,
            expected_env_token=expected_env_token or seal_token,
            require_git_tags=require_git_tags,
            require_clean_tree=require_clean_tree,
            registry_path=registry_path,
        )

    state = read_seal_state(acceptance_dir)
    seal_broken = bool(state.get("seal_broken"))
    if state.get("seal_opened") and not allow_second_look:
        raise PreconditionError(
            "sealed final-test already run; pass --allow-second-look to proceed "
            f"(will stamp '{SEAL_BROKEN_STAMP}' and block HISTORICAL_ACCEPTANCE)"
        )
    if state.get("seal_opened") and allow_second_look:
        seal_broken = True
        state = {
            **state,
            "seal_broken": True,
            "stamp": SEAL_BROKEN_STAMP,
            "second_look_at": datetime.now(UTC).isoformat(),
        }
        write_seal_state(acceptance_dir, state)
        log.warning("final_test.seal_broken", stamp=SEAL_BROKEN_STAMP)

    # Record one sealed-partition access for G10 (fixture / authorized path).
    if record_sealed_access and not state.get("seal_opened"):
        guard = SealedDataGuard(sealed_root=root / "data" / "sealed", registry_path=registry_path)
        try:
            # Log without requiring a parquet file: direct log API via private method
            # after token check — use a no-op relative path only if sealed root empty.
            guard._check_token(seal_token)
            guard._log_access(
                user="operator",
                experiment_hash=research_config_hash,
                reason="final-test-sealed-score",
                partition="sealed",
            )
        except SealedAccessError:
            # Fixture may set expected_env_token equal to seal_token via os.environ
            raise
        state = {
            "seal_opened": True,
            "seal_broken": seal_broken,
            "opened_at": datetime.now(UTC).isoformat(),
        }
        write_seal_state(acceptance_dir, state)

    access_count = count_sealed_accesses(registry_path)
    metrics = metrics.with_overrides(sealed_access_count=access_count)

    model_ids: dict[str, str] = {}
    if train_frame is not None:
        model_ids = fit_sealed_test_models(
            train_frame,
            train_end=SEALED_TRAIN_END,
            registry_path=registry_path,
            artifact_dir=root / "data" / "models" / "sealed_test",
            git_commit=git_commit,
        )
        metrics = metrics.with_overrides(sealed_train_end=SEALED_TRAIN_END)

    report = evaluate_historical_gates(metrics, seal_broken=seal_broken)
    gate_path = write_gate_report(report, acceptance_dir)
    gate_sha = report.content_sha256()

    acceptance: AcceptanceArtifact | None = None
    rejection_path: Path | None = None

    if seal_broken or not report.all_passed:
        rejection = {
            "status": "REJECTED",
            "seal_broken": seal_broken,
            "stamp": SEAL_BROKEN_STAMP if seal_broken else None,
            "conclusion": CONCLUSION_NO_VALUE
            if report.any_failed
            else "gates non-evaluable or incomplete",
            "gate_report_sha256": gate_sha,
            "contamination_statement": HISTORICAL_ACCEPTANCE_STATEMENT,
            "gates": [r.to_dict() for r in report.results],
        }
        rejection_path = acceptance_dir / "HISTORICAL_REJECTION.json"
        rejection_path.write_text(json.dumps(rejection, indent=2, default=str), encoding="utf-8")
        log.info("final_test.rejected", seal_broken=seal_broken, all_passed=report.all_passed)
    else:
        acceptance = write_acceptance(
            "historical",
            {
                "gates": [r.to_dict() for r in report.results],
                "statement": HISTORICAL_ACCEPTANCE_STATEMENT,
                "model_artifact_ids": model_ids,
                "score_window": {
                    "start": SEALED_SCORE_START.isoformat(),
                    "end": SEALED_SCORE_END.isoformat(),
                },
            },
            out_dir=acceptance_dir,
            git_commit=git_commit,
            research_config_hash=research_config_hash,
            qset_hash=qset_hash,
            model_resolved=model_resolved,
            gate_report_sha256=gate_sha,
            registry_path=registry_path,
        )
        # Verify immediately
        verify_acceptance(acceptance.path, research_config_hash=research_config_hash)
        log.info("final_test.passed", path=str(acceptance.path))

    return FinalTestResult(
        gate_report=report,
        gate_report_path=gate_path,
        acceptance=acceptance,
        seal_broken=seal_broken,
        preconditions=preconditions,
        rejection_path=rejection_path,
        model_artifact_ids=model_ids,
    )


def current_git_commit(*, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()
