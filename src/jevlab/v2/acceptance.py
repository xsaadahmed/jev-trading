"""Rule-based V2 sealed acceptance evaluator (no human discretion)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from jevlab.evaluation.metrics import _round_trip_pnl
from jevlab.v2.statistics import (
    MAX_DRAWDOWN,
    MIN_SHARPE_ADVANTAGE,
    P_THRESHOLD,
    e1_pass,
    e2_pass,
    e3_pass,
    e4_pass,
    e5_pass,
)

# Evaluator IDs: frozen names plus explicit splits required for independent tests.
# Frozen mapping: R1_frozen=mechanics → R2; PS1 → S1; C1 → S2∧S3; hash integrity → R1.
GateId = Literal["R1", "R2", "E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "Z1"]
GATE_IDS: tuple[GateId, ...] = (
    "R1",
    "R2",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "S1",
    "S2",
    "S3",
    "Z1",
)
FROZEN_GATE_MAP: dict[str, tuple[GateId, ...]] = {
    "R1": ("R2",),
    "E1": ("E1",),
    "E2": ("E2",),
    "E3": ("E3",),
    "E4": ("E4",),
    "E5": ("E5",),
    "PS1": ("S1",),
    "C1": ("S2", "S3"),
    "Z1": ("Z1",),
}


@dataclass
class V2GateMetrics:
    """Inputs for V2 hard gates. ``None`` ⇒ non-evaluable ⇒ fail closed."""

    hash_integrity_ok: bool | None = True
    tag_integrity_ok: bool | None = True
    research_config_hash_ok: bool | None = True
    mechanics_ok: bool | None = True
    leakage_ok: bool | None = True
    ledger_fast_ok: bool | None = True
    p_value: float | None = 0.01
    sharpe_advantage: float | None = 0.20
    cagr_normal: float | None = 0.05
    cagr_pessimistic: float | None = 0.01
    max_drawdown_dec: float | None = -0.20
    phases_beat_null_median: int | None = 2
    top5_profitable_share: float | None = 0.10
    max_etf_positive_share: float | None = 0.20
    positive_pnl: float | None = 1.0
    seal_one_shot_ok: bool | None = True
    seal_broken: bool = False
    sealed_access_count: int | None = 1
    predictive_log_loss: float | None = 0.9
    predictive_auc: float | None = 0.4
    robustness_notes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def with_overrides(self, **overrides: Any) -> V2GateMetrics:
        base = self.to_dict()
        base.update(overrides)
        return V2GateMetrics(**base)


FROZEN_ALIAS: dict[GateId, str] = {
    "R1": "hash_tag_config_integrity",
    "R2": "R1",  # frozen R1 = mechanics / no leakage
    "E1": "E1",
    "E2": "E2",
    "E3": "E3",
    "E4": "E4",
    "E5": "E5",
    "S1": "PS1",
    "S2": "C1_top5",
    "S3": "C1_etf",
    "Z1": "Z1",
}


@dataclass
class GateResult:
    gate_id: GateId
    passed: bool | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.gate_id,
            "frozen_id": FROZEN_ALIAS[self.gate_id],
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class V2GateReport:
    results: list[GateResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: list[str] = field(default_factory=list)
    label: str = "confirmatory"

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed is True for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "label": self.label,
            "all_passed": self.all_passed,
            "notes": list(self.notes),
            "gates": [r.to_dict() for r in self.results],
            "predictive_metrics_are_gates": False,
            "robustness_variants_are_gates": False,
        }


def _fail_closed(gate: GateId, reason: str) -> GateResult:
    return GateResult(gate, False, f"fail-closed: {reason}")


def evaluate_r1(m: V2GateMetrics) -> GateResult:
    if None in (m.hash_integrity_ok, m.tag_integrity_ok, m.research_config_hash_ok):
        return _fail_closed("R1", "hash/tag fields missing")
    if not (m.hash_integrity_ok and m.tag_integrity_ok and m.research_config_hash_ok):
        return GateResult("R1", False, "candidate/config/tag/hash integrity failed")
    return GateResult("R1", True, "hash/tag/config integrity ok")


def evaluate_r2(m: V2GateMetrics) -> GateResult:
    if None in (m.mechanics_ok, m.leakage_ok, m.ledger_fast_ok):
        return _fail_closed("R2", "mechanics fields missing")
    if not (m.mechanics_ok and m.leakage_ok and m.ledger_fast_ok):
        return GateResult("R2", False, "mechanics/reproducibility/leakage failed")
    return GateResult("R2", True, "mechanics ok")


def evaluate_e1(m: V2GateMetrics) -> GateResult:
    if m.p_value is None or not math.isfinite(m.p_value):
        return _fail_closed("E1", "p_value missing/non-finite")
    ok = e1_pass(m.p_value, threshold=P_THRESHOLD)
    return GateResult("E1", ok, f"p={m.p_value} threshold={P_THRESHOLD}")


def evaluate_e2(m: V2GateMetrics) -> GateResult:
    if m.sharpe_advantage is None or not math.isfinite(m.sharpe_advantage):
        return _fail_closed("E2", "advantage missing/non-finite")
    ok = e2_pass(m.sharpe_advantage, threshold=MIN_SHARPE_ADVANTAGE)
    return GateResult("E2", ok, f"advantage={m.sharpe_advantage}")


def evaluate_e3(m: V2GateMetrics) -> GateResult:
    if m.cagr_normal is None:
        return _fail_closed("E3", "cagr_normal missing")
    ok = e3_pass(m.cagr_normal)
    return GateResult("E3", ok, f"cagr_normal={m.cagr_normal}")


def evaluate_e4(m: V2GateMetrics) -> GateResult:
    if m.cagr_pessimistic is None:
        return _fail_closed("E4", "cagr_pessimistic missing")
    ok = e4_pass(m.cagr_pessimistic)
    return GateResult("E4", ok, f"cagr_pessimistic={m.cagr_pessimistic}")


def evaluate_e5(m: V2GateMetrics) -> GateResult:
    if m.max_drawdown_dec is None:
        return _fail_closed("E5", "max_drawdown missing")
    ok = e5_pass(m.max_drawdown_dec, cap=MAX_DRAWDOWN)
    return GateResult("E5", ok, f"max_dd={m.max_drawdown_dec}")


def evaluate_s1(m: V2GateMetrics) -> GateResult:
    if m.phases_beat_null_median is None:
        return _fail_closed("S1", "phase stability missing")
    ok = int(m.phases_beat_null_median) >= 2
    return GateResult("S1", ok, f"phases_beat_null={m.phases_beat_null_median}")


def evaluate_s2(m: V2GateMetrics) -> GateResult:
    if m.positive_pnl is None or m.top5_profitable_share is None:
        return _fail_closed("S2", "concentration fields missing")
    if not math.isfinite(m.positive_pnl) or m.positive_pnl <= 0:
        return GateResult("S2", False, "P+ <= 0; concentration unevaluable → fail")
    ok = m.top5_profitable_share < 0.25
    return GateResult("S2", ok, f"top5_profitable_share={m.top5_profitable_share}")


def evaluate_s3(m: V2GateMetrics) -> GateResult:
    if m.positive_pnl is None or m.max_etf_positive_share is None:
        return _fail_closed("S3", "ETF concentration fields missing")
    if not math.isfinite(m.positive_pnl) or m.positive_pnl <= 0:
        return GateResult("S3", False, "P+ <= 0; concentration unevaluable → fail")
    ok = m.max_etf_positive_share <= 0.35
    return GateResult("S3", ok, f"max_etf_positive_share={m.max_etf_positive_share}")


def evaluate_z1(m: V2GateMetrics) -> GateResult:
    if m.seal_broken:
        return GateResult("Z1", False, "seal BROKEN; cannot grant V2_SEALED_PASS")
    if m.seal_one_shot_ok is None or m.sealed_access_count is None:
        return _fail_closed("Z1", "seal fields missing")
    if m.sealed_access_count != 1:
        return GateResult("Z1", False, f"sealed_access_count={m.sealed_access_count}")
    if not m.seal_one_shot_ok:
        return GateResult("Z1", False, "one-shot seal integrity failed")
    return GateResult("Z1", True, "one-shot seal ok")


EVALUATORS = {
    "R1": evaluate_r1,
    "R2": evaluate_r2,
    "E1": evaluate_e1,
    "E2": evaluate_e2,
    "E3": evaluate_e3,
    "E4": evaluate_e4,
    "E5": evaluate_e5,
    "S1": evaluate_s1,
    "S2": evaluate_s2,
    "S3": evaluate_s3,
    "Z1": evaluate_z1,
}


def evaluate_v2_gates(metrics: V2GateMetrics, *, label: str = "confirmatory") -> V2GateReport:
    """Evaluate every hard gate. Predictive/robustness fields are ignored."""
    _ = metrics.predictive_log_loss, metrics.predictive_auc, metrics.robustness_notes
    results = [EVALUATORS[gid](metrics) for gid in GATE_IDS]
    return V2GateReport(results=results, label=label)


def concentration_from_trades(trades: Any) -> dict[str, float]:
    """Positive-P&L concentration using V1 round-trip P&L semantics."""
    pnl = _round_trip_pnl(trades)
    if pnl is None or pnl.empty:
        return {
            "positive_pnl": 0.0,
            "top5_profitable_share": float("nan"),
            "max_etf_positive_share": float("nan"),
        }
    pos = pnl.loc[pnl["pnl_usd"] > 0]
    p_plus = float(pos["pnl_usd"].sum()) if not pos.empty else 0.0
    if p_plus <= 0:
        return {
            "positive_pnl": p_plus,
            "top5_profitable_share": float("nan"),
            "max_etf_positive_share": float("nan"),
        }
    top5 = float(pos["pnl_usd"].astype(float).nlargest(min(5, len(pos))).sum())
    by_etf = pos.groupby("symbol")["pnl_usd"].sum()
    max_etf = float(by_etf.max()) if len(by_etf) else 0.0
    return {
        "positive_pnl": p_plus,
        "top5_profitable_share": top5 / p_plus,
        "max_etf_positive_share": max_etf / p_plus,
    }


def phases_beating_null(
    phase_sharpe_a: dict[int, float],
    phase_null_medians: dict[int, float],
) -> int:
    n = 0
    for phase in (0, 1, 2):
        a = phase_sharpe_a.get(phase)
        med = phase_null_medians.get(phase)
        if a is None or med is None or not math.isfinite(a) or not math.isfinite(med):
            continue
        if a > med:
            n += 1
    return n
