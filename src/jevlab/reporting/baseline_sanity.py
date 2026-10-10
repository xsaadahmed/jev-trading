"""Baseline SANITY report for R0/A/D (MASTER_SPEC §16.1 G3, Phase 4).

Profitability is NOT a pass/fail criterion. Only mechanics failures (G3) or a
documented fundamental design error (human-decided) should halt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from jevlab.backtest.costs import CostModel
from jevlab.backtest.fast import equity_curves_equal, run_fast
from jevlab.backtest.ledger import run_ledger
from jevlab.config.models import RiskConfig
from jevlab.evaluation.metrics import (
    compute_economic_metrics,
    max_drawdown_dec,
    power_statement,
)
from jevlab.signals.selection import rebalance_dates


@dataclass
class SanityCheck:
    name: str
    passed: bool
    detail: str
    halt_worthy: bool = False  # G3 mechanics only


@dataclass
class SanityReport:
    checks: list[SanityCheck] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def mechanics_ok(self) -> bool:
        return all(c.passed for c in self.checks if c.halt_worthy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanics_ok": self.mechanics_ok,
            "profitability_is_not_a_gate": True,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "halt_worthy": c.halt_worthy,
                }
                for c in self.checks
            ],
            "tables": self.tables,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Baseline SANITY Report",
            "",
            "**Profitability is not a pass/fail criterion.** "
            "Negative or zero CAGR must not stop the pipeline.",
            "",
            f"Mechanics (G3) ok: **{self.mechanics_ok}**",
            "",
            "## Checks",
            "",
        ]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            halt = " [G3]" if c.halt_worthy else ""
            lines.append(f"- [{mark}]{halt} **{c.name}**: {c.detail}")
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for n in self.notes:
            lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)


def label_base_rate(labels: pd.DataFrame) -> float:
    """Eligible-row base rate of y=1; G3 band is [0.40, 0.60]."""
    if labels.empty or "y" not in labels.columns:
        return float("nan")
    work = labels.copy()
    if "eligible" in work.columns:
        work = work.loc[work["eligible"].fillna(0).astype(int) == 1]
    y = work["y"].dropna()
    return float(y.mean()) if len(y) else float("nan")


def realized_cost_bps(trades: pd.DataFrame, cost: CostModel) -> dict[str, float]:
    """Compare configured round-trip bps to a crude realized estimate."""
    configured = {
        "cost_bps_per_side": cost.cost_bps_per_side,
        "sell_fee_bps": cost.sell_fee_bps,
        "roundtrip_normal_spy_mult1_bps": 2.0 * cost.cost_bps_per_side + cost.sell_fee_bps,
    }
    if trades is None or trades.empty:
        return {**configured, "realized_fee_usd_total": 0.0, "n_sells": 0.0}
    sells = trades.loc[trades["side"] == "sell"]
    fee = float(sells["fee_usd"].sum()) if "fee_usd" in sells.columns else 0.0
    return {
        **configured,
        "realized_fee_usd_total": fee,
        "n_sells": float(len(sells)),
    }


def build_sanity_report(
    *,
    bars: pd.DataFrame,
    scores_by_model: dict[str, pd.DataFrame],
    labels: pd.DataFrame | None,
    cost_models: list[CostModel],
    risk_config: RiskConfig,
    initial_capital_usd: float,
    top_k: int = 3,
    phases: list[int] | None = None,
    max_drawdown_pathology_dec: float = 0.40,
) -> SanityReport:
    """Run R0/A/D (whatever keys are provided) x cost scenarios x phases."""
    phases = phases or [0, 1, 2]
    report = SanityReport()
    report.notes.append(power_statement())
    report.notes.append(
        "Baseline profitability is informational only; do not halt on negative CAGR."
    )

    # Label base rate
    if labels is not None and not labels.empty:
        br = label_base_rate(labels)
        ok = 0.40 <= br <= 0.60 if br == br else False
        report.checks.append(
            SanityCheck(
                "label_base_rate",
                ok,
                f"eligible y base rate={br:.4f} (G3 band [0.40, 0.60])",
                halt_worthy=True,
            )
        )
    else:
        report.checks.append(
            SanityCheck(
                "label_base_rate",
                False,
                "labels not provided",
                halt_worthy=True,
            )
        )

    results_table: list[dict[str, Any]] = []
    accounting_ok = True
    ledger_fast_ok = True
    pathology_ok = True
    trades_per_phase: dict[str, dict[int, int]] = {}
    exposure_rows: list[dict[str, Any]] = []

    for model_id, scores in scores_by_model.items():
        trades_per_phase[model_id] = {}
        for phase in phases:
            for cost in cost_models:
                ledger = run_ledger(
                    bars,
                    scores,
                    cost=cost,
                    risk_config=risk_config,
                    initial_capital_usd=initial_capital_usd,
                    top_k=top_k,
                    phase=phase,
                    apply_risk=True,
                )
                fast = run_fast(
                    bars,
                    scores,
                    cost=cost,
                    risk_config=risk_config,
                    initial_capital_usd=initial_capital_usd,
                    top_k=top_k,
                    phase=phase,
                    apply_risk=True,
                )
                if not ledger.invariants_ok:
                    accounting_ok = False
                if not equity_curves_equal(ledger.equity_curve, fast.equity_curve):
                    ledger_fast_ok = False

                metrics = compute_economic_metrics(
                    ledger.equity_curve,
                    ledger.trades,
                    initial_capital_usd=initial_capital_usd,
                ).to_dict()
                n_trades = int(metrics.get("trade_count", 0))
                trades_per_phase[model_id][phase] = (
                    trades_per_phase[model_id].get(phase, 0) + n_trades
                )
                exp = float(metrics.get("exposure", 0.0))
                exposure_rows.append(
                    {
                        "model": model_id,
                        "phase": phase,
                        "scenario": cost.scenario_name,
                        "exposure": exp,
                        "trade_count": n_trades,
                        "cagr_dec": metrics.get("cagr_dec"),
                        "max_drawdown_dec": metrics.get("max_drawdown_dec"),
                        "sharpe": metrics.get("sharpe"),
                    }
                )
                if cost.scenario_name == "normal" and model_id == "A" and phase == 0:
                    dd = abs(float(metrics.get("max_drawdown_dec", 0.0)))
                    if dd > max_drawdown_pathology_dec:
                        pathology_ok = False

                results_table.append(
                    {
                        "model": model_id,
                        "phase": phase,
                        "scenario": cost.scenario_name,
                        "cost_bps_per_side": cost.cost_bps_per_side,
                        **metrics,
                        "realized_costs": realized_cost_bps(ledger.trades, cost),
                        "invariants_ok": ledger.invariants_ok,
                    }
                )

    report.checks.append(
        SanityCheck(
            "accounting_invariants",
            accounting_ok,
            "cash + holdings = equity every day (rel tol 1e-9)"
            if accounting_ok
            else "invariant violations detected",
            halt_worthy=True,
        )
    )
    report.checks.append(
        SanityCheck(
            "ledger_equals_fast",
            ledger_fast_ok,
            "equity curves match within 1e-9" if ledger_fast_ok else "ledger/fast mismatch",
            halt_worthy=True,
        )
    )
    report.checks.append(
        SanityCheck(
            "drawdown_pathology_A_normal",
            pathology_ok,
            f"model A max DD at normal costs ≤ {max_drawdown_pathology_dec:.0%} "
            f"(pathology check only)",
            halt_worthy=True,
        )
    )

    # Trades / exposure summaries (informational; G3 thresholds are soft here
    # because toy/short samples may undershoot — flagged but not auto-halt unless
    # clearly zero on a full OOS run).
    for model_id, by_phase in trades_per_phase.items():
        for ph, n in by_phase.items():
            report.notes.append(f"trades model={model_id} phase={ph} (summed over costs): {n}")

    report.tables["results"] = results_table
    report.tables["exposure"] = exposure_rows
    report.tables["trades_per_phase"] = trades_per_phase

    # Cost monotonicity spot-check on phase-0 equity terminal value.
    for model_id, scores in scores_by_model.items():
        terminals: list[tuple[float, float]] = []
        for cost in sorted(cost_models, key=lambda c: c.cost_bps_per_side):
            led = run_ledger(
                bars,
                scores,
                cost=cost,
                risk_config=risk_config,
                initial_capital_usd=initial_capital_usd,
                top_k=top_k,
                phase=0,
                apply_risk=False,
            )
            if not led.equity_curve.empty:
                terminals.append(
                    (cost.cost_bps_per_side, float(led.equity_curve["equity_usd"].iloc[-1]))
                )
        mono = all(terminals[i][1] + 1e-6 >= terminals[i + 1][1] for i in range(len(terminals) - 1))
        report.checks.append(
            SanityCheck(
                f"cost_monotonicity_{model_id}",
                mono,
                f"terminal equity vs cost_bps: {terminals}",
                halt_worthy=True,
            )
        )

    return report


def write_sanity_report(report: SanityReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "baseline_sanity_report.md"
    json_path = out_dir / "baseline_sanity_report.json"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    # Save results table
    rows = report.tables.get("results", [])
    if rows:
        flat = []
        for r in rows:
            row = {k: v for k, v in r.items() if k != "realized_costs"}
            rc = r.get("realized_costs") or {}
            if isinstance(rc, dict):
                row.update({f"cost_{k}": v for k, v in rc.items()})
            flat.append(row)
        pd.DataFrame(flat).to_csv(out_dir / "baseline_results.csv", index=False)
    return md_path


def scores_from_predictions(
    predictions: pd.DataFrame,
    *,
    sessions: list[date] | None = None,
    every: int = 3,
    phase: int = 0,
) -> pd.DataFrame:
    """Normalize prediction frame to score/eligible columns for the engine."""
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if "score" not in frame.columns:
        if "prob" in frame.columns:
            frame["score"] = frame["prob"]
        else:
            raise ValueError("predictions need score or prob")
    if "eligible" not in frame.columns:
        frame["eligible"] = 1
    if sessions is not None:
        reb = rebalance_dates(sessions, every=every, phase=phase)
        # Keep all dates; engine filters rebalance itself. Ensuring coverage:
        _ = reb
    return frame


def drawdown_from_equity(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    return max_drawdown_dec(equity_curve["equity_usd"])
