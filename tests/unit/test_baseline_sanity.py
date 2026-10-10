"""Baseline sanity report on a tiny synthetic panel (Phase 4)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from jevlab.backtest.costs import CostModel
from jevlab.config.models import RiskConfig
from jevlab.reporting.baseline_sanity import (
    build_sanity_report,
    label_base_rate,
    write_sanity_report,
)


def _risk() -> RiskConfig:
    return RiskConfig.model_validate(
        {
            "max_position_pct": 40.0,
            "max_gross_exposure_pct": 100.0,
            "max_new_orders_per_day": 6,
            "max_daily_loss_pct": 5.0,
            "max_drawdown_pct": 50.0,
            "min_live_position_notional_usd": 1.0,
            "stale_data_halt": True,
            "max_missing_symbols": 2,
            "submit_window_start_et": "08:30",
            "submit_window_end_et": "09:25",
            "kill_switch_file": "KILL_SWITCH",
            "kill_switch_env_var": "TRADING_HALT",
        }
    )


def _panel(n_days: int = 30) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    start = date(2019, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(n_days)]
    symbols = ["AAA", "BBB", "CCC"]
    bar_rows = []
    score_rows = []
    label_rows = []
    rng = np.random.default_rng(0)
    px = {s: 100.0 for s in symbols}
    for d in sessions:
        for s in symbols:
            shock = 1.0 + float(rng.normal(0, 0.005))
            o = px[s]
            c = o * shock
            px[s] = c
            bar_rows.append({"symbol": s, "date": d, "adj_open": o, "adj_close": c})
            score_rows.append(
                {
                    "symbol": s,
                    "date": d,
                    "score": float(rng.random()),
                    "eligible": 1,
                }
            )
            label_rows.append(
                {
                    "symbol": s,
                    "date": d,
                    "y": int(rng.random() > 0.48),
                    "eligible": 1,
                }
            )
    bars = pd.DataFrame(bar_rows)
    scores = pd.DataFrame(score_rows)
    labels = pd.DataFrame(label_rows)
    scores_by = {"R0": scores.copy(), "A": scores.copy(), "D": scores.copy()}
    # R0-like: invert a fake ret
    scores_by["R0"]["score"] = -scores_by["R0"]["score"]
    return bars, scores_by, labels


def test_label_base_rate_band() -> None:
    labels = pd.DataFrame(
        {
            "y": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "eligible": [1] * 10,
        }
    )
    assert label_base_rate(labels) == 0.5


def test_sanity_report_mechanics(tmp_path: object) -> None:
    from pathlib import Path

    bars, scores_by, labels = _panel()
    costs = [
        CostModel(
            cost_bps_per_side=bps,
            sell_fee_bps=0.3,
            per_symbol_cost_multiplier={"AAA": 1.0, "BBB": 1.0, "CCC": 1.0},
            scenario_name=name,
        )
        for name, bps in [
            ("optimistic", 1.0),
            ("normal", 2.5),
            ("pessimistic", 6.0),
            ("stress", 12.0),
        ]
    ]
    report = build_sanity_report(
        bars=bars,
        scores_by_model=scores_by,
        labels=labels,
        cost_models=costs,
        risk_config=_risk(),
        initial_capital_usd=10_000.0,
        top_k=3,
        phases=[0, 1, 2],
        max_drawdown_pathology_dec=0.90,  # loose for random toy
    )
    assert report.mechanics_ok
    assert any(c.name == "ledger_equals_fast" and c.passed for c in report.checks)
    assert any(c.name == "accounting_invariants" and c.passed for c in report.checks)
    assert "results" in report.tables
    assert len(report.tables["results"]) == 3 * 3 * 4  # models x phases x costs
    out = write_sanity_report(report, Path(str(tmp_path)) / "sanity")
    assert out.is_file()
    assert (out.parent / "baseline_results.csv").is_file()
