"""Economic metrics and MDE helper tests (Phase 4 / §12.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jevlab.evaluation.metrics import (
    beta_alpha_hac,
    compute_economic_metrics,
    max_drawdown_dec,
    minimum_detectable_effect,
    power_statement,
)


def test_max_drawdown_and_metrics_smoke() -> None:
    eq = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "cash_usd": [100.0, 50.0, 50.0, 50.0, 50.0],
            "market_value_usd": [0.0, 40.0, 30.0, 45.0, 60.0],
            "equity_usd": [100.0, 90.0, 80.0, 95.0, 110.0],
        }
    )
    assert max_drawdown_dec(eq["equity_usd"]) == pytest.approx(-0.2)
    trades = pd.DataFrame(
        [
            {
                "session": "2020-01-02",
                "symbol": "SPY",
                "side": "buy",
                "qty": 1.0,
                "price_usd": 50.0,
                "fee_usd": 0.0,
                "notional_usd": 50.0,
                "client_order_id": "b",
                "cohort_id": "c1",
                "decision_date": "2020-01-01",
            },
            {
                "session": "2020-01-05",
                "symbol": "SPY",
                "side": "sell",
                "qty": 1.0,
                "price_usd": 60.0,
                "fee_usd": 0.0,
                "notional_usd": 60.0,
                "client_order_id": "s",
                "cohort_id": "c1",
                "decision_date": "2020-01-01",
            },
        ]
    )
    m = compute_economic_metrics(eq, trades, initial_capital_usd=100.0).to_dict()
    assert m["total_return_dec"] == pytest.approx(0.1)
    assert m["trade_count"] == 1.0
    assert m["hit_rate"] == 1.0


def test_beta_alpha_hac_runs() -> None:
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, 200))
    strat = 0.001 + 1.2 * bench + rng.normal(0, 0.005, 200)
    out = beta_alpha_hac(strat, bench)
    assert abs(out["beta"] - 1.2) < 0.15
    assert out["beta_se_hac"] > 0


def test_mde_and_power_statement() -> None:
    mde = minimum_detectable_effect(n_blocks=419, corr=0.9)
    assert mde["mde_delta_sharpe_unit"] > 0
    text = power_statement()
    assert "inconclusive" in text
    assert "date-blocks" in text
