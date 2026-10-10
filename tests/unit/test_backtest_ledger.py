"""Ledger / fast / costs tests (MASTER_SPEC Phase 4)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from jevlab.backtest.costs import CostModel, require_cost_model
from jevlab.backtest.fast import equity_curves_equal, run_fast
from jevlab.backtest.ledger import EQUITY_TOL_REL, run_ledger
from jevlab.config.models import RiskConfig


def _risk(**kwargs: object) -> RiskConfig:
    base = dict(
        max_position_pct=34.0,
        max_gross_exposure_pct=100.0,
        max_new_orders_per_day=6,
        max_daily_loss_pct=5.0,
        max_drawdown_pct=20.0,
        min_live_position_notional_usd=1.0,
        stale_data_halt=True,
        max_missing_symbols=2,
        submit_window_start_et="08:30",
        submit_window_end_et="09:25",
        kill_switch_file="KILL_SWITCH",
        kill_switch_env_var="TRADING_HALT",
    )
    base.update(kwargs)
    return RiskConfig.model_validate(base)


def _cost(bps: float = 0.0, fee: float = 0.0, name: str = "test") -> CostModel:
    return CostModel(
        cost_bps_per_side=bps,
        sell_fee_bps=fee,
        per_symbol_cost_multiplier={"AAA": 1.0, "BBB": 1.0, "CCC": 1.0},
        scenario_name=name,
    )


def _toy_market() -> tuple[pd.DataFrame, pd.DataFrame, list[date]]:
    """3 assets, 10 sessions — deterministic prices for hand checks.

    Sessions d0..d9. Phase 0 rebalance on d0, d3, d6, d9.
    On d0 scores pick AAA,BBB,CCC (all eligible). Entry d1 open, exit d4 open.
    Flat prices except AAA rises so P&L is known with zero costs.
    """
    start = date(2020, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(10)]  # calendar days ok for toy
    rows: list[dict[object, object]] = []
    # Constant prices: AAA=10, BBB=20, CCC=40 except AAA open jumps on d4.
    opens = {
        "AAA": [10.0] * 10,
        "BBB": [20.0] * 10,
        "CCC": [40.0] * 10,
    }
    closes = {
        "AAA": [10.0] * 10,
        "BBB": [20.0] * 10,
        "CCC": [40.0] * 10,
    }
    # Exit open for first cohort (d4): AAA +10% → 11.0
    opens["AAA"][4] = 11.0
    closes["AAA"][4] = 11.0
    for i, d in enumerate(sessions):
        for sym in ("AAA", "BBB", "CCC"):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "adj_open": opens[sym][i],
                    "adj_close": closes[sym][i],
                }
            )
    bars = pd.DataFrame(rows)
    # Scores: prefer AAA > BBB > CCC on every day; all eligible.
    score_rows = []
    for d in sessions:
        for sym, sc in (("AAA", 3.0), ("BBB", 2.0), ("CCC", 1.0)):
            score_rows.append({"symbol": sym, "date": d, "score": sc, "eligible": 1})
    scores = pd.DataFrame(score_rows)
    return bars, scores, sessions


def test_costs_required() -> None:
    with pytest.raises(ValueError, match="required"):
        require_cost_model(None)


def test_toy_market_hand_computed_pnl() -> None:
    """With zero costs and K=3, first cohort: AAA +10%, BBB/CCC flat.

    Capital 300 → $100 each. Buy at d1: AAA 10 shares, BBB 5, CCC 2.5.
    Exit d4: AAA 10*11=110, BBB 100, CCC 100 → cash 310. Equity 310.
    """
    bars, scores, _sessions = _toy_market()
    result = run_ledger(
        bars,
        scores,
        cost=_cost(0.0, 0.0),
        risk_config=_risk(max_position_pct=40.0),  # allow ~33% with headroom
        initial_capital_usd=300.0,
        top_k=3,
        phase=0,
        apply_risk=True,
    )
    assert result.invariants_ok
    trades = result.trades
    buys = trades.loc[trades["side"] == "buy"]
    # First cohort decision d0 → entry session index 1
    first_entry = sorted(buys["session"].unique())[0]
    cohort_buys = buys.loc[buys["session"] == first_entry]
    assert set(cohort_buys["symbol"]) == {"AAA", "BBB", "CCC"}
    aaa_buy = cohort_buys.loc[cohort_buys["symbol"] == "AAA"].iloc[0]
    assert aaa_buy["qty"] == pytest.approx(10.0)
    assert aaa_buy["notional_usd"] == pytest.approx(100.0)

    sells = trades.loc[trades["side"] == "sell"]
    aaa_sell = sells.loc[sells["symbol"] == "AAA"].iloc[0]
    assert aaa_sell["notional_usd"] == pytest.approx(110.0)

    # After first exit session, equity should be 310 before next entries consume cash.
    exit_day = aaa_sell["session"]
    eq_at_exit = result.equity_curve.loc[
        result.equity_curve["date"] == exit_day, "equity_usd"
    ].iloc[0]
    # Same day may also open new cohort; equity still marks correctly.
    assert eq_at_exit == pytest.approx(310.0, rel=1e-9) or eq_at_exit == pytest.approx(
        310.0, abs=1e-6
    )


def test_accounting_invariant_every_day() -> None:
    bars, scores, _ = _toy_market()
    result = run_ledger(
        bars,
        scores,
        cost=_cost(2.5, 0.3),
        risk_config=_risk(max_position_pct=40.0),
        initial_capital_usd=10_000.0,
        top_k=3,
        phase=0,
    )
    eq = result.equity_curve
    recon = eq["cash_usd"] + eq["market_value_usd"]
    denom = eq["equity_usd"].abs().clip(lower=1e-12)
    assert ((recon - eq["equity_usd"]).abs() / denom <= EQUITY_TOL_REL).all()
    assert result.invariants_ok


def test_ledger_equals_fast() -> None:
    bars, scores, _ = _toy_market()
    kwargs = dict(
        cost=_cost(2.5, 0.3, "normal"),
        risk_config=_risk(max_position_pct=40.0),
        initial_capital_usd=10_000.0,
        top_k=3,
        phase=1,
        apply_risk=True,
    )
    led = run_ledger(bars, scores, **kwargs)  # type: ignore[arg-type]
    fast = run_fast(bars, scores, **kwargs)  # type: ignore[arg-type]
    assert equity_curves_equal(led.equity_curve, fast.equity_curve)


def test_costs_never_raise_equity() -> None:
    bars, scores, _ = _toy_market()
    risk = _risk(max_position_pct=40.0)
    zero = run_ledger(
        bars, scores, cost=_cost(0.0, 0.0), risk_config=risk, initial_capital_usd=10_000.0, phase=0
    )
    high = run_ledger(
        bars,
        scores,
        cost=_cost(12.0, 0.3, "stress"),
        risk_config=risk,
        initial_capital_usd=10_000.0,
        phase=0,
    )
    z = float(zero.equity_curve["equity_usd"].iloc[-1])
    h = float(high.equity_curve["equity_usd"].iloc[-1])
    assert h <= z + 1e-6


def test_delayed_execution_differs() -> None:
    bars, scores, sessions = _toy_market()
    # Make close[t+1] differ from open[t+1] for AAA on entry day of first cohort (index 1).
    bars = bars.copy()
    entry = sessions[1]
    mask = (bars["date"] == entry) & (bars["symbol"] == "AAA")
    bars.loc[mask, "adj_close"] = 10.5
    risk = _risk(max_position_pct=40.0)
    cost = _cost(0.0, 0.0)
    normal = run_ledger(
        bars, scores, cost=cost, risk_config=risk, initial_capital_usd=300.0, fill_mode="open_t1"
    )
    delayed = run_ledger(
        bars, scores, cost=cost, risk_config=risk, initial_capital_usd=300.0, fill_mode="delayed"
    )
    nb = normal.trades.loc[
        (normal.trades["side"] == "buy") & (normal.trades["symbol"] == "AAA")
    ].iloc[0]
    db = delayed.trades.loc[
        (delayed.trades["side"] == "buy") & (delayed.trades["symbol"] == "AAA")
    ].iloc[0]
    assert nb["price_usd"] == pytest.approx(10.0)
    assert db["price_usd"] == pytest.approx(10.5)
    assert not equity_curves_equal(normal.equity_curve, delayed.equity_curve)
