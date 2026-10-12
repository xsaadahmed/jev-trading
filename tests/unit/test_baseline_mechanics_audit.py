"""Regression tests for research backtest mechanics bugs found in real-data audit.

Bugs covered:
1. Risk/sizing treated still-open prior cohorts as committed capital on the
   decision day before same-morning exits → mass ``max_position`` /
   ``max_gross_exposure`` rejections and cost-dependent trade counts.
2. Sizing used full equity while caps used ``min(capital, equity)`` → after
   modest gains every new K=3 book was rejected.
3. Economic metrics averaged idle warm-up years into exposure.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from jevlab.backtest.costs import CostModel
from jevlab.backtest.ledger import run_ledger
from jevlab.config.models import RiskConfig
from jevlab.evaluation.metrics import compute_economic_metrics
from jevlab.signals.selection import rebalance_dates


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


def _cost(bps: float, name: str) -> CostModel:
    return CostModel(
        cost_bps_per_side=bps,
        sell_fee_bps=0.3,
        per_symbol_cost_multiplier={"AAA": 1.0, "BBB": 1.0, "CCC": 1.0},
        scenario_name=name,
    )


def _growing_market(n_sessions: int = 16) -> tuple[pd.DataFrame, pd.DataFrame, list[date]]:
    """Rising market so equity exceeds initial capital before later rebalances."""
    start = date(2020, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(n_sessions)]
    rows: list[dict[object, object]] = []
    for i, d in enumerate(sessions):
        # Slow drift upward so later cohorts size against larger equity.
        px = 10.0 * (1.0 + 0.01 * i)
        for sym in ("AAA", "BBB", "CCC"):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "adj_open": px,
                    "adj_close": px,
                }
            )
    bars = pd.DataFrame(rows)
    score_rows = []
    for d in sessions:
        for sym, sc in (("AAA", 3.0), ("BBB", 2.0), ("CCC", 1.0)):
            score_rows.append({"symbol": sym, "date": d, "score": sc, "eligible": 1})
    return bars, pd.DataFrame(score_rows), sessions


def test_second_rebalance_not_rejected_at_34pct_cap() -> None:
    """Prior cohort still open at decision time must not block next K=3 book."""
    bars, scores, sessions = _growing_market()
    reb = rebalance_dates(sessions, every=3, phase=0)
    assert len(reb) >= 3
    result = run_ledger(
        bars,
        scores,
        cost=_cost(0.0, "zero"),
        risk_config=_risk(),
        initial_capital_usd=300.0,
        top_k=3,
        phase=0,
        apply_risk=True,
        enable_drawdown_kill=False,
    )
    buys = result.trades.loc[result.trades["side"] == "buy"]
    # At least two full cohorts (6+ buys) — second rebalance must fill.
    assert len(buys) >= 6
    max_pos_rejects = [r for r in result.rejections if r["reason"] == "max_position"]
    max_gross_rejects = [r for r in result.rejections if r["reason"] == "max_gross_exposure"]
    assert max_pos_rejects == []
    assert max_gross_rejects == []


def test_trade_count_invariant_across_cost_scenarios() -> None:
    bars, scores, _ = _growing_market()
    counts: list[int] = []
    for bps, name in ((1.0, "optimistic"), (2.5, "normal"), (6.0, "pessimistic"), (12.0, "stress")):
        led = run_ledger(
            bars,
            scores,
            cost=_cost(bps, name),
            risk_config=_risk(),
            initial_capital_usd=300.0,
            top_k=3,
            phase=0,
            apply_risk=True,
            enable_drawdown_kill=False,
        )
        metrics = compute_economic_metrics(led.equity_curve, led.trades).to_dict()
        counts.append(int(metrics["trade_count"]))
    assert len(set(counts)) == 1
    assert counts[0] > 0


def test_oos_metric_window_does_not_dilute_exposure() -> None:
    """Idle pre-window days must not pull mean exposure toward zero."""
    bars, scores, sessions = _growing_market(n_sessions=20)
    led = run_ledger(
        bars,
        scores,
        cost=_cost(0.0, "zero"),
        risk_config=_risk(),
        initial_capital_usd=300.0,
        top_k=3,
        phase=0,
        apply_risk=True,
        enable_drawdown_kill=False,
    )
    # Pretend OOS starts at session index 9 (after some idle if scores missing).
    # Use a window covering only the active trading stretch.
    oos_start = sessions[6]
    oos_end = sessions[-1]
    full = compute_economic_metrics(led.equity_curve, led.trades).to_dict()
    oos = compute_economic_metrics(
        led.equity_curve,
        led.trades,
        metric_start=oos_start,
        metric_end=oos_end,
    ).to_dict()
    # With continuous phase-0 coverage in-window, OOS exposure should be high.
    assert oos["exposure"] >= 0.30
    assert oos["trade_count"] == full["trade_count"] or oos["trade_count"] > 0


def test_research_path_ignores_live_drawdown_kill_threshold() -> None:
    """evaluate_orders must not kill_switch-reject when enable_drawdown_kill=False."""
    bars, scores, _ = _growing_market(n_sessions=20)
    # Force a large adverse move after first cohort so DD >> 20%.
    bars = bars.copy()
    sessions = sorted(bars["date"].unique())
    crash_days = sessions[5:]
    for d in crash_days:
        mask = bars["date"] == d
        bars.loc[mask, "adj_open"] = bars.loc[mask, "adj_open"].astype(float) * 0.5
        bars.loc[mask, "adj_close"] = bars.loc[mask, "adj_close"].astype(float) * 0.5
    led = run_ledger(
        bars,
        scores,
        cost=_cost(0.0, "zero"),
        risk_config=_risk(max_drawdown_pct=20.0),
        initial_capital_usd=300.0,
        top_k=3,
        phase=0,
        apply_risk=True,
        enable_drawdown_kill=False,
    )
    kill_rej = [r for r in led.rejections if r["reason"] == "kill_switch"]
    assert kill_rej == []
    assert len(led.trades.loc[led.trades["side"] == "buy"]) >= 6


def test_sizing_uses_strategy_capital_cap_not_unbounded_equity() -> None:
    bars, scores, _ = _growing_market(n_sessions=20)
    led = run_ledger(
        bars,
        scores,
        cost=_cost(0.0, "zero"),
        risk_config=_risk(),
        initial_capital_usd=300.0,
        top_k=3,
        phase=0,
        apply_risk=True,
        enable_drawdown_kill=False,
    )
    buys = led.trades.loc[led.trades["side"] == "buy"]
    # Each leg sized to capital/K = 100, never ~equity/K after large gains.
    assert (buys["notional_usd"].astype(float) <= 100.0 + 1e-6).all()
