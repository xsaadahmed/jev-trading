"""Random-null engine: seed policy, selection-only randomization, uniformity."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from jevlab.backtest.costs import CostModel
from jevlab.config.models import RiskConfig
from jevlab.signals.selection import rebalance_dates
from jevlab.v2.null_engine import (
    BASE_SEED,
    child_seed,
    randomize_ranking,
    rng_for,
    run_null_distribution,
    run_null_draw,
    seed_policy_dict,
)
from jevlab.v2.statistics import FROZEN_N


def _risk() -> RiskConfig:
    return RiskConfig.model_validate(
        dict(
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
    )


def _toy(n_days: int = 18, n_sym: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list[date]]:
    start = date(2020, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(n_days)]
    symbols = [f"S{i}" for i in range(n_sym)]
    bars = []
    scores = []
    for i, d in enumerate(sessions):
        for j, s in enumerate(symbols):
            bars.append(
                {
                    "symbol": s,
                    "date": d,
                    "adj_open": 10.0 + j + 0.01 * i,
                    "adj_close": 10.0 + j + 0.011 * i,
                }
            )
            scores.append(
                {
                    "symbol": s,
                    "date": d,
                    "score": float(n_sym - j),
                    "eligible": 0 if (i % 7 == 0 and j == 0) else 1,
                }
            )
    return pd.DataFrame(bars), pd.DataFrame(scores), sessions


def test_seed_policy_frozen() -> None:
    policy = seed_policy_dict()
    assert policy["n"] == FROZEN_N == 2000
    assert policy["base_seed"] == BASE_SEED == 15224036858585574915
    assert child_seed(1, 0) != child_seed(2, 0)
    assert child_seed(1, 0) != child_seed(1, 1)
    assert child_seed(1, 0) == (BASE_SEED + 1 + 0) % 2**64


def test_rng_repeatable_and_distinct_streams() -> None:
    a = rng_for(1, 0).random(8)
    b = rng_for(1, 0).random(8)
    c = rng_for(2, 0).random(8)
    d = rng_for(1, 1).random(8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert not np.allclose(a, d)


def test_selection_only_randomization() -> None:
    _bars, scores, sessions = _toy()
    out1 = randomize_ranking(scores, draw_id=1, phase=0, sessions=sessions)
    out2 = randomize_ranking(scores, draw_id=1, phase=0, sessions=sessions)
    pd.testing.assert_frame_equal(
        out1.sort_values(["date", "symbol"]).reset_index(drop=True),
        out2.sort_values(["date", "symbol"]).reset_index(drop=True),
    )
    reb = set(rebalance_dates(sessions, every=3, phase=0))
    for dt, g in out1.groupby("date"):
        elig = g.loc[g["eligible"] == 1]
        orig = scores.loc[scores["date"] == dt]
        assert set(elig["symbol"]) <= set(orig.loc[orig["eligible"] == 1, "symbol"])
        n_elig = int((orig["eligible"] == 1).sum())
        if dt in reb:
            assert int(g["selected"].sum()) == min(3, n_elig)
        ineligible_selected = g.loc[(g["eligible"] != 1) & (g["selected"] == 1)]
        assert ineligible_selected.empty
    # Decision calendar is the frozen rebalance dates - not randomized.
    assert set(out1["date"]) == reb


def test_costs_do_not_change_selections() -> None:
    bars, scores, sessions = _toy()
    sel = randomize_ranking(scores, draw_id=3, phase=1, sessions=sessions)
    cheap = CostModel(1.0, 0.0, {}, "optimistic")
    dear = CostModel(12.0, 0.3, {}, "stress")
    d1 = run_null_draw(
        bars,
        scores,
        draw_id=3,
        sessions=sessions,
        cost=cheap,
        risk_config=_risk(),
        metric_start=sessions[0],
        metric_end=sessions[-1],
        phases=(1,),
    )
    d2 = run_null_draw(
        bars,
        scores,
        draw_id=3,
        sessions=sessions,
        cost=dear,
        risk_config=_risk(),
        metric_start=sessions[0],
        metric_end=sessions[-1],
        phases=(1,),
    )
    a = d1.selections[1].sort_values(["date", "symbol"])
    b = d2.selections[1].sort_values(["date", "symbol"])
    pd.testing.assert_series_equal(
        a["selected"].reset_index(drop=True), b["selected"].reset_index(drop=True)
    )
    _ = sel


def test_ledger_fast_agree_on_null_draw() -> None:
    bars, scores, sessions = _toy()
    cost = CostModel(2.5, 0.3, {}, "normal")
    draw = run_null_draw(
        bars,
        scores,
        draw_id=1,
        sessions=sessions,
        cost=cost,
        risk_config=_risk(),
        metric_start=sessions[0],
        metric_end=sessions[-1],
        check_fast=True,
        phases=(0, 1, 2),
    )
    assert set(draw.phase_sharpes) == {0, 1, 2}


def test_randomize_rejects_synthetic_sealed_dates() -> None:
    from jevlab.v2.dev_io import SealedPathError

    sessions = [date(2023, 12, 22), date(2024, 1, 2)]
    scores = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": sessions,
            "score": [1.0, 1.0],
            "eligible": [1, 1],
        }
    )
    with __import__("pytest").raises(SealedPathError):
        randomize_ranking(scores, draw_id=1, phase=0, sessions=sessions)


def test_null_distribution_refuses_silent_n_change() -> None:
    bars, scores, sessions = _toy()
    cost = CostModel(2.5, 0.3, {}, "normal")
    with __import__("pytest").raises(ValueError, match="frozen random-null N"):
        run_null_distribution(
            bars,
            scores,
            sessions=sessions,
            cost=cost,
            n=8,
            risk_config=_risk(),
            metric_start=sessions[0],
            metric_end=sessions[-1],
        )
    arr = run_null_distribution(
        bars,
        scores,
        sessions=sessions,
        cost=cost,
        n=3,
        allow_reduced_n=True,
        risk_config=_risk(),
        metric_start=sessions[0],
        metric_end=sessions[-1],
        phases=(0,),
    )
    assert len(arr) == 3


def test_uniformity_fixture() -> None:
    """On always-eligible 4 names, K=3, selection frequencies are near-uniform."""
    start = date(2020, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(12)]
    symbols = ["A", "B", "C", "D"]
    scores = []
    for d in sessions:
        for s in symbols:
            scores.append({"symbol": s, "date": d, "score": 1.0, "eligible": 1})
    frame = pd.DataFrame(scores)
    counts = {s: 0 for s in symbols}
    n_draws = 80
    for r in range(1, n_draws + 1):
        out = randomize_ranking(frame, draw_id=r, phase=0, sessions=sessions, k=3)
        reb0 = rebalance_dates(sessions, every=3, phase=0)[0]
        picked = out.loc[(out["date"] == reb0) & (out["selected"] == 1), "symbol"]
        for s in picked:
            counts[s] += 1
    freqs = np.array(list(counts.values()), dtype=float) / n_draws
    # Each name selected with prob 3/4 = 0.75; allow Monte Carlo noise.
    assert freqs.min() > 0.45
    assert freqs.max() < 0.95
    assert abs(freqs.mean() - 0.75) < 0.08
