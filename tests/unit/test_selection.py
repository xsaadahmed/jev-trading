"""Top-K selection tests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from jevlab.signals.selection import rebalance_dates, select_top_k


def test_select_top_k_ties_broken_by_symbol_order() -> None:
    scores = pd.DataFrame(
        {
            "date": [date(2019, 1, 2)] * 4,
            "symbol": ["QQQ", "SPY", "IWM", "DIA"],
            "score": [0.5, 0.5, 0.1, 0.9],
            "eligible": [1, 1, 1, 1],
        }
    )
    out = select_top_k(scores, k=2)
    selected = out.loc[out["selected"] == 1, "symbol"].tolist()
    # DIA has highest score; SPY and QQQ tie at 0.5 → QQQ before SPY by symbol order.
    assert selected[0] == "DIA"
    assert selected[1] == "QQQ"


def test_select_top_k_ignores_ineligible() -> None:
    scores = pd.DataFrame(
        {
            "date": [date(2019, 1, 3)] * 3,
            "symbol": ["AAA", "BBB", "CCC"],
            "score": [0.9, 0.8, 0.7],
            "eligible": [0, 1, 1],
        }
    )
    out = select_top_k(scores, k=2)
    assert set(out.loc[out["selected"] == 1, "symbol"]) == {"BBB", "CCC"}


def test_rebalance_dates_phase() -> None:
    sessions = [date(2019, 1, d) for d in (2, 3, 4, 7, 8, 9)]
    # Use real sorted list
    reb = rebalance_dates(sessions, every=3, phase=0)
    assert reb[0] == sessions[0]
    assert all(sessions.index(d) % 3 == 0 for d in reb)
