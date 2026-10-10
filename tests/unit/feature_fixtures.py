"""Shared synthetic market fixtures for feature tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from jevlab.data.availability import assign_bars_available_at, assign_fred_available_at
from jevlab.utils.calendar import nyse_session_dates


def make_sessions(n: int, start: date = date(2018, 1, 2)) -> list[date]:
    """Return ``n`` NYSE sessions starting at or after ``start``."""
    end = start + timedelta(days=int(n * 2.2) + 60)
    sessions = nyse_session_dates(start, end)
    return sessions[:n]


def make_bars(
    symbols: list[str],
    sessions: list[date],
    *,
    seed: int = 0,
    price0: float = 100.0,
) -> pd.DataFrame:
    """Synthetic OHLCV with available_at for each symbol/session."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for j, sym in enumerate(symbols):
        px = price0 * (1.0 + 0.01 * j)
        for _i, d in enumerate(sessions):
            shock = float(rng.normal(0.0005, 0.01))
            px = max(px * float(np.exp(shock)), 1.0)
            o = px * float(np.exp(rng.normal(0, 0.001)))
            h = max(o, px) * 1.001
            low = min(o, px) * 0.999
            vol = float(1_000_000 + int(rng.integers(0, 100_000)))
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "vendor": "tiingo",
                    "raw_open": o,
                    "raw_high": h,
                    "raw_low": low,
                    "raw_close": px,
                    "raw_volume": int(vol),
                    "adj_open": o,
                    "adj_high": h,
                    "adj_low": low,
                    "adj_close": px,
                    "adj_volume": vol,
                    "div_cash": 0.0,
                    "split_factor": 1.0,
                }
            )
    return assign_bars_available_at(pd.DataFrame(rows))


def make_vix(sessions: list[date], *, seed: int = 1) -> pd.DataFrame:
    """Synthetic VIXCLS-like series on the same session dates."""
    rng = np.random.default_rng(seed)
    level = 15.0
    rows = []
    for d in sessions:
        level = max(level + float(rng.normal(0, 0.5)), 9.0)
        rows.append({"series_id": "VIXCLS", "obs_date": d, "value": level})
    return assign_fred_available_at(pd.DataFrame(rows))
