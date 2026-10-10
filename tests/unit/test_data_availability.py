"""Availability policy and as_of filtering (MASTER_SPEC §6.1)."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from jevlab.data.availability import (
    as_of,
    assign_bars_available_at,
    assign_fred_available_at,
    bars_available_at,
    decision_time,
    fred_available_at,
)
from jevlab.utils.calendar import is_nyse_session, next_nyse_session

ET = ZoneInfo("America/New_York")


def test_bars_available_at_is_session_1800_et() -> None:
    ts = bars_available_at(date(2024, 7, 5))
    assert ts.tz_convert(ET).hour == 18
    assert ts.tz_convert(ET).minute == 0
    assert ts.date() == date(2024, 7, 5)


def test_fred_available_at_next_session_0800_across_weekend() -> None:
    # Friday observation → Monday 08:00 ET
    friday = date(2024, 7, 5)
    assert is_nyse_session(friday)
    avail = fred_available_at(friday)
    assert next_nyse_session(friday) == date(2024, 7, 8)
    assert avail.tz_convert(ET).date() == date(2024, 7, 8)
    assert avail.tz_convert(ET).hour == 8


def test_fred_available_at_across_july4_holiday() -> None:
    # 2024-07-03 (Wed) → next session 2024-07-05 (Thu; Jul 4 holiday)
    wed = date(2024, 7, 3)
    assert not is_nyse_session(date(2024, 7, 4))
    avail = fred_available_at(wed)
    assert avail.tz_convert(ET).date() == date(2024, 7, 5)
    assert avail.tz_convert(ET).hour == 8


def test_half_day_still_uses_1800_et_policy() -> None:
    # 2024-07-03 was a full day; 2024-11-29 is day after Thanksgiving (half day historically).
    half = date(2024, 11, 29)
    assert is_nyse_session(half)
    ts = bars_available_at(half)
    assert ts.tz_convert(ET).hour == 18


def test_as_of_excludes_rows_after_decision_time() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": [date(2024, 7, 3), date(2024, 7, 5)],
            "adj_close": [100.0, 101.0],
        }
    )
    bars = assign_bars_available_at(bars)
    dt = decision_time(date(2024, 7, 3))  # 18:30 ET on Jul 3
    filtered = as_of(bars, dt)
    assert list(filtered["date"]) == [date(2024, 7, 3)]


def test_as_of_excludes_same_date_vix_at_decision_time() -> None:
    """VIX(t) is not available at decision_time(t); VIX(t-1) is (§6.1)."""
    session_t = date(2024, 7, 5)  # Friday
    prev = date(2024, 7, 3)  # Wednesday (Thu Jul 4 holiday)
    series = pd.DataFrame(
        {
            "series_id": ["VIXCLS", "VIXCLS"],
            "obs_date": [prev, session_t],
            "value": [12.0, 13.0],
        }
    )
    series = assign_fred_available_at(series)
    dt = decision_time(session_t)
    filtered = as_of(series, dt)
    assert list(filtered["obs_date"]) == [prev]
    # Same-date VIX becomes available next session 08:00 ET — after decision_time(t).
    assert fred_available_at(session_t) > dt


def test_as_of_requires_timezone_aware_decision_time() -> None:
    table = pd.DataFrame(
        {
            "available_at": [pd.Timestamp("2024-07-05 18:00", tz=ET)],
            "value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        as_of(table, datetime(2024, 7, 5, 18, 30))
