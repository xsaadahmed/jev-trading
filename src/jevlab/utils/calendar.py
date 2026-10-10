"""NYSE session calendar helpers via exchange_calendars."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Any

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=1)
def _nyse() -> Any:
    return xcals.get_calendar("XNYS")


def nyse_session_dates(
    start: date | datetime | str,
    end: date | datetime | str,
) -> list[date]:
    """Return NYSE session dates in ``[start, end]`` inclusive (date labels)."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError(f"end {end_ts.date()} is before start {start_ts.date()}")
    sessions = _nyse().sessions_in_range(start_ts, end_ts)
    return [date(ts.year, ts.month, ts.day) for ts in sessions]


def is_nyse_session(day: date | datetime | str) -> bool:
    """Return True if ``day`` is an NYSE trading session."""
    day_ts = pd.Timestamp(day).normalize()
    return bool(_nyse().is_session(day_ts))


def next_nyse_session(day: date | datetime | str) -> date:
    """Return the next NYSE session strictly after ``day``."""
    day_ts = pd.Timestamp(day).normalize()
    nxt = _nyse().date_to_session(day_ts, direction="next")
    # date_to_session with direction=next returns day itself if it is a session.
    if nxt.normalize() == day_ts and _nyse().is_session(day_ts):
        nxt = _nyse().next_session(day_ts)
    elif nxt.normalize() == day_ts:
        nxt = _nyse().date_to_session(day_ts + pd.Timedelta(days=1), direction="next")
    return date(nxt.year, nxt.month, nxt.day)


def previous_nyse_session(day: date | datetime | str) -> date:
    """Return the previous NYSE session strictly before ``day``."""
    day_ts = pd.Timestamp(day).normalize()
    prev = _nyse().date_to_session(day_ts, direction="previous")
    if prev.normalize() == day_ts and _nyse().is_session(day_ts):
        prev = _nyse().previous_session(day_ts)
    return date(prev.year, prev.month, prev.day)
