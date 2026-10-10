"""Backtest session helpers: entry/exit offsets relative to decision date.

Decision after close of session t; entry open is t+1; exit open is t+4
(horizon_sessions=3 ⇒ entry_lag 1 + hold 3).
"""

from __future__ import annotations

from datetime import date


def session_index_map(sessions: list[date]) -> dict[date, int]:
    ordered = sorted(sessions)
    return {d: i for i, d in enumerate(ordered)}


def offset_session(
    decision: date,
    *,
    sessions: list[date],
    offset: int,
) -> date | None:
    """Return the session ``offset`` sessions after ``decision``, or None if OOB."""
    idx = session_index_map(sessions)
    i = idx.get(decision)
    if i is None:
        return None
    j = i + offset
    if j < 0 or j >= len(sessions):
        return None
    ordered = sorted(sessions)
    return ordered[j]


def entry_session(decision: date, sessions: list[date], entry_lag: int = 1) -> date | None:
    """Fill session for entry: open of decision + entry_lag."""
    return offset_session(decision, sessions=sessions, offset=entry_lag)


def exit_session(
    decision: date,
    sessions: list[date],
    *,
    entry_lag: int = 1,
    horizon_sessions: int = 3,
) -> date | None:
    """Fill session for exit: open of decision + entry_lag + horizon_sessions."""
    return offset_session(
        decision,
        sessions=sessions,
        offset=entry_lag + horizon_sessions,
    )
