"""Purge / embargo helpers for walk-forward CV (MASTER_SPEC §10.4)."""

from __future__ import annotations

from datetime import date

import pandas as pd


def embargo_cutoff_date(
    sessions: list[date],
    test_start: date,
    *,
    embargo_sessions: int = 5,
) -> date | None:
    """Latest decision date allowed in training for a test fold starting at ``test_start``.

    Training rows must satisfy ``s(t) <= s(T0) - embargo_sessions`` so a training
    label's window (open t+1 .. open t+4) does not intersect the test period.
    """
    if test_start not in sessions:
        # Use the first session >= test_start as T0.
        later = [d for d in sessions if d >= test_start]
        if not later:
            return None
        test_start = later[0]
    i0 = sessions.index(test_start)
    train_last_i = i0 - embargo_sessions
    if train_last_i < 0:
        return None
    return sessions[train_last_i]


def filter_train_dates(
    dates: list[date] | pd.Series,
    *,
    sessions: list[date],
    test_start: date,
    embargo_sessions: int = 5,
) -> list[date]:
    """Return dates from ``dates`` that are valid training dates under the embargo."""
    cutoff = embargo_cutoff_date(sessions, test_start, embargo_sessions=embargo_sessions)
    if cutoff is None:
        return []
    date_list = list(dates) if not isinstance(dates, pd.Series) else dates.tolist()
    return [d for d in date_list if d <= cutoff]


def label_window_intersects_test(
    decision_date: date,
    *,
    sessions: list[date],
    test_start: date,
    test_end: date,
    entry_lag_sessions: int = 1,
    horizon_sessions: int = 3,
) -> bool:
    """True if the label window [entry, exit] overlaps the test decision period.

    The label for decision t uses opens from entry=t+lag through exit=t+lag+horizon.
    Overlap with the test period is defined as any of those sessions falling on or
    after ``test_start`` while the decision date is before ``test_start`` (training
    leakage into test), or more generally if entry/exit land inside [test_start, test_end]
    when decision is outside — we check: exit_date >= test_start and decision_date < test_start.
    """
    if decision_date not in sessions:
        return False
    i = sessions.index(decision_date)
    exit_offset = entry_lag_sessions + horizon_sessions
    if i + exit_offset >= len(sessions):
        return False
    entry_date = sessions[i + entry_lag_sessions]
    exit_date = sessions[i + exit_offset]
    # Intersects test if the open-to-open window touches any session >= test_start
    # before the decision itself is in the test set.
    if decision_date >= test_start:
        return False
    return exit_date >= test_start and entry_date <= test_end


def assert_no_train_label_intersects_test(
    train_dates: list[date],
    *,
    sessions: list[date],
    test_start: date,
    test_end: date,
    entry_lag_sessions: int = 1,
    horizon_sessions: int = 3,
) -> None:
    """Raise if any training decision's label window intersects the test fold."""
    offenders = [
        d
        for d in train_dates
        if label_window_intersects_test(
            d,
            sessions=sessions,
            test_start=test_start,
            test_end=test_end,
            entry_lag_sessions=entry_lag_sessions,
            horizon_sessions=horizon_sessions,
        )
    ]
    if offenders:
        raise AssertionError(
            f"{len(offenders)} training dates have label windows intersecting "
            f"test [{test_start}, {test_end}]; e.g. {offenders[:5]}"
        )
