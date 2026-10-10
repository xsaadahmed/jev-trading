"""Expanding walk-forward folds with purge/embargo (MASTER_SPEC §10.4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from jevlab.cv.purge import assert_no_train_label_intersects_test, filter_train_dates

# Canonical V1 bounds (§10.3 / §21A).
FIRST_TRAIN_START: date = date(2016, 1, 4)
FIRST_TRAIN_END: date = date(2018, 12, 31)
OOS_START: date = date(2019, 1, 2)
OOS_END: date = date(2023, 12, 29)
SEALED_AFTER: date = date(2023, 12, 29)  # data after this date is sealed
DEV_MAX_DATE: date = date(2023, 12, 29)


@dataclass(frozen=True, slots=True)
class Fold:
    """One purged expanding-window fold."""

    fold_id: str
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]
    test_start: date
    test_end: date
    embargo_sessions: int

    @property
    def n_train(self) -> int:
        return len(self.train_dates)

    @property
    def n_test(self) -> int:
        return len(self.test_dates)


def _quarter_start(year: int, quarter: int) -> date:
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
    return date(year, month, 1)


def _quarter_end(year: int, quarter: int) -> date:
    month = {1: 3, 2: 6, 3: 9, 4: 12}[quarter]
    day = {1: 31, 2: 30, 3: 30, 4: 31}[quarter]
    return date(year, month, day)


def iter_quarter_bounds(
    start: date = OOS_START,
    end: date = OOS_END,
) -> list[tuple[date, date, str]]:
    """Return (q_start, q_end, fold_id) for calendar quarters covering OOS."""
    bounds: list[tuple[date, date, str]] = []
    year = start.year
    quarter = (start.month - 1) // 3 + 1
    while True:
        qs = _quarter_start(year, quarter)
        qe = _quarter_end(year, quarter)
        if qs > end:
            break
        fold_id = f"{year}Q{quarter}"
        bounds.append((max(qs, start), min(qe, end), fold_id))
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    return bounds


def make_folds(
    dates: list[date] | pd.Series,
    first_train_end: date | str = FIRST_TRAIN_END,
    test_freq: str = "Q",
    embargo_sessions: int = 5,
    *,
    oos_start: date = OOS_START,
    oos_end: date = OOS_END,
    train_start: date = FIRST_TRAIN_START,
) -> list[Fold]:
    """Build expanding purged folds: train through first_train_end, then quarterly OOS.

    ``test_freq`` must be ``\"Q\"`` (V1). Returns 20 folds for 2019Q1-2023Q4 when
    ``dates`` covers that span.
    """
    if test_freq != "Q":
        raise ValueError("V1 walk-forward only supports test_freq='Q'")
    sessions = sorted({pd.Timestamp(d).date() for d in dates})
    first_end = pd.Timestamp(first_train_end).date()

    folds: list[Fold] = []
    for q_start, q_end, fold_id in iter_quarter_bounds(oos_start, oos_end):
        test_dates = tuple(d for d in sessions if q_start <= d <= q_end)
        if not test_dates:
            continue
        t0 = test_dates[0]
        # Expanding train: from train_start through embargo cutoff before T0,
        # and at least through first_train_end for the first fold's spirit —
        # actually expanding means all history before embargo, starting train_start.
        train_pool = [d for d in sessions if train_start <= d <= max(first_end, t0)]
        train_dates = tuple(
            filter_train_dates(
                train_pool,
                sessions=sessions,
                test_start=t0,
                embargo_sessions=embargo_sessions,
            )
        )
        # Keep only dates strictly before test and respecting embargo.
        train_dates = tuple(d for d in train_dates if d < t0)
        if not train_dates:
            continue
        assert_no_train_label_intersects_test(
            list(train_dates),
            sessions=sessions,
            test_start=t0,
            test_end=test_dates[-1],
        )
        folds.append(
            Fold(
                fold_id=fold_id,
                train_dates=train_dates,
                test_dates=test_dates,
                test_start=t0,
                test_end=test_dates[-1],
                embargo_sessions=embargo_sessions,
            )
        )
    return folds


def make_inner_folds(
    train_dates: list[date] | tuple[date, ...],
    *,
    sessions: list[date],
    n_splits: int = 3,
    embargo_sessions: int = 5,
) -> list[Fold]:
    """Inner expanding-window CV on a training set (same purge rule), ``n_splits`` tests."""
    ordered = sorted(set(train_dates))
    if n_splits < 2 or len(ordered) < n_splits + embargo_sessions + 5:
        return []
    # Split the latter portion into n_splits contiguous test blocks.
    # Keep an initial seed train chunk, then expanding.
    n = len(ordered)
    # Reserve ~40% as minimum initial train.
    min_train = max(n // 2, embargo_sessions + 10)
    test_region = ordered[min_train:]
    if len(test_region) < n_splits:
        return []
    block = len(test_region) // n_splits
    folds: list[Fold] = []
    for i in range(n_splits):
        start_i = i * block
        end_i = (i + 1) * block if i < n_splits - 1 else len(test_region)
        test_block = tuple(test_region[start_i:end_i])
        if not test_block:
            continue
        t0 = test_block[0]
        train_block = tuple(
            filter_train_dates(
                ordered,
                sessions=sessions,
                test_start=t0,
                embargo_sessions=embargo_sessions,
            )
        )
        train_block = tuple(d for d in train_block if d < t0)
        if not train_block:
            continue
        folds.append(
            Fold(
                fold_id=f"inner_{i}",
                train_dates=train_block,
                test_dates=test_block,
                test_start=t0,
                test_end=test_block[-1],
                embargo_sessions=embargo_sessions,
            )
        )
    return folds
