"""Walk-forward, purge, and sealed-guard tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from jevlab.cv.purge import (
    assert_no_train_label_intersects_test,
    embargo_cutoff_date,
    filter_train_dates,
    label_window_intersects_test,
)
from jevlab.cv.sealed_guard import SealedAccessError, SealedDataGuard
from jevlab.cv.walkforward import DEV_MAX_DATE, make_folds, make_inner_folds
from jevlab.utils.calendar import nyse_session_dates


@pytest.fixture(scope="module")
def sessions_2016_2023() -> list[date]:
    return nyse_session_dates(date(2016, 1, 4), date(2023, 12, 29))


def test_make_folds_deterministic_and_twenty_quarters(sessions_2016_2023: list[date]) -> None:
    folds_a = make_folds(sessions_2016_2023)
    folds_b = make_folds(sessions_2016_2023)
    assert len(folds_a) == 20
    assert [f.fold_id for f in folds_a] == [f.fold_id for f in folds_b]
    assert folds_a[0].fold_id == "2019Q1"
    assert folds_a[-1].fold_id == "2023Q4"
    assert folds_a[0].test_dates == folds_b[0].test_dates
    # Expanding: later folds have more train dates.
    assert folds_a[-1].n_train > folds_a[0].n_train


def test_embargo_prevents_label_intersection(sessions_2016_2023: list[date]) -> None:
    folds = make_folds(sessions_2016_2023)
    for fold in folds:
        assert_no_train_label_intersects_test(
            list(fold.train_dates),
            sessions=sessions_2016_2023,
            test_start=fold.test_start,
            test_end=fold.test_end,
        )
        cutoff = embargo_cutoff_date(sessions_2016_2023, fold.test_start, embargo_sessions=5)
        assert cutoff is not None
        assert max(fold.train_dates) <= cutoff
        # Direct check: no training date's exit reaches test_start.
        for t in fold.train_dates:
            assert not label_window_intersects_test(
                t,
                sessions=sessions_2016_2023,
                test_start=fold.test_start,
                test_end=fold.test_end,
            )


def test_filter_train_dates_embargo() -> None:
    sessions = nyse_session_dates(date(2019, 1, 2), date(2019, 3, 29))
    t0 = sessions[20]
    train = filter_train_dates(sessions, sessions=sessions, test_start=t0, embargo_sessions=5)
    assert max(train) == sessions[20 - 5]
    assert all(d < t0 for d in train)


def test_inner_folds_deterministic(sessions_2016_2023: list[date]) -> None:
    train = [d for d in sessions_2016_2023 if d <= date(2018, 12, 31)]
    a = make_inner_folds(train, sessions=sessions_2016_2023, n_splits=3)
    b = make_inner_folds(train, sessions=sessions_2016_2023, n_splits=3)
    assert len(a) == 3
    assert [f.test_dates for f in a] == [f.test_dates for f in b]


def test_sealed_guard_blocks_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed_root = tmp_path / "sealed"
    sealed_root.mkdir()
    path = sealed_root / "bars.parquet"
    pd.DataFrame({"date": [date(2024, 1, 2)], "x": [1]}).to_parquet(path, index=False)
    monkeypatch.setenv("JEV_SEAL_TOKEN", "correct-token")
    guard = SealedDataGuard(
        sealed_root=sealed_root,
        registry_path=tmp_path / "experiments.sqlite",
    )
    with pytest.raises(SealedAccessError):
        guard.load_sealed_parquet("bars.parquet", token=None, reason="test")
    with pytest.raises(SealedAccessError):
        guard.load_sealed_parquet("bars.parquet", token="wrong", reason="test")


def test_sealed_guard_logs_authorized_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed_root = tmp_path / "sealed"
    sealed_root.mkdir()
    path = sealed_root / "bars.parquet"
    pd.DataFrame({"date": [date(2024, 6, 3)], "x": [1.0]}).to_parquet(path, index=False)
    monkeypatch.setenv("JEV_SEAL_TOKEN", "seal-ok")
    guard = SealedDataGuard(
        sealed_root=sealed_root,
        registry_path=tmp_path / "experiments.sqlite",
    )
    frame = guard.load_sealed_parquet(
        "bars.parquet",
        token="seal-ok",
        reason="unit-test",
        user="tester",
        experiment_hash="abc",
    )
    assert len(frame) == 1
    log_rows = guard.list_access_log()
    assert len(log_rows) == 1
    assert log_rows[0]["reason"] == "unit-test"
    assert log_rows[0]["user"] == "tester"


def test_dev_filter_drops_2024_plus(tmp_path: Path) -> None:
    guard = SealedDataGuard(
        sealed_root=tmp_path / "sealed",
        registry_path=tmp_path / "experiments.sqlite",
    )
    frame = pd.DataFrame(
        {
            "date": [date(2023, 12, 29), date(2024, 1, 2), date(2025, 1, 2)],
            "v": [1, 2, 3],
        }
    )
    filtered = guard.filter_dev_frame(frame)
    assert list(filtered["date"]) == [date(2023, 12, 29)]
    with pytest.raises(SealedAccessError):
        guard.assert_dev_dates(frame["date"])
    guard.assert_dev_dates([DEV_MAX_DATE])
