"""Validator fault-injection and reconciliation tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from jevlab.data.availability import assign_bars_available_at
from jevlab.data.validators import (
    reconcile_closes,
    validate_adj_raw_ratio_jumps,
    validate_bars,
    validate_duplicate_dates,
    validate_missing_sessions,
    validate_non_positive_prices,
    validate_ohlc_consistency,
    validate_volume_anomalies,
)
from jevlab.utils.calendar import nyse_session_dates


def _clean_bars(start: date, end: date, symbol: str = "SPY") -> pd.DataFrame:
    sessions = nyse_session_dates(start, end)
    rows = []
    px = 100.0
    for i, d in enumerate(sessions):
        px = 100.0 + i * 0.1
        rows.append(
            {
                "symbol": symbol,
                "date": d,
                "vendor": "tiingo",
                "raw_open": px,
                "raw_high": px + 1.0,
                "raw_low": px - 1.0,
                "raw_close": px + 0.2,
                "raw_volume": 1_000_000 + i,
                "adj_open": px,
                "adj_high": px + 1.0,
                "adj_low": px - 1.0,
                "adj_close": px + 0.2,
                "adj_volume": float(1_000_000 + i),
                "div_cash": 0.0,
                "split_factor": 1.0,
            }
        )
    return assign_bars_available_at(pd.DataFrame(rows))


def test_validate_duplicate_dates_detects_fault() -> None:
    bars = _clean_bars(date(2024, 1, 2), date(2024, 1, 5))
    bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    issues = validate_duplicate_dates(bars)
    assert any(i.code == "duplicate_dates" for i in issues)


def test_validate_missing_sessions_detects_fault() -> None:
    bars = _clean_bars(date(2024, 1, 2), date(2024, 1, 10))
    bars = bars.iloc[1:].reset_index(drop=True)  # drop first session
    issues = validate_missing_sessions(
        bars, start=date(2024, 1, 2), end=date(2024, 1, 10), symbol="SPY"
    )
    assert any(i.code == "missing_sessions" for i in issues)


def test_validate_ohlc_consistency_detects_fault() -> None:
    bars = _clean_bars(date(2024, 1, 2), date(2024, 1, 5))
    bars.loc[0, "raw_high"] = bars.loc[0, "raw_low"] - 1.0
    issues = validate_ohlc_consistency(bars)
    assert any(i.code == "ohlc_consistency" for i in issues)


def test_validate_non_positive_prices_detects_fault() -> None:
    bars = _clean_bars(date(2024, 1, 2), date(2024, 1, 5))
    bars.loc[0, "adj_close"] = 0.0
    issues = validate_non_positive_prices(bars)
    assert any(i.code == "non_positive_prices" for i in issues)


def test_validate_adj_raw_ratio_jumps_detects_fault() -> None:
    bars = _clean_bars(date(2024, 1, 2), date(2024, 1, 10))
    # Inject unexplained adj/raw jump with no split/div flag.
    bars.loc[3, "adj_close"] = bars.loc[3, "raw_close"] * 1.5
    issues = validate_adj_raw_ratio_jumps(bars, jump_threshold=0.10)
    assert any(i.code == "adj_raw_ratio_jump" for i in issues)


def test_validate_volume_anomalies_detects_non_positive() -> None:
    bars = _clean_bars(date(2024, 1, 2), date(2024, 1, 5))
    bars.loc[0, "raw_volume"] = 0
    issues = validate_volume_anomalies(bars)
    assert any(i.code == "volume_non_positive" for i in issues)


def test_validate_bars_clean_passes() -> None:
    start, end = date(2024, 1, 2), date(2024, 1, 31)
    bars = _clean_bars(start, end)
    report = validate_bars(bars, start=start, end=end)
    assert report.ok
    assert report.issues == []


def test_reconcile_closes_flags_soft_and_hard_bps() -> None:
    primary = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "QQQ"],
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "adj_close": [100.0, 100.0, 200.0],
        }
    )
    other = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "QQQ"],
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            # 6bp, 60bp, 0bp
            "adj_close": [100.06, 100.60, 200.0],
        }
    )
    report = reconcile_closes(primary, other, price_col="adj_close")
    assert report.n_overlapping_days == 3
    assert report.share_diff_gt_5bp == pytest.approx(2 / 3)
    assert len(report.days_gt_50bp) == 1
    assert report.days_gt_50bp[0]["symbol"] == "SPY"
    assert report.days_gt_50bp[0]["date"] == "2024-01-03"
