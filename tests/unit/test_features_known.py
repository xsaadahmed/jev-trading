"""Known-value tests for RSI, SMA, and own trailing percentile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jevlab.features.rolling import own_trailing_percentile, sma, wilder_rsi


def test_sma_known_values() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_own_trailing_percentile_known_values() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = own_trailing_percentile(s, window=5)
    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx(1.0)

    s2 = pd.Series([10.0, 20.0, 30.0, 25.0])
    out2 = own_trailing_percentile(s2, window=4)
    assert out2.iloc[3] == pytest.approx(0.75)


def test_wilder_rsi_known_short_series() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 12.5, 12.0, 12.5])
    rsi = wilder_rsi(close, period=3)
    assert rsi.iloc[:3].isna().all()
    assert rsi.iloc[3] == pytest.approx(100.0)
    assert rsi.iloc[4] == pytest.approx(80.0)
