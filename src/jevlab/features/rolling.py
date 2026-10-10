"""Trailing-window helpers: percentiles, Wilder RSI, streaks (as-of close of t)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.Series, periods: int) -> pd.Series:
    """Trailing log return over ``periods`` sessions: log(P_t / P_{t-periods})."""
    close_f = close.astype("float64")
    ratio = close_f / close_f.shift(periods)
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=close_f.index, dtype="float64")


def simple_return(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Simple return numer/denom - 1 with float64 dtype."""
    out = numer.astype("float64") / denom.astype("float64") - 1.0
    return pd.Series(out.to_numpy(dtype=float), index=numer.index, dtype="float64")


def sma(series: pd.Series, window: int) -> pd.Series:
    """Trailing simple moving average inclusive of t; NaN until ``window`` obs."""
    out = series.astype("float64").rolling(window=window, min_periods=window).mean()
    return pd.Series(out.to_numpy(dtype=float), index=series.index, dtype="float64")


def own_trailing_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """Own trailing percentile of today's value in the last ``window`` values (incl. t).

    Defined as ``mean(window_values <= x_t)`` over a full window of length ``window``.
    Warm-up rows (fewer than ``window`` observations) are NaN.
    """

    def _pct(arr: np.ndarray) -> float:
        x = float(arr[-1])
        if np.isnan(x) or np.isnan(arr).any():
            return np.nan
        return float(np.mean(arr <= x))

    out = series.astype("float64").rolling(window=window, min_periods=window).apply(_pct, raw=True)
    return pd.Series(out.to_numpy(dtype=float), index=series.index, dtype="float64")


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI on adjusted closes (as of close of t).

    First average gain/loss is the SMA of the first ``period`` changes; subsequent
    values use Wilder RMA: ``avg = (prev * (period - 1) + current) / period``.
    """
    close_f = close.astype("float64")
    delta = close_f.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = pd.Series(np.nan, index=close_f.index, dtype="float64")
    avg_loss = pd.Series(np.nan, index=close_f.index, dtype="float64")
    rsi = pd.Series(np.nan, index=close_f.index, dtype="float64")

    if len(close_f) <= period:
        return rsi

    avg_gain.iloc[period] = float(gain.iloc[1 : period + 1].mean())
    avg_loss.iloc[period] = float(loss.iloc[1 : period + 1].mean())

    for i in range(period + 1, len(close_f)):
        avg_gain.iloc[i] = (
            float(avg_gain.iloc[i - 1]) * (period - 1) + float(gain.iloc[i])
        ) / period
        avg_loss.iloc[i] = (
            float(avg_loss.iloc[i - 1]) * (period - 1) + float(loss.iloc[i])
        ) / period

    for i in range(period, len(close_f)):
        ag = float(avg_gain.iloc[i])
        al = float(avg_loss.iloc[i])
        if al == 0.0 and ag == 0.0:
            rsi.iloc[i] = 50.0
        elif al == 0.0:
            rsi.iloc[i] = 100.0
        else:
            rs = ag / al
            rsi.iloc[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def down_streak(close: pd.Series) -> pd.Series:
    """Count of consecutive strictly-down closes ending at t (0 if today not down)."""
    close_f = close.astype("float64")
    values: list[float] = []
    streak = 0
    for i in range(len(close_f)):
        if i == 0 or pd.isna(close_f.iloc[i]) or pd.isna(close_f.iloc[i - 1]):
            values.append(np.nan)
            streak = 0
            continue
        if float(close_f.iloc[i]) < float(close_f.iloc[i - 1]):
            streak += 1
        else:
            streak = 0
        values.append(float(streak))
    return pd.Series(values, index=close_f.index, dtype="float64")


def realized_vol(log_ret_1d: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized vol: std of 1d log returns * sqrt(252) over trailing window."""
    vol = log_ret_1d.astype("float64").rolling(window=window, min_periods=window).std(ddof=0)
    return pd.Series(vol.to_numpy(dtype=float) * np.sqrt(252.0), index=log_ret_1d.index)
