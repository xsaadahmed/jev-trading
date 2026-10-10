"""Per-symbol technical features from as-of bars (MASTER_SPEC §6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jevlab.features.rolling import (
    down_streak,
    log_return,
    own_trailing_percentile,
    realized_vol,
    simple_return,
    sma,
    wilder_rsi,
)


def compute_symbol_technicals(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol technical columns from a single-symbol bar history.

    Input ``bars`` must already be filtered with ``as_of(decision_time)`` and sorted
    by session date ascending. Uses adjusted prices for returns/ratios/RSI and raw
    volume / raw close for volume features (§5.2).
    """
    if bars.empty:
        return pd.DataFrame()

    frame = bars.sort_values("date").reset_index(drop=True)
    adj_close = frame["adj_close"]
    adj_open = frame["adj_open"]
    raw_close = frame["raw_close"]
    raw_volume = frame["raw_volume"].astype("float64")

    ret_1d = log_return(adj_close, 1)
    out = pd.DataFrame(
        {
            "date": frame["date"].to_numpy(),
            "symbol": frame["symbol"].to_numpy(),
            "ret_1d": ret_1d,
            "ret_3d": log_return(adj_close, 3),
            "ret_5d": log_return(adj_close, 5),
            "ret_20d": log_return(adj_close, 20),
            "gap_1d": simple_return(adj_open, adj_close.shift(1)),
            "px_vs_sma20": simple_return(adj_close, sma(adj_close, 20)),
            "px_vs_sma50": simple_return(adj_close, sma(adj_close, 50)),
            "px_vs_sma200": simple_return(adj_close, sma(adj_close, 200)),
            "rsi_3": wilder_rsi(adj_close, 3),
            "rsi_14": wilder_rsi(adj_close, 14),
            "down_streak": down_streak(adj_close),
        }
    )
    out["rv_20"] = realized_vol(ret_1d, 20)
    out["rv_pct_252"] = own_trailing_percentile(out["rv_20"], 252)
    vol_mean_20 = raw_volume.rolling(window=20, min_periods=20).mean()
    out["vol_ratio_20"] = raw_volume / vol_mean_20
    dollar = (raw_close.astype("float64") * raw_volume).rolling(window=20, min_periods=20).mean()
    out["dollar_vol_20"] = np.log(dollar.replace(0.0, np.nan))
    # Carry SMA200 for eligibility (not a stored feature column).
    out["sma200"] = sma(adj_close, 200)
    out["adj_close"] = adj_close.astype("float64")
    return out
