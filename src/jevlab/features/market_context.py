"""Market-context features: SPY and lagged VIX (MASTER_SPEC §6 / §6.1)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from jevlab.features.rolling import log_return, own_trailing_percentile, simple_return, sma
from jevlab.utils.calendar import previous_nyse_session


def compute_spy_context(spy_bars: pd.DataFrame) -> pd.DataFrame:
    """SPY ``spy_ret_5d`` and ``spy_px_vs_sma200`` from as-of SPY bars."""
    if spy_bars.empty:
        return pd.DataFrame(columns=["date", "spy_ret_5d", "spy_px_vs_sma200"])
    frame = spy_bars.sort_values("date").reset_index(drop=True)
    adj = frame["adj_close"]
    return pd.DataFrame(
        {
            "date": frame["date"].to_numpy(),
            "spy_ret_5d": log_return(adj, 5),
            "spy_px_vs_sma200": simple_return(adj, sma(adj, 200)),
        }
    )


def compute_vix_lag1_pct_252(
    vix_as_of: pd.DataFrame,
    session_dates: list[date],
) -> pd.DataFrame:
    """VIX lag-1 own-252 percentile for each decision session.

    For session t, uses the VIX observation whose ``obs_date`` equals the previous
    NYSE session (never same-day VIX). Percentile is taken in the trailing 252
    VIX observations available in ``vix_as_of`` ending at that lag observation.

    ``vix_as_of`` must already be filtered with ``as_of(decision_time(t))`` for the
    *latest* session when called once for a panel, or contain only rows available
    at each t. This function uses the full provided series and looks up VIX(t-1);
    callers that build day-by-day pass the as_of slice for that day (which ends at
    t-1), and the percentile window is taken from that history.
    """
    if vix_as_of.empty:
        return pd.DataFrame(
            {"date": session_dates, "vix_lag1_pct_252": [float("nan")] * len(session_dates)}
        )

    series = vix_as_of.sort_values("obs_date").reset_index(drop=True)
    values = series["value"].astype("float64")
    pct = own_trailing_percentile(values, 252)
    by_obs = {
        d: float(p) if pd.notna(p) else float("nan")
        for d, p in zip(series["obs_date"].tolist(), pct.tolist(), strict=True)
    }
    # Also allow raw value lookup for sessions where percentile is still warming up
    # but we still need NaN for incomplete windows (percentile already NaN).

    rows: list[dict[str, object]] = []
    for t in session_dates:
        lag_date = previous_nyse_session(t)
        rows.append(
            {
                "date": t,
                "vix_lag1_pct_252": by_obs.get(lag_date, float("nan")),
            }
        )
    return pd.DataFrame(rows)
