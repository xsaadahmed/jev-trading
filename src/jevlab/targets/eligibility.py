"""Eligibility flag: adjusted close above SMA200 (MASTER_SPEC §4.2 / §6)."""

from __future__ import annotations

import pandas as pd


def compute_eligible(adj_close: pd.Series, sma200: pd.Series) -> pd.Series:
    """Return boolean eligibility: ``adj_close > sma200`` as of close of t.

    Rows where ``sma200`` is NaN (warm-up) are False / not eligible.
    """
    return (adj_close.astype("float64") > sma200.astype("float64")).fillna(False)


def attach_eligible(features: pd.DataFrame) -> pd.DataFrame:
    """Attach integer ``eligible`` column (1/0) from ``adj_close`` and ``sma200``."""
    out = features.copy()
    if "adj_close" not in out.columns or "sma200" not in out.columns:
        raise ValueError("features must include adj_close and sma200 to compute eligibility")
    out["eligible"] = compute_eligible(out["adj_close"], out["sma200"]).astype("int64")
    return out
