"""Relative and cross-sectional features (MASTER_SPEC §6)."""

from __future__ import annotations

import pandas as pd


def add_relative_and_xs_ranks(features: pd.DataFrame) -> pd.DataFrame:
    """Add ``rel_ret_5d`` and ``xs_rank_ret_5d`` to a multi-symbol feature panel.

    ``rel_ret_5d = ret_5d - spy_ret_5d``.
    ``xs_rank_ret_5d`` is the cross-sectional percentile rank of ``ret_5d`` across
    symbols on the same date (pandas ``rank(pct=True, method='average')``).
    """
    out = features.copy()
    out["rel_ret_5d"] = out["ret_5d"] - out["spy_ret_5d"]
    out["xs_rank_ret_5d"] = out.groupby("date", sort=False)["ret_5d"].transform(
        lambda s: s.rank(method="average", pct=True)
    )
    return out
