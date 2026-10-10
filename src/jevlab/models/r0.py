"""Deterministic reference rule R0: lowest 5d return among eligible (§4.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_r0(features: pd.DataFrame, *, ret_col: str = "ret_5d") -> pd.DataFrame:
    """Assign R0 scores: ``score = -ret_5d`` so lower returns rank higher.

    Non-eligible rows get ``score = -inf`` equivalent (very low) via NaN then filled
    after selection filters on eligible.
    """
    out = features.copy()
    if ret_col not in out.columns:
        raise ValueError(f"R0 requires column {ret_col}")
    out["score"] = -out[ret_col].astype("float64")
    out["prob"] = out["score"].rank(method="average", pct=True)
    out["model_id"] = "R0"
    # Ineligible: keep score for diagnostics but selection ignores them.
    mask = out.get("eligible", 1)
    if "eligible" in out.columns:
        out.loc[out["eligible"].fillna(0).astype(int) != 1, "prob"] = np.nan
    _ = mask
    return out
