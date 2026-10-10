"""Build anonymized ``state_v1`` from features (MASTER_SPEC §7.1).

Each feature (except down_streak) is mapped via the asset's own trailing-252
percentile into a closed-vocabulary English phrase. ``down_streak`` is spelled
as an ordinal word. The resulting state contains **no JSON numbers**.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from jevlab.features.columns import STATE_V1_SOURCE_COLUMNS
from jevlab.features.rolling import own_trailing_percentile
from jevlab.jev.vocabulary import (
    DEFAULT_BUCKET_EDGES,
    FIELD_PHRASE_BANK,
    FUND_TYPE,
    SCHEMA_VALUE,
    STREAK_WORDS,
    bucket_index,
)


def streak_word(down_streak: float | int) -> str:
    """Map a non-negative streak count to a closed-vocabulary word."""
    if down_streak != down_streak:  # NaN
        raise ValueError("down_streak is NaN")
    n = int(down_streak)
    if n <= 0:
        return STREAK_WORDS[0]
    if n >= 5:
        return STREAK_WORDS[5]
    return STREAK_WORDS[n]


def phrase_for_field(
    field: str,
    percentile: float,
    *,
    edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
) -> str:
    """Select the closed phrase for ``field`` given a trailing percentile."""
    bank = FIELD_PHRASE_BANK[field]
    idx = bucket_index(percentile, edges)
    return bank[idx]


def build_state_from_percentiles(
    percentiles: Mapping[str, float],
    *,
    down_streak: float | int,
    edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    schema: str = SCHEMA_VALUE,
    fund_type: str = FUND_TYPE,
) -> dict[str, str]:
    """Build ``state_v1`` from precomputed trailing percentiles per source column.

    ``percentiles`` keys are **feature column names** (e.g. ``ret_1d``), not state
    field names. Missing/NaN percentiles raise.
    """
    state: dict[str, str] = {"schema": schema, "fund_type": fund_type}
    for field, src in STATE_V1_SOURCE_COLUMNS.items():
        if field == "consecutive_down_days":
            continue
        if src not in percentiles:
            raise KeyError(f"missing percentile for feature {src}")
        state[field] = phrase_for_field(field, float(percentiles[src]), edges=edges)
    state["consecutive_down_days"] = streak_word(down_streak)
    return state


def build_state_row(
    row: Mapping[str, Any],
    percentiles: Mapping[str, float],
    *,
    edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
) -> dict[str, str]:
    """Build state for one feature row given its trailing percentiles."""
    return build_state_from_percentiles(
        percentiles,
        down_streak=float(row["down_streak"]),
        edges=edges,
    )


def compute_trailing_percentiles(
    features: pd.DataFrame,
    *,
    window: int = 252,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Own-trailing percentile for each feature column, per symbol."""
    cols = list(columns) if columns is not None else list(STATE_V1_SOURCE_COLUMNS.values())
    # down_streak is not percentiled for the state (spelled instead), but may
    # still appear in cols via the map — exclude it.
    cols = [c for c in cols if c != "down_streak"]
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    out_parts: list[pd.DataFrame] = []
    for _sym, group in frame.groupby("symbol", sort=True):
        g = group.sort_values("date")
        piece = g.loc[:, ["symbol", "date"]].copy()
        for col in cols:
            if col not in g.columns:
                raise KeyError(f"features missing column {col}")
            piece[col] = own_trailing_percentile(g[col].astype("float64"), window=window).to_numpy()
        out_parts.append(piece)
    if not out_parts:
        return pd.DataFrame(columns=["symbol", "date", *cols])
    return pd.concat(out_parts, ignore_index=True)


def build_states_frame(
    features: pd.DataFrame,
    *,
    window: int = 252,
    edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
) -> pd.DataFrame:
    """Return one ``state`` dict (as object column) per feature row with warm history.

    Rows without a full trailing window (NaN percentiles) are dropped.
    """
    pct = compute_trailing_percentiles(features, window=window)
    pct_cols = [c for c in pct.columns if c not in {"symbol", "date"}]
    pct_renamed = pct.rename(columns={c: f"{c}__pct" for c in pct_cols})
    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged = merged.merge(pct_renamed, on=["symbol", "date"], how="inner")
    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        percentiles: dict[str, float] = {}
        skip = False
        for col in pct_cols:
            val = r[f"{col}__pct"]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                skip = True
                break
            percentiles[col] = float(val)
        if skip:
            continue
        state = build_state_from_percentiles(
            percentiles,
            down_streak=float(r["down_streak"]),
            edges=edges,
        )
        rows.append(
            {
                "symbol": r["symbol"],
                "date": r["date"],
                "state": state,
            }
        )
    return pd.DataFrame(rows)
