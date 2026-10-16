"""Attribution controls: placebo slots (F) and bucket one-hot (G).

Model F is interface-only until Phase 10 fills placebo Jev columns.
Model G adds drop-first one-hots of the semantic buckets sent to Jev.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from jevlab.features.columns import MODEL_A_COLUMNS, STATE_V1_SOURCE_COLUMNS
from jevlab.features.rolling import own_trailing_percentile
from jevlab.jev.vocabulary import DEFAULT_BUCKET_EDGES, STREAK_WORDS, bucket_index
from jevlab.models.column_sets import PLACEBO_JEV_COLUMNS
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

# Percentile-bucketed state fields (excludes consecutive_down_days / streak).
BUCKETED_STATE_FIELDS: tuple[str, ...] = tuple(
    field
    for field, src in STATE_V1_SOURCE_COLUMNS.items()
    if field != "consecutive_down_days" and src != "down_streak"
)

N_BUCKETS: int = len(DEFAULT_BUCKET_EDGES) + 1  # 7 ordered bins


def placebo_column_names() -> tuple[str, ...]:
    """Stable names for Model F's eight placebo slots."""
    return PLACEBO_JEV_COLUMNS


def ensure_placebo_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure placebo columns exist (NaN) so Model F's schema is registered.

    Values are filled in Phase 10; Phase 8 only provides the interface.
    """
    out = frame.copy()
    for col in PLACEBO_JEV_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def fill_placebo_jev_columns(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Fill Model F's eight ``placebo_j_*`` slots with AR(1)-matched noise.

    Matches each real Jev column's per-asset autocorrelation (same construction
    as placebo P4), written into the dedicated placebo column names so Model F
    = A + 8 placebo features without mutating the real Jev columns used by B.
    """
    from jevlab.evaluation.placebo import ar1_matched_noise
    from jevlab.jev.parse import JEV_FEATURE_COLUMNS

    out = ensure_placebo_columns(frame)
    noisy = ar1_matched_noise(out, columns=JEV_FEATURE_COLUMNS, seed=seed)
    for real, placebo in zip(JEV_FEATURE_COLUMNS, PLACEBO_JEV_COLUMNS, strict=True):
        out[placebo] = noisy[real].to_numpy()
    return out


def _sanitize_token(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")


def onehot_column_name(field: str, bucket: int) -> str:
    """Drop-first naming: ``goh_{field}_b{bucket}`` for bucket in 1..N-1."""
    return f"goh_{_sanitize_token(field)}_b{bucket}"


def expected_g_onehot_columns(
    *,
    edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    include_streak: bool = True,
) -> tuple[str, ...]:
    """Deterministic list of drop-first one-hot column names for Model G."""
    n_bins = len(edges) + 1
    names: list[str] = []
    for field in BUCKETED_STATE_FIELDS:
        for b in range(1, n_bins):
            names.append(onehot_column_name(field, b))
    if include_streak:
        for word in STREAK_WORDS[1:]:
            names.append(f"goh_streak_{word}")
    return tuple(names)


def model_g_columns(
    *,
    edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    include_streak: bool = True,
) -> tuple[str, ...]:
    """A (17) + drop-first bucket one-hots."""
    return MODEL_A_COLUMNS + expected_g_onehot_columns(edges=edges, include_streak=include_streak)


def build_bucket_onehot_frame(
    features: pd.DataFrame,
    *,
    window: int = 252,
    edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    include_streak: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build drop-first one-hot columns of Jev state buckets from Model-A features.

    Returns ``(frame_with_onehots, meta)`` where ``meta`` logs the dummy count.
    Warm-up rows without a full trailing window keep zeros for one-hots.
    """
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    onehot_names = list(expected_g_onehot_columns(edges=edges, include_streak=include_streak))
    n_bins = len(edges) + 1
    src_cols = [STATE_V1_SOURCE_COLUMNS[f] for f in BUCKETED_STATE_FIELDS]

    oh_blocks: list[pd.DataFrame] = []
    for _sym, group in work.groupby("symbol", sort=True):
        g = group.sort_values("date")
        n = len(g)
        oh = np.zeros((n, len(onehot_names)), dtype=float)
        name_to_j = {name: j for j, name in enumerate(onehot_names)}

        pct: dict[str, np.ndarray] = {}
        for src in src_cols:
            if src not in g.columns:
                raise KeyError(f"features missing {src} for bucket one-hot")
            pct[src] = own_trailing_percentile(g[src].astype("float64"), window=window).to_numpy()

        for i in range(n):
            for field in BUCKETED_STATE_FIELDS:
                src = STATE_V1_SOURCE_COLUMNS[field]
                val = pct[src][i]
                if val != val:  # NaN warm-up
                    continue
                b = int(np.clip(bucket_index(float(val), edges), 0, n_bins - 1))
                if b > 0:
                    oh[i, name_to_j[onehot_column_name(field, b)]] = 1.0
            if include_streak and "down_streak" in g.columns:
                streak = g["down_streak"].iloc[i]
                if streak == streak:
                    sn = int(streak)
                    if sn <= 0:
                        word = STREAK_WORDS[0]
                    elif sn >= 5:
                        word = STREAK_WORDS[5]
                    else:
                        word = STREAK_WORDS[sn]
                    if word != STREAK_WORDS[0]:
                        col = f"goh_streak_{word}"
                        oh[i, name_to_j[col]] = 1.0

        block = pd.DataFrame(oh, columns=onehot_names, index=g.index)
        oh_blocks.append(block)

    if oh_blocks:
        oh_all = pd.concat(oh_blocks).sort_index()
        for name in onehot_names:
            work[name] = oh_all[name].to_numpy()
    else:
        for name in onehot_names:
            work[name] = 0.0

    meta: dict[str, Any] = {
        "n_onehot_columns": len(onehot_names),
        "n_bucketed_fields": len(BUCKETED_STATE_FIELDS),
        "n_bins": n_bins,
        "include_streak": include_streak,
        "onehot_columns": onehot_names,
        "drop_first": True,
    }
    log.info(
        "model_g_onehot",
        n_onehot=meta["n_onehot_columns"],
        n_fields=meta["n_bucketed_fields"],
    )
    return work, meta
