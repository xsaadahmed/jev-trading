"""Causal sealed feature construction from historical context + opened observations.

Never uses a later observation to build an earlier feature. Pre-2024 history may
warm trailing windows so the sealed partition itself need not contain 252 sessions
of warm-up.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from jevlab.data.availability import decision_time
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.features.pipeline import build_feature_store
from jevlab.targets.labels import build_labels
from jevlab.v2.dev_io import SEALED_MAX_DATE, assert_frame_date_window


def combine_history_and_sealed_bars(
    history_bars: pd.DataFrame,
    sealed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Concat historical context with sealed bars; drop duplicate symbol/date."""
    hist = history_bars.copy()
    sealed = sealed_bars.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    sealed["date"] = pd.to_datetime(sealed["date"]).dt.date
    combined = pd.concat([hist, sealed], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def combine_history_and_sealed_features(
    history_features: pd.DataFrame,
    sealed_features: pd.DataFrame,
) -> pd.DataFrame:
    hist = history_features.copy()
    sealed = sealed_features.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    sealed["date"] = pd.to_datetime(sealed["date"]).dt.date
    combined = pd.concat([hist, sealed], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def assert_causal_available_at(frame: pd.DataFrame, *, session: date) -> None:
    """Every input row used for session ``t`` must have available_at <= decision_time(t)."""
    if "available_at" not in frame.columns or frame.empty:
        return
    dt = decision_time(session)
    latest = pd.to_datetime(frame["available_at"], utc=True).max()
    if latest > dt.tz_convert("UTC"):
        raise AssertionError(
            f"non-causal input for {session}: available_at {latest} > decision_time {dt}"
        )


def build_causal_score_features(
    *,
    history_bars: pd.DataFrame,
    sealed_bars: pd.DataFrame,
    vix: pd.DataFrame,
    symbols: list[str],
    score_sessions: list[date],
    spy_symbol: str = "SPY",
) -> pd.DataFrame:
    """Build V1 features for score sessions from history + sealed bars (as_of)."""
    combined = combine_history_and_sealed_bars(history_bars, sealed_bars)
    assert_frame_date_window(combined, allow_sealed_dates=True)
    if (pd.to_datetime(combined["date"]).dt.date > SEALED_MAX_DATE).any():
        raise AssertionError("causal builder refused post-2025 bars")
    return build_feature_store(
        combined,
        vix,
        symbols=symbols,
        sessions=score_sessions,
        spy_symbol=spy_symbol,
    )


def drop_unevaluable_end_window(
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    window_end: date = SEALED_MAX_DATE,
) -> pd.DataFrame:
    """Keep only rows whose t+1 entry and t+4 exit exist inside the sealed window.

    Frozen V1/ledger semantics: missing entry/exit is not completed with 2026 data.
    """
    work = labels.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    bar_dates = set(pd.to_datetime(bars["date"]).dt.date.tolist())
    bar_dates = {d for d in bar_dates if d <= window_end}
    return work.loc[work["date"].map(lambda d: d <= window_end)].reset_index(drop=True)


def construct_combined_study_frames(
    *,
    history: dict[str, Any],
    sealed: dict[str, Any],
    allow_sealed_dates: bool,
) -> dict[str, pd.DataFrame]:
    """Combine allowed historical context with opened sealed observations."""
    hist_feat = pd.DataFrame(history.get("features", history.get("labels", [])))
    hist_bars = pd.DataFrame(history.get("bars", []))
    sealed_feat = pd.DataFrame(sealed.get("features", sealed.get("labels", [])))
    sealed_bars = pd.DataFrame(sealed.get("bars", []))
    if hist_feat.empty and sealed_feat.empty:
        raise ValueError("combined study requires features on history or sealed source")
    if hist_bars.empty and sealed_bars.empty:
        raise ValueError("combined study requires bars on history or sealed source")
    features = combine_history_and_sealed_features(
        hist_feat if not hist_feat.empty else sealed_feat.iloc[0:0],
        sealed_feat if not sealed_feat.empty else hist_feat.iloc[0:0],
    )
    bars = combine_history_and_sealed_bars(
        hist_bars if not hist_bars.empty else sealed_bars.iloc[0:0],
        sealed_bars if not sealed_bars.empty else hist_bars.iloc[0:0],
    )
    missing = [c for c in MODEL_A_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"combined features missing Model A columns: {missing}")
    if "eligible" not in features.columns:
        raise ValueError("combined features missing eligibility")
    assert_frame_date_window(features, allow_sealed_dates=allow_sealed_dates)
    assert_frame_date_window(bars, allow_sealed_dates=allow_sealed_dates)
    universe = sorted(features["symbol"].astype(str).unique().tolist())
    labels = build_labels(
        bars,
        eligible=features.loc[:, ["symbol", "date", "eligible"]],
        universe_symbols=universe,
    )
    if labels.empty and "labels" in sealed:
        extra = pd.DataFrame(sealed["labels"])
        extra["date"] = pd.to_datetime(extra["date"]).dt.date
        labels = extra
    if labels.empty and "labels" in history:
        extra = pd.DataFrame(history["labels"])
        extra["date"] = pd.to_datetime(extra["date"]).dt.date
        labels = extra
    if labels.empty:
        raise ValueError("could not construct targets from combined bars")
    labels = drop_unevaluable_end_window(labels, bars)
    assert_frame_date_window(labels, allow_sealed_dates=allow_sealed_dates)
    return {"features": features, "labels": labels, "bars": bars}
