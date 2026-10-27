"""Causal sealed feature construction from historical context + opened observations.

Never uses a later observation to build an earlier feature. Pre-2024 history may
warm trailing windows so the sealed partition itself need not contain 252 sessions
of warm-up. Provenance columns are audit-only and are stripped before Model A.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from jevlab.data.availability import (
    assign_bars_available_at,
    assign_fred_available_at,
    decision_time,
)
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.features.pipeline import build_feature_store
from jevlab.targets.labels import build_labels
from jevlab.v2.dev_io import SEALED_MAX_DATE, assert_frame_date_window
from jevlab.v2.payload import (
    HISTORICAL_ORIGIN,
    SEALED_ORIGIN,
    EmptySealedScoringError,
    SealedPayloadError,
    strip_audit_columns,
)

_CAUSAL_BUILD_CALLS: list[dict[str, Any]] = []


def causal_builder_call_count() -> int:
    return len(_CAUSAL_BUILD_CALLS)


def reset_causal_builder_calls() -> None:
    _CAUSAL_BUILD_CALLS.clear()


def causal_builder_calls() -> list[dict[str, Any]]:
    return list(_CAUSAL_BUILD_CALLS)


def combine_history_and_sealed_bars(
    history_bars: pd.DataFrame,
    sealed_bars: pd.DataFrame,
    *,
    key: tuple[str, ...] = ("symbol", "date"),
    date_col: str = "date",
) -> pd.DataFrame:
    """Concat historical context with sealed bars; sealed rows win on duplicate keys."""
    if sealed_bars is None or sealed_bars.empty:
        raise EmptySealedScoringError(
            "sealed bars are empty; refusing to evaluate historical context alone"
        )
    hist = history_bars.copy() if history_bars is not None else pd.DataFrame()
    sealed = sealed_bars.copy()
    if not hist.empty and date_col in hist.columns:
        hist[date_col] = pd.to_datetime(hist[date_col]).dt.date
        hist["row_origin"] = HISTORICAL_ORIGIN
    if date_col in sealed.columns:
        sealed[date_col] = pd.to_datetime(sealed[date_col]).dt.date
    sealed["row_origin"] = SEALED_ORIGIN
    combined = sealed if hist.empty else pd.concat([hist, sealed], ignore_index=True)
    key_cols = [c for c in key if c in combined.columns]
    if len(key_cols) != len(key):
        raise SealedPayloadError(f"combined bars missing merge keys {key}")
    # hist then sealed, keep last → never replace sealed with pre-sealed.
    combined = combined.drop_duplicates(subset=list(key), keep="last")
    overlap = combined.duplicated(subset=list(key), keep=False)
    if bool(overlap.any()):
        raise SealedPayloadError("combined bars still contain duplicate keys")
    sort_cols = [c for c in key if c in combined.columns]
    return combined.sort_values(sort_cols).reset_index(drop=True)


def combine_history_and_sealed_vix(
    history_vix: pd.DataFrame,
    sealed_vix: pd.DataFrame,
) -> pd.DataFrame:
    hist = history_vix.copy() if history_vix is not None else pd.DataFrame()
    sealed = sealed_vix.copy() if sealed_vix is not None else pd.DataFrame()
    if hist.empty and sealed.empty:
        raise SealedPayloadError("combined vix is empty; frozen V1 features require VIX")
    if not hist.empty:
        if "obs_date" not in hist.columns and "date" in hist.columns:
            hist = hist.rename(columns={"date": "obs_date"})
        hist["obs_date"] = pd.to_datetime(hist["obs_date"]).dt.date
        hist["row_origin"] = HISTORICAL_ORIGIN
    if not sealed.empty:
        if "obs_date" not in sealed.columns and "date" in sealed.columns:
            sealed = sealed.rename(columns={"date": "obs_date"})
        sealed["obs_date"] = pd.to_datetime(sealed["obs_date"]).dt.date
        sealed["row_origin"] = SEALED_ORIGIN
    if hist.empty:
        combined = sealed
    elif sealed.empty:
        combined = hist
    else:
        combined = pd.concat([hist, sealed], ignore_index=True)
    combined = combined.drop_duplicates(subset=["obs_date"], keep="last")
    return combined.sort_values(["obs_date"]).reset_index(drop=True)


def combine_history_and_sealed_features(
    history_features: pd.DataFrame,
    sealed_features: pd.DataFrame,
) -> pd.DataFrame:
    raise SealedPayloadError(
        "precomputed feature bags are not a legal r4 sealed input; "
        "call build_causal_score_features"
    )


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


def _ensure_available_at_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if "available_at" in bars.columns:
        return bars
    return assign_bars_available_at(bars)


def _ensure_available_at_vix(vix: pd.DataFrame) -> pd.DataFrame:
    if "available_at" in vix.columns:
        return vix
    date_col = "obs_date" if "obs_date" in vix.columns else "date"
    return assign_fred_available_at(vix, date_col=date_col)


def build_causal_score_features(
    *,
    history_bars: pd.DataFrame,
    sealed_bars: pd.DataFrame,
    vix: pd.DataFrame,
    symbols: list[str],
    score_sessions: list[date],
    spy_symbol: str = "SPY",
    history_vix: pd.DataFrame | None = None,
    sealed_vix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build V1 features for score sessions from history + sealed bars (as_of)."""
    if not score_sessions:
        raise EmptySealedScoringError("causal builder received zero sealed score sessions")
    combined = combine_history_and_sealed_bars(history_bars, sealed_bars)
    n_sealed = 0
    if "row_origin" in combined.columns:
        n_sealed = int((combined["row_origin"] == SEALED_ORIGIN).sum())
    if n_sealed == 0:
        raise EmptySealedScoringError(
            "combined market history contains zero sealed rows; refusing historical-only fallback"
        )
    assert_frame_date_window(combined, allow_sealed_dates=True)
    if (pd.to_datetime(combined["date"]).dt.date > SEALED_MAX_DATE).any():
        raise AssertionError("causal builder refused post-2025 bars")
    if history_vix is not None or sealed_vix is not None:
        combined_vix = combine_history_and_sealed_vix(
            history_vix if history_vix is not None else pd.DataFrame(),
            sealed_vix if sealed_vix is not None else pd.DataFrame(),
        )
    else:
        combined_vix = vix
    combined_vix = _ensure_available_at_vix(combined_vix)
    pipeline_bars = strip_audit_columns(_ensure_available_at_bars(combined))
    pipeline_vix = strip_audit_columns(combined_vix)
    for session in score_sessions:
        assert_causal_available_at(
            pipeline_bars.loc[pd.to_datetime(pipeline_bars["date"]).dt.date <= session],
            session=session,
        )
    _CAUSAL_BUILD_CALLS.append(
        {
            "n_history": int(len(history_bars) if history_bars is not None else 0),
            "n_sealed": len(sealed_bars),
            "n_combined": len(combined),
            "n_score_sessions": len(score_sessions),
            "n_sealed_origin_rows": n_sealed,
        }
    )
    features = build_feature_store(
        pipeline_bars,
        pipeline_vix,
        symbols=symbols,
        sessions=score_sessions,
        spy_symbol=spy_symbol,
    )
    if features.empty:
        raise EmptySealedScoringError("causal feature builder produced zero rows")
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.date
    if (features["date"] > SEALED_MAX_DATE).any():
        raise AssertionError("causal features include dates after 2025-12-31")
    score_set = set(score_sessions)
    if any(d not in score_set for d in features["date"].tolist()):
        raise EmptySealedScoringError(
            "causal features include dates outside the sealed score window"
        )
    features["decision_origin"] = SEALED_ORIGIN
    features["used_historical_context"] = True
    return features


def drop_unevaluable_end_window(
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    window_end: date = SEALED_MAX_DATE,
) -> pd.DataFrame:
    """Keep only rows whose t+1 entry and t+4 exit exist inside the sealed window.

    Frozen V1/ledger semantics: missing entry/exit is not completed with 2026 data.
    ``build_labels`` already omits incomplete windows; this drops any leftover
    decision dates that cannot realize inside ``window_end``.
    """
    work = labels.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    bar_dates = {
        d for d in pd.to_datetime(bars["date"]).dt.date.tolist() if d <= window_end
    }
    sessions = sorted(bar_dates)
    keep: list[date] = []
    for d in work["date"].tolist():
        if d not in bar_dates or d > window_end:
            continue
        i = sessions.index(d)
        if i + 4 >= len(sessions):
            continue
        if sessions[i + 1] > window_end or sessions[i + 4] > window_end:
            continue
        keep.append(d)
    keep_set = set(keep)
    return work.loc[work["date"].map(lambda d: d in keep_set)].reset_index(drop=True)


def construct_combined_study_frames(
    *,
    history: dict[str, Any],
    sealed: dict[str, Any],
    allow_sealed_dates: bool,
) -> dict[str, pd.DataFrame]:
    """Legacy r3 combiner. Refuses anonymous ``frames`` bags."""
    if "frames" in sealed and "bars" not in sealed:
        raise SealedPayloadError(
            "anonymous sealed frames bag cannot enter construct_combined_study_frames; "
            "classify into ProductionSealedPayload first"
        )
    hist_bars = pd.DataFrame(history.get("bars", []))
    sealed_bars = pd.DataFrame(sealed.get("bars", []))
    if hist_bars.empty:
        raise SealedPayloadError("combined study requires historical-context bars")
    if sealed_bars.empty:
        raise EmptySealedScoringError("combined study received zero sealed bars")
    hist_vix = pd.DataFrame(history.get("vix", []))
    sealed_vix = pd.DataFrame(sealed.get("vix", []))
    symbols = sorted(
        pd.concat([hist_bars, sealed_bars], ignore_index=True)["symbol"].astype(str).unique()
    )
    sealed_dates = sorted(set(pd.to_datetime(sealed_bars["date"]).dt.date.tolist()))
    if hist_vix.empty and sealed_vix.empty:
        combined_vix = sealed_vix
    else:
        combined_vix = pd.concat([hist_vix, sealed_vix], ignore_index=True)
    features = build_causal_score_features(
        history_bars=hist_bars,
        sealed_bars=sealed_bars,
        vix=combined_vix,
        history_vix=hist_vix,
        sealed_vix=sealed_vix,
        symbols=symbols,
        score_sessions=sealed_dates,
    )
    missing = [c for c in MODEL_A_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"combined features missing Model A columns: {missing}")
    bars = combine_history_and_sealed_bars(hist_bars, sealed_bars)
    bars_for_labels = strip_audit_columns(bars)
    assert_frame_date_window(features, allow_sealed_dates=allow_sealed_dates)
    assert_frame_date_window(bars_for_labels, allow_sealed_dates=allow_sealed_dates)
    labels = build_labels(
        bars_for_labels,
        eligible=features.loc[:, ["symbol", "date", "eligible"]],
        universe_symbols=symbols,
    )
    if labels.empty:
        raise ValueError("could not construct targets from combined bars")
    labels = drop_unevaluable_end_window(labels, bars_for_labels)
    labels = labels.copy()
    labels["target_origin"] = SEALED_ORIGIN
    assert_frame_date_window(labels, allow_sealed_dates=allow_sealed_dates)
    return {"features": features, "labels": labels, "bars": bars_for_labels, "combined_bars": bars}
