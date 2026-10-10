"""Feature pipeline causality, availability, VIX-lag, invariance, digest tests."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jevlab.data.availability import decision_time
from jevlab.features.columns import (
    MODEL_A_COLUMNS,
    STATE_V1_SOURCE_COLUMNS,
    STORED_FEATURE_COLUMNS,
    STORED_ONLY_COLUMNS,
)
from jevlab.features.pipeline import (
    build_feature_store,
    build_features_for_session,
    feature_store_digest,
    read_feature_store,
    write_feature_store,
)
from jevlab.features.rolling import sma
from tests.unit.feature_fixtures import make_bars, make_sessions, make_vix

ET = ZoneInfo("America/New_York")
SYMBOLS = ["SPY", "QQQ", "IWM"]


@pytest.fixture(scope="module")
def panel() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    # 280 sessions ≈ enough for SMA200 + percentile 252 warm-up with residual history.
    sessions = make_sessions(280)
    bars = make_bars(SYMBOLS, sessions, seed=7)
    vix = make_vix(sessions, seed=11)
    return bars, vix, sessions


def test_model_a_column_invariants() -> None:
    assert len(MODEL_A_COLUMNS) == 17
    assert len(STORED_FEATURE_COLUMNS) == 20
    assert set(STORED_ONLY_COLUMNS) == {"gap_1d", "rv_20", "dollar_vol_20"}
    assert set(STATE_V1_SOURCE_COLUMNS.values()).issubset(MODEL_A_COLUMNS)


def test_build_features_schema_and_warmup_nans(panel: tuple) -> None:
    bars, vix, sessions = panel
    # Build on the last 30 sessions only (history still full via as_of).
    target = sessions[-30:]
    feats = build_feature_store(bars, vix, symbols=SYMBOLS, sessions=target)
    assert not feats.empty
    assert set(MODEL_A_COLUMNS).issubset(feats.columns)
    assert set(STORED_ONLY_COLUMNS).issubset(feats.columns)
    assert feats["feature_version"].eq("v1").all()
    assert feats["decision_time"].notna().all()
    # Warm-up: early calendar sessions in a short history still NaN for long windows;
    # on late sessions with 280 days of history, long-window features should be finite.
    late = feats.loc[feats["date"] == target[-1]]
    for col in ("ret_1d", "px_vs_sma20", "rsi_14", "vol_ratio_20", "xs_rank_ret_5d"):
        assert late[col].notna().all(), col
    assert late["eligible"].isin([0, 1]).all()


def test_truncate_and_compare_causality(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-5]
    full = build_features_for_session(bars, vix, t, symbols=SYMBOLS)
    trunc_bars = bars.loc[bars["date"] <= t].copy()
    trunc_vix = vix.loc[vix["obs_date"] <= t].copy()
    trunc = build_features_for_session(trunc_bars, trunc_vix, t, symbols=SYMBOLS)
    full = full.sort_values("symbol").reset_index(drop=True)
    trunc = trunc.sort_values("symbol").reset_index(drop=True)
    for col in STORED_FEATURE_COLUMNS:
        assert np.allclose(
            full[col].to_numpy(dtype=float),
            trunc[col].to_numpy(dtype=float),
            equal_nan=True,
            rtol=0,
            atol=0,
        ), col


def test_perturb_future_leaves_features_at_t_unchanged(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-10]
    base = build_features_for_session(bars, vix, t, symbols=SYMBOLS).sort_values("symbol")

    bars_p = bars.copy()
    future = bars_p["date"] > t
    rng = np.random.default_rng(99)
    noise = 1.0 + rng.normal(0, 0.2, size=int(future.sum()))
    for col in ("adj_close", "adj_open", "adj_high", "adj_low", "raw_close"):
        bars_p.loc[future, col] = bars_p.loc[future, col].to_numpy(dtype=float) * noise
    bars_p.loc[future, "raw_volume"] = (
        bars_p.loc[future, "raw_volume"].to_numpy(dtype=float) * noise
    ).astype("int64")
    vix_p = vix.copy()
    vix_future = vix_p["obs_date"] > t
    vix_p.loc[vix_future, "value"] = vix_p.loc[vix_future, "value"] * 3.0

    pert = build_features_for_session(bars_p, vix_p, t, symbols=SYMBOLS).sort_values("symbol")
    for col in STORED_FEATURE_COLUMNS:
        assert np.allclose(
            base[col].to_numpy(dtype=float),
            pert[col].to_numpy(dtype=float),
            equal_nan=True,
            rtol=0,
            atol=0,
        ), col


def test_availability_future_available_at_never_influences(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-3]
    base = build_features_for_session(bars, vix, t, symbols=SYMBOLS).sort_values("symbol")

    # Inject a same-date SPY bar that would change adj_close if used, but mark it
    # unavailable until after decision_time(t).
    leak = bars.loc[(bars["symbol"] == "SPY") & (bars["date"] == t)].copy()
    assert len(leak) == 1
    leak["adj_close"] = leak["adj_close"] * 10.0
    leak["raw_close"] = leak["raw_close"] * 10.0
    leak["available_at"] = decision_time(t) + pd.Timedelta(hours=1)
    bars_leak = pd.concat([bars, leak], ignore_index=True)

    out = build_features_for_session(bars_leak, vix, t, symbols=SYMBOLS).sort_values("symbol")
    for col in STORED_FEATURE_COLUMNS:
        assert np.allclose(
            base[col].to_numpy(dtype=float),
            out[col].to_numpy(dtype=float),
            equal_nan=True,
        ), col


def test_vix_lag_changing_vix_on_t_changes_no_feature_at_t(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-4]
    base = build_features_for_session(bars, vix, t, symbols=SYMBOLS).sort_values("symbol")

    vix_p = vix.copy()
    mask = vix_p["obs_date"] == t
    assert mask.any()
    vix_p.loc[mask, "value"] = vix_p.loc[mask, "value"] * 5.0

    out = build_features_for_session(bars, vix_p, t, symbols=SYMBOLS).sort_values("symbol")
    for col in STORED_FEATURE_COLUMNS:
        assert np.allclose(
            base[col].to_numpy(dtype=float),
            out[col].to_numpy(dtype=float),
            equal_nan=True,
        ), col
    # And explicitly the VIX feature must be unchanged.
    assert np.allclose(
        base["vix_lag1_pct_252"].to_numpy(dtype=float),
        out["vix_lag1_pct_252"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_determinism(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-2]
    a = build_features_for_session(bars, vix, t, symbols=SYMBOLS)
    b = build_features_for_session(bars, vix, t, symbols=SYMBOLS)
    pd.testing.assert_frame_equal(
        a.sort_values("symbol").reset_index(drop=True),
        b.sort_values("symbol").reset_index(drop=True),
    )


def test_adjustment_invariance_under_adj_price_scale(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-6]
    base = build_features_for_session(bars, vix, t, symbols=SYMBOLS).sort_values("symbol")

    scaled = bars.copy()
    for col in ("adj_open", "adj_high", "adj_low", "adj_close"):
        scaled[col] = scaled[col] * 1.25
    out = build_features_for_session(scaled, vix, t, symbols=SYMBOLS).sort_values("symbol")

    # Scale-invariant adj-derived features must match; dollar_vol uses raw prices.
    invariant = [c for c in STORED_FEATURE_COLUMNS if c not in {"dollar_vol_20"}]
    for col in invariant:
        assert np.allclose(
            base[col].to_numpy(dtype=float),
            out[col].to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        ), col


@given(st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=15, deadline=None)
def test_hypothesis_scale_invariance_price_scaling(scale: float) -> None:
    sessions = make_sessions(260)
    bars = make_bars(["SPY", "QQQ"], sessions, seed=3)
    vix = make_vix(sessions, seed=4)
    t = sessions[-1]
    base = build_features_for_session(bars, vix, t, symbols=["SPY", "QQQ"]).sort_values("symbol")

    scaled = bars.copy()
    price_cols = [
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
    ]
    for col in price_cols:
        scaled[col] = scaled[col] * scale
    out = build_features_for_session(scaled, vix, t, symbols=["SPY", "QQQ"]).sort_values("symbol")
    # dollar_vol_20 scales with prices; exclude it.
    for col in STORED_FEATURE_COLUMNS:
        if col == "dollar_vol_20":
            continue
        assert np.allclose(
            base[col].to_numpy(dtype=float),
            out[col].to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-9,
            atol=1e-9,
        ), col


def test_logical_digest_stable_across_rebuild_and_parquet_encoding(
    panel: tuple, tmp_path: Path
) -> None:
    bars, vix, sessions = panel
    target = sessions[-20:]
    feats1 = build_feature_store(bars, vix, symbols=SYMBOLS, sessions=target)
    feats2 = build_feature_store(bars, vix, symbols=SYMBOLS, sessions=target)
    d1 = feature_store_digest(feats1)
    d2 = feature_store_digest(feats2)
    assert d1 == d2

    p1 = tmp_path / "a"
    p2 = tmp_path / "b"
    write_feature_store(feats1, p1)
    # Rewrite with a different compression via manual parquet round-trip.
    loaded = read_feature_store(p1)
    write_feature_store(loaded, p2)
    # Force alternate compression file
    alt = tmp_path / "alt.parquet"
    loaded.to_parquet(alt, index=False, compression="gzip")
    loaded_alt = pd.read_parquet(alt)
    # Align types for digest
    loaded_alt["date"] = pd.to_datetime(loaded_alt["date"]).dt.date
    loaded_alt["decision_time"] = pd.to_datetime(loaded_alt["decision_time"], utc=True)
    assert feature_store_digest(feats1) == feature_store_digest(loaded_alt)


def test_warmup_nans_only_early() -> None:
    sessions = make_sessions(30)
    bars = make_bars(["SPY"], sessions, seed=1)
    vix = make_vix(sessions, seed=2)
    feats = build_feature_store(bars, vix, symbols=["SPY"], sessions=sessions)
    # Long-window features must be all-NaN on a 30-session sample (need 200/252).
    assert feats["px_vs_sma200"].isna().all()
    assert feats["rv_pct_252"].isna().all()
    # Short windows become finite after warm-up length.
    assert feats["ret_1d"].isna().sum() == 1  # first row only
    # No NaNs appearing after the warm-up point for ret_1d.
    assert feats["ret_1d"].iloc[1:].notna().all()


def test_eligibility_matches_adj_close_gt_sma200(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-1]
    feats = build_features_for_session(bars, vix, t, symbols=["SPY"])
    spy = bars.loc[bars["symbol"] == "SPY"].sort_values("date")
    sma200 = sma(spy["adj_close"], 200)
    expected = bool(spy["adj_close"].iloc[-1] > sma200.iloc[-1])
    assert int(feats.iloc[0]["eligible"]) == int(expected)


def test_decision_time_is_1830_et(panel: tuple) -> None:
    bars, vix, sessions = panel
    t = sessions[-1]
    feats = build_features_for_session(bars, vix, t, symbols=["SPY"])
    dt = pd.Timestamp(feats.iloc[0]["decision_time"]).tz_convert(ET)
    assert dt.hour == 18 and dt.minute == 30
    assert dt.date() == t
