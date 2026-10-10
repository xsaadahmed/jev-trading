"""Feature column registries (MASTER_SPEC §6 / §21A)."""

from __future__ import annotations

FEATURE_VERSION: str = "v1"

# Model A uses exactly these 17 columns (§21A / Phase 2 prompt).
MODEL_A_COLUMNS: tuple[str, ...] = (
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_20d",
    "down_streak",
    "px_vs_sma20",
    "px_vs_sma50",
    "px_vs_sma200",
    "rsi_3",
    "rsi_14",
    "rv_pct_252",
    "vol_ratio_20",
    "vix_lag1_pct_252",
    "spy_ret_5d",
    "spy_px_vs_sma200",
    "rel_ret_5d",
    "xs_rank_ret_5d",
)

# Stored in the feature store but not Model A inputs in V1.
STORED_ONLY_COLUMNS: tuple[str, ...] = (
    "gap_1d",
    "rv_20",
    "dollar_vol_20",
)

# All 20 numeric feature columns (excluding eligibility).
STORED_FEATURE_COLUMNS: tuple[str, ...] = MODEL_A_COLUMNS + STORED_ONLY_COLUMNS

# state_v1 field → source Model-A column (information parity; Phase 5 builds the text).
STATE_V1_SOURCE_COLUMNS: dict[str, str] = {
    "long_term_trend": "px_vs_sma200",
    "medium_term_trend": "px_vs_sma50",
    "short_term_position": "px_vs_sma20",
    "one_day_return": "ret_1d",
    "three_day_return": "ret_3d",
    "five_day_return": "ret_5d",
    "twenty_day_return": "ret_20d",
    "consecutive_down_days": "down_streak",
    "short_term_momentum_oscillator": "rsi_3",
    "medium_term_momentum_oscillator": "rsi_14",
    "volatility": "rv_pct_252",
    "volume_vs_recent_average": "vol_ratio_20",
    "versus_broad_market_five_day": "rel_ret_5d",
    "broad_market_five_day_return": "spy_ret_5d",
    "broad_market_trend": "spy_px_vs_sma200",
    "market_volatility_index_previous_close": "vix_lag1_pct_252",
}

assert len(MODEL_A_COLUMNS) == 17
assert len(STORED_FEATURE_COLUMNS) == 20
assert len(set(MODEL_A_COLUMNS)) == 17
assert set(STORED_ONLY_COLUMNS).isdisjoint(MODEL_A_COLUMNS)
assert set(STATE_V1_SOURCE_COLUMNS.values()).issubset(MODEL_A_COLUMNS)
