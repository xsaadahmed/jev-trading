"""Closed vocabulary for ``state_v1`` leaf strings (MASTER_SPEC §7.1).

Every key and every leaf value that may appear in a state must be listed here.
Horizon descriptors (one_day, three_day, five_day, twenty_day, short/medium/long
term) are allowed as *fixed feature semantics* in field names and phrases.
"""

from __future__ import annotations

from collections.abc import Sequence

from jevlab.features.columns import STATE_V1_SOURCE_COLUMNS

# Constant fund descriptor (never varies by observation).
FUND_TYPE: str = "a diversified US equity exchange-traded fund"
SCHEMA_VALUE: str = "state_v1"
SCHEMA_VALUE_COARSE: str = "state_v1c"

# Bucket edges from config (inclusive upper bounds for first six bins).
DEFAULT_BUCKET_EDGES: tuple[float, ...] = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)

STATE_KEYS: tuple[str, ...] = (
    "schema",
    "fund_type",
    *tuple(STATE_V1_SOURCE_COLUMNS.keys()),
)

# Spelled streak integers (not percentile buckets).
STREAK_WORDS: tuple[str, ...] = (
    "none",
    "one",
    "two",
    "three",
    "four",
    "five_or_more",
)

# Seven ordered phrases per semantic family (index 0 = lowest percentile bin).
RETURN_PHRASES: tuple[str, ...] = (
    "very large decline",
    "large decline",
    "moderate decline",
    "small decline",
    "small advance",
    "moderate advance",
    "large advance",
)

TREND_PHRASES_LONG: tuple[str, ...] = (
    "price is well below its long-term average",
    "price is moderately below its long-term average",
    "price is slightly below its long-term average",
    "price is near its long-term average",
    "price is slightly above its long-term average",
    "price is moderately above its long-term average",
    "price is well above its long-term average",
)

TREND_PHRASES_MEDIUM: tuple[str, ...] = (
    "price is well below its medium-term average",
    "price is moderately below its medium-term average",
    "price is slightly below its medium-term average",
    "price is near its medium-term average",
    "price is slightly above its medium-term average",
    "price is moderately above its medium-term average",
    "price is well above its medium-term average",
)

TREND_PHRASES_SHORT: tuple[str, ...] = (
    "price is well below its short-term average",
    "price is moderately below its short-term average",
    "price is slightly below its short-term average",
    "price is near its short-term average",
    "price is slightly above its short-term average",
    "price is moderately above its short-term average",
    "price is well above its short-term average",
)

# Broad-market trend uses the long-term average phrasing family but shorter forms
# appear in the §7.1 example ("price is above its long-term average").
BROAD_TREND_PHRASES: tuple[str, ...] = (
    "price is well below its long-term average",
    "price is below its long-term average",
    "price is slightly below its long-term average",
    "price is near its long-term average",
    "price is slightly above its long-term average",
    "price is above its long-term average",
    "price is well above its long-term average",
)

OSCILLATOR_PHRASES: tuple[str, ...] = (
    "deeply oversold",
    "oversold",
    "mildly oversold",
    "neutral",
    "mildly overbought",
    "overbought",
    "deeply overbought",
)

VOLATILITY_PHRASES: tuple[str, ...] = (
    "well below its own one-year norm",
    "below its own one-year norm",
    "slightly below its own one-year norm",
    "near its own one-year norm",
    "slightly above its own one-year norm",
    "above its own one-year norm",
    "well above its own one-year norm",
)

VOLUME_PHRASES: tuple[str, ...] = (
    "very light",
    "light",
    "somewhat light",
    "average",
    "somewhat heavy",
    "heavy",
    "very heavy",
)

RELATIVE_PHRASES: tuple[str, ...] = (
    "much weaker than the broad market",
    "weaker than the broad market",
    "slightly weaker than the broad market",
    "in line with the broad market",
    "slightly stronger than the broad market",
    "stronger than the broad market",
    "much stronger than the broad market",
)

VIX_PHRASES: tuple[str, ...] = (
    "very low compared with the past year",
    "low compared with the past year",
    "somewhat low compared with the past year",
    "typical compared with the past year",
    "somewhat elevated compared with the past year",
    "elevated compared with the past year",
    "extreme compared with the past year",
)

FIELD_PHRASE_BANK: dict[str, tuple[str, ...]] = {
    "long_term_trend": TREND_PHRASES_LONG,
    "medium_term_trend": TREND_PHRASES_MEDIUM,
    "short_term_position": TREND_PHRASES_SHORT,
    "one_day_return": RETURN_PHRASES,
    "three_day_return": RETURN_PHRASES,
    "five_day_return": RETURN_PHRASES,
    "twenty_day_return": RETURN_PHRASES,
    "short_term_momentum_oscillator": OSCILLATOR_PHRASES,
    "medium_term_momentum_oscillator": OSCILLATOR_PHRASES,
    "volatility": VOLATILITY_PHRASES,
    "volume_vs_recent_average": VOLUME_PHRASES,
    "versus_broad_market_five_day": RELATIVE_PHRASES,
    "broad_market_five_day_return": RETURN_PHRASES,
    "broad_market_trend": BROAD_TREND_PHRASES,
    "market_volatility_index_previous_close": VIX_PHRASES,
}


def bucket_index(percentile: float, edges: Sequence[float] = DEFAULT_BUCKET_EDGES) -> int:
    """Map a trailing percentile in ``[0, 1]`` to a bucket index ``0..len(edges)``."""
    if percentile != percentile:  # NaN
        raise ValueError("percentile is NaN")
    p = float(percentile)
    for i, edge in enumerate(edges):
        if p <= float(edge):
            return i
    return len(edges)


def build_closed_vocabulary(
    *,
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
) -> frozenset[str]:
    """Return every allowed key and leaf string for ``state_v1``.

    ``bucket_edges`` is accepted for API stability; phrase banks are fixed for V1.
    """
    _ = bucket_edges
    leaves: set[str] = {
        SCHEMA_VALUE,
        SCHEMA_VALUE_COARSE,
        FUND_TYPE,
        *STATE_KEYS,
        *STREAK_WORDS,
    }
    for phrases in FIELD_PHRASE_BANK.values():
        leaves.update(phrases)
    return frozenset(leaves)


CLOSED_VOCABULARY: frozenset[str] = build_closed_vocabulary()
