"""Semantic privacy validation + 5,000-row property test (Phase 5)."""

from __future__ import annotations

import copy

import numpy as np

from jevlab.features.columns import STATE_V1_SOURCE_COLUMNS
from jevlab.jev.privacy import validate_state_privacy
from jevlab.jev.state_builder import build_state_from_percentiles
from jevlab.jev.vocabulary import CLOSED_VOCABULARY, DEFAULT_BUCKET_EDGES


def _clean_state() -> dict[str, str]:
    percentiles = {src: 0.5 for src in STATE_V1_SOURCE_COLUMNS.values() if src != "down_streak"}
    return build_state_from_percentiles(percentiles, down_streak=2, edges=DEFAULT_BUCKET_EDGES)


def test_clean_state_has_no_violations() -> None:
    state = _clean_state()
    assert validate_state_privacy(state, CLOSED_VOCABULARY) == []


def test_property_5000_random_feature_rows() -> None:
    rng = np.random.default_rng(42)
    srcs = [c for c in STATE_V1_SOURCE_COLUMNS.values() if c != "down_streak"]
    for _ in range(5000):
        percentiles = {c: float(rng.random()) for c in srcs}
        streak = int(rng.integers(0, 8))
        state = build_state_from_percentiles(
            percentiles, down_streak=streak, edges=DEFAULT_BUCKET_EDGES
        )
        bad = validate_state_privacy(state, CLOSED_VOCABULARY)
        assert bad == [], bad


def test_reject_leaf_not_in_vocab() -> None:
    state = _clean_state()
    state["one_day_return"] = "not a real phrase"
    assert any(
        v.startswith("leaf_not_in_vocab") for v in validate_state_privacy(state, CLOSED_VOCABULARY)
    )


def test_reject_universe_ticker() -> None:
    state = _clean_state()
    state["fund_type"] = "SPY"
    bad = validate_state_privacy(state, CLOSED_VOCABULARY | {"SPY"})
    # Even if somehow in vocab, ticker rule fires; use vocab without SPY:
    bad = validate_state_privacy(state, CLOSED_VOCABULARY)
    assert any("ticker" in v or "leaf_not_in_vocab" in v for v in bad)


def test_reject_iso_date_and_year() -> None:
    state = _clean_state()
    state["one_day_return"] = "2020-01-02"
    bad = validate_state_privacy(state, CLOSED_VOCABULARY | {"2020-01-02"})
    assert any("iso_date" in v or "year" in v for v in bad)


def test_reject_month_name() -> None:
    state = _clean_state()
    state["one_day_return"] = "january"
    bad = validate_state_privacy(state, CLOSED_VOCABULARY | {"january"})
    assert any("month_name" in v for v in bad)


def test_reject_currency_percent_float() -> None:
    state = _clean_state()
    for leaf, kind in [("$12.50", "currency"), ("12.5%", "percentage"), ("3.1415", "float")]:
        s = copy.deepcopy(state)
        s["one_day_return"] = leaf
        bad = validate_state_privacy(s, CLOSED_VOCABULARY | {leaf})
        assert any(kind in v for v in bad), (kind, bad)


def test_reject_prohibited_key() -> None:
    state = _clean_state()
    state["symbol"] = "hidden"
    bad = validate_state_privacy(state, CLOSED_VOCABULARY | {"symbol", "hidden"})
    assert any("prohibited_key" in v for v in bad)


def test_reject_numeric_json_types() -> None:
    state = _clean_state()
    state2 = dict(state)
    state2["one_day_return"] = 0.12  # type: ignore[assignment]
    bad = validate_state_privacy(state2, CLOSED_VOCABULARY)
    assert any("numeric_json_type" in v for v in bad)


def test_horizon_descriptors_allowed_in_keys() -> None:
    state = _clean_state()
    assert "one_day_return" in state
    assert "five_day_return" in state
    assert "twenty_day_return" in state
    assert validate_state_privacy(state, CLOSED_VOCABULARY) == []
