"""Choice order-averaging and confidence formula tests."""

from __future__ import annotations

import pytest

from jevlab.jev.parse import (
    average_choice_probs,
    choice_confidence,
    noul_confidence,
    parse_response,
    score_confidence,
    score_expectation,
    score_p_ge,
    total_variation_distance,
)


def test_choice_confidence_even_split_is_zero() -> None:
    assert choice_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)


def test_choice_confidence_certainty() -> None:
    assert choice_confidence({"a": 1.0, "b": 0.0}) == pytest.approx(1.0)


def test_noul_confidence() -> None:
    assert noul_confidence(0.5) == pytest.approx(0.0)
    assert noul_confidence(1.0) == pytest.approx(1.0)
    assert noul_confidence(0.0) == pytest.approx(1.0)


def test_score_confidence_neighbors_vs_ends() -> None:
    # Three levels: neighbours (0, 0.5, 0.5) → 0.25; ends (0.5, 0, 0.5) → 0
    neigh = score_confidence({"0": 0.0, "1": 0.5, "2": 0.5})
    ends = score_confidence({"0": 0.5, "1": 0.0, "2": 0.5})
    assert neigh == pytest.approx(0.25)
    assert ends == pytest.approx(0.0)


def test_score_expectation_and_p3() -> None:
    probs = {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.3}
    assert score_expectation(probs) == pytest.approx(2.6)
    assert score_p_ge(probs, 3) == pytest.approx(0.6)


def test_average_and_tv() -> None:
    a = {"x": 0.7, "y": 0.3}
    b = {"x": 0.3, "y": 0.7}
    avg = average_choice_probs(a, b)
    assert avg["x"] == pytest.approx(0.5)
    assert total_variation_distance(a, b) == pytest.approx(0.4)


def test_parse_response_order_average() -> None:
    resp = {
        "model": "jev-1.13.0",
        "answers": {
            "dislocation": {"type": "noul", "noul": 0.6},
            "regime_supportive": {"type": "noul", "noul": 0.55},
            "downside_risk": {
                "type": "score",
                "score": 2.0,
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.2, "4": 0.2},
                "legend": {str(i): str(i) for i in range(5)},
                "confidence": 0.5,
            },
            "setup_type_a": {
                "type": "choice",
                "choice": "pullback_in_uptrend",
                "probabilities": {
                    "pullback_in_uptrend": 0.6,
                    "trend_breakdown": 0.2,
                    "sideways_noise": 0.1,
                    "extended_rally_pause": 0.1,
                },
                "confidence": 0.5,
            },
            "setup_type_b": {
                "type": "choice",
                "choice": "trend_breakdown",
                "probabilities": {
                    "pullback_in_uptrend": 0.4,
                    "trend_breakdown": 0.4,
                    "sideways_noise": 0.1,
                    "extended_rally_pause": 0.1,
                },
                "confidence": 0.2,
            },
        },
    }
    feats = parse_response(resp)
    assert feats["j_dislocation"] == pytest.approx(0.6)
    assert feats["j_pullback_p"] == pytest.approx(0.5)
    assert feats["j_breakdown_p"] == pytest.approx(0.3)
    assert feats["j_order_disagree"] == pytest.approx(0.2)
    assert "j_setup_conf" in feats
