"""Schema contract tests from MASTER_SPEC §2.4 documented examples."""

from __future__ import annotations

from jevlab.jev.schemas import (
    DOCUMENTED_RESPONSE_EXAMPLE,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
)


def test_documented_response_example_parses() -> None:
    resp = SystemOneResponse.model_validate(DOCUMENTED_RESPONSE_EXAMPLE)
    assert resp.model == "jev-1.13.0"
    noul = resp.parse_answer("q_noul")
    assert isinstance(noul, NoulAnswer)
    assert noul.noul == 0.95
    choice = resp.parse_answer("q_choice")
    assert isinstance(choice, ChoiceAnswer)
    assert choice.choice == "billing"
    assert choice.probabilities["billing"] == 0.88
    score = resp.parse_answer("q_score")
    assert isinstance(score, ScoreAnswer)
    assert score.score == 1.05
