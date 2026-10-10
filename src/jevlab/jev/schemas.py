"""Pydantic models for System One request/response (MASTER_SPEC §2.3-2.4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionType = Literal["noul", "choice", "score"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulCriteria(_Strict):
    true: str | dict[str, Any] | list[Any] | None = None
    false: str | dict[str, Any] | list[Any] | None = None


class QuestionSpec(_Strict):
    type: QuestionType
    instructions: str | dict[str, Any] | list[Any]
    criteria: Any = None

    @model_validator(mode="after")
    def _criteria_by_type(self) -> QuestionSpec:
        if self.type == "choice" and not isinstance(self.criteria, dict):
            raise ValueError("choice questions require criteria map option->description")
        if self.type == "score" and (
            not isinstance(self.criteria, list) or not (2 <= len(self.criteria) <= 10)
        ):
            raise ValueError("score questions require criteria list of 2..10 levels")
        return self


class SystemOneRequest(_Strict):
    model: str
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, QuestionSpec]


class NoulAnswer(_Strict):
    type: Literal["noul"] = "noul"
    noul: float

    @field_validator("noul")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError("noul probability must be in [0, 1]")
        return float(v)


class ChoiceAnswer(_Strict):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None


class ScoreAnswer(_Strict):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float]
    confidence: float | None = None


AnswerModel = NoulAnswer | ChoiceAnswer | ScoreAnswer


class UsageInfo(_Strict):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(_Strict):
    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, Any]
    usage: UsageInfo | None = None

    def parse_answer(self, question_id: str) -> AnswerModel:
        raw = self.answers[question_id]
        if not isinstance(raw, dict):
            raise TypeError(f"answer for {question_id} must be an object")
        qtype = raw.get("type")
        if qtype == "noul":
            return NoulAnswer.model_validate(raw)
        if qtype == "choice":
            return ChoiceAnswer.model_validate(raw)
        if qtype == "score":
            return ScoreAnswer.model_validate(raw)
        raise ValueError(f"unknown answer type for {question_id}: {qtype!r}")


# Documented §2.4 example payload (for contract tests).
DOCUMENTED_RESPONSE_EXAMPLE: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "q_noul": {"type": "noul", "noul": 0.95},
        "q_choice": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.88, "technical": 0.12},
            "confidence": 0.81,
        },
        "q_score": {
            "type": "score",
            "score": 1.05,
            "legend": {"0": "Calm", "1": "Frustrated"},
            "probabilities": {"0": 0.0, "1": 0.95},
            "confidence": 0.92,
        },
    },
    "usage": {"input_tokens": 296, "output_tokens": 20},
}
