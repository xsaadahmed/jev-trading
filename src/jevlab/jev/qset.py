"""Question-set loader with canonical hashing (MASTER_SPEC §7.2, §7.4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jevlab.jev.schemas import QuestionSpec
from jevlab.utils.hashing import canonical_json, sha256_of

Purpose = Literal["research", "audit"]

REQUIRED_RESEARCH_IDS: frozenset[str] = frozenset(
    {
        "dislocation",
        "downside_risk",
        "regime_supportive",
        "setup_type_a",
        "setup_type_b",
    }
)


class QSetHashMismatchError(ValueError):
    """Frozen qset hash in config does not match file contents."""


@dataclass(frozen=True, slots=True)
class QuestionSet:
    qset_version: str
    purpose: Purpose
    questions: dict[str, QuestionSpec]
    path: Path
    content_hash: str

    def as_request_questions(self) -> dict[str, QuestionSpec]:
        return dict(self.questions)

    def to_cacheable_dict(self) -> dict[str, Any]:
        return {qid: spec.model_dump(mode="json") for qid, spec in sorted(self.questions.items())}


def compute_qset_hash(payload: dict[str, Any]) -> str:
    """Canonical hash of the qset document (stable across key order)."""
    return sha256_of(canonical_json(payload))


def load_qset(
    path: Path | str,
    *,
    expected_hash: str | None = None,
    expected_purpose: Purpose | None = None,
) -> QuestionSet:
    """Load a qset JSON file; abort if a frozen ``expected_hash`` mismatches."""
    qpath = Path(path)
    if not qpath.is_file():
        raise FileNotFoundError(f"qset not found: {qpath}")
    raw = json.loads(qpath.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("qset root must be an object")
    content_hash = compute_qset_hash(raw)
    if expected_hash and content_hash != expected_hash:
        raise QSetHashMismatchError(
            f"qset hash mismatch for {qpath}: expected {expected_hash}, got {content_hash}"
        )
    version = str(raw.get("qset_version", ""))
    purpose = str(raw.get("purpose", "research"))
    if purpose not in {"research", "audit"}:
        raise ValueError(f"invalid qset purpose: {purpose}")
    if expected_purpose is not None and purpose != expected_purpose:
        raise ValueError(f"qset purpose {purpose} != expected {expected_purpose}")
    questions_raw = raw.get("questions")
    if not isinstance(questions_raw, dict):
        raise ValueError("qset.questions must be an object")
    questions: dict[str, QuestionSpec] = {
        qid: QuestionSpec.model_validate(spec) for qid, spec in questions_raw.items()
    }
    if purpose == "research":
        missing = REQUIRED_RESEARCH_IDS - set(questions)
        if missing:
            raise ValueError(f"research qset missing questions: {sorted(missing)}")
        # setup_type_b must reverse option order relative to setup_type_a
        crit_a = questions["setup_type_a"].criteria
        crit_b = questions["setup_type_b"].criteria
        if not isinstance(crit_a, dict) or not isinstance(crit_b, dict):
            raise ValueError("setup_type criteria must be maps")
        a = list(crit_a.keys())
        b = list(crit_b.keys())
        if b != list(reversed(a)):
            raise ValueError("setup_type_b criteria keys must reverse setup_type_a order")
    purpose_lit: Purpose = "audit" if purpose == "audit" else "research"
    return QuestionSet(
        qset_version=version,
        purpose=purpose_lit,
        questions=questions,
        path=qpath,
        content_hash=content_hash,
    )
