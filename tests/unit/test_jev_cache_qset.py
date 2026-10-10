"""Cache hit/miss, hash stability, qset loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevlab.jev.cache import JevCache, request_hash
from jevlab.jev.qset import QSetHashMismatchError, compute_qset_hash, load_qset
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.paths import get_project_root


def test_request_hash_stable_across_key_order() -> None:
    state_a = {"b": 1, "a": 2}
    state_b = {"a": 2, "b": 1}
    questions = {"q": {"type": "noul", "instructions": "x"}}
    h1 = request_hash(
        route="systemone", model_requested="jev-1.13.0", state=state_a, questions=questions
    )
    h2 = request_hash(
        route="systemone", model_requested="jev-1.13.0", state=state_b, questions=questions
    )
    assert h1 == h2
    assert h1 == sha256_of(
        canonical_json(
            {
                "route": "systemone",
                "model_requested": "jev-1.13.0",
                "state": state_a,
                "questions": questions,
            }
        )
    )


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    h = request_hash(route="r", model_requested="m", state={"s": "x"}, questions={"q": 1})
    assert cache.get(h) is None
    rec = cache.put(
        req_hash=h,
        route="r",
        model_requested="m",
        model_resolved="jev-1.13.0",
        purpose="research",
        qset_version="qset_v1",
        state_schema_version="state_v1",
        response={"model": "jev-1.13.0", "answers": {}},
    )
    assert rec.attempt == 1
    hit = cache.get(h)
    assert hit is not None
    assert hit.response["model"] == "jev-1.13.0"
    # Immutable: same attempt rejected; new attempt ok
    with pytest.raises(ValueError, match="immutable"):
        cache.put(
            req_hash=h,
            route="r",
            model_requested="m",
            model_resolved="jev-1.13.0",
            purpose="research",
            qset_version="qset_v1",
            state_schema_version="state_v1",
            response={"model": "jev-1.13.0", "answers": {"x": 1}},
            attempt=1,
        )
    rec2 = cache.put(
        req_hash=h,
        route="r",
        model_requested="m",
        model_resolved="jev-1.13.0",
        purpose="research",
        qset_version="qset_v1",
        state_schema_version="state_v1",
        response={"model": "jev-1.13.0", "answers": {"repeat": True}},
    )
    assert rec2.attempt == 2


def test_load_qset_v1_and_audit() -> None:
    root = get_project_root()
    qset = load_qset(root / "config" / "qset_v1.json", expected_purpose="research")
    assert qset.qset_version == "qset_v1"
    assert set(qset.questions) >= {
        "dislocation",
        "downside_risk",
        "regime_supportive",
        "setup_type_a",
        "setup_type_b",
    }
    audit = load_qset(root / "config" / "audit_qset.json", expected_purpose="audit")
    assert audit.purpose == "audit"
    assert "era_class" in audit.questions
    assert "outcome_outperform" in audit.questions


def test_qset_hash_mismatch_aborts(tmp_path: Path) -> None:
    root = get_project_root()
    src = (root / "config" / "qset_v1.json").read_text(encoding="utf-8")
    path = tmp_path / "qset.json"
    path.write_text(src, encoding="utf-8")
    real = compute_qset_hash(__import__("json").loads(src))
    with pytest.raises(QSetHashMismatchError):
        load_qset(path, expected_hash="0" * 64)
    # Matching hash succeeds
    loaded = load_qset(path, expected_hash=real)
    assert loaded.content_hash == real
