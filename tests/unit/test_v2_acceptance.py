"""V2 acceptance evaluator: independent failures, boundaries, fail-closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevlab.experiments.artifacts import ArtifactError
from jevlab.v2.acceptance import GATE_IDS, V2GateMetrics, evaluate_v2_gates, passing_fixture_metrics
from jevlab.v2.artifacts import verify_result_artifact, verify_v2_sealed_pass, write_v2_sealed_pass
from jevlab.v2.diagnostics import gates_ignore_diagnostics


def _pass(**overrides: object) -> V2GateMetrics:
    m = passing_fixture_metrics()
    return m.with_overrides(**overrides) if overrides else m


FAIL_OVERRIDES: dict[str, dict[str, object]] = {
    "R1": {"hash_integrity_ok": False},
    "R2": {"mechanics_ok": False},
    "E1": {"p_value": 0.06},
    "E2": {"sharpe_advantage": 0.09},
    "E3": {"cagr_normal": 0.0},
    "E4": {"cagr_pessimistic": -0.01},
    "E5": {"max_drawdown_dec": -0.36},
    "S1": {"phases_beat_null_median": 1},
    "S2": {"top5_profitable_share": 0.25},
    "S3": {"max_etf_positive_share": 0.36},
    "Z1": {"seal_broken": True},
}


def test_all_gates_pass_writes_artifact(tmp_path: Path) -> None:
    report = evaluate_v2_gates(_pass())
    assert report.all_passed
    art = write_v2_sealed_pass(
        report,
        out_dir=tmp_path,
        git_commit="test",
        sealed_result_digest="abc",
        training_artifact_digest="def",
        random_null_artifact_digest="ghi",
    )
    body = verify_v2_sealed_pass(art.path)
    assert body["status"] == "V2_SEALED_PASS"
    dumped = json.dumps(body).lower()
    assert "signature" not in dumped or "not a cryptographic" in dumped
    assert "integrity digest" in json.dumps(body).lower()


@pytest.mark.parametrize("gate_id", list(GATE_IDS))
def test_each_gate_fail_blocks_artifact(tmp_path: Path, gate_id: str) -> None:
    metrics = _pass(**FAIL_OVERRIDES[gate_id])
    report = evaluate_v2_gates(metrics)
    assert report.all_passed is False
    failed = [r for r in report.results if r.passed is not True]
    assert any(r.gate_id == gate_id for r in failed)
    others = [r for r in report.results if r.gate_id != gate_id]
    assert all(r.passed is True for r in others)
    with pytest.raises(ArtifactError):
        write_v2_sealed_pass(
            report,
            out_dir=tmp_path,
            git_commit="test",
            sealed_result_digest="x",
            training_artifact_digest="y",
            random_null_artifact_digest="z",
        )
    assert not (tmp_path / "V2_SEALED_PASS.json").exists()


def test_nan_missing_fail_closed(tmp_path: Path) -> None:
    fields = (
        "p_value",
        "sharpe_advantage",
        "cagr_normal",
        "cagr_pessimistic",
        "max_drawdown_dec",
    )
    for field in fields:
        report = evaluate_v2_gates(_pass(**{field: float("nan")}))
        assert report.all_passed is False
        with pytest.raises(ArtifactError):
            write_v2_sealed_pass(
                report,
                out_dir=tmp_path / field,
                git_commit="t",
                sealed_result_digest="x",
                training_artifact_digest="y",
                random_null_artifact_digest="z",
            )
    report = evaluate_v2_gates(_pass(p_value=None))
    assert report.all_passed is False
    report = evaluate_v2_gates(_pass(cagr_normal=float("inf")))
    assert report.all_passed is False


def test_tampered_acceptance_fails_verify(tmp_path: Path) -> None:
    report = evaluate_v2_gates(_pass())
    art = write_v2_sealed_pass(
        report,
        out_dir=tmp_path,
        git_commit="test",
        sealed_result_digest="abc",
        training_artifact_digest="def",
        random_null_artifact_digest="ghi",
    )
    body = json.loads(art.path.read_text(encoding="utf-8"))
    body["gate_results"][0]["passed"] = False
    art.path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ArtifactError):
        verify_v2_sealed_pass(art.path)


def test_tampered_result_fails_verify(tmp_path: Path) -> None:
    from jevlab.utils.hashing import canonical_json, sha256_of

    rest = {"hello": "world", "n": 1}
    body = dict(rest)
    body["integrity_digest"] = sha256_of(canonical_json(rest))
    path = tmp_path / "SEALED_RESULT.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    verify_result_artifact(path)
    body["hello"] = "tampered"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ArtifactError):
        verify_result_artifact(path)


def test_predictive_and_robustness_do_not_affect_gates() -> None:
    gates_ignore_diagnostics(_pass())
    failed = _pass(cagr_normal=-0.1)
    gates_ignore_diagnostics(failed)
    assert evaluate_v2_gates(failed).all_passed is False
    assert evaluate_v2_gates(_pass()).all_passed is True


def test_rehearsal_label_cannot_write_pass(tmp_path: Path) -> None:
    report = evaluate_v2_gates(_pass(), label="development_rehearsal_only")
    with pytest.raises(ArtifactError):
        write_v2_sealed_pass(
            report,
            out_dir=tmp_path,
            git_commit="t",
            sealed_result_digest="x",
            training_artifact_digest="y",
            random_null_artifact_digest="z",
        )
