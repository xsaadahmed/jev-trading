"""Phase 12: sealed final-test gates, acceptance integrity digest, second-look."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV
from jevlab.experiments.acceptance import (
    GATE_IDS,
    SEAL_BROKEN_STAMP,
    SEALED_TRAIN_END,
    HistoricalGateMetrics,
    evaluate_historical_gates,
    fit_sealed_test_models,
    run_final_test,
    snapshot_trial_ledger_hash,
)
from jevlab.experiments.artifacts import (
    HISTORICAL_ACCEPTANCE_STATEMENT,
    ArtifactError,
    verify_acceptance,
    write_acceptance,
)
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.registry import init_registry

SEAL_PHRASE = "phase12-fixture-seal-phrase"


def _passing_metrics(**overrides: object) -> HistoricalGateMetrics:
    m = HistoricalGateMetrics()
    if overrides:
        m = m.with_overrides(**overrides)
    return m


def _tmp_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "proj"
    (root / "docs").mkdir(parents=True)
    (root / "data" / "sealed").mkdir(parents=True)
    (root / "data" / "models").mkdir(parents=True)
    (root / "reports" / "audit").mkdir(parents=True)
    (root / "reports" / "acceptance").mkdir(parents=True)
    (root / "registry").mkdir(parents=True)
    disp = root / "docs" / "contamination_disposition.md"
    disp.write_text(
        "- [x] `proceed`\n- [ ] `proceed_with_caveats`\n- [ ] `stop`\n",
        encoding="utf-8",
    )
    (root / "reports" / "audit" / "AUDIT_REPORT.md").write_text(
        "# AUDIT\nwarning signal, not proof\n",
        encoding="utf-8",
    )
    reg = root / "registry" / "experiments.sqlite"
    init_registry(reg)
    # Seed one trial_ledger row so snapshot hash is non-empty
    from sqlalchemy import text

    engine = init_registry(reg)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trial_ledger (
                    trial_id, created_at, qset_version, state_schema_version,
                    feature_set, model_class, grid_json, universe_version,
                    horizon, purpose, experiment_id, result_summary
                ) VALUES (
                    't1', '2020-01-01T00:00:00+00:00', 'qset_v1', 'state_v1',
                    'A', 'logistic', '{}', 'v1', 3, 'exploratory', 'e1', '{}'
                )
                """
            )
        )
    return root, root / "reports" / "acceptance", reg


@pytest.fixture(autouse=True)
def _seal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL_PHRASE)


def test_all_gates_pass_writes_acceptance(tmp_path: Path) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    result = run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=_passing_metrics(),
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        skip_preconditions=False,
        require_git_tags=False,
        require_clean_tree=False,
    )
    assert result.gate_report.all_passed
    assert result.acceptance is not None
    body = verify_acceptance(result.acceptance.path, research_config_hash="fixture_config")
    assert body["status"] == "HISTORICAL_RESEARCH_PASS"
    assert HISTORICAL_ACCEPTANCE_STATEMENT in json.dumps(body)
    assert "signed" not in json.dumps(body).lower() or "not a" in json.dumps(body).lower()
    assert "integrity digest" in json.dumps(body).lower()
    assert result.gate_report.to_dict()["forward_session_count_required"] is False


@pytest.mark.parametrize("gate_id", list(GATE_IDS))
def test_each_gate_fail_blocks_artifact(tmp_path: Path, gate_id: str) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    fail_overrides: dict[str, object] = {
        "G1": {"disposition_decision": "stop"},
        "G2": {"reproducibility_ok": False},
        "G3": {"mechanics_ok": False},
        "G4": {"delta_log_loss": 0.01},
        "G5": {"folds_nonpositive_dll": 10},
        "G6": {"cagr_b": -0.01},
        "G7": {"top5_trade_share": 0.50},
        "G8": {"param_stability_fraction": 0.5},
        "G9": {"max_dd_b": 0.50, "max_dd_a": 0.10},
        "G10": {"sealed_placebo_p1_p": 0.50},
    }
    metrics = _passing_metrics(**fail_overrides[gate_id])
    # Avoid sealed access count override fighting G10 — still record access once
    result = run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=metrics,
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        require_git_tags=False,
        require_clean_tree=False,
    )
    assert result.acceptance is None
    assert result.rejection_path is not None
    assert not result.gate_report.all_passed
    failed = [r for r in result.gate_report.results if r.gate_id == gate_id]
    assert failed and failed[0].passed is False


def test_non_evaluable_gate_blocks(tmp_path: Path) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    metrics = _passing_metrics(delta_log_loss=None)
    result = run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=metrics,
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        require_git_tags=False,
        require_clean_tree=False,
    )
    assert result.acceptance is None
    g4 = next(r for r in result.gate_report.results if r.gate_id == "G4")
    assert g4.passed is None


def test_second_run_requires_allow_second_look(tmp_path: Path) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=_passing_metrics(),
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        require_git_tags=False,
        require_clean_tree=False,
    )
    with pytest.raises(PreconditionError, match="allow-second-look"):
        run_final_test(
            root=root,
            acceptance_dir=acc_dir,
            registry_path=reg,
            metrics=_passing_metrics(),
            seal_token=SEAL_PHRASE,
            expected_env_token=SEAL_PHRASE,
            require_git_tags=False,
            require_clean_tree=False,
            allow_second_look=False,
        )


def test_second_look_stamps_seal_broken_blocks_artifact(tmp_path: Path) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    first = run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=_passing_metrics(),
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        require_git_tags=False,
        require_clean_tree=False,
    )
    assert first.acceptance is not None
    # Remove acceptance so we can observe block on second look even if gates pass
    first.acceptance.path.unlink()
    second = run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=_passing_metrics(),
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        require_git_tags=False,
        require_clean_tree=False,
        allow_second_look=True,
        record_sealed_access=False,
    )
    assert second.seal_broken is True
    assert second.acceptance is None
    assert second.rejection_path is not None
    text = second.rejection_path.read_text(encoding="utf-8")
    assert SEAL_BROKEN_STAMP in text
    state = json.loads((acc_dir / "seal_state.json").read_text(encoding="utf-8"))
    assert state.get("seal_broken") is True


def test_tampered_acceptance_fails_verify(tmp_path: Path) -> None:
    _root, acc_dir, reg = _tmp_root(tmp_path)
    art = write_acceptance(
        "historical",
        {"ok": True},
        out_dir=acc_dir,
        git_commit="abc",
        research_config_hash="cfg",
        qset_hash="q",
        model_resolved="jev-1.13.0",
        gate_report_sha256="deadbeef",
        registry_path=reg,
    )
    verify_acceptance(art.path, research_config_hash="cfg")
    body = json.loads(art.path.read_text(encoding="utf-8"))
    body["payload"]["ok"] = False
    art.path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ArtifactError, match="integrity_digest"):
        verify_acceptance(art.path, research_config_hash="cfg")


def test_tampered_gate_report_sha_mismatch(tmp_path: Path) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    result = run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=_passing_metrics(),
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        require_git_tags=False,
        require_clean_tree=False,
    )
    assert result.acceptance is not None
    # Tamper gate report after the fact
    gp = result.gate_report_path
    data = json.loads(gp.read_text(encoding="utf-8"))
    data["notes"].append("tamper")
    gp.write_text(json.dumps(data), encoding="utf-8")
    # Stored sha in acceptance no longer matches recomputed gate report

    # Re-load acceptance and compare to current file hash
    acc = json.loads(result.acceptance.path.read_text(encoding="utf-8"))
    current = json.loads(gp.read_text(encoding="utf-8"))
    from jevlab.utils.hashing import canonical_json, sha256_of

    # Build hash of tampered content != stored
    assert sha256_of(canonical_json(current)) != acc["gate_report_sha256"]


def test_sealed_access_exactly_once(tmp_path: Path) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    result = run_final_test(
        root=root,
        acceptance_dir=acc_dir,
        registry_path=reg,
        metrics=_passing_metrics(),
        seal_token=SEAL_PHRASE,
        expected_env_token=SEAL_PHRASE,
        require_git_tags=False,
        require_clean_tree=False,
    )
    g10 = next(r for r in result.gate_report.results if r.gate_id == "G10")
    assert g10.passed is True
    assert True
    # Fail if count != 1
    from jevlab.experiments.acceptance import count_sealed_accesses

    assert count_sealed_accesses(reg) == 1


def test_sealed_test_models_train_end_guard(tmp_path: Path) -> None:
    root, _acc_dir, reg = _tmp_root(tmp_path)
    rng = np.random.default_rng(0)
    n = 80
    rows = []
    for i in range(n):
        d = date(2023, 1, 3)
        # all <= train end
        row = {
            "symbol": f"E{i % 4}",
            "date": d,
            "y": int(rng.integers(0, 2)),
        }
        for c in MODEL_A_COLUMNS:
            row[c] = float(rng.normal())
        for c in JEV_FEATURE_COLUMNS:
            row[c] = float(rng.uniform(0.05, 0.95))
        rows.append(row)
    frame = pd.DataFrame(rows)
    ids = fit_sealed_test_models(
        frame,
        train_end=SEALED_TRAIN_END,
        registry_path=reg,
        artifact_dir=root / "data" / "models" / "sealed_test",
        git_commit="test",
    )
    assert set(ids) == {"A", "B"}
    # Overwrite refused
    with pytest.raises(ArtifactError, match="never be overwritten"):
        fit_sealed_test_models(
            frame,
            train_end=SEALED_TRAIN_END,
            registry_path=reg,
            artifact_dir=root / "data" / "models" / "sealed_test",
            git_commit="test",
        )
    # Dates after train_end refused
    bad = frame.copy()
    bad.loc[0, "date"] = date(2024, 6, 1)
    with pytest.raises(ArtifactError, match="after"):
        # use fresh registry path for this check only — actually same kind/model exists
        # so clear by using new model path... archive will fail on overwrite first.
        # Callfit with different registry:
        reg2 = tmp_path / "reg2.sqlite"
        init_registry(reg2)
        fit_sealed_test_models(
            bad,
            train_end=SEALED_TRAIN_END,
            registry_path=reg2,
            artifact_dir=root / "data" / "models" / "sealed_test2",
            git_commit="test",
        )


def test_disposition_stop_blocks_preconditions(tmp_path: Path) -> None:
    root, acc_dir, reg = _tmp_root(tmp_path)
    (root / "docs" / "contamination_disposition.md").write_text(
        "- [ ] `proceed`\n- [ ] `proceed_with_caveats`\n- [x] `stop`\n",
        encoding="utf-8",
    )
    with pytest.raises(PreconditionError, match="stop"):
        run_final_test(
            root=root,
            acceptance_dir=acc_dir,
            registry_path=reg,
            metrics=_passing_metrics(),
            seal_token=SEAL_PHRASE,
            expected_env_token=SEAL_PHRASE,
            require_git_tags=False,
            require_clean_tree=False,
        )


def test_evaluate_gates_independent_unit() -> None:
    report = evaluate_historical_gates(_passing_metrics())
    assert report.all_passed
    assert len(report.results) == 10
    assert snapshot_trial_ledger_hash  # smoke import


def test_no_forward_session_dependency_in_gate_report() -> None:
    report = evaluate_historical_gates(_passing_metrics())
    d = report.to_dict()
    assert d["forward_session_count_required"] is False
    assert "forward_sessions" not in d
