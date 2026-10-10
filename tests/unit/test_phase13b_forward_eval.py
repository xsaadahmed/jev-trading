"""Phase 13B: production-forward models, forward eval, SCALE gates."""

from __future__ import annotations

import ast
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevlab.constants import TINY_LIVE_CEILING_USD
from jevlab.experiments.acceptance import SEALED_TRAIN_END, fit_sealed_test_models
from jevlab.experiments.artifacts import ArtifactError, verify_acceptance, write_acceptance
from jevlab.experiments.forward_eval import (
    CHECKPOINT_60_FILENAME,
    SCALE_DOES_NOT_RAISE_CEILING,
    SCALE_GATE_IDS,
    ForwardLabelJoinError,
    ScaleGateMetrics,
    assert_can_join_forward_labels,
    evaluate_scale,
    evaluate_scale_gates,
    join_forward_labels,
    mark_first_label_join,
    run_forward_report,
    score_forward_frame,
    write_60_session_checkpoint,
)
from jevlab.experiments.forward_registry import register_deployment
from jevlab.experiments.production_models import (
    PRODUCTION_RATIONALE,
    assert_production_differs_from_sealed_only_by_sample,
    build_production_models,
    compute_production_train_end,
    filter_predeployment_training_rows,
    list_model_artifacts,
    production_hashes_path,
    read_production_hashes,
    write_production_hashes,
)
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.utils.registry import init_registry


def _synth_frame(
    dates: list[date],
    *,
    n_symbols: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for d in dates:
        for i in range(n_symbols):
            row: dict[str, object] = {
                "symbol": f"E{i}",
                "date": d,
                "eligible": 1,
                "y": int(rng.integers(0, 2)),
                "R": float(rng.normal(0.001, 0.01)),
                "Rbar": float(rng.normal(0.0, 0.01)),
                "y_cost": int(rng.integers(0, 2)),
                "exit_date": d,
            }
            for c in MODEL_A_COLUMNS:
                row[c] = float(rng.normal())
            for c in JEV_FEATURE_COLUMNS:
                row[c] = float(rng.uniform(0.05, 0.95))
            rows.append(row)
    return pd.DataFrame(rows)


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "proj"
    (root / "data" / "models").mkdir(parents=True)
    (root / "data" / "forward").mkdir(parents=True)
    (root / "reports" / "acceptance").mkdir(parents=True)
    (root / "registry").mkdir(parents=True)
    reg = root / "registry" / "experiments.sqlite"
    init_registry(reg)
    return root, root / "data" / "forward", reg


def _deploy(reg: Path, *, forward_start: date = date(2026, 10, 1)) -> None:
    # deployment a few days before forward_start so compute works
    ts = datetime(2026, 9, 28, 20, 0, tzinfo=UTC)
    register_deployment(
        registry_path=reg,
        jev_model_id="jev-1.13.0",
        qset_hash="fixture_qset",
        feature_version="v1",
        state_schema_version="state_v1",
        git_commit="fixture",
        deployment_timestamp=ts,
    )
    # Override forward_start if calendar resolved differently; re-read and assert
    from sqlalchemy import text

    from jevlab.experiments.forward_registry import read_deployment

    row = read_deployment(reg)
    assert row is not None
    engine = init_registry(reg)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE forward_registry SET forward_start_date = :d WHERE registry_id = :id"),
            {"d": forward_start.isoformat(), "id": row.registry_id},
        )


def _sealed_and_production(
    root: Path, reg: Path, forward_root: Path, *, forward_start: date = date(2026, 10, 1)
) -> object:
    _deploy(reg, forward_start=forward_start)
    from jevlab.experiments.forward_registry import read_deployment

    deployment = read_deployment(reg)
    assert deployment is not None
    # Sealed train on 2023 dates only
    sealed_dates = [date(2023, 1, 3) + timedelta(days=i) for i in range(0, 40, 1)]
    sealed_dates = [d for d in sealed_dates if d <= SEALED_TRAIN_END][:20]
    sealed_frame = _synth_frame(sealed_dates, seed=1)
    fit_sealed_test_models(
        sealed_frame,
        train_end=SEALED_TRAIN_END,
        registry_path=reg,
        artifact_dir=root / "data" / "models" / "sealed_test",
        git_commit="fixture",
    )
    # Production can use more pre-deployment rows (through train_end before forward)
    train_end = compute_production_train_end(forward_start)
    prod_dates = [
        *sealed_dates,
        date(2024, 6, 3),
        date(2024, 6, 4),
        date(2025, 3, 3),
        train_end,
    ]
    prod_frame = _synth_frame(prod_dates, seed=2)
    return build_production_models(
        prod_frame,
        registry_path=reg,
        artifact_dir=root / "data" / "models" / "production_forward",
        forward_root=forward_root,
        deployment=deployment,
        git_commit="fixture",
    )


def test_production_differs_from_sealed_only_by_sample(tmp_path: Path) -> None:
    root, fwd, reg = _setup(tmp_path)
    _sealed_and_production(root, reg, fwd)
    sealed = list_model_artifacts(reg, kind="sealed_test")
    prod = list_model_artifacts(reg, kind="production_forward")
    assert set(sealed) == {"A", "B"}
    assert set(prod) == {"A", "B"}
    report = assert_production_differs_from_sealed_only_by_sample(reg)
    assert report["same_procedure_different_sample"] is True
    for mid in ("A", "B"):
        assert sealed[mid]["columns_json"] == prod[mid]["columns_json"]
        assert sealed[mid]["train_end"] != prod[mid]["train_end"]
        assert sealed[mid]["content_digest"] != prod[mid]["content_digest"]
    # sealed untouched count
    assert len(sealed) == 2
    hashes = read_production_hashes(production_hashes_path(fwd))
    assert hashes is not None
    assert "rationale" in hashes
    assert PRODUCTION_RATIONALE[:40] in hashes["rationale"]


def test_no_forward_row_in_training(tmp_path: Path) -> None:
    root, fwd, reg = _setup(tmp_path)
    _deploy(reg, forward_start=date(2026, 10, 1))
    from jevlab.experiments.forward_registry import read_deployment

    deployment = read_deployment(reg)
    assert deployment is not None
    fit_sealed_test_models(
        _synth_frame([date(2023, 1, 3)], seed=0),
        registry_path=reg,
        artifact_dir=root / "data" / "models" / "sealed_test",
        git_commit="x",
    )
    bad = _synth_frame([date(2026, 10, 1), date(2026, 9, 15)], seed=3)
    with pytest.raises(ArtifactError, match="forward rows"):
        build_production_models(
            bad,
            registry_path=reg,
            artifact_dir=root / "data" / "models" / "production_forward",
            forward_root=fwd,
            deployment=deployment,
            git_commit="x",
        )
    with pytest.raises(ArtifactError, match="forward rows"):
        filter_predeployment_training_rows(
            bad,
            train_end=date(2026, 9, 20),
            forward_start_date=date(2026, 10, 1),
        )


def test_forward_label_join_refused_before_hashes(tmp_path: Path) -> None:
    _root, fwd, _reg = _setup(tmp_path)
    with pytest.raises(ForwardLabelJoinError, match="absent"):
        assert_can_join_forward_labels(fwd)
    scores = pd.DataFrame(
        {"symbol": ["E0"], "date": [date(2026, 10, 2)], "model_id": ["A"], "score": [0.5]}
    )
    labels = pd.DataFrame(
        {
            "symbol": ["E0"],
            "date": [date(2026, 10, 2)],
            "y": [1],
            "R": [0.01],
            "Rbar": [0.0],
            "y_cost": [1],
            "exit_date": [date(2026, 10, 6)],
        }
    )
    with pytest.raises(ForwardLabelJoinError, match="absent"):
        join_forward_labels(scores, labels, forward_root=fwd)


def test_forward_label_join_refused_if_hashes_newer_than_first_join(tmp_path: Path) -> None:
    _root, fwd, _reg = _setup(tmp_path)
    early = datetime(2026, 10, 1, tzinfo=UTC)
    late = datetime(2026, 10, 5, tzinfo=UTC)
    write_production_hashes(
        production_hashes_path(fwd),
        artifact_ids={"A": "a1", "B": "b1"},
        content_digests={"A": "d1", "B": "d2"},
        train_end=date(2026, 9, 20),
        git_commit="c",
        recorded_at=early,
    )
    hashes = read_production_hashes(production_hashes_path(fwd))
    assert hashes is not None
    mark_first_label_join(fwd, hashes=hashes, joined_at=early + timedelta(hours=1))
    # Rewrite hashes with newer recorded_at
    write_production_hashes(
        production_hashes_path(fwd),
        artifact_ids={"A": "a1", "B": "b1"},
        content_digests={"A": "d1", "B": "d2"},
        train_end=date(2026, 9, 20),
        git_commit="c",
        recorded_at=late,
    )
    with pytest.raises(ForwardLabelJoinError, match="newer"):
        assert_can_join_forward_labels(fwd)


def test_forward_score_and_report(tmp_path: Path) -> None:
    root, fwd, reg = _setup(tmp_path)
    _sealed_and_production(root, reg, fwd)
    # Forward session features (post deployment)
    day = date(2026, 10, 2)
    frame = _synth_frame([day], seed=9)
    result = score_forward_frame(
        frame,
        registry_path=reg,
        artifact_dir=root / "data" / "models" / "production_forward",
        forward_root=fwd,
        session_date=day,
        top_k=3,
    )
    assert result.n_scored == 6
    assert result.scores_path.is_file()
    scores = pd.read_parquet(result.scores_path)
    assert set(scores["model_id"]) == {"A", "B"}
    assert int(scores["selected"].sum()) >= 3

    # Build labeled scores for report
    labeled_scores = scores.copy()
    labels = frame.loc[:, ["symbol", "date", "y", "R", "Rbar", "y_cost", "exit_date"]]
    baseline = frame.loc[:, list(JEV_FEATURE_COLUMNS)]
    # Expand baseline / forward slightly
    baseline = pd.concat([baseline, baseline], ignore_index=True)
    report = run_forward_report(
        forward_root=fwd,
        scores=labeled_scores,
        labels=labels,
        baseline_jev=baseline,
        forward_jev=baseline,
        report_dir=root / "reports" / "forward",
        write_checkpoint_at=1,  # force checkpoint for fixture
    )
    assert report.labels_joined is True
    assert report.checkpoint_60_path is not None
    ck = json.loads(report.checkpoint_60_path.read_text(encoding="utf-8"))
    assert ck["cannot_grant_status"] is True
    assert ck["grants_scale_ready"] is False
    assert ck["informational_only"] is True


def test_60_session_checkpoint_cannot_grant_status(tmp_path: Path) -> None:
    path = write_60_session_checkpoint(tmp_path, metrics={"delta_ic": -0.01}, n_sessions=60)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["grants_scale_ready"] is False
    assert body["cannot_grant_status"] is True
    assert path.name == CHECKPOINT_60_FILENAME
    # Must not write SCALE_ACCEPTANCE
    assert not (tmp_path / "SCALE_ACCEPTANCE.json").exists()


def _tiny_live(acc_dir: Path, reg: Path, *, cfg: str = "cfg") -> Path:
    art = write_acceptance(
        "tiny_live",
        {"ok": True},
        out_dir=acc_dir,
        git_commit="g",
        research_config_hash=cfg,
        qset_hash="q",
        model_resolved="jev-1.13.0",
        gate_report_sha256="deadbeef",
        parent_artifact_digest="parent_hist",
        registry_path=reg,
    )
    return art.path


def test_scale_all_pass_writes_acceptance(tmp_path: Path) -> None:
    root, _fwd, reg = _setup(tmp_path)
    acc = root / "reports" / "acceptance"
    tiny = _tiny_live(acc, reg)
    ceiling_before = TINY_LIVE_CEILING_USD
    result = evaluate_scale(
        metrics=ScaleGateMetrics(),
        acceptance_dir=acc,
        registry_path=reg,
        tiny_live_path=tiny,
        git_commit="g",
        research_config_hash="cfg",
        qset_hash="q",
        model_resolved="jev-1.13.0",
    )
    assert result.passed
    assert result.acceptance is not None
    body = verify_acceptance(result.acceptance.path, research_config_hash="cfg")
    assert body["status"] == "SCALE_READY"
    assert body["parent_artifact_digest"] == verify_acceptance(tiny)["integrity_digest"]
    expires = datetime.fromisoformat(str(body["expires_at"]))
    created = datetime.fromisoformat(str(body["created_at"]))
    assert (expires - created).days == 90
    assert TINY_LIVE_CEILING_USD == ceiling_before == 10.00
    assert SCALE_DOES_NOT_RAISE_CEILING in json.dumps(body)


def _fail_override(gate_id: str) -> dict[str, object]:
    return {
        "S1": {"n_completed_live_round_trips": 5},
        "S2": {"n_forward_sessions_labeled": 60},
        # Fail S3 via log-loss only so S4 z stays below threshold.
        "S3": {"delta_log_loss": 0.01},
        "S4": {"se_fwd": 0.001, "ic_hist": 0.5, "delta_ic": 0.01},
        "S5": {"ks_jev_max": 0.5},
        "S6": {"cum_net_return_b": -0.10},
        "S7": {"mean_realized_cost_bps": 9.0},
        "S8": {"human_signoff": False},
    }[gate_id]


@pytest.mark.parametrize("gate_id", list(SCALE_GATE_IDS))
def test_each_scale_gate_blocks_independently(tmp_path: Path, gate_id: str) -> None:
    root, _fwd, reg = _setup(tmp_path)
    acc = root / "reports" / "acceptance"
    tiny = _tiny_live(acc, reg)
    metrics = ScaleGateMetrics().with_overrides(**_fail_override(gate_id))
    gates = evaluate_scale_gates(metrics)
    by_id = {g.gate_id: g for g in gates}
    assert by_id[gate_id].passed is False
    # Target gate is the sole failure (independent blocking).
    failed = [g.gate_id for g in gates if not g.passed]
    assert failed == [gate_id]
    result = evaluate_scale(
        metrics=metrics,
        acceptance_dir=acc,
        registry_path=reg,
        tiny_live_path=tiny,
        git_commit="g",
        research_config_hash="cfg",
        qset_hash="q",
        model_resolved="jev-1.13.0",
    )
    assert result.acceptance is None
    assert result.rejection_path is not None
    assert not (acc / "SCALE_ACCEPTANCE.json").exists()


def test_scale_digest_verification(tmp_path: Path) -> None:
    root, _fwd, reg = _setup(tmp_path)
    acc = root / "reports" / "acceptance"
    tiny = _tiny_live(acc, reg)
    result = evaluate_scale(
        metrics=ScaleGateMetrics(),
        acceptance_dir=acc,
        registry_path=reg,
        tiny_live_path=tiny,
        git_commit="g",
        research_config_hash="cfg",
        qset_hash="q",
        model_resolved="jev-1.13.0",
    )
    assert result.acceptance is not None
    verify_acceptance(result.acceptance.path, research_config_hash="cfg")
    body = json.loads(result.acceptance.path.read_text(encoding="utf-8"))
    body["payload"]["gates"] = []
    result.acceptance.path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ArtifactError, match="integrity_digest"):
        verify_acceptance(result.acceptance.path, research_config_hash="cfg")


def test_no_broker_imports_in_phase13b_modules() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "jevlab" / "experiments"
    for name in ("production_models.py", "forward_eval.py"):
        src = (root / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "broker" not in alias.name
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "broker" not in mod
                assert not mod.startswith("jevlab.brokers")


def test_sealed_required_before_production(tmp_path: Path) -> None:
    root, fwd, reg = _setup(tmp_path)
    _deploy(reg)
    from jevlab.experiments.forward_registry import read_deployment

    deployment = read_deployment(reg)
    assert deployment is not None
    with pytest.raises(ArtifactError, match="sealed_test"):
        build_production_models(
            _synth_frame([date(2023, 1, 3)]),
            registry_path=reg,
            artifact_dir=root / "data" / "models" / "production_forward",
            forward_root=fwd,
            deployment=deployment,
            git_commit="x",
        )
