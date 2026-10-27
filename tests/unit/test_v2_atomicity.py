"""Fault-injection: crash after seal consume cannot mint V2_SEALED_PASS."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.registry import init_registry
from jevlab.v2.model_artifact import fit_presealed_model, write_sealed_test_model
from jevlab.v2.rehearsal import two_source_causal_fixture
from jevlab.v2.sealed_run import InjectedCrash, run_v2_final_test
from jevlab.v2.sealed_source import CountingPreSealedSource, CountingSealedSource

SEAL = "v2-fixture-seal"


def _root(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "proj"
    (root / "data" / "sealed").mkdir(parents=True)
    acc = root / "reports" / "acceptance"
    seal = root / "reports" / "v2_sealed"
    acc.mkdir(parents=True)
    seal.mkdir(parents=True)
    (root / "registry").mkdir()
    reg = root / "registry" / "experiments.sqlite"
    init_registry(reg)
    return root, acc, seal, reg


def _prebuild(tmp_path: Path) -> tuple[dict, dict, object]:
    _payload, history, sealed = two_source_causal_fixture(seed=3)
    fitted, meta = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=_payload["t0"],
        source_manifest_digest="atomicity",
    )
    art = write_sealed_test_model(
        fitted=fitted, meta=meta, out_dir=tmp_path / "model", git_commit="t"
    )
    return history, sealed["production"], art


def _kwargs(
    root: Path,
    acc: Path,
    seal: Path,
    reg: Path,
    src: CountingSealedSource,
    *,
    history: dict,
    artifact: object,
) -> dict[str, object]:
    return dict(
        root=root,
        registry_path=reg,
        acceptance_dir=acc,
        seal_dir=seal,
        seal_token=SEAL,
        sealed_source=src,
        training_source=CountingPreSealedSource(history),
        model_artifact=artifact,
        require_prebuilt_model=True,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
        require_clean_tree=False,
        require_authorization=False,
        n=4,
        allow_reduced_n=True,
        run_robustness=False,
        check_fast_first_null=False,
        result_label="development_rehearsal_only",
        t0=src.payload["t0"],
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL)


@pytest.mark.parametrize(
    "step",
    (
        "append_sealed_access_record",
        "read_sealed_or_fixture",
        "classify_sealed_payload",
        "merge_historical_context",
        "causal_feature_build",
        "construct_features_labels",
        "load_prebuilt_model",
        "score_entire_period",
        "candidate_backtest_phase_0",
        "candidate_backtest_phase_1",
        "candidate_backtest_phase_2",
        "candidate_backtest",
        "first_null_draw",
        "midpoint_null_draw",
        "final_null_draw",
        "run_random_nulls",
        "calculate_hard_gates",
        "generate_result_report",
        "conditionally_write_acceptance",
    ),
)
def test_crash_after_step_does_not_write_pass(tmp_path: Path, step: str) -> None:
    root, acc, seal, reg = _root(tmp_path)
    history, sealed, art = _prebuild(tmp_path)
    src = CountingSealedSource(sealed)
    with pytest.raises(InjectedCrash):
        run_v2_final_test(
            **_kwargs(root, acc, seal, reg, src, history=history, artifact=art),
            fail_after=step,
        )  # type: ignore[arg-type]
    assert not (acc / "V2_SEALED_PASS.json").exists()
    state = (seal / "v2_seal_state.json").read_text(encoding="utf-8")
    assert "seal_opened" in state
    with pytest.raises(PreconditionError, match="already run"):
        run_v2_final_test(
            **_kwargs(root, acc, seal, reg, src, history=history, artifact=art),
        )  # type: ignore[arg-type]
