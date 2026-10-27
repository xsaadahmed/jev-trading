"""V2 one-shot sealed guard with synthetic stores only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV, SealedAccessError
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.registry import init_registry
from jevlab.v2.model_artifact import fit_presealed_model, write_sealed_test_model
from jevlab.v2.rehearsal import two_source_causal_fixture
from jevlab.v2.sealed_run import (
    ATOMIC_STEPS,
    SEAL_BROKEN_STAMP,
    VOID_AUTHORIZATION_PHRASES,
    ForbiddenRealSealedSource,
    assert_fresh_r4_authorization,
    run_v2_final_test,
)
from jevlab.v2.sealed_source import CountingPreSealedSource, CountingSealedSource

SEAL = "v2-fixture-seal"


class FakeSource:
    def __init__(self) -> None:
        self.loads = 0

    def load(self) -> dict[str, object]:
        self.loads += 1
        return {"fixture": True, "rows": 0}


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


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL)


def test_first_run_fixture_consumes_seal_without_pass(tmp_path: Path) -> None:
    root, acc, seal, reg = _root(tmp_path)
    src = FakeSource()
    with pytest.raises(ValueError, match="fixture stub"):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=src,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
        )
    assert src.loads == 1
    assert not (acc / "V2_SEALED_PASS.json").is_file()
    state = json.loads((seal / "v2_seal_state.json").read_text(encoding="utf-8"))
    assert state["seal_opened"] is True


def test_second_run_refuses_before_load(tmp_path: Path) -> None:
    root, acc, seal, reg = _root(tmp_path)
    src = FakeSource()
    with pytest.raises(ValueError, match="fixture stub"):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=src,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
        )
    src2 = FakeSource()
    with pytest.raises(PreconditionError, match="already run"):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=src2,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
        )
    assert src2.loads == 0


def test_allow_second_look_breaks_seal_and_never_writes_pass(tmp_path: Path) -> None:
    root, acc, seal, reg = _root(tmp_path)
    with pytest.raises(ValueError, match="fixture stub"):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=FakeSource(),
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
        )
    assert not (acc / "V2_SEALED_PASS.json").exists()
    with pytest.raises(ValueError, match="fixture stub"):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=FakeSource(),
            allow_second_look=True,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
        )
    assert not (acc / "V2_SEALED_PASS.json").exists()
    state = json.loads((seal / "v2_seal_state.json").read_text(encoding="utf-8"))
    assert state["stamp"] == SEAL_BROKEN_STAMP
    assert state["seal_broken"] is True


def test_failed_precondition_does_not_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, acc, seal, reg = _root(tmp_path)
    monkeypatch.setenv(SEAL_TOKEN_ENV, "other")
    src = FakeSource()
    with pytest.raises(SealedAccessError):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=src,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
        )
    assert src.loads == 0


def test_forbidden_real_source_never_loads_sealed() -> None:
    with pytest.raises(SealedAccessError, match="not authorized"):
        ForbiddenRealSealedSource().load()


def test_void_authorization_is_rejected() -> None:
    with pytest.raises(PreconditionError, match="VOID"):
        assert_fresh_r4_authorization(next(iter(VOID_AUTHORIZATION_PHRASES)))
    with pytest.raises(PreconditionError, match="fresh"):
        assert_fresh_r4_authorization("")


def test_real_compute_records_atomic_steps(tmp_path: Path) -> None:
    root, acc, seal, reg = _root(tmp_path)
    payload, history, sealed = two_source_causal_fixture(seed=5)
    fitted, meta = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=payload["t0"],
        source_manifest_digest="guard",
    )
    art = write_sealed_test_model(
        fitted=fitted, meta=meta, out_dir=tmp_path / "model", git_commit="t"
    )
    src = CountingSealedSource(sealed["production"])
    result = run_v2_final_test(
        root=root,
        registry_path=reg,
        acceptance_dir=acc,
        seal_dir=seal,
        seal_token=SEAL,
        sealed_source=src,
        training_source=CountingPreSealedSource(history),
        model_artifact=art,
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
        t0=payload["t0"],
    )
    assert src.loads == 1
    assert result.acceptance_written is False
    assert not (acc / "V2_SEALED_PASS.json").exists()
    assert result.step_log.index("validate_preconditions") < result.step_log.index(
        "read_sealed_or_fixture"
    )
    assert result.step_log.index("read_sealed_or_fixture") < result.step_log.index(
        "classify_sealed_payload"
    )
    assert result.step_log.index("classify_sealed_payload") < result.step_log.index(
        "causal_feature_build"
    )
    assert result.step_log.index("causal_feature_build") < result.step_log.index(
        "construct_features_labels"
    )
    assert result.step_log.index("candidate_backtest") < result.step_log.index("run_random_nulls")
    recorded = [s for s in result.step_log if s in ATOMIC_STEPS]
    assert recorded == list(ATOMIC_STEPS)
