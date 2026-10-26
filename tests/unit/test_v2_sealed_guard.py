"""V2 one-shot sealed guard with synthetic stores only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV, SealedAccessError
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.registry import init_registry
from jevlab.v2.acceptance import V2GateMetrics
from jevlab.v2.sealed_run import (
    ATOMIC_STEPS,
    SEAL_BROKEN_STAMP,
    ForbiddenRealSealedSource,
    run_v2_final_test,
)

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


def test_first_run_fixture_consumes_seal(tmp_path: Path) -> None:
    root, acc, seal, reg = _root(tmp_path)
    src = FakeSource()
    result = run_v2_final_test(
        root=root,
        registry_path=reg,
        acceptance_dir=acc,
        seal_dir=seal,
        seal_token=SEAL,
        metrics=V2GateMetrics(),
        sealed_source=src,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
        require_clean_tree=False,
    )
    assert src.loads == 1
    assert result.acceptance_written is True
    assert (acc / "V2_SEALED_PASS.json").is_file()
    assert "read_sealed_or_fixture" in result.step_log
    assert result.step_log.index("validate_preconditions") < result.step_log.index(
        "read_sealed_or_fixture"
    )
    assert result.step_log.index("read_sealed_or_fixture") < result.step_log.index(
        "run_random_nulls"
    )
    assert result.step_log.index("candidate_backtest") < result.step_log.index("run_random_nulls")
    recorded = [s for s in result.step_log if s in ATOMIC_STEPS]
    assert recorded == list(ATOMIC_STEPS)


def test_second_run_refuses_before_load(tmp_path: Path) -> None:
    root, acc, seal, reg = _root(tmp_path)
    src = FakeSource()
    run_v2_final_test(
        root=root,
        registry_path=reg,
        acceptance_dir=acc,
        seal_dir=seal,
        seal_token=SEAL,
        metrics=V2GateMetrics(),
        sealed_source=src,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
        require_clean_tree=False,
    )
    src2 = FakeSource()
    with pytest.raises(PreconditionError, match="already run"):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            metrics=V2GateMetrics(),
            sealed_source=src2,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
        )
    assert src2.loads == 0


def test_allow_second_look_breaks_seal_and_never_writes_pass(tmp_path: Path) -> None:
    root, acc, seal, reg = _root(tmp_path)
    run_v2_final_test(
        root=root,
        registry_path=reg,
        acceptance_dir=acc,
        seal_dir=seal,
        seal_token=SEAL,
        metrics=V2GateMetrics(),
        sealed_source=FakeSource(),
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
        require_clean_tree=False,
    )
    assert (acc / "V2_SEALED_PASS.json").is_file()
    result = run_v2_final_test(
        root=root,
        registry_path=reg,
        acceptance_dir=acc,
        seal_dir=seal,
        seal_token=SEAL,
        metrics=V2GateMetrics(),
        sealed_source=FakeSource(),
        allow_second_look=True,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
        require_clean_tree=False,
    )
    assert result.seal_broken is True
    assert result.acceptance_written is False
    assert not (acc / "V2_SEALED_PASS.json").exists()
    assert (acc / "V2_SEALED_PASS.INVALIDATED.json").is_file()
    state = json.loads((seal / "v2_seal_state.json").read_text(encoding="utf-8"))
    assert state["stamp"] == SEAL_BROKEN_STAMP


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
            metrics=V2GateMetrics(),
            sealed_source=src,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
        )
    assert src.loads == 0


def test_forbidden_real_source_never_loads_sealed() -> None:
    with pytest.raises(SealedAccessError, match="not authorized"):
        ForbiddenRealSealedSource().load()
