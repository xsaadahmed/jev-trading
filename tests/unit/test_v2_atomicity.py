"""Fault-injection: crash after seal consume cannot mint V2_SEALED_PASS."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.registry import init_registry
from jevlab.v2.acceptance import V2GateMetrics
from jevlab.v2.sealed_run import InjectedCrash, run_v2_final_test
from jevlab.v2.sealed_source import CountingSealedSource

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


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL)


@pytest.mark.parametrize(
    "step",
    (
        "append_sealed_access_record",
        "read_sealed_or_fixture",
        "run_random_nulls",
        "generate_result_report",
        "conditionally_write_acceptance",
    ),
)
def test_crash_after_step_does_not_write_pass(tmp_path: Path, step: str) -> None:
    root, acc, seal, reg = _root(tmp_path)
    src = CountingSealedSource()
    with pytest.raises(InjectedCrash):
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
            fail_after=step,
        )
    assert not (acc / "V2_SEALED_PASS.json").exists()
    state = (seal / "v2_seal_state.json").read_text(encoding="utf-8")
    assert "seal_opened" in state
    src2 = CountingSealedSource()
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
