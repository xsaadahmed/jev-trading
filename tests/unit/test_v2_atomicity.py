"""Fault-injection: crash after seal consume cannot mint V2_SEALED_PASS."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.registry import init_registry
from jevlab.v2.rehearsal import synthetic_pre2024_payload
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


def _kwargs(
    root: Path, acc: Path, seal: Path, reg: Path, src: CountingSealedSource
) -> dict[str, object]:
    return dict(
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
        "construct_features_labels",
        "fit_once_pre_sealed",
        "score_entire_period",
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
    src = CountingSealedSource(synthetic_pre2024_payload(n_sessions=40, n_symbols=4, seed=3))
    with pytest.raises(InjectedCrash):
        run_v2_final_test(**_kwargs(root, acc, seal, reg, src), fail_after=step)  # type: ignore[arg-type]
    assert not (acc / "V2_SEALED_PASS.json").exists()
    state = (seal / "v2_seal_state.json").read_text(encoding="utf-8")
    assert "seal_opened" in state
    src2 = CountingSealedSource(synthetic_pre2024_payload(n_sessions=40, n_symbols=4, seed=3))
    with pytest.raises(PreconditionError, match="already run"):
        run_v2_final_test(**_kwargs(root, acc, seal, reg, src2))  # type: ignore[arg-type]
    assert src2.loads == 0
