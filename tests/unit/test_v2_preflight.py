"""Preflight and production source construction must not read sealed files."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV, SealedAccessError
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.registry import init_registry
from jevlab.v2.hashes import FROZEN_COMMIT, FROZEN_TAG_DESIGN, FROZEN_TAG_PREREG
from jevlab.v2.sealed_run import assert_v2_final_test_preconditions, run_v2_preflight
from jevlab.v2.sealed_source import ProductionSealedSource


def test_constructing_production_source_does_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = tmp_path / "data" / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "secret.parquet").write_bytes(b"not-a-parquet")

    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("filesystem read during construction")

    monkeypatch.setattr(Path, "glob", boom)
    monkeypatch.setattr("pandas.read_parquet", boom)
    src = ProductionSealedSource(sealed)
    assert src.filesystem_reads == 0
    assert src.configured_path_string() == str(sealed)
    with pytest.raises(SealedAccessError):
        src.load()
    assert src.filesystem_reads == 0


def test_preflight_no_seal_token_zero_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEAL_TOKEN_ENV, raising=False)
    root = tmp_path / "proj"
    sealed = root / "data" / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "year=2024").mkdir()
    (root / "registry").mkdir()
    (root / "reports" / "acceptance").mkdir(parents=True)
    (root / "reports" / "v2_sealed").mkdir(parents=True)
    init_registry(root / "registry" / "experiments.sqlite")
    src = ProductionSealedSource(sealed)
    reads_before = src.filesystem_reads

    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("sealed glob/read during preflight")

    monkeypatch.setattr("pandas.read_parquet", boom)
    out = run_v2_preflight(
        root=root,
        registry_path=root / "registry" / "experiments.sqlite",
        configured_sealed_path=sealed,
        require_clean_tree=False,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
    )
    assert out["preflight"] is True
    assert out["sealed_read"] is False
    assert out["filesystem_reads"] == 0
    assert src.filesystem_reads == reads_before
    with pytest.raises(SealedAccessError):
        src.load()


def test_original_design_tags_are_not_sufficient_for_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "proj"
    (root / "reports" / "acceptance").mkdir(parents=True)
    (root / "reports" / "v2_sealed").mkdir(parents=True)
    (root / "registry").mkdir()
    reg = root / "registry" / "experiments.sqlite"
    init_registry(reg)

    def fake_exists(tag: str, cwd: Path | None = None) -> bool:
        _ = cwd
        return tag in {FROZEN_TAG_PREREG, FROZEN_TAG_DESIGN}

    monkeypatch.setattr("jevlab.v2.sealed_run.git_tag_exists", fake_exists)
    monkeypatch.setattr(
        "jevlab.v2.sealed_run.peel_tag", lambda tag, cwd=None: FROZEN_COMMIT
    )
    monkeypatch.setattr("jevlab.v2.sealed_run.is_ancestor", lambda commit, cwd=None: True)
    with pytest.raises(PreconditionError, match="prereg-v2-baseline-r1"):
        assert_v2_final_test_preconditions(
            root=root,
            registry_path=reg,
            seal_token="",
            require_clean_tree=False,
            require_git_tags=True,
            require_preseal_tag=False,
            require_hashes=False,
            require_seal_token=False,
        )
