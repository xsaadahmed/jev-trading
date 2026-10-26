"""Guard: V2 implementation tests must not touch sealed 2024-2025 files."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from jevlab.utils.paths import get_project_root
from jevlab.utils.registry import init_registry
from jevlab.v2.dev_io import DEV_YEAR_MAX, SealedPathError, _reject_sealed_path


def test_reject_sealed_year_paths() -> None:
    with __import__("pytest").raises(SealedPathError):
        _reject_sealed_path(Path("data/features/v1/symbol=SPY/year=2024/part.parquet"))
    with __import__("pytest").raises(SealedPathError):
        _reject_sealed_path(Path("data/sealed/bars.parquet"))
    _reject_sealed_path(Path("data/features/v1/symbol=SPY/year=2023/part.parquet"))


def test_dev_year_max_is_2023() -> None:
    assert DEV_YEAR_MAX == 2023


def test_real_registry_sealed_access_still_zero() -> None:
    root = get_project_root()
    db = root / "registry" / "experiments.sqlite"
    if not db.is_file():
        return
    engine = init_registry(db)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM sealed_access_log")).scalar_one()
    assert int(n) == 0


def test_no_v2_acceptance_in_repo() -> None:
    root = get_project_root()
    assert not (root / "reports" / "acceptance" / "V2_SEALED_PASS.json").is_file()
