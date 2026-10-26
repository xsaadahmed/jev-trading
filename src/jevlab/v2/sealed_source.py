"""Production sealed-data source. Construction does not read sealed contents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevlab.cv.sealed_guard import SealedAccessError

SEALED_SCORE_YEARS: tuple[int, ...] = (2024, 2025)


class ProductionSealedSource:
    """Adapter for the configured sealed store.

    ``__init__`` stores the path string only. It must not list, glob, stat-walk,
    or read files. ``load()`` may read only after ``authorize_for_atomic_path``.
    """

    def __init__(self, configured_path: Path | str) -> None:
        self.configured_path = Path(configured_path)
        self.filesystem_reads = 0
        self._authorized = False

    def configured_path_string(self) -> str:
        return str(self.configured_path)

    def authorize_for_atomic_path(self) -> None:
        """Called by the atomic runner after preconditions + SEAL token pass."""
        self._authorized = True

    def load(self) -> dict[str, Any]:
        if not self._authorized:
            raise SealedAccessError(
                "ProductionSealedSource.load requires atomic-path authorization "
                "after preconditions and human SEAL token"
            )
        return self._read_authorized()

    def _read_authorized(self) -> dict[str, Any]:
        """Read 2024-2025 partitions. Must not run during the pre-seal freeze."""
        import pandas as pd

        self.filesystem_reads += 1
        root = self.configured_path
        files: list[Path] = []
        for year in SEALED_SCORE_YEARS:
            files.extend(sorted(root.glob(f"**/*year={year}*/**/*.parquet")))
            files.extend(sorted(root.glob(f"**/*{year}*.parquet")))
        unique = sorted(set(files))
        frames = [pd.read_parquet(p) for p in unique]
        return {
            "configured_path": str(root),
            "n_files": len(unique),
            "frames": frames,
        }


class CountingSealedSource:
    """Test double: records load() calls; never touches a real store."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.loads = 0
        self.filesystem_reads = 0
        self.payload = payload if payload is not None else {"fixture": True}
        self._authorized = False

    def authorize_for_atomic_path(self) -> None:
        self._authorized = True

    def load(self) -> dict[str, Any]:
        self.loads += 1
        self.filesystem_reads += 1
        return dict(self.payload)
