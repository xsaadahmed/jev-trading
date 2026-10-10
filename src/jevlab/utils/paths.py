"""Project path resolution rooted at the repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("could not locate project root containing pyproject.toml")


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Canonical project directories (data/ is git-ignored; manifests may be committed)."""

    root: Path
    config: Path
    data: Path
    data_raw: Path
    data_processed: Path
    data_features: Path
    data_jev_cache: Path
    data_sealed: Path
    data_forward: Path
    data_models: Path
    registry: Path
    reports: Path
    reports_acceptance: Path
    docs: Path
    tests: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        root = root or get_project_root()
        data = root / "data"
        reports = root / "reports"
        return cls(
            root=root,
            config=root / "config",
            data=data,
            data_raw=data / "raw",
            data_processed=data / "processed",
            data_features=data / "features",
            data_jev_cache=data / "jev_cache",
            data_sealed=data / "sealed",
            data_forward=data / "forward",
            data_models=data / "models",
            registry=root / "registry",
            reports=reports,
            reports_acceptance=reports / "acceptance",
            docs=root / "docs",
            tests=root / "tests",
        )

    def ensure_runtime_dirs(self) -> None:
        """Create git-ignored runtime directories if missing."""
        for path in (
            self.data_raw,
            self.data_processed,
            self.data_features,
            self.data_jev_cache,
            self.data_sealed,
            self.data_forward,
            self.data_models,
            self.registry,
            self.reports_acceptance,
        ):
            path.mkdir(parents=True, exist_ok=True)
