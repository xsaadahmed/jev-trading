"""Repo-wide scan for committed secret-like strings."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jevlab.utils.paths import get_project_root

# Patterns that strongly suggest a real API key / secret was committed.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    re.compile(r"(?i)sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
)

_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "node_modules",
    "data",
    ".uv",
}

_SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".parquet",
    ".sqlite",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".gz",
    ".zst",
}

# Files that intentionally contain the placeholder key *names* or test fixtures.
_ALLOWLIST_PATH_PARTS = {
    Path("tests") / "unit" / "test_secret_scan.py",
    Path("tests") / "unit" / "test_utils.py",
    Path(".env.example"),
    Path("docs") / "MASTER_SPEC.md",
    Path("docs") / "PHASE_0_REPORT.md",
    Path("docs") / "PHASE_1_REPORT.md",
    Path("docs") / "DATA_NOTES.md",
    Path("docs") / "PREREGISTRATION.md",
    Path("tests") / "unit" / "test_data_download.py",
    Path("tests") / "unit" / "test_data_validators.py",
    Path("tests") / "unit" / "test_data_availability.py",
    Path("tests") / "unit" / "test_features_known.py",
    Path("tests") / "unit" / "test_features_pipeline.py",
    Path("tests") / "unit" / "feature_fixtures.py",
    Path("tests") / "unit" / "test_labels.py",
    Path("tests") / "unit" / "test_cv.py",
    Path("tests") / "unit" / "test_models_baselines.py",
    Path("tests") / "unit" / "test_selection.py",
}


def _is_allowlisted(rel: Path) -> bool:
    return rel.as_posix() in {p.as_posix() for p in _ALLOWLIST_PATH_PARTS}


def _iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        if path.name == "uv.lock":
            continue
        files.append(path)
    return files


@pytest.mark.secret_scan
def test_repo_has_no_committed_secret_like_strings() -> None:
    root = get_project_root()
    hits: list[str] = []
    for path in _iter_text_files(root):
        rel = path.relative_to(root)
        if _is_allowlisted(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)[:80]
                # Placeholder empties and documentation of env *names* are fine if value short.
                if re.search(r"[:=]\s*['\"]?\s*$", snippet):
                    continue
                if "YOUR_" in snippet.upper() or "PLACEHOLDER" in snippet.upper():
                    continue
                if "example" in snippet.lower() or "redacted" in snippet.lower():
                    continue
                hits.append(f"{rel}: {snippet}")
    assert hits == [], "possible secrets committed:\n" + "\n".join(hits)
