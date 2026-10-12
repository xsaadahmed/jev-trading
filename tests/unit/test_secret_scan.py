"""Scan git-tracked files only for committed secret-like strings.

Local gitignored files such as ``.env`` are intentionally excluded so operators
can keep real keys on disk without failing CI or leaking values into pytest output.
"""

from __future__ import annotations

import re
import subprocess
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


def _iter_tracked_text_files(root: Path) -> list[Path]:
    """Return only git-tracked files (excludes ``.env`` and other gitignored paths)."""
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    files: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode("utf-8", errors="surrogateescape"))
        path = root / rel
        if not path.is_file():
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        if path.name == "uv.lock":
            continue
        # Never scan local env files even if somehow tracked.
        if path.name != ".env.example" and (path.name == ".env" or path.name.startswith(".env.")):
            continue
        files.append(path)
    return files


def _redact_secret_snippet(snippet: str) -> str:
    """Keep path diagnostics useful without echoing secret material."""
    return re.sub(
        r"([:=]\s*['\"]?)([A-Za-z0-9_\-\.]{8,})",
        lambda m: f"{m.group(1)}<redacted len={len(m.group(2))}>",
        snippet,
    )


@pytest.mark.secret_scan
def test_repo_has_no_committed_secret_like_strings() -> None:
    root = get_project_root()
    hits: list[str] = []
    for path in _iter_tracked_text_files(root):
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
                hits.append(f"{rel}: {_redact_secret_snippet(snippet)}")
    assert hits == [], "possible secrets in tracked files:\n" + "\n".join(hits)
