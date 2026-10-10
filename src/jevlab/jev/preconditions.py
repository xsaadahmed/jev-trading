"""Preconditions for bulk historical Jev inference (MASTER_SPEC Phase 7)."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jevlab.jev.qset import load_qset


class PreconditionError(RuntimeError):
    """A required Phase 7 gate failed."""


REQUIRED_TAGS: tuple[str, ...] = ("prereg-v1", "qset-v1-frozen")
DECISION_PATTERN = re.compile(
    r"^-\s*\[(?P<mark>[xX])\]\s*`(?P<decision>proceed|proceed_with_caveats|stop)`",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class PreconditionReport:
    ok: bool
    checks: dict[str, str]
    qset_hash: str | None = None
    disposition_decision: str | None = None


def git_tag_exists(tag: str, *, cwd: Path | None = None) -> bool:
    proc = subprocess.run(
        ["git", "tag", "--list", tag],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and tag in {t.strip() for t in proc.stdout.splitlines()}


def git_working_tree_clean(*, cwd: Path | None = None) -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    return proc.stdout.strip() == ""


def parse_disposition_decision(text: str) -> str | None:
    """Return the human-checked decision, or None if none / ambiguous."""
    found = [(m.group("decision"), m.group("mark")) for m in DECISION_PATTERN.finditer(text)]
    checked = [d for d, mark in found if mark.lower() == "x"]
    if len(checked) != 1:
        return None
    return checked[0]


def assert_bulk_preconditions(
    *,
    root: Path,
    qset_path: Path,
    frozen_hash: str | None,
    disposition_path: Path | None = None,
    require_git_tags: bool = True,
    require_clean_tree: bool = True,
) -> PreconditionReport:
    """Enforce Phase 7 gates. Raises ``PreconditionError`` on failure."""
    checks: dict[str, str] = {}
    disposition_path = disposition_path or (root / "docs" / "contamination_disposition.md")

    if not frozen_hash:
        raise PreconditionError(
            "qset_hash is not configured in jev.yaml; run `jev jev freeze-qset` first"
        )
    qset = load_qset(qset_path, expected_hash=frozen_hash, expected_purpose="research")
    checks["qset_hash"] = "ok"
    qset_hash = qset.content_hash

    if not disposition_path.is_file():
        raise PreconditionError(f"missing contamination disposition: {disposition_path}")
    decision = parse_disposition_decision(disposition_path.read_text(encoding="utf-8"))
    if decision is None:
        raise PreconditionError(
            "contamination_disposition.md must have exactly one checked decision "
            "among proceed / proceed_with_caveats / stop"
        )
    if decision == "stop":
        raise PreconditionError("contamination disposition decision is 'stop'; bulk refused")
    checks["disposition"] = decision

    if require_git_tags:
        for tag in REQUIRED_TAGS:
            if not git_tag_exists(tag, cwd=root):
                raise PreconditionError(f"required git tag missing: {tag}")
            checks[f"tag:{tag}"] = "ok"

    if require_clean_tree and not git_working_tree_clean(cwd=root):
        raise PreconditionError("git working tree is dirty; commit or stash before bulk")
    if require_clean_tree:
        checks["working_tree"] = "clean"

    return PreconditionReport(
        ok=True,
        checks=checks,
        qset_hash=qset_hash,
        disposition_decision=decision,
    )
