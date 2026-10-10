"""Operator arm token (.live_arm) — typed phrase, ≤ 24 h TTL (MASTER_SPEC §15.3 L5)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_ARM_PHRASE: str = "I-ACCEPT-REAL-MONEY-RISK-USD-10"
DEFAULT_ARM_FILENAME: str = ".live_arm"
DEFAULT_TTL_HOURS: float = 24.0


class ArmError(RuntimeError):
    """Arm / disarm failure."""


def arm_path(root: Path) -> Path:
    return Path(root) / DEFAULT_ARM_FILENAME


def write_arm(
    root: Path,
    *,
    phrase: str,
    expected_phrase: str = DEFAULT_ARM_PHRASE,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    now: datetime | None = None,
) -> Path:
    if phrase != expected_phrase:
        raise ArmError("arm phrase mismatch; refused")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    expires = now + timedelta(hours=ttl_hours)
    path = arm_path(root)
    payload = {
        "armed_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "phrase_accepted": True,
        "ttl_hours": ttl_hours,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def disarm(root: Path) -> None:
    path = arm_path(root)
    if path.is_file():
        path.unlink()


def read_arm(root: Path) -> dict[str, Any] | None:
    path = arm_path(root)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def is_armed(
    root: Path,
    *,
    now: datetime | None = None,
) -> bool:
    body = read_arm(root)
    if body is None:
        return False
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    expires_raw = body.get("expires_at")
    if not expires_raw:
        return False
    expires = datetime.fromisoformat(str(expires_raw))
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return now <= expires
