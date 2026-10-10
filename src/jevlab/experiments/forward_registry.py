"""Immutable forward deployment registry (MASTER_SPEC §18 / Phase 13A).

The first successful forward-capture run writes exactly one row. A second write
raises. ``FORWARD_START_DATE`` is the first NYSE session whose decision time
(18:30 ET) is strictly after ``deployment_timestamp``.

Forward evidence is valid only for the pinned Jev version recorded here; a new
Jev version restarts the forward window (§11.4, §16.3 S8).
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from jevlab.data.availability import decision_time
from jevlab.utils.calendar import nyse_session_dates
from jevlab.utils.registry import init_registry


class ForwardRegistryError(RuntimeError):
    """Base error for forward registry operations."""


class ForwardRegistryImmutableError(ForwardRegistryError):
    """Raised when a second deployment row would be written."""


@dataclass(frozen=True, slots=True)
class ForwardRegistryRow:
    """One immutable forward-deployment record."""

    registry_id: str
    deployment_timestamp: datetime
    jev_model_id: str
    qset_hash: str
    feature_version: str
    state_schema_version: str
    git_commit: str
    forward_start_date: date


def git_head_commit(cwd: Path | None = None) -> str:
    """Return ``git rev-parse HEAD``, or ``unknown`` if unavailable."""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    commit = proc.stdout.strip()
    return commit if commit else "unknown"


def compute_forward_start_date(deployment_timestamp: datetime | pd.Timestamp) -> date:
    """First NYSE session whose ``decision_time`` is after ``deployment_timestamp``.

    Time semantics: decision_time(t) = t 18:30 America/New_York. Comparison is
    performed in UTC. Sessions with decision_time <= deployment are not clean
    forward evidence.
    """
    dt = pd.Timestamp(deployment_timestamp)
    dt = dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")

    et_day = dt.tz_convert("America/New_York").date()
    search_start = et_day - timedelta(days=1)
    search_end = et_day + timedelta(days=21)
    for session in nyse_session_dates(search_start, search_end):
        if decision_time(session).tz_convert("UTC") > dt:
            return session
    raise ForwardRegistryError(
        f"could not resolve FORWARD_START_DATE after deployment_timestamp={dt.isoformat()}"
    )


def _row_from_db(row: object) -> ForwardRegistryRow:
    mapping = row._mapping  # type: ignore[attr-defined]
    ts = pd.Timestamp(mapping["deployment_timestamp"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ForwardRegistryRow(
        registry_id=str(mapping["registry_id"]),
        deployment_timestamp=ts.to_pydatetime(),
        jev_model_id=str(mapping["jev_model_id"]),
        qset_hash=str(mapping["qset_hash"]),
        feature_version=str(mapping["feature_version"]),
        state_schema_version=str(mapping["state_schema_version"]),
        git_commit=str(mapping["git_commit"]),
        forward_start_date=pd.Timestamp(mapping["forward_start_date"]).date(),
    )


def read_deployment(registry_path: Path | str) -> ForwardRegistryRow | None:
    """Return the sole forward_registry row, or None if empty."""
    path = Path(registry_path)
    if not path.exists():
        return None
    engine = init_registry(path)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT registry_id, deployment_timestamp, jev_model_id, qset_hash, "
                "feature_version, state_schema_version, git_commit, forward_start_date "
                "FROM forward_registry ORDER BY deployment_timestamp ASC LIMIT 1"
            )
        ).fetchone()
    if row is None:
        return None
    return _row_from_db(row)


def read_forward_start_date(registry_path: Path | str) -> date | None:
    """Return ``forward_start_date`` from the registry, if present."""
    row = read_deployment(registry_path)
    return None if row is None else row.forward_start_date


def register_deployment(
    *,
    registry_path: Path | str,
    jev_model_id: str,
    qset_hash: str,
    feature_version: str,
    state_schema_version: str,
    git_commit: str | None = None,
    deployment_timestamp: datetime | None = None,
    registry_id: str | None = None,
    engine: Engine | None = None,
    project_root: Path | None = None,
) -> ForwardRegistryRow:
    """Write the immutable forward_registry row (first successful capture only).

    Raises ``ForwardRegistryImmutableError`` if a row already exists.
    """
    path = Path(registry_path)
    eng = engine or init_registry(path)
    existing = read_deployment(path)
    if existing is not None:
        raise ForwardRegistryImmutableError(
            "forward_registry is immutable once written; "
            f"existing registry_id={existing.registry_id} "
            f"deployment_timestamp={existing.deployment_timestamp.isoformat()} "
            f"forward_start_date={existing.forward_start_date.isoformat()}"
        )

    ts = deployment_timestamp or datetime.now(UTC)
    ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)

    commit = git_commit if git_commit is not None else git_head_commit(project_root)
    fwd_start = compute_forward_start_date(ts)
    rid = registry_id or uuid.uuid4().hex

    row = ForwardRegistryRow(
        registry_id=rid,
        deployment_timestamp=ts,
        jev_model_id=jev_model_id,
        qset_hash=qset_hash,
        feature_version=feature_version,
        state_schema_version=state_schema_version,
        git_commit=commit,
        forward_start_date=fwd_start,
    )

    with eng.begin() as conn:
        # Re-check inside the transaction against races.
        again = conn.execute(text("SELECT COUNT(*) AS n FROM forward_registry")).fetchone()
        if again is not None and int(again[0]) > 0:
            raise ForwardRegistryImmutableError(
                "forward_registry is immutable once written (concurrent write detected)"
            )
        conn.execute(
            text(
                """
                INSERT INTO forward_registry (
                    registry_id, deployment_timestamp, jev_model_id, qset_hash,
                    feature_version, state_schema_version, git_commit, forward_start_date
                ) VALUES (
                    :registry_id, :deployment_timestamp, :jev_model_id, :qset_hash,
                    :feature_version, :state_schema_version, :git_commit, :forward_start_date
                )
                """
            ),
            {
                "registry_id": row.registry_id,
                "deployment_timestamp": row.deployment_timestamp.isoformat(),
                "jev_model_id": row.jev_model_id,
                "qset_hash": row.qset_hash,
                "feature_version": row.feature_version,
                "state_schema_version": row.state_schema_version,
                "git_commit": row.git_commit,
                "forward_start_date": row.forward_start_date.isoformat(),
            },
        )
    return row
