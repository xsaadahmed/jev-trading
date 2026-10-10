"""Integrity-digested acceptance artifacts (MASTER_SPEC §16.0).

``integrity_digest`` is an integrity digest (detects accidental edits / stale
configs), **not** a cryptographic signature. Never call these artifacts
'signed'.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text

from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.registry import init_registry

AcceptanceKind = Literal["historical", "tiny_live", "scale"]

STATUS_BY_KIND: dict[AcceptanceKind, str] = {
    "historical": "HISTORICAL_RESEARCH_PASS",
    "tiny_live": "TINY_LIVE_READY",
    "scale": "SCALE_READY",
}

EXPIRY_DAYS: dict[AcceptanceKind, int] = {
    "historical": 180,
    "tiny_live": 90,
    "scale": 90,
}

FILENAME_BY_KIND: dict[AcceptanceKind, str] = {
    "historical": "HISTORICAL_ACCEPTANCE.json",
    "tiny_live": "TINY_LIVE_ACCEPTANCE.json",
    "scale": "SCALE_ACCEPTANCE.json",
}

# Required statement on HISTORICAL_RESEARCH_PASS (§16.1).
HISTORICAL_ACCEPTANCE_STATEMENT: str = (
    "Historical Jev evidence is potentially training-contaminated; "
    "this status does not establish that the edge is genuine."
)


class ArtifactError(RuntimeError):
    """Acceptance artifact write/verify failure."""


@dataclass(frozen=True, slots=True)
class AcceptanceArtifact:
    path: Path
    status: str
    integrity_digest: str
    body: dict[str, Any]


def _now_utc() -> datetime:
    return datetime.now(UTC)


def compute_integrity_digest(body_without_digest: Mapping[str, Any]) -> str:
    """``integrity_digest = sha256(canonical_json(all other fields))``."""
    return sha256_of(canonical_json(dict(body_without_digest)))


def write_acceptance(
    kind: AcceptanceKind,
    payload: Mapping[str, Any],
    *,
    out_dir: Path,
    git_commit: str,
    research_config_hash: str,
    qset_hash: str,
    model_resolved: str,
    gate_report_sha256: str,
    parent_artifact_digest: str | None = None,
    registry_path: Path | None = None,
    created_at: datetime | None = None,
) -> AcceptanceArtifact:
    """Write an integrity-digested acceptance artifact (only call when gates pass).

    Never labels the digest as a signature. Historical artifacts include the
    §16.1 contamination caveat in ``payload``.
    """
    if kind not in STATUS_BY_KIND:
        raise ArtifactError(f"unknown acceptance kind: {kind}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = created_at or _now_utc()
    expires = created + timedelta(days=EXPIRY_DAYS[kind])
    status = STATUS_BY_KIND[kind]

    payload_body = dict(payload)
    if kind == "historical":
        payload_body.setdefault("statement", HISTORICAL_ACCEPTANCE_STATEMENT)
        payload_body.setdefault(
            "note",
            "integrity_digest is an integrity digest, not a cryptographic signature",
        )

    body: dict[str, Any] = {
        "status": status,
        "created_at": created.isoformat(),
        "expires_at": expires.isoformat(),
        "git_commit": git_commit,
        "research_config_hash": research_config_hash,
        "qset_hash": qset_hash,
        "model_resolved": model_resolved,
        "gate_report_sha256": gate_report_sha256,
        "parent_artifact_digest": parent_artifact_digest,
        "payload": payload_body,
    }
    digest = compute_integrity_digest(body)
    body["integrity_digest"] = digest

    path = out_dir / FILENAME_BY_KIND[kind]
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")

    if registry_path is not None:
        engine = init_registry(registry_path)
        artifact_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO acceptance_artifacts (
                        artifact_id, status, path, integrity_digest, parent_digest,
                        gate_report_sha256, research_config_hash, git_commit,
                        created_at, expires_at
                    ) VALUES (
                        :artifact_id, :status, :path, :integrity_digest, :parent_digest,
                        :gate_report_sha256, :research_config_hash, :git_commit,
                        :created_at, :expires_at
                    )
                    """
                ),
                {
                    "artifact_id": artifact_id,
                    "status": status,
                    "path": str(path),
                    "integrity_digest": digest,
                    "parent_digest": parent_artifact_digest,
                    "gate_report_sha256": gate_report_sha256,
                    "research_config_hash": research_config_hash,
                    "git_commit": git_commit,
                    "created_at": created.isoformat(),
                    "expires_at": expires.isoformat(),
                },
            )

    return AcceptanceArtifact(path=path, status=status, integrity_digest=digest, body=body)


def verify_acceptance(
    path: Path | str,
    *,
    research_config_hash: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify integrity digest, expiry, and optional config lineage.

    Raises ``ArtifactError`` on any mismatch. Does not claim cryptographic
    signature verification.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"acceptance artifact missing: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    if "integrity_digest" not in body:
        raise ArtifactError("missing integrity_digest")
    claimed = str(body["integrity_digest"])
    rest = {k: v for k, v in body.items() if k != "integrity_digest"}
    expected = compute_integrity_digest(rest)
    if claimed != expected:
        raise ArtifactError(
            "integrity_digest mismatch (artifact tampered or rewritten; "
            "this is an integrity digest check, not a signature)"
        )
    now = now or _now_utc()
    expires_raw = body.get("expires_at")
    if not expires_raw:
        raise ArtifactError("missing expires_at")
    expires = datetime.fromisoformat(str(expires_raw))
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if now > expires:
        raise ArtifactError(f"acceptance artifact expired at {expires.isoformat()}")
    if (
        research_config_hash is not None
        and str(body.get("research_config_hash", "")) != research_config_hash
    ):
        raise ArtifactError("research_config_hash lineage mismatch")
    return body


def archive_model_artifact(
    *,
    registry_path: Path,
    kind: Literal["sealed_test", "production_forward"],
    model_id: str,
    columns: list[str] | tuple[str, ...],
    train_end: date,
    hyperparams: Mapping[str, Any],
    content_digest: str,
    git_commit: str,
    artifact_dir: Path,
    blob: Mapping[str, Any] | None = None,
) -> str:
    """Persist a model artifact row + optional JSON blob. Never overwrites.

    Sealed-test artifacts are permanent: if a row already exists for the same
    (kind, model_id), raise rather than replace.
    """
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        existing = conn.execute(
            text(
                """
                SELECT model_artifact_id FROM model_artifacts
                WHERE kind = :kind AND model_id = :model_id
                """
            ),
            {"kind": kind, "model_id": model_id},
        ).fetchone()
    if existing is not None:
        raise ArtifactError(
            f"model artifact kind={kind} model_id={model_id} already archived "
            f"({existing[0]}); sealed_test artifacts must never be overwritten"
        )

    artifact_id = str(uuid.uuid4())
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_artifact_id": artifact_id,
        "kind": kind,
        "model_id": model_id,
        "columns": list(columns),
        "train_end": train_end.isoformat(),
        "hyperparams": dict(hyperparams),
        "content_digest": content_digest,
        "git_commit": git_commit,
        "blob": dict(blob) if blob else {},
    }
    path = artifact_dir / f"{kind}_{model_id}_{artifact_id[:8]}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_artifacts (
                    model_artifact_id, kind, model_id, columns_json, train_end,
                    hyperparams_json, content_digest, git_commit, created_at
                ) VALUES (
                    :model_artifact_id, :kind, :model_id, :columns_json, :train_end,
                    :hyperparams_json, :content_digest, :git_commit, :created_at
                )
                """
            ),
            {
                "model_artifact_id": artifact_id,
                "kind": kind,
                "model_id": model_id,
                "columns_json": canonical_json(list(columns)),
                "train_end": train_end.isoformat(),
                "hyperparams_json": canonical_json(dict(hyperparams)),
                "content_digest": content_digest,
                "git_commit": git_commit,
                "created_at": _now_utc().isoformat(),
            },
        )
    return artifact_id
