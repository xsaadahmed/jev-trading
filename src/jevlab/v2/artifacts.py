"""V2_SEALED_PASS.json - written only when every frozen hard gate passes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jevlab.experiments.artifacts import ArtifactError, compute_integrity_digest
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.v2.acceptance import GATE_IDS, V2GateReport
from jevlab.v2.hashes import (
    FROZEN_ACCEPTANCE_GATE_HASH,
    FROZEN_CANDIDATE_HASH,
    FROZEN_COMMIT,
    FROZEN_COST_HASH,
    FROZEN_FEATURE_SET_HASH,
    FROZEN_RANDOM_NULL_HASH,
    FROZEN_RESEARCH_CONFIG_HASH,
    FROZEN_STRATEGY_HASH,
    FROZEN_TARGET_HASH,
    FROZEN_UNIVERSE_HASH,
)

V2_SEALED_PASS_STATUS = "V2_SEALED_PASS"
V2_SEALED_PASS_FILENAME = "V2_SEALED_PASS.json"


@dataclass(frozen=True, slots=True)
class V2AcceptanceArtifact:
    path: Path
    status: str
    integrity_digest: str
    body: dict[str, Any]


def write_v2_sealed_pass(
    report: V2GateReport,
    *,
    out_dir: Path,
    git_commit: str,
    sealed_result_digest: str,
    training_artifact_digest: str,
    random_null_artifact_digest: str,
    created_at: datetime | None = None,
) -> V2AcceptanceArtifact:
    """Write V2_SEALED_PASS.json. Raises if any gate failed/missing/unevaluable."""
    if report.seal_broken if hasattr(report, "seal_broken") else False:
        raise ArtifactError("seal broken; cannot write V2_SEALED_PASS")
    if not report.results:
        raise ArtifactError("no gate results")
    present = {r.gate_id for r in report.results}
    missing = [g for g in GATE_IDS if g not in present]
    if missing:
        raise ArtifactError(f"missing gates: {missing}")
    if any(r.passed is not True for r in report.results):
        raise ArtifactError("not every hard gate passed; V2_SEALED_PASS not written")
    if report.label == "development_rehearsal_only":
        raise ArtifactError("rehearsal results cannot create V2_SEALED_PASS")

    created = created_at or datetime.now(UTC)
    body: dict[str, Any] = {
        "status": V2_SEALED_PASS_STATUS,
        "created_at": created.isoformat(),
        "frozen_v2_git_commit": FROZEN_COMMIT,
        "implementation_git_commit": git_commit,
        "research_config_hash": FROZEN_RESEARCH_CONFIG_HASH,
        "candidate_hash": FROZEN_CANDIDATE_HASH,
        "universe_hash": FROZEN_UNIVERSE_HASH,
        "feature_hash": FROZEN_FEATURE_SET_HASH,
        "target_hash": FROZEN_TARGET_HASH,
        "strategy_hash": FROZEN_STRATEGY_HASH,
        "cost_hash": FROZEN_COST_HASH,
        "random_null_hash": FROZEN_RANDOM_NULL_HASH,
        "acceptance_gate_hash": FROZEN_ACCEPTANCE_GATE_HASH,
        "sealed_result_report_digest": sealed_result_digest,
        "training_artifact_digest": training_artifact_digest,
        "random_null_artifact_digest": random_null_artifact_digest,
        "gate_results": [r.to_dict() for r in report.results],
        "note": "integrity_digest is an integrity digest, not a cryptographic signature",
    }
    digest = compute_integrity_digest(body)
    body["integrity_digest"] = digest
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / V2_SEALED_PASS_FILENAME
    path.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    return V2AcceptanceArtifact(
        path=path, status=V2_SEALED_PASS_STATUS, integrity_digest=digest, body=body
    )


def verify_v2_sealed_pass(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"missing {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(body.get("integrity_digest", ""))
    rest = {k: v for k, v in body.items() if k != "integrity_digest"}
    expected = compute_integrity_digest(rest)
    if claimed != expected:
        raise ArtifactError(
            "integrity_digest mismatch (artifact tampered; this is an integrity "
            "digest check, not a signature)"
        )
    if body.get("status") != V2_SEALED_PASS_STATUS:
        raise ArtifactError("status is not V2_SEALED_PASS")
    if body.get("research_config_hash") != FROZEN_RESEARCH_CONFIG_HASH:
        raise ArtifactError("research_config_hash lineage mismatch")
    return dict(body)


def digest_payload(payload: Mapping[str, Any]) -> str:
    return sha256_of(canonical_json(dict(payload)))


def invalidate_v2_sealed_pass(acceptance_dir: Path, *, reason: str = "SEAL BROKEN") -> Path | None:
    """Move any existing V2_SEALED_PASS.json aside; verification of the original path fails."""
    path = Path(acceptance_dir) / V2_SEALED_PASS_FILENAME
    if not path.is_file():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    dest = Path(acceptance_dir) / "V2_SEALED_PASS.INVALIDATED.json"
    body["status"] = "V2_SEALED_PASS_INVALIDATED"
    body["invalidation_reason"] = reason
    body["original_integrity_digest"] = body.get("integrity_digest")
    dest.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    path.unlink()
    return dest


def verify_result_artifact(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(body.get("integrity_digest", ""))
    rest = {k: v for k, v in body.items() if k != "integrity_digest"}
    expected = compute_integrity_digest(rest)
    if claimed != expected:
        raise ArtifactError("sealed result integrity_digest mismatch")
    return dict(body)
