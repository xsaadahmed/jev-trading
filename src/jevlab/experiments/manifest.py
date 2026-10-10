"""Experiment manifests and trial-ledger writers (MASTER_SPEC §18 / Phase 8)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text

from jevlab.models.column_sets import CONTAMINATION_LABEL
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.registry import init_registry

TrialPurpose = Literal["confirmatory", "exploratory", "audit", "placebo"]


def write_manifest(payload: dict[str, Any], out_dir: Path) -> Path:
    """Write ``manifest.json`` under ``out_dir``; stamp contamination label."""
    out_dir.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("contamination_label", CONTAMINATION_LABEL)
    body.setdefault("potentially_training_contaminated", True)
    body.setdefault("created_at", datetime.now(UTC).isoformat())
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return path


def manifest_completeness_keys() -> tuple[str, ...]:
    """Required keys for Phase 8 lineup manifests."""
    return (
        "experiment_id",
        "models",
        "folds",
        "C_grid",
        "seed",
        "contamination_label",
        "potentially_training_contaminated",
        "column_counts",
        "purpose_by_model",
        "metrics_by_model",
    )


def assert_manifest_complete(payload: dict[str, Any]) -> None:
    missing = [k for k in manifest_completeness_keys() if k not in payload]
    if missing:
        raise AssertionError(f"manifest missing keys: {missing}")


def insert_trial_ledger(
    registry_path: Path,
    *,
    purpose: TrialPurpose,
    feature_set: str,
    model_class: str,
    grid: dict[str, Any],
    experiment_id: str,
    result_summary: str,
    qset_version: str = "qset_v1",
    state_schema_version: str = "state_v1",
    universe_version: str = "v1",
    horizon: int = 3,
) -> str:
    """Append one ``trial_ledger`` row; return ``trial_id``."""
    engine = init_registry(registry_path)
    trial_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trial_ledger (
                    trial_id, created_at, qset_version, state_schema_version,
                    feature_set, model_class, grid_json, universe_version,
                    horizon, purpose, experiment_id, result_summary
                ) VALUES (
                    :trial_id, :created_at, :qset_version, :state_schema_version,
                    :feature_set, :model_class, :grid_json, :universe_version,
                    :horizon, :purpose, :experiment_id, :result_summary
                )
                """
            ),
            {
                "trial_id": trial_id,
                "created_at": datetime.now(UTC).isoformat(),
                "qset_version": qset_version,
                "state_schema_version": state_schema_version,
                "feature_set": feature_set,
                "model_class": model_class,
                "grid_json": canonical_json(grid),
                "universe_version": universe_version,
                "horizon": horizon,
                "purpose": purpose,
                "experiment_id": experiment_id,
                "result_summary": result_summary,
            },
        )
    return trial_id


def content_digest(payload: dict[str, Any]) -> str:
    return sha256_of(canonical_json(payload))
