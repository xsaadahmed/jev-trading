"""Immutable pre-sealed Model A artifact for the V2 sealed test.

Build and archive BEFORE any 2024-2025 access. The sealed runner loads this
artifact and must not refit.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from jevlab.config.v2_research import PROTOCOL_REVISION
from jevlab.cv.purge import embargo_cutoff_date
from jevlab.experiments.artifacts import compute_integrity_digest
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.v2.dev_io import assert_frame_pre_sealed
from jevlab.v2.hashes import (
    FROZEN_CANDIDATE_HASH,
    FROZEN_FEATURE_SET_HASH,
    FROZEN_IMPLEMENTATION_REVISION,
    FROZEN_RESEARCH_CONFIG_HASH,
    FROZEN_STRATEGY_HASH,
    FROZEN_TARGET_HASH,
    MODEL_FREEZE_IMPLEMENTATION_REVISION,
    assert_frozen_research_hashes,
)
from jevlab.v2.training import (
    EXPECTED_CUTOFF,
    SEALED_T0,
    FittedModelA,
    TrainingLeakError,
    assert_no_sealed_training_years,
    assert_sealed_training_cutoff,
    filter_training_rows,
    fit_once_model_a,
    logical_model_digest,
    preprocessing_logical_digest,
    training_row_digest,
)

ARTIFACT_KIND = "v2_sealed_test_model"
DEFAULT_MANIFEST_REL = Path("reports") / "v2_preseal" / "SEALED_TEST_MODEL.json"
DEFAULT_PIPELINE_REL = Path("reports") / "v2_preseal" / "SEALED_TEST_MODEL.joblib"
FREEZE_REL = Path("config") / "v2" / "SEALED_TEST_MODEL_FREEZE.json"


def load_model_freeze(root: Path | None = None) -> dict[str, Any]:
    from jevlab.utils.paths import get_project_root

    path = (root or get_project_root()) / FREEZE_REL
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded)


def environment_digest() -> str:
    def _ver(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "missing"

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": _ver("numpy"),
            "pandas": _ver("pandas"),
            "scikit-learn": _ver("scikit-learn"),
            "scipy": _ver("scipy"),
        },
    }
    return sha256_of(canonical_json(payload))


def _git_head(root: Path) -> str:
    import subprocess

    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


@dataclass(frozen=True, slots=True)
class SealedTestModelArtifact:
    manifest: dict[str, Any]
    fitted: FittedModelA
    path: Path

    @property
    def integrity_digest(self) -> str:
        return str(self.manifest["integrity_digest"])

    @property
    def chosen_C(self) -> float:
        return float(self.manifest["chosen_C"])


def fit_presealed_model(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    t0: date = SEALED_T0,
    source_manifest_digest: str = "",
) -> tuple[FittedModelA, dict[str, Any]]:
    """Train + inner CV + one final fit. Refuses 2024/2025 training rows."""
    assert_frozen_research_hashes()
    assert_frame_pre_sealed(features)
    assert_frame_pre_sealed(labels)
    assert_frame_pre_sealed(bars)
    feat = features.copy()
    lab = labels.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.date
    lab["date"] = pd.to_datetime(lab["date"]).dt.date
    label_cols = [
        c for c in ("symbol", "date", "y", "X", "R", "Rbar", "y_cost") if c in lab.columns
    ]
    frame = feat.merge(
        lab.loc[:, label_cols],
        on=["symbol", "date"],
        how="left",
        suffixes=("", "_lab"),
    )
    if "eligible" not in frame.columns:
        raise TrainingLeakError("training frame missing eligibility")
    bar_dates = pd.to_datetime(bars["date"]).dt.date.tolist()
    sessions = sorted(set(frame["date"].tolist()) | set(bar_dates))
    if t0 not in sessions:
        sessions = sorted(set(sessions) | {t0})
    cutoff = embargo_cutoff_date(sessions, t0, embargo_sessions=5)
    if cutoff is None:
        raise TrainingLeakError("could not derive embargo cutoff")
    if t0 == SEALED_T0:
        cal = assert_sealed_training_cutoff()
        if cutoff != cal or cutoff != EXPECTED_CUTOFF:
            raise TrainingLeakError(f"final training decision date {cutoff} != {EXPECTED_CUTOFF}")
        if cutoff != date(2023, 12, 22):
            raise TrainingLeakError("final training decision date must be 2023-12-22")
    pre_t0 = frame.loc[frame["date"] < t0].copy()
    assert_frame_pre_sealed(pre_t0)
    train = filter_training_rows(pre_t0, cutoff=cutoff, t0=t0)
    n_2024, n_2025 = assert_no_sealed_training_years(train)
    if train["date"].max() != cutoff and t0 == SEALED_T0:
        raise TrainingLeakError(f"final training decision date {train['date'].max()} != {cutoff}")
    fitted = fit_once_model_a(train, sessions=sessions, t0=t0, cutoff=cutoff)
    if fitted.fit_count != 1:
        raise TrainingLeakError("fit_count != 1")
    meta = {
        "cutoff": cutoff,
        "final_training_decision_date": str(fitted.cutoff),
        "n_train_rows": fitted.n_train_rows,
        "n_2024_train": n_2024,
        "n_2025_train": n_2025,
        "training_row_digest": training_row_digest(train),
        "preprocessing_digest": preprocessing_logical_digest(fitted),
        "model_logical_digest": logical_model_digest(fitted),
        "training_source_manifest_digest": source_manifest_digest,
        "chosen_C": fitted.C,
        "fit_count": 1,
    }
    return fitted, meta


def build_manifest(
    *,
    fitted: FittedModelA,
    meta: dict[str, Any],
    git_commit: str,
    env_digest: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "protocol_revision": PROTOCOL_REVISION,
        "implementation_revision": FROZEN_IMPLEMENTATION_REVISION,
        "research_config_hash": FROZEN_RESEARCH_CONFIG_HASH,
        "candidate_hash": FROZEN_CANDIDATE_HASH,
        "feature_hash": FROZEN_FEATURE_SET_HASH,
        "target_hash": FROZEN_TARGET_HASH,
        "strategy_hash": FROZEN_STRATEGY_HASH,
        "training_source_manifest_digest": meta["training_source_manifest_digest"],
        "training_row_digest": meta["training_row_digest"],
        "final_training_decision_date": meta["final_training_decision_date"],
        "training_observation_count": int(fitted.n_train_rows),
        "chosen_C": float(fitted.C),
        "preprocessing_digest": meta["preprocessing_digest"],
        "fitted_model_digest": meta["model_logical_digest"],
        "source_code_commit": git_commit,
        "python_package_environment_digest": env_digest,
        "created_at": datetime.now(UTC).isoformat(),
        "fit_count": 1,
        "not_an_acceptance_artifact": True,
        "contains_2024_2025": False,
    }
    body["integrity_digest"] = compute_integrity_digest(
        {k: v for k, v in body.items() if k != "integrity_digest"}
    )
    return body


def write_sealed_test_model(
    *,
    fitted: FittedModelA,
    meta: dict[str, Any],
    out_dir: Path,
    git_commit: str,
    env_digest: str | None = None,
) -> SealedTestModelArtifact:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = env_digest if env_digest is not None else environment_digest()
    manifest = build_manifest(fitted=fitted, meta=meta, git_commit=git_commit, env_digest=env)
    pipe_path = out_dir / "SEALED_TEST_MODEL.joblib"
    man_path = out_dir / "SEALED_TEST_MODEL.json"
    joblib.dump(  # type: ignore[no-untyped-call]
        {
            "pipeline": fitted.pipeline,
            "C": fitted.C,
            "cutoff": fitted.cutoff.isoformat(),
            "n_train_rows": fitted.n_train_rows,
            "fit_count": fitted.fit_count,
            "seed": fitted.seed,
            "columns": list(fitted.columns),
        },
        pipe_path,
    )
    man_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return SealedTestModelArtifact(manifest=manifest, fitted=fitted, path=man_path)


def load_sealed_test_model(manifest_path: Path) -> SealedTestModelArtifact:
    man_path = Path(manifest_path)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    expected = compute_integrity_digest(
        {k: v for k, v in manifest.items() if k != "integrity_digest"}
    )
    if expected != manifest.get("integrity_digest"):
        raise TrainingLeakError("sealed-test-model integrity_digest mismatch")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise TrainingLeakError("artifact is not v2_sealed_test_model")
    if int(manifest.get("protocol_revision", -1)) != PROTOCOL_REVISION:
        raise TrainingLeakError("sealed-test-model protocol revision drifted")
    vintage = int(manifest.get("implementation_revision", -1))
    if vintage not in {MODEL_FREEZE_IMPLEMENTATION_REVISION, FROZEN_IMPLEMENTATION_REVISION}:
        raise TrainingLeakError(
            "sealed-test-model implementation revision is not the frozen Model A vintage "
            f"({MODEL_FREEZE_IMPLEMENTATION_REVISION}) or the current runner "
            f"({FROZEN_IMPLEMENTATION_REVISION})"
        )
    if manifest.get("contains_2024_2025"):
        raise TrainingLeakError("sealed-test-model must not contain 2024-2025 information")
    pipe_path = man_path.with_name("SEALED_TEST_MODEL.joblib")
    blob = joblib.load(pipe_path)  # type: ignore[no-untyped-call]
    pipeline = blob["pipeline"]
    if not isinstance(pipeline, Pipeline):
        raise TrainingLeakError("sealed-test-model pipeline missing")
    fitted = FittedModelA(
        pipeline=pipeline,
        C=float(blob["C"]),
        cutoff=date.fromisoformat(str(blob["cutoff"])),
        n_train_rows=int(blob["n_train_rows"]),
        fit_count=int(blob["fit_count"]),
        seed=int(blob["seed"]),
        columns=tuple(blob["columns"]),
    )
    if fitted.fit_count != 1:
        raise TrainingLeakError("loaded model fit_count != 1")
    live_logical = logical_model_digest(fitted)
    if live_logical != manifest["fitted_model_digest"]:
        raise TrainingLeakError("loaded model logical digest drifted from manifest")
    if preprocessing_logical_digest(fitted) != manifest["preprocessing_digest"]:
        raise TrainingLeakError("loaded preprocessing digest drifted from manifest")
    if list(fitted.columns) != list(MODEL_A_COLUMNS):
        raise TrainingLeakError("loaded model columns are not MODEL_A_COLUMNS")
    return SealedTestModelArtifact(manifest=manifest, fitted=fitted, path=man_path)


def assert_artifact_matches_freeze(
    artifact: SealedTestModelArtifact,
    *,
    expected: dict[str, Any],
) -> None:
    """Fail closed if the prebuilt model drifted from the r3 freeze record."""
    checks = {
        "fitted_model_digest": artifact.manifest["fitted_model_digest"],
        "preprocessing_digest": artifact.manifest["preprocessing_digest"],
        "training_row_digest": artifact.manifest["training_row_digest"],
        "chosen_C": artifact.manifest["chosen_C"],
        "training_source_manifest_digest": artifact.manifest["training_source_manifest_digest"],
    }
    for key, got in checks.items():
        if key not in expected:
            continue
        if got != expected[key]:
            raise TrainingLeakError(
                f"sealed-test-model freeze mismatch for {key}: live={got} frozen={expected[key]}"
            )
