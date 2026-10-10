"""Production-forward model artifacts (MASTER_SPEC §10.7 / Phase 13B).

Rationale (documented for the ledger and reports):
The sealed evaluation measured the ex-ante frozen *procedure* trained on
rows <= 2023-12-29 and scored on data it had never seen; that artifact and
its result are preserved. Refitting the same frozen procedure on more
pre-deployment data after the seal is opened does not alter that evidence
and is evaluated only on forward data that post-dates its training cutoff.
Any change beyond the training sample is a new trial and cannot inherit
HISTORICAL_RESEARCH_PASS.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from jevlab.experiments.artifacts import ArtifactError, archive_model_artifact
from jevlab.experiments.forward_registry import ForwardRegistryRow
from jevlab.experiments.manifest import insert_trial_ledger
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import JEV_LOGIT_COLUMNS, MODEL_B_COLUMNS
from jevlab.models.transforms import make_lineup_logistic_pipeline
from jevlab.utils.calendar import previous_nyse_session
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger
from jevlab.utils.registry import init_registry

log = get_logger(__name__)

PRODUCTION_RATIONALE: str = (
    "The sealed evaluation measured the ex-ante frozen procedure trained on "
    "rows <= 2023-12-29 and scored on held-out sealed data; that artifact stays "
    "untouched. Refitting the same frozen procedure on all pre-deployment rows "
    "whose labels are known by deployment_timestamp (embargo applied) does not "
    "alter the sealed evidence and is evaluated only on forward data after the "
    "training cutoff."
)

PRODUCTION_HASHES_FILENAME: str = "production_model_hashes.json"
DEFAULT_EMBARGO_SESSIONS: int = 5
DEFAULT_C: float = 1.0
DEFAULT_SEED: int = 42


@dataclass(frozen=True, slots=True)
class ProductionBuildResult:
    artifact_ids: dict[str, str]
    train_end: date
    n_train_rows: int
    hashes_path: Path
    content_digests: dict[str, str]
    rationale: str = PRODUCTION_RATIONALE


def require_sealed_test_artifacts(registry_path: Path) -> dict[str, str]:
    """Return {model_id: artifact_id} for sealed_test A and B; raise if missing."""
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT model_id, model_artifact_id FROM model_artifacts WHERE kind = 'sealed_test'"
            )
        ).fetchall()
    found = {str(r[0]): str(r[1]) for r in rows}
    missing = [m for m in ("A", "B") if m not in found]
    if missing:
        raise ArtifactError(
            f"sealed_test artifacts missing for {missing}; run final-test before "
            "build-production-models"
        )
    return found


def list_model_artifacts(
    registry_path: Path,
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT model_artifact_id, model_id, columns_json, train_end, "
                "hyperparams_json, content_digest, git_commit, created_at "
                "FROM model_artifacts WHERE kind = :kind"
            ),
            {"kind": kind},
        ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[str(r[1])] = {
            "model_artifact_id": str(r[0]),
            "columns_json": str(r[2]),
            "train_end": str(r[3]),
            "hyperparams_json": str(r[4]),
            "content_digest": str(r[5]),
            "git_commit": str(r[6]),
            "created_at": str(r[7]),
        }
    return out


def compute_production_train_end(
    forward_start_date: date,
    *,
    embargo_sessions: int = DEFAULT_EMBARGO_SESSIONS,
) -> date:
    """Last session eligible for production training.

    No forward row (date >= forward_start) may enter training. Embargo pushes
    train_end back ``embargo_sessions`` NYSE sessions before forward_start.
    """
    cursor = previous_nyse_session(forward_start_date)
    for _ in range(max(embargo_sessions - 1, 0)):
        cursor = previous_nyse_session(cursor)
    return cursor


def filter_predeployment_training_rows(
    frame: pd.DataFrame,
    *,
    train_end: date,
    forward_start_date: date,
) -> pd.DataFrame:
    """Keep rows with date <= train_end and strictly before forward_start."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    if (out["date"] >= forward_start_date).any():
        raise ArtifactError(
            "training frame contains forward rows (date >= forward_start_date); refused"
        )
    filtered = out.loc[out["date"] <= train_end].reset_index(drop=True)
    if filtered.empty:
        raise ArtifactError("no pre-deployment training rows after train_end filter")
    return filtered


def production_hashes_path(forward_root: Path) -> Path:
    return Path(forward_root) / PRODUCTION_HASHES_FILENAME


def write_production_hashes(
    path: Path,
    *,
    artifact_ids: Mapping[str, str],
    content_digests: Mapping[str, str],
    train_end: date,
    git_commit: str,
    recorded_at: datetime | None = None,
) -> Path:
    """Record production hashes BEFORE any forward-label join (§10.7)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": (recorded_at or datetime.now(UTC)).isoformat(),
        "train_end": train_end.isoformat(),
        "git_commit": git_commit,
        "artifact_ids": dict(artifact_ids),
        "content_digests": dict(content_digests),
        "rationale": PRODUCTION_RATIONALE,
        "kind": "production_forward",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_production_hashes(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else None


def pipeline_path(artifact_dir: Path, model_id: str, artifact_id: str) -> Path:
    return Path(artifact_dir) / f"production_forward_{model_id}_{artifact_id[:8]}.pkl"


def build_production_models(
    train_frame: pd.DataFrame,
    *,
    registry_path: Path,
    artifact_dir: Path,
    forward_root: Path,
    deployment: ForwardRegistryRow,
    git_commit: str,
    C: float = DEFAULT_C,
    seed: int = DEFAULT_SEED,
    embargo_sessions: int = DEFAULT_EMBARGO_SESSIONS,
    qset_version: str = "qset_v1",
    state_schema_version: str = "state_v1",
) -> ProductionBuildResult:
    """Retrain frozen A/B procedure on pre-deployment labeled rows.

    Requires sealed_test artifacts to already exist (untouched). Records hashes
    in trial_ledger and ``production_model_hashes.json`` before any label join.
    """
    sealed = require_sealed_test_artifacts(registry_path)
    train_end = compute_production_train_end(
        deployment.forward_start_date, embargo_sessions=embargo_sessions
    )
    frame = filter_predeployment_training_rows(
        train_frame,
        train_end=train_end,
        forward_start_date=deployment.forward_start_date,
    )
    y = frame["y"].to_numpy()
    artifact_ids: dict[str, str] = {}
    digests: dict[str, str] = {}
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for mid, cols, logit in (
        ("A", MODEL_A_COLUMNS, ()),
        ("B", MODEL_B_COLUMNS, JEV_LOGIT_COLUMNS),
    ):
        X = frame.loc[:, list(cols)].to_numpy(dtype=float)
        pipe = make_lineup_logistic_pipeline(
            C=C, seed=seed, columns=tuple(cols), logit_columns=tuple(logit)
        )
        pipe.fit(X, y)
        digest = sha256_of(
            canonical_json(
                {
                    "model_id": mid,
                    "kind": "production_forward",
                    "train_end": train_end.isoformat(),
                    "C": C,
                    "seed": seed,
                    "n_rows": len(frame),
                    "columns": list(cols),
                    "forward_start_date": deployment.forward_start_date.isoformat(),
                }
            )
        )
        digests[mid] = digest
        aid = archive_model_artifact(
            registry_path=registry_path,
            kind="production_forward",
            model_id=mid,
            columns=list(cols),
            train_end=train_end,
            hyperparams={"C": C, "seed": seed, "embargo_sessions": embargo_sessions},
            content_digest=digest,
            git_commit=git_commit,
            artifact_dir=artifact_dir,
            blob={
                "n_train_rows": len(frame),
                "sealed_test_artifact_id": sealed[mid],
                "rationale": PRODUCTION_RATIONALE,
                "jev_columns": list(JEV_FEATURE_COLUMNS),
            },
        )
        artifact_ids[mid] = aid
        with pipeline_path(artifact_dir, mid, aid).open("wb") as fh:
            pickle.dump(pipe, fh, protocol=pickle.HIGHEST_PROTOCOL)

    hashes_path = write_production_hashes(
        production_hashes_path(forward_root),
        artifact_ids=artifact_ids,
        content_digests=digests,
        train_end=train_end,
        git_commit=git_commit,
    )

    # Ledger BEFORE any forward-label join.
    for mid, aid in artifact_ids.items():
        insert_trial_ledger(
            registry_path,
            purpose="exploratory",
            feature_set=f"production_forward_{mid}",
            model_class="logistic",
            grid={
                "C": C,
                "seed": seed,
                "train_end": train_end.isoformat(),
                "artifact_id": aid,
                "content_digest": digests[mid],
            },
            experiment_id=f"production-forward-{mid}",
            result_summary=json.dumps(
                {
                    "kind": "production_forward",
                    "artifact_id": aid,
                    "rationale": PRODUCTION_RATIONALE,
                    "sealed_test_artifact_id": sealed[mid],
                }
            ),
            qset_version=qset_version,
            state_schema_version=state_schema_version,
        )

    log.info(
        "production_models.built",
        train_end=str(train_end),
        n_rows=len(frame),
        artifacts=artifact_ids,
    )
    return ProductionBuildResult(
        artifact_ids=artifact_ids,
        train_end=train_end,
        n_train_rows=len(frame),
        hashes_path=hashes_path,
        content_digests=digests,
    )


def load_production_pipeline(artifact_dir: Path, model_id: str, artifact_id: str) -> Any:
    path = pipeline_path(artifact_dir, model_id, artifact_id)
    if not path.is_file():
        raise ArtifactError(f"missing production pipeline file: {path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def assert_production_differs_from_sealed_only_by_sample(
    registry_path: Path,
) -> dict[str, Any]:
    """Compare sealed_test vs production_forward metadata (same procedure)."""
    sealed = list_model_artifacts(registry_path, kind="sealed_test")
    prod = list_model_artifacts(registry_path, kind="production_forward")
    if set(sealed) != {"A", "B"} or set(prod) != {"A", "B"}:
        raise ArtifactError("need sealed_test and production_forward for A and B")
    report: dict[str, Any] = {"models": {}}
    for mid in ("A", "B"):
        s_cols = sealed[mid]["columns_json"]
        p_cols = prod[mid]["columns_json"]
        if s_cols != p_cols:
            raise ArtifactError(f"{mid}: column lists differ between sealed and production")
        s_hp = json.loads(sealed[mid]["hyperparams_json"])
        p_hp = json.loads(prod[mid]["hyperparams_json"])
        for key in ("C", "seed"):
            if s_hp.get(key) != p_hp.get(key):
                raise ArtifactError(
                    f"{mid}: hyperparam {key} differs (sealed={s_hp.get(key)} "
                    f"production={p_hp.get(key)})"
                )
        report["models"][mid] = {
            "sealed_train_end": sealed[mid]["train_end"],
            "production_train_end": prod[mid]["train_end"],
            "columns_identical": True,
            "hyperparams_C_seed_identical": True,
            "content_digest_sealed": sealed[mid]["content_digest"],
            "content_digest_production": prod[mid]["content_digest"],
            "digests_differ": sealed[mid]["content_digest"] != prod[mid]["content_digest"],
        }
    report["same_procedure_different_sample"] = True
    return report
