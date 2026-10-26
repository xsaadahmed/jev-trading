"""Pre-seal environment snapshot. Does not read sealed 2024-2025 contents."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sqlalchemy import text

from jevlab.experiments.artifacts import compute_integrity_digest
from jevlab.features.columns import FEATURE_VERSION, MODEL_A_COLUMNS
from jevlab.utils.hashing import canonical_json, logical_table_digest, sha256_of
from jevlab.utils.paths import ProjectPaths, get_project_root
from jevlab.v2.dev_io import DEV_YEAR_MAX, DEV_YEAR_MIN, iter_dev_year_paths, read_dev_feature_store
from jevlab.v2.hashes import FROZEN_COMMIT, assert_frozen_research_hashes, frozen_hash_payload
from jevlab.v2.training import EXPECTED_CUTOFF, SEALED_T0, assert_sealed_training_cutoff

SNAPSHOT_REL = Path("reports") / "v2_preseal" / "PRESEAL_SNAPSHOT.json"

CODE_FILES = (
    "src/jevlab/v2/candidate.py",
    "src/jevlab/v2/null_engine.py",
    "src/jevlab/v2/seed_policy.py",
    "src/jevlab/v2/acceptance.py",
    "src/jevlab/v2/sealed_run.py",
    "src/jevlab/v2/sealed_source.py",
    "src/jevlab/v2/training.py",
    "src/jevlab/v2/statistics.py",
    "src/jevlab/v2/evaluate.py",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def build_preseal_snapshot(*, root: Path | None = None) -> dict[str, Any]:
    root = root or get_project_root()
    paths = ProjectPaths.from_root(root)
    hashes = assert_frozen_research_hashes(root=root)
    cutoff = assert_sealed_training_cutoff()
    uv_lock = root / "uv.lock"
    feat_root = paths.data_features / FEATURE_VERSION
    feat_files = iter_dev_year_paths(feat_root, "symbol=*/")
    file_manifest = [
        {
            "rel": str(p.relative_to(root)).replace("\\", "/"),
            "size": int(p.stat().st_size),
        }
        for p in feat_files
    ]
    features = read_dev_feature_store()
    keep = ["symbol", "date", "eligible", *MODEL_A_COLUMNS]
    keep = [c for c in keep if c in features.columns]
    logical = (
        logical_table_digest(features.loc[:, keep], primary_key=["symbol", "date"])
        if not features.empty
        else {"content_sha256": "empty", "schema_sha256": "empty"}
    )
    from jevlab.utils.registry import init_registry

    engine = init_registry(paths.registry / "experiments.sqlite")
    with engine.connect() as conn:
        sealed_n = int(conn.execute(text("SELECT COUNT(1) FROM sealed_access_log")).scalar_one())
    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "implementation_git_commit": _git_head(root),
        "frozen_design_commit": FROZEN_COMMIT,
        "research_hashes": {**frozen_hash_payload(), "live": hashes},
        "uv_lock_sha256": _file_sha256(uv_lock) if uv_lock.is_file() else None,
        "python_version": sys.version,
        "package_versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "scipy": _package_version("scipy"),
            "exchange-calendars": _package_version("exchange-calendars"),
        },
        "os_runtime": platform.platform(),
        "v1_model_a_parent_commit": "e5724df4b943006b3f73b4d6333a5177a7b77510",
        "training_cutoff": str(cutoff),
        "t0": str(SEALED_T0),
        "expected_cutoff": str(EXPECTED_CUTOFF),
        "pre2024_feature_file_manifest_digest": sha256_of(canonical_json(file_manifest)),
        "pre2024_feature_logical_digest": logical,
        "dev_year_min": DEV_YEAR_MIN,
        "dev_year_max": DEV_YEAR_MAX,
        "feature_version": FEATURE_VERSION,
        "calendar": "exchange_calendars.XNYS / jevlab.utils.calendar.nyse_session_dates",
        "code_hashes": {
            rel: _file_sha256(root / rel) for rel in CODE_FILES if (root / rel).is_file()
        },
        "configured_sealed_path_string": str(paths.data_sealed),
        "sealed_access_log_count": sealed_n,
        "note": (
            "configured_sealed_path_string is a path identifier only; "
            "contents were not listed or read"
        ),
    }
    payload["integrity_digest"] = compute_integrity_digest(
        {k: v for k, v in payload.items() if k != "integrity_digest"}
    )
    return payload


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


def write_preseal_snapshot(*, root: Path | None = None) -> dict[str, Any]:
    root = root or get_project_root()
    payload = build_preseal_snapshot(root=root)
    out = root / SNAPSHOT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return {"path": str(out), "integrity_digest": payload["integrity_digest"]}
