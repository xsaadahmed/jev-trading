"""One-shot integrity checks for Phase 10 completion checkpoint."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from jevlab.config.loader import load_all_configs
from jevlab.experiments.forward_registry import read_deployment, read_forward_start_date
from jevlab.jev.qset import load_qset
from jevlab.utils.hashing import sha256_of, canonical_json
from jevlab.utils.paths import ProjectPaths, get_project_root


def main() -> None:
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    q = load_qset(paths.root / configs.jev.qset_path, expected_hash=configs.jev.qset_hash)
    print("qset_hash", q.content_hash)
    print("model", configs.jev.model_id)

    p9 = json.loads((paths.reports / "phase9_stats" / "stats_summary.json").read_text(encoding="utf-8"))
    print("phase9_content_digest", p9.get("content_digest"))
    print("phase9_point", p9.get("point_estimates"))

    db = paths.registry / "experiments.sqlite"
    con = sqlite3.connect(db)
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
    print("tables", tables)
    if "sealed_access_log" in tables:
        print("sealed_access", con.execute("select count(*) from sealed_access_log").fetchone()[0])
    if "trial_ledger" in tables:
        print("trial_ledger", con.execute("select count(*) from trial_ledger").fetchone()[0])
        rows = con.execute(
            "select trial_id, purpose, feature_set, experiment_id from trial_ledger "
            "where experiment_id like 'phase10%' order by created_at"
        ).fetchall()
        for r in rows:
            print(" trial", r)

    try:
        dep = read_deployment(db)
        print("deployment", dep)
        print("FORWARD_START_DATE", read_forward_start_date(db))
    except Exception as exc:  # noqa: BLE001
        print("forward_registry_error", exc)

    print("HISTORICAL_ACCEPTANCE", (paths.reports / "HISTORICAL_ACCEPTANCE.json").exists())
    jev = paths.data_features / "jev" / configs.jev.qset_version / "jev_features.parquet"
    print("jev_features_sha256", sha256_of(jev.read_bytes()) if jev.is_file() else None)

    # paraphrase hashes distinct from frozen
    for p in sorted((paths.root / "config" / "paraphrase").glob("*.json")):
        h = sha256_of(canonical_json(json.loads(p.read_text(encoding="utf-8"))))
        print("paraphrase_file", p.name, h)

    placebo = json.loads((paths.reports / "robustness_report" / "placebo_results.json").read_text())
    print("P6", placebo["P6"].get("status"), "warn", placebo["P6"].get("pipeline_integrity_warning"))
    print("placebo_digest", (paths.reports / "robustness_report" / "placebo_results.digest").read_text().strip())
    print("robustness_digest", (paths.reports / "robustness_report" / "robustness_data.digest").read_text().strip())


if __name__ == "__main__":
    main()
