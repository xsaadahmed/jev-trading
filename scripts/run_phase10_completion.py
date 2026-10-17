"""Fill Phase 10 gaps: horizon ±1, delayed open_t2, live P6, live paraphrase.

Does not rerun completed placebos/robustness. Never opens sealed dates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevlab.backtest.costs import CostModel
from jevlab.config.loader import load_all_configs
from jevlab.cv.sealed_guard import SealedDataGuard
from jevlab.cv.walkforward import make_folds
from jevlab.data.downloader import default_store
from jevlab.evaluation.phase10_completion import (
    delayed_execution_full,
    execute_p6_live,
    execute_paraphrase_live,
    horizon_sweep,
)
from jevlab.evaluation.placebo import CONTAMINATION_LABEL, train_b_oos, write_placebo_results
from jevlab.experiments.manifest import insert_trial_ledger
from jevlab.experiments.runner import _join_jev_features
from jevlab.features.columns import FEATURE_VERSION
from jevlab.features.pipeline import read_feature_store
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import column_set_registry
from jevlab.models.controls import ensure_placebo_columns
from jevlab.models.pipelines import prepare_model_frame, run_column_set_logistic
from jevlab.targets.labels import build_labels
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import configure_logging, get_logger
from jevlab.utils.paths import ProjectPaths, get_project_root

log = get_logger(__name__)


def main() -> int:
    load_dotenv(ROOT / ".env")
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    out_dir = paths.reports / "robustness_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    guard = SealedDataGuard(
        sealed_root=paths.data_sealed,
        registry_path=paths.registry / "experiments.sqlite",
    )
    store = default_store(paths, configs.data)
    features = guard.filter_dev_frame(read_feature_store(paths.data_features / FEATURE_VERSION))
    bars = guard.filter_dev_frame(store.read_bars(vendor=configs.data.primary_vendor))
    scenario = next(c for c in configs.backtest.cost_scenarios if c.name == "normal")
    elig = features.loc[:, ["symbol", "date", "eligible"]]
    labels = guard.filter_dev_frame(
        build_labels(
            bars,
            eligible=elig,
            universe_symbols=list(configs.universe.symbols),
            cost_bps_per_side=scenario.cost_bps_per_side,
            sell_fee_bps=configs.backtest.sell_fee_bps,
            per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
        )
    )
    jev_path = paths.data_features / "jev" / configs.jev.qset_version / "jev_features.parquet"
    jev = pd.read_parquet(jev_path) if jev_path.is_file() else None
    if jev is not None:
        jev = guard.filter_dev_frame(jev)

    log.info(
        "phase10_completion.loaded",
        n_features=len(features),
        n_bars=len(bars),
        n_labels=len(labels),
        jev_rows=0 if jev is None else len(jev),
    )

    # --- Offline: horizon ±1 ---
    log.info("phase10_completion.horizon_start")
    horizon_rows = horizon_sweep(bars, features, configs=configs, jev_features=jev, seed=42)
    (out_dir / "horizon_sweep.json").write_text(
        json.dumps(horizon_rows, indent=2, default=str), encoding="utf-8"
    )
    log.info("phase10_completion.horizon_done", n=len(horizon_rows), rows=horizon_rows)

    # --- Offline: delayed execution including open_t2 ---
    feat = ensure_placebo_columns(_join_jev_features(features, jev))
    frame = prepare_model_frame(feat, labels, extra_columns=list(JEV_FEATURE_COLUMNS))
    sessions = sorted(frame["date"].unique().tolist())
    folds = make_folds(sessions, embargo_sessions=5)
    C = list(configs.strategy.logistic_C_grid)
    reg = column_set_registry()
    preds_a = run_column_set_logistic(
        frame, folds, reg["A"], sessions=sessions, C_grid=C, seed=42, embargo_sessions=5
    ).predictions
    preds_b = train_b_oos(frame, folds, sessions=sessions, C_grid=C, seed=42, embargo_sessions=5)
    cost = CostModel(
        cost_bps_per_side=scenario.cost_bps_per_side,
        sell_fee_bps=configs.backtest.sell_fee_bps,
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
        scenario_name="normal",
    )
    delayed_rows = delayed_execution_full(
        bars, preds_a, preds_b, cost=cost, risk_config=configs.risk, top_k=configs.strategy.top_k
    )
    (out_dir / "delayed_execution.json").write_text(
        json.dumps(delayed_rows, indent=2, default=str), encoding="utf-8"
    )
    log.info("phase10_completion.delayed_done", rows=delayed_rows)

    # --- Live P6 ---
    log.info("phase10_completion.p6_start")
    p6 = execute_p6_live(
        features,
        labels,
        configs=configs,
        paths=paths,
        n_states=3000,
        seed=42,
        budget_usd=25.0,
        max_workers=4,
        out_dir=out_dir / "p6_live",
    )
    placebo_path = out_dir / "placebo_results.json"
    placebo = json.loads(placebo_path.read_text(encoding="utf-8"))
    placebo["P6"] = p6
    write_placebo_results(placebo, out_dir)
    insert_trial_ledger(
        paths.registry / "experiments.sqlite",
        purpose="placebo",
        feature_set="placebo_P6",
        model_class="state_shuffle_live",
        grid={"n_states": 3000, "seed": 42, "live": True},
        experiment_id="phase10-P6-live",
        result_summary=json.dumps(
            {
                "status": p6.get("status"),
                "n_ok": p6.get("n_ok"),
                "pipeline_integrity_warning": p6.get("pipeline_integrity_warning"),
                "spent_usd": p6.get("spent_usd"),
            },
            default=str,
        ),
        qset_version=configs.jev.qset_version,
        state_schema_version=configs.jev.state_schema_version,
    )
    log.info("phase10_completion.p6_done", status=p6.get("status"), n_ok=p6.get("n_ok"))

    # --- Live paraphrase ---
    log.info("phase10_completion.paraphrase_start")
    para = execute_paraphrase_live(
        features,
        labels,
        configs=configs,
        paths=paths,
        n_states=3000,
        seed=42,
        budget_usd=25.0,
        max_workers=4,
        out_dir=out_dir / "paraphrase_live",
    )
    (out_dir / "paraphrase.json").write_text(
        json.dumps([para], indent=2, default=str), encoding="utf-8"
    )
    rob_path = out_dir / "robustness_data.json"
    rob = json.loads(rob_path.read_text(encoding="utf-8"))
    rob["delayed_execution"] = delayed_rows
    rob["paraphrase"] = [para]
    rob["horizon_sweep"] = horizon_rows
    rob["meta"] = {**(rob.get("meta") or {}), "contamination_label": CONTAMINATION_LABEL}
    clean = json.loads(json.dumps(rob, default=str))
    rob_path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    digest = sha256_of(canonical_json(clean))
    (out_dir / "robustness_data.digest").write_text(digest + "\n", encoding="utf-8")

    insert_trial_ledger(
        paths.registry / "experiments.sqlite",
        purpose="exploratory",
        feature_set="paraphrase_sensitivity",
        model_class="jev_paraphrase_live",
        grid={"n_states": 3000, "n_paraphrases": 2, "seed": 42},
        experiment_id="phase10-paraphrase-live",
        result_summary=json.dumps(
            {
                "status": para.get("status"),
                "variants": [
                    {
                        "qset_version": v.get("qset_version"),
                        "qset_hash": v.get("qset_hash"),
                        "n_ok": v.get("n_ok"),
                        "spent_usd": v.get("spent_usd"),
                    }
                    for v in para.get("variants", [])
                ],
                "materially_wording_dependent": para.get("materially_wording_dependent"),
            },
            default=str,
        ),
        qset_version=configs.jev.qset_version,
        state_schema_version=configs.jev.state_schema_version,
    )
    log.info("phase10_completion.paraphrase_done", status=para.get("status"))

    summary = {
        "horizon": horizon_rows,
        "delayed_execution": delayed_rows,
        "P6": {
            "status": p6.get("status"),
            "n_ok": p6.get("n_ok"),
            "pipeline_integrity_warning": p6.get("pipeline_integrity_warning"),
            "spent_usd": p6.get("spent_usd"),
            "resolved_models": p6.get("resolved_models"),
        },
        "paraphrase": {
            "status": para.get("status"),
            "materially_wording_dependent": para.get("materially_wording_dependent"),
            "variants": [
                {
                    "qset_version": v.get("qset_version"),
                    "n_ok": v.get("n_ok"),
                    "hash": v.get("qset_hash"),
                }
                for v in para.get("variants", [])
            ],
        },
        "robustness_data_digest": digest,
        "contamination_label": CONTAMINATION_LABEL,
    }
    (out_dir / "phase10_completion_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    ok = p6.get("status") == "complete" and para.get("status") == "complete"
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
