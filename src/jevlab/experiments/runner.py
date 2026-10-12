"""End-to-end lineup runner: folds → OOS → selection → backtests → ledger.

MASTER_SPEC Phase 8. All Jev-touching outputs are labeled
``potentially_training_contaminated``. Purpose is confirmatory only for A vs B;
everything else is exploratory. Model F is interface-only (Phase 10).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from jevlab.backtest.costs import CostModel
from jevlab.backtest.ledger import run_ledger
from jevlab.config.models import RiskConfig
from jevlab.cv.sealed_guard import SealedDataGuard
from jevlab.cv.walkforward import OOS_END, OOS_START, Fold, make_folds
from jevlab.evaluation.metrics import compute_economic_metrics
from jevlab.experiments.manifest import (
    assert_manifest_complete,
    insert_trial_ledger,
    write_manifest,
)
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import (
    CONTAMINATION_LABEL,
    ColumnSetSpec,
    assert_ab_column_diff,
    column_set_registry,
)
from jevlab.models.controls import (
    build_bucket_onehot_frame,
    ensure_placebo_columns,
    model_g_columns,
)
from jevlab.models.pipelines import (
    WalkForwardResult,
    assert_identical_folds,
    prepare_model_frame,
    run_lineup_walkforward,
)
from jevlab.signals.selection import rebalance_dates, select_top_k
from jevlab.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LineupRunResult:
    experiment_id: str
    results: dict[str, WalkForwardResult]
    folds: list[Fold]
    registry: dict[str, ColumnSetSpec]
    comparison: pd.DataFrame
    manifest_path: Path | None = None
    trial_ids: list[str] = field(default_factory=list)
    backtest_rows: list[dict[str, Any]] = field(default_factory=list)
    g_onehot_meta: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def build_comparison_table(
    results: dict[str, WalkForwardResult],
    *,
    registry: dict[str, ColumnSetSpec],
    backtest_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Numeric comparison only — **no conclusion language**."""
    rows: list[dict[str, Any]] = []
    for mid, wf in results.items():
        spec = registry.get(mid)
        row: dict[str, Any] = {
            "model_id": mid,
            "n_columns": spec.n_columns if spec else None,
            "learner": spec.learner if spec else None,
            "purpose": spec.purpose if spec else None,
            "log_loss": wf.metrics.get("log_loss"),
            "auc": wf.metrics.get("auc"),
            "rank_ic": wf.metrics.get("rank_ic"),
            "n_oos_rows": wf.metrics.get("n_rows"),
            "n_folds_reported": len(wf.fold_report),
            "contamination_label": CONTAMINATION_LABEL,
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if backtest_rows:
        bt = pd.DataFrame(backtest_rows)
        if not bt.empty and "model_id" in bt.columns:
            # Aggregate normal / phase-0 CAGR + Sharpe when present
            normal = bt.loc[
                (bt.get("scenario", pd.Series(dtype=str)) == "normal")
                & (bt.get("phase", pd.Series(dtype=int)) == 0)
            ]
            if not normal.empty:
                agg = normal.groupby("model_id", as_index=False).agg(
                    cagr_dec_normal_p0=("cagr_dec", "mean"),
                    sharpe_normal_p0=("sharpe", "mean"),
                    trade_count_normal_p0=("trade_count", "sum"),
                )
                table = table.merge(agg, on="model_id", how="left")
    return table


def comparison_table_markdown(table: pd.DataFrame) -> str:
    """Markdown table with an explicit no-conclusion banner."""
    lines = [
        "# Lineup comparison (Phase 8)",
        "",
        f"All Jev-touching results are **{CONTAMINATION_LABEL}**.",
        "",
        "_This table reports metrics only. It contains no conclusion language._",
        "",
    ]
    if table.empty:
        lines.append("(empty)")
        return "\n".join(lines)
    cols = [c for c in table.columns if c != "contamination_label"] + ["contamination_label"]
    show = table.loc[:, [c for c in cols if c in table.columns]]
    headers = list(show.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, r in show.iterrows():
        cells: list[str] = []
        for h in headers:
            v = r[h]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if v == v else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _join_jev_features(
    features: pd.DataFrame,
    jev_features: pd.DataFrame | None,
) -> pd.DataFrame:
    if jev_features is None or jev_features.empty:
        return features
    jev = jev_features.copy()
    jev["date"] = pd.to_datetime(jev["date"]).dt.date
    keep = ["symbol", "date", *[c for c in JEV_FEATURE_COLUMNS if c in jev.columns]]
    jev = jev.loc[:, keep]
    feat = features.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.date
    return feat.merge(jev, on=["symbol", "date"], how="left")


def run_model_set(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    out_dir: Path,
    registry_path: Path,
    C_grid: list[float] | None = None,
    lgbm_params: dict[str, Any] | None = None,
    seed: int = 42,
    embargo_sessions: int = 5,
    sealed_guard: SealedDataGuard | None = None,
    jev_features: pd.DataFrame | None = None,
    bars: pd.DataFrame | None = None,
    cost_models: Sequence[CostModel] | None = None,
    risk_config: RiskConfig | None = None,
    initial_capital_usd: float = 100_000.0,
    top_k: int = 3,
    phases: Sequence[int] = (0,),
    model_ids: Sequence[str] | None = None,
    qset_version: str = "qset_v1",
    state_schema_version: str = "state_v1",
    build_g_onehot: bool = True,
    run_backtests: bool = True,
    oos_start: date | None = None,
    oos_end: date | None = None,
    first_train_end: date | None = None,
) -> LineupRunResult:
    """Folds → OOS predictions → selection → optional backtests → manifest + ledger."""
    assert_ab_column_diff()
    C_grid = list(C_grid) if C_grid is not None else [0.01, 0.1, 1.0]
    experiment_id = str(uuid.uuid4())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = _join_jev_features(features, jev_features)
    feat = ensure_placebo_columns(feat)
    g_meta: dict[str, Any] = {}
    g_cols = model_g_columns()
    if build_g_onehot:
        feat, g_meta = build_bucket_onehot_frame(feat)
        g_cols = MODEL_A_COLUMNS + tuple(g_meta["onehot_columns"])
    else:
        # Expect caller to have already attached goh_* columns when G is requested.
        present = [c for c in feat.columns if str(c).startswith("goh_")]
        if present:
            g_cols = MODEL_A_COLUMNS + tuple(sorted(present))
            g_meta = {"n_onehot_columns": len(present), "onehot_columns": sorted(present)}

    registry = column_set_registry(model_g_columns=g_cols)
    assert registry["A"].n_columns == 17
    assert registry["B"].n_columns == 25
    assert registry["C"].n_columns == 8
    assert registry["D"].n_columns == 17
    assert registry["E"].n_columns == 25
    assert registry["F"].n_columns == 25
    assert registry["G"].n_columns == len(g_cols)
    if g_meta:
        assert registry["G"].n_columns == 17 + int(g_meta["n_onehot_columns"])

    extra = list(JEV_FEATURE_COLUMNS) + list(g_meta.get("onehot_columns", []))
    frame = prepare_model_frame(feat, labels, extra_columns=extra)
    if sealed_guard is not None:
        sealed_guard.assert_dev_dates(frame["date"])
    else:
        # Hard refuse sealed dates even without an injected guard.
        from jevlab.models.pipelines import _assert_no_sealed

        _assert_no_sealed(frame)

    sessions = sorted(frame["date"].unique().tolist())
    fold_kwargs: dict[str, Any] = {"embargo_sessions": embargo_sessions}
    if oos_start is not None:
        fold_kwargs["oos_start"] = oos_start
    if oos_end is not None:
        fold_kwargs["oos_end"] = oos_end
    if first_train_end is not None:
        fold_kwargs["first_train_end"] = first_train_end
    folds = make_folds(sessions, **fold_kwargs)
    if not folds:
        raise RuntimeError("no walk-forward folds produced")

    # Shared folds for A and B (identical indices by construction).
    assert_identical_folds(folds, folds)

    wanted = list(model_ids) if model_ids is not None else ["A", "B", "C", "D", "E", "G"]
    results = run_lineup_walkforward(
        frame,
        folds,
        registry,  # type: ignore[arg-type]
        sessions=sessions,
        C_grid=C_grid,
        lgbm_params=lgbm_params,
        seed=seed,
        embargo_sessions=embargo_sessions,
        model_ids=wanted,
    )

    # Persist OOS predictions + signals
    reb = rebalance_dates(sessions, every=3, phase=0)
    for mid, wf in results.items():
        pred_path = out_dir / f"oos_{mid}.parquet"
        wf.predictions.to_parquet(pred_path, index=False)
        if not wf.predictions.empty:
            selected = select_top_k(wf.predictions, k=top_k, rebalance_dates=reb)
            selected.to_parquet(out_dir / f"signals_{mid}.parquet", index=False)

    backtest_rows: list[dict[str, Any]] = []
    if run_backtests and bars is not None and cost_models and risk_config is not None:
        for mid, wf in results.items():
            if wf.predictions.empty:
                continue
            scores = wf.predictions.copy()
            if "score" not in scores.columns and "prob" in scores.columns:
                scores["score"] = scores["prob"]
            for phase in phases:
                for cost in cost_models:
                    ledger = run_ledger(
                        bars,
                        scores,
                        cost=cost,
                        risk_config=risk_config,
                        initial_capital_usd=initial_capital_usd,
                        top_k=top_k,
                        phase=int(phase),
                        apply_risk=True,
                        enable_drawdown_kill=False,
                    )
                    metrics = compute_economic_metrics(
                        ledger.equity_curve,
                        ledger.trades,
                        initial_capital_usd=initial_capital_usd,
                        metric_start=OOS_START,
                        metric_end=OOS_END,
                    ).to_dict()
                    backtest_rows.append(
                        {
                            "model_id": mid,
                            "phase": int(phase),
                            "scenario": cost.scenario_name,
                            "cagr_dec": metrics.get("cagr_dec"),
                            "sharpe": metrics.get("sharpe"),
                            "max_drawdown_dec": metrics.get("max_drawdown_dec"),
                            "trade_count": int(metrics.get("trade_count", 0)),
                            "contamination_label": CONTAMINATION_LABEL,
                        }
                    )

    comparison = build_comparison_table(results, registry=registry, backtest_rows=backtest_rows)
    comparison.to_csv(out_dir / "comparison_table.csv", index=False)
    (out_dir / "comparison_table.md").write_text(
        comparison_table_markdown(comparison), encoding="utf-8"
    )

    purpose_by_model = {mid: registry[mid].purpose for mid in results}  # type: ignore[index]
    column_counts = {mid: registry[mid].n_columns for mid in registry}  # type: ignore[index]
    metrics_by_model = {mid: wf.metrics for mid, wf in results.items()}

    manifest_payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "models": sorted(results.keys()),
        "folds": [
            {
                "fold_id": f.fold_id,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "test_start": str(f.test_start),
                "test_end": str(f.test_end),
            }
            for f in folds
        ],
        "C_grid": C_grid,
        "seed": seed,
        "lgbm_params": lgbm_params,
        "contamination_label": CONTAMINATION_LABEL,
        "potentially_training_contaminated": True,
        "column_counts": column_counts,
        "purpose_by_model": purpose_by_model,
        "metrics_by_model": metrics_by_model,
        "g_onehot": g_meta,
        "ab_assert": {
            "A_n": 17,
            "B_n": 25,
            "B_minus_A": list(JEV_FEATURE_COLUMNS),
            "identical_folds": True,
            "identical_C_grid": True,
            "identical_seed": True,
        },
        "interface_only_models": ["F"],
        "backtest_rows": backtest_rows,
    }
    assert_manifest_complete(manifest_payload)
    manifest_path = write_manifest(manifest_payload, out_dir)

    trial_ids: list[str] = []
    # Confirmatory: A vs B as a paired trial; also log each exploratory model.
    if "A" in results and "B" in results:
        summary = {
            "A": results["A"].metrics,
            "B": results["B"].metrics,
            "delta_log_loss_B_minus_A": (
                float(results["B"].metrics.get("log_loss", float("nan")))
                - float(results["A"].metrics.get("log_loss", float("nan")))
            ),
            "delta_rank_ic_B_minus_A": (
                float(results["B"].metrics.get("rank_ic", float("nan")))
                - float(results["A"].metrics.get("rank_ic", float("nan")))
            ),
            "contamination_label": CONTAMINATION_LABEL,
        }
        trial_ids.append(
            insert_trial_ledger(
                registry_path,
                purpose="confirmatory",
                feature_set="A_vs_B",
                model_class="logistic",
                grid={"C_grid": C_grid, "seed": seed},
                experiment_id=experiment_id,
                result_summary=json.dumps(summary, default=str),
                qset_version=qset_version,
                state_schema_version=state_schema_version,
            )
        )
    for mid, wf in results.items():
        if mid in {"A", "B"}:
            continue
        spec = registry[mid]  # type: ignore[index]
        trial_ids.append(
            insert_trial_ledger(
                registry_path,
                purpose="exploratory",
                feature_set=mid,
                model_class=spec.learner,
                grid={"C_grid": C_grid, "seed": seed, "n_columns": spec.n_columns},
                experiment_id=experiment_id,
                result_summary=json.dumps(
                    {"metrics": wf.metrics, "contamination_label": CONTAMINATION_LABEL},
                    default=str,
                ),
                qset_version=qset_version,
                state_schema_version=state_schema_version,
            )
        )

    log.info(
        "lineup.finish",
        experiment_id=experiment_id,
        models=list(results.keys()),
        contamination=CONTAMINATION_LABEL,
    )
    return LineupRunResult(
        experiment_id=experiment_id,
        results=results,
        folds=folds,
        registry=registry,  # type: ignore[arg-type]
        comparison=comparison,
        manifest_path=manifest_path,
        trial_ids=trial_ids,
        backtest_rows=backtest_rows,
        g_onehot_meta=g_meta,
        meta={"contamination_label": CONTAMINATION_LABEL},
    )
