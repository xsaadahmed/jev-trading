"""CLI: ``jev run baselines|sanity|lineup|forward-capture`` — experiment runners."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import pandas as pd
import typer

from jevlab.backtest.costs import CostModel
from jevlab.config.loader import load_all_configs
from jevlab.cv.sealed_guard import SealedDataGuard
from jevlab.data.downloader import default_store
from jevlab.evaluation.bootstrap import DEFAULT_N_DRAWS
from jevlab.evaluation.stats_summary import (
    build_ab_stats_summary,
    log_confirmatory_trial,
    write_stats_summary,
)
from jevlab.experiments.runner import comparison_table_markdown, run_model_set
from jevlab.features.columns import FEATURE_VERSION
from jevlab.features.pipeline import read_feature_store
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import CONTAMINATION_LABEL
from jevlab.models.pipelines import run_baselines
from jevlab.models.r0 import score_r0
from jevlab.reporting.baseline_sanity import (
    build_sanity_report,
    scores_from_predictions,
    write_sanity_report,
)
from jevlab.signals.selection import rebalance_dates, select_top_k
from jevlab.targets.labels import build_labels
from jevlab.utils.logging import configure_logging
from jevlab.utils.paths import ProjectPaths, get_project_root

app = typer.Typer(
    help=(
        "Experiment runners (baselines, sanity, lineup, stats, robustness, "
        "forward-capture, build-production-models, forward-score/report, evaluate-scale)."
    ),
)


def _guard(paths: ProjectPaths) -> SealedDataGuard:
    return SealedDataGuard(
        sealed_root=paths.data_sealed,
        registry_path=paths.registry / "experiments.sqlite",
    )


@app.command("baselines")
def baselines_cmd(
    cost_scenario: str = typer.Option("normal", help="Cost scenario name from backtest.yaml"),
) -> None:
    """Fit R0/A/D walk-forward on features+labels; write OOS predictions and metrics."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    guard = _guard(paths)

    features = read_feature_store(paths.data_features / FEATURE_VERSION)
    features = guard.filter_dev_frame(features)
    bars = store.read_bars(vendor=configs.data.primary_vendor)
    bars = guard.filter_dev_frame(bars)
    if features.empty or bars.empty:
        typer.echo("Need features and bars; run data download + features build first.", err=True)
        raise typer.Exit(code=1)

    scenario = next(
        (c for c in configs.backtest.cost_scenarios if c.name == cost_scenario),
        None,
    )
    if scenario is None:
        raise typer.BadParameter(f"unknown cost scenario: {cost_scenario}")

    elig = features.loc[:, ["symbol", "date", "eligible"]]
    labels = build_labels(
        bars,
        eligible=elig,
        universe_symbols=list(configs.universe.symbols),
        cost_bps_per_side=scenario.cost_bps_per_side,
        sell_fee_bps=configs.backtest.sell_fee_bps,
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
    )
    labels = guard.filter_dev_frame(labels)

    results = run_baselines(
        features,
        labels,
        C_grid=list(configs.strategy.logistic_C_grid),
        lgbm_params=dict(configs.strategy.lightgbm_params),
        seed=configs.strategy.seed,
        sealed_guard=guard,
    )

    out_dir = paths.reports / "phase3_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}
    sessions = sorted(features["date"].unique().tolist())
    reb = rebalance_dates(
        sessions,
        every=configs.strategy.rebalance_every_sessions,
        phase=0,
    )
    for model_id, result in results.items():
        pred_path = out_dir / f"oos_{model_id}.parquet"
        result.predictions.to_parquet(pred_path, index=False)
        selected = select_top_k(
            result.predictions,
            k=configs.strategy.top_k,
            rebalance_dates=reb,
        )
        sel_path = out_dir / f"signals_{model_id}.parquet"
        selected.to_parquet(sel_path, index=False)
        summary[model_id] = {
            "metrics": result.metrics,
            "n_predictions": len(result.predictions),
            "predictions_path": str(pred_path),
            "signals_path": str(sel_path),
            "fold_report": result.fold_report,
            "chosen_C": result.chosen_C,
        }

    summary_path = out_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    typer.echo(json.dumps({k: v["metrics"] for k, v in summary.items()}, indent=2))
    typer.echo(f"wrote {summary_path}")


@app.command("sanity")
def sanity_cmd(
    use_saved_predictions: bool = typer.Option(
        True,
        help="Prefer reports/phase3_baselines OOS parquet when present",
    ),
) -> None:
    """Run R0/A/D under 4 cost scenarios; write baseline_sanity_report (Phase 4)."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    guard = _guard(paths)

    features = read_feature_store(paths.data_features / FEATURE_VERSION)
    features = guard.filter_dev_frame(features)
    bars = store.read_bars(vendor=configs.data.primary_vendor)
    bars = guard.filter_dev_frame(bars)
    if features.empty or bars.empty:
        typer.echo("Need features and bars; run data download + features build first.", err=True)
        raise typer.Exit(code=1)

    scenario_normal = next(c for c in configs.backtest.cost_scenarios if c.name == "normal")
    elig = features.loc[:, ["symbol", "date", "eligible"]]
    labels = build_labels(
        bars,
        eligible=elig,
        universe_symbols=list(configs.universe.symbols),
        cost_bps_per_side=scenario_normal.cost_bps_per_side,
        sell_fee_bps=configs.backtest.sell_fee_bps,
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
    )
    labels = guard.filter_dev_frame(labels)

    pred_dir = paths.reports / "phase3_baselines"
    scores_by_model: dict[str, pd.DataFrame] = {}
    if use_saved_predictions and pred_dir.is_dir():
        for model_id in ("R0", "A", "D"):
            path = pred_dir / f"oos_{model_id}.parquet"
            if path.is_file():
                scores_by_model[model_id] = scores_from_predictions(pd.read_parquet(path))

    if len(scores_by_model) < 3:
        # Fit baselines in-process when OOS artifacts are missing.
        results = run_baselines(
            features,
            labels,
            C_grid=list(configs.strategy.logistic_C_grid),
            lgbm_params=dict(configs.strategy.lightgbm_params),
            seed=configs.strategy.seed,
            sealed_guard=guard,
        )
        for model_id, result in results.items():
            scores_by_model[model_id] = scores_from_predictions(result.predictions)
        # R0 can also be scored directly on features for full calendar coverage.
        if "R0" not in scores_by_model:
            scores_by_model["R0"] = scores_from_predictions(score_r0(features))

    cost_models = [
        CostModel(
            cost_bps_per_side=sc.cost_bps_per_side,
            sell_fee_bps=configs.backtest.sell_fee_bps,
            per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
            scenario_name=sc.name,
        )
        for sc in configs.backtest.cost_scenarios
    ]

    report = build_sanity_report(
        bars=bars,
        scores_by_model=scores_by_model,
        labels=labels,
        cost_models=cost_models,
        risk_config=configs.risk,
        initial_capital_usd=configs.backtest.initial_capital_usd,
        top_k=configs.strategy.top_k,
        phases=list(configs.strategy.phases),
    )
    out_dir = paths.reports / "phase4_sanity"
    md_path = write_sanity_report(report, out_dir)
    typer.echo(report.to_markdown())
    typer.echo(f"wrote {md_path}")
    if not report.mechanics_ok:
        typer.echo(
            "G3 mechanics checks failed — human halt decision required "
            "(profitability alone is not a halt).",
            err=True,
        )
        raise typer.Exit(code=2)


@app.command("forward-capture")
def forward_capture_cmd(
    session: str | None = typer.Option(
        None,
        help="NYSE session YYYY-MM-DD (default: today ET if session day after 18:30)",
    ),
    no_download: bool = typer.Option(
        False,
        help="Skip market-data download; use existing store only",
    ),
    now: str | None = typer.Option(
        None,
        help="Override wall clock ISO datetime (tests/ops); timezone-aware preferred",
    ),
) -> None:
    """Capture forward INPUTS for one completed session (Phase 13A; no orders)."""
    # Import here so ``jev run baselines|sanity`` does not load forward-capture deps
    # through this module's top level in a way that couples brokers (it never does).
    from jevlab.experiments.forward_capture import run_forward_capture

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    paths.ensure_runtime_dirs()

    wall: datetime | None = None
    if now is not None:
        wall = datetime.fromisoformat(now)
        if wall.tzinfo is None:
            wall = wall.replace(tzinfo=ZoneInfo("America/New_York"))

    result = run_forward_capture(
        paths=paths,
        now=wall,
        session=date.fromisoformat(session) if session else None,
        download=not no_download,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
    if result.status in {"too_early", "skipped_holiday", "skipped_idempotent"}:
        raise typer.Exit(code=0)
    if result.status != "complete":
        raise typer.Exit(code=1)


@app.command("lineup")
def lineup_cmd(
    models: str = typer.Option(
        "R0,A,B,C,D,E,G",
        help="Comma-separated model ids (F is interface-only and skipped).",
    ),
    skip_backtests: bool = typer.Option(
        False,
        help="Skip cost-scenario ledger backtests (predictive metrics only).",
    ),
) -> None:
    """Run A-G attribution lineup on identical folds (Phase 8)."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    guard = _guard(paths)

    features = read_feature_store(paths.data_features / FEATURE_VERSION)
    features = guard.filter_dev_frame(features)
    bars = store.read_bars(vendor=configs.data.primary_vendor)
    bars = guard.filter_dev_frame(bars)
    if features.empty or bars.empty:
        typer.echo("Need features and bars; run data download + features build first.", err=True)
        raise typer.Exit(code=1)

    scenario_normal = next(c for c in configs.backtest.cost_scenarios if c.name == "normal")
    elig = features.loc[:, ["symbol", "date", "eligible"]]
    labels = build_labels(
        bars,
        eligible=elig,
        universe_symbols=list(configs.universe.symbols),
        cost_bps_per_side=scenario_normal.cost_bps_per_side,
        sell_fee_bps=configs.backtest.sell_fee_bps,
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
    )
    labels = guard.filter_dev_frame(labels)

    jev_path = paths.data_features / "jev" / configs.jev.qset_version / "jev_features.parquet"
    jev_features = pd.read_parquet(jev_path) if jev_path.is_file() else None

    cost_models = [
        CostModel(
            cost_bps_per_side=sc.cost_bps_per_side,
            sell_fee_bps=configs.backtest.sell_fee_bps,
            per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
            scenario_name=sc.name,
        )
        for sc in configs.backtest.cost_scenarios
    ]

    model_ids = [m.strip() for m in models.split(",") if m.strip() and m.strip() != "F"]
    # Frozen strategy phases 0/1/2 (acceptance uses the mean of the three).
    phases = [int(p) for p in configs.strategy.phases]
    out_dir = paths.reports / "phase8_lineup"
    result = run_model_set(
        features,
        labels,
        out_dir=out_dir,
        registry_path=paths.registry / "experiments.sqlite",
        C_grid=list(configs.strategy.logistic_C_grid),
        lgbm_params=dict(configs.strategy.lightgbm_params),
        seed=configs.strategy.seed,
        sealed_guard=guard,
        jev_features=jev_features,
        bars=bars,
        cost_models=cost_models,
        risk_config=configs.risk,
        initial_capital_usd=configs.backtest.initial_capital_usd,
        top_k=configs.strategy.top_k,
        phases=phases,
        model_ids=model_ids,
        qset_version=configs.jev.qset_version,
        state_schema_version=configs.jev.state_schema_version,
        run_backtests=not skip_backtests,
    )
    typer.echo(comparison_table_markdown(result.comparison))
    typer.echo(f"contamination_label={CONTAMINATION_LABEL}")
    typer.echo(f"experiment_id={result.experiment_id}")
    typer.echo(f"manifest={result.manifest_path}")
    typer.echo(f"wrote {out_dir}")


@app.command("robustness")
def robustness_cmd(
    seed: int = typer.Option(42, help="RNG seed for placebos / robustness."),
    n_p1: int = typer.Option(1000, help="P1 draws (spec default 1000)."),
    n_p2: int = typer.Option(500, help="P2 draws (spec default 500)."),
    n_p3: int = typer.Option(500, help="P3 draws (spec default 500)."),
    n_p4: int = typer.Option(500, help="P4 draws (spec default 500)."),
    skip_placebos: bool = typer.Option(False, help="Skip P1-P5 placebo loops."),
    skip_loo: bool = typer.Option(False, help="Skip leave-one-ETF-out."),
    live_p6: bool = typer.Option(
        False, "--live-p6", help="Arm P6 live work queue (budget-capped)."
    ),
    budget_usd: float = typer.Option(25.0, help="USD budget cap for P6 live path."),
) -> None:
    """Phase 10: placebo P1-P6 + robustness suite. Writes data-only robustness_report/."""
    from jevlab.cv.walkforward import make_folds
    from jevlab.evaluation.placebo import (
        run_p5_projection,
        run_placebo_loop,
        train_b_oos,
        write_placebo_results,
    )
    from jevlab.evaluation.robustness import run_robustness_suite, write_robustness_report
    from jevlab.experiments.manifest import insert_trial_ledger
    from jevlab.experiments.runner import _join_jev_features
    from jevlab.jev.audit import run_p6_state_shuffle
    from jevlab.models.column_sets import column_set_registry
    from jevlab.models.controls import ensure_placebo_columns
    from jevlab.models.pipelines import prepare_model_frame, run_column_set_logistic

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    guard = _guard(paths)
    embargo = 5

    features = read_feature_store(paths.data_features / FEATURE_VERSION)
    features = guard.filter_dev_frame(features)
    bars = store.read_bars(vendor=configs.data.primary_vendor)
    bars = guard.filter_dev_frame(bars)
    if features.empty:
        typer.echo("Need features; run data download + features build first.", err=True)
        raise typer.Exit(code=1)

    scenario_normal = next(c for c in configs.backtest.cost_scenarios if c.name == "normal")
    elig = features.loc[:, ["symbol", "date", "eligible"]]
    labels = build_labels(
        bars,
        eligible=elig,
        universe_symbols=list(configs.universe.symbols),
        cost_bps_per_side=scenario_normal.cost_bps_per_side,
        sell_fee_bps=configs.backtest.sell_fee_bps,
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
    )
    labels = guard.filter_dev_frame(labels)

    jev_path = paths.data_features / "jev" / configs.jev.qset_version / "jev_features.parquet"
    jev_features = pd.read_parquet(jev_path) if jev_path.is_file() else None

    feat = ensure_placebo_columns(_join_jev_features(features, jev_features))
    frame = prepare_model_frame(feat, labels, extra_columns=list(JEV_FEATURE_COLUMNS))
    frame = guard.filter_dev_frame(frame)
    sessions = sorted(frame["date"].unique().tolist())
    folds = make_folds(sessions, embargo_sessions=embargo)
    if not folds:
        typer.echo("No walk-forward folds; check date coverage.", err=True)
        raise typer.Exit(code=1)

    C_grid = list(configs.strategy.logistic_C_grid)
    registry = column_set_registry()
    preds_a = run_column_set_logistic(
        frame,
        folds,
        registry["A"],
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo,
    ).predictions
    preds_b = train_b_oos(
        frame,
        folds,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo,
    )

    cost = CostModel(
        cost_bps_per_side=scenario_normal.cost_bps_per_side,
        sell_fee_bps=configs.backtest.sell_fee_bps,
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
        scenario_name="normal",
    )
    out_dir = paths.reports / "robustness_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    placebo_payload: dict[str, Any] = {
        "contamination_label": CONTAMINATION_LABEL,
        "narrative_conclusions": None,
        "counts": {"P1": n_p1, "P2": n_p2, "P3": n_p3, "P4": n_p4, "P5": 1, "P6": 1},
    }
    if not skip_placebos:
        for kind, n in (("P1", n_p1), ("P2", n_p2), ("P3", n_p3), ("P4", n_p4)):
            typer.echo(f"running {kind} n={n} ...")
            result = run_placebo_loop(
                frame,
                folds,
                kind=kind,  # type: ignore[arg-type]
                preds_a=preds_a,
                sessions=sessions,
                C_grid=C_grid,
                n=n,
                seed=seed,
                embargo_sessions=embargo,
                bars=bars if not bars.empty else None,
                cost=cost,
                risk_config=configs.risk,
                top_k=configs.strategy.top_k,
                compute_sharpe=not bars.empty,
            )
            placebo_payload[kind] = result.to_dict()
            insert_trial_ledger(
                paths.registry / "experiments.sqlite",
                purpose="placebo",
                feature_set=f"placebo_{kind}",
                model_class="logistic_B",
                grid={"n": n, "seed": seed, "kind": kind},
                experiment_id=f"phase10-{kind}",
                result_summary=json.dumps(
                    {
                        "pvalue_log_loss": result.pvalue_log_loss,
                        "observed_delta_log_loss_gain": result.observed_delta_log_loss_gain,
                    }
                ),
                qset_version=configs.jev.qset_version,
                state_schema_version=configs.jev.state_schema_version,
            )
        p5 = run_p5_projection(
            frame,
            folds,
            preds_a=preds_a,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo,
        )
        placebo_payload["P5"] = p5.to_dict()
        insert_trial_ledger(
            paths.registry / "experiments.sqlite",
            purpose="placebo",
            feature_set="placebo_P5",
            model_class="logistic_B_projection",
            grid={"seed": seed},
            experiment_id="phase10-P5",
            result_summary=json.dumps(p5.to_dict(), default=str),
            qset_version=configs.jev.qset_version,
            state_schema_version=configs.jev.state_schema_version,
        )

    p6 = run_p6_state_shuffle(
        features,
        live=live_p6,
        budget_usd=budget_usd,
        seed=seed,
        labels=labels,
    )
    placebo_payload["P6"] = p6
    insert_trial_ledger(
        paths.registry / "experiments.sqlite",
        purpose="placebo",
        feature_set="placebo_P6",
        model_class="state_shuffle",
        grid={"live": live_p6, "budget_usd": budget_usd, "seed": seed},
        experiment_id="phase10-P6",
        result_summary=json.dumps(
            {k: p6[k] for k in ("status", "n_shuffled_states", "mismatch_rate") if k in p6},
            default=str,
        ),
        qset_version=configs.jev.qset_version,
        state_schema_version=configs.jev.state_schema_version,
    )
    write_placebo_results(placebo_payload, out_dir)

    typer.echo("running robustness suite ...")
    bundle = run_robustness_suite(
        frame,
        folds,
        preds_a=preds_a,
        preds_b=preds_b,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo,
        bars=bars if not bars.empty else None,
        cost=cost,
        risk_config=configs.risk,
        top_k=configs.strategy.top_k,
        include_loo=not skip_loo,
    )
    path = write_robustness_report(bundle, out_dir)
    # Paraphrase plan logged as exploratory trial (live execution separate).
    insert_trial_ledger(
        paths.registry / "experiments.sqlite",
        purpose="exploratory",
        feature_set="paraphrase_sensitivity",
        model_class="jev_paraphrase",
        grid=bundle.paraphrase[0] if bundle.paraphrase else {},
        experiment_id="phase10-paraphrase",
        result_summary=json.dumps(bundle.paraphrase, default=str),
        qset_version=configs.jev.qset_version,
        state_schema_version=configs.jev.state_schema_version,
    )
    typer.echo(f"contamination_label={CONTAMINATION_LABEL}")
    typer.echo(f"robustness_report={path}")
    typer.echo("narrative_conclusions=null")


@app.command("stats")
def stats_cmd(
    n_draws: int = typer.Option(DEFAULT_N_DRAWS, help="Bootstrap draws (default 10000)."),
    seed: int = typer.Option(42, help="Bootstrap RNG seed."),
) -> None:
    """A vs B confirmatory statistical summary (Phase 9). Writes stats_summary.json."""
    import uuid

    from sqlalchemy import text

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    guard = _guard(paths)

    pred_a = paths.reports / "phase8_lineup" / "oos_A.parquet"
    pred_b = paths.reports / "phase8_lineup" / "oos_B.parquet"
    if not pred_a.is_file() or not pred_b.is_file():
        # Fall back to Phase 3 baseline A if B missing is not acceptable —
        # require lineup artifacts for confirmatory A vs B.
        typer.echo(
            f"Need OOS predictions at {pred_a} and {pred_b}; run `jev run lineup` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    preds_a = guard.filter_dev_frame(pd.read_parquet(pred_a))
    preds_b = guard.filter_dev_frame(pd.read_parquet(pred_b))

    registry_path = paths.registry / "experiments.sqlite"
    from jevlab.utils.registry import init_registry

    engine = init_registry(registry_path)
    with engine.connect() as conn:
        n_trials = int(conn.execute(text("SELECT COUNT(*) FROM trial_ledger")).scalar_one())

    summary = build_ab_stats_summary(
        preds_a,
        preds_b,
        n_trials_ledger=max(n_trials, 1),
        n_draws=n_draws,
        seed=seed,
    )
    out_dir = paths.reports / "phase9_stats"
    path = write_stats_summary(summary, out_dir)
    experiment_id = str(uuid.uuid4())
    trial_id = log_confirmatory_trial(
        summary,
        registry_path=registry_path,
        experiment_id=experiment_id,
        qset_version=configs.jev.qset_version,
        state_schema_version=configs.jev.state_schema_version,
    )
    typer.echo(json.dumps(summary.get("point_estimates", {}), indent=2, default=str))
    typer.echo(f"contamination_label={CONTAMINATION_LABEL}")
    typer.echo(f"stats_summary={path}")
    typer.echo(f"trial_id={trial_id}")


@app.command("final-test")
def final_test_cmd(
    seal_token: Annotated[str, typer.Option("--seal-token", help="Human-provided SEAL token.")],
    allow_second_look: Annotated[
        bool,
        typer.Option(
            "--allow-second-look",
            help="Permit a second sealed look (stamps SEAL BROKEN; blocks acceptance).",
        ),
    ] = False,
    fixture_metrics: Annotated[
        Path | None,
        typer.Option(
            "--fixture-metrics",
            help="JSON metrics for fixture evaluation (no real sealed Jev spend).",
        ),
    ] = None,
    skip_git_checks: Annotated[
        bool,
        typer.Option("--skip-git-checks", help="Skip clean-tree/tag checks (tests / local dry)."),
    ] = False,
) -> None:
    """Phase 12 one-shot sealed final test + HISTORICAL_RESEARCH_PASS evaluation.

    Does not depend on any forward-session count. Prefer ``--fixture-metrics`` for
    dry evaluation; live sealed Jev inference is operator-gated and budgeted.
    """
    import os

    from jevlab.config.loader import load_all_configs
    from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV
    from jevlab.experiments.acceptance import (
        HistoricalGateMetrics,
        current_git_commit,
        run_final_test,
    )
    from jevlab.experiments.artifacts import HISTORICAL_ACCEPTANCE_STATEMENT
    from jevlab.jev.preconditions import PreconditionError

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    paths.ensure_runtime_dirs()

    if fixture_metrics is None:
        typer.echo(
            "Live sealed final-test requires operator-prepared sealed Jev features "
            "and full OOS metrics. Use --fixture-metrics for dry/fixture evaluation.",
            err=True,
        )
        raise typer.Exit(code=2)

    raw = json.loads(Path(fixture_metrics).read_text(encoding="utf-8"))
    metrics = HistoricalGateMetrics.from_dict(raw)
    # Ensure env token matches for SealedDataGuard
    os.environ[SEAL_TOKEN_ENV] = seal_token

    try:
        result = run_final_test(
            root=paths.root,
            acceptance_dir=paths.reports_acceptance,
            registry_path=paths.registry / "experiments.sqlite",
            metrics=metrics,
            seal_token=seal_token,
            git_commit=current_git_commit(cwd=paths.root),
            research_config_hash=configs.research_config_hash,
            qset_hash=str(configs.jev.qset_hash or ""),
            model_resolved=str(configs.jev.model_id),
            disposition_path=paths.docs / "contamination_disposition.md",
            audit_report_path=paths.reports / "audit" / "AUDIT_REPORT.md",
            allow_second_look=allow_second_look,
            require_git_tags=not skip_git_checks,
            require_clean_tree=not skip_git_checks,
            expected_env_token=seal_token,
        )
    except PreconditionError as exc:
        typer.echo(f"precondition failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"all_passed={result.gate_report.all_passed}")
    typer.echo(f"seal_broken={result.seal_broken}")
    typer.echo(f"gate_report={result.gate_report_path}")
    typer.echo(HISTORICAL_ACCEPTANCE_STATEMENT)
    if result.acceptance is not None:
        typer.echo(f"acceptance={result.acceptance.path}")
        typer.echo(f"integrity_digest={result.acceptance.integrity_digest}")
        typer.echo("note=integrity_digest is an integrity digest, not a signature")
    elif result.rejection_path is not None:
        typer.echo(f"rejection={result.rejection_path}")


@app.command("build-production-models")
def build_production_models_cmd(
    train_parquet: Annotated[
        Path | None,
        typer.Option(
            "--train-parquet",
            help="Labeled training frame (symbol,date,y + model columns). Fixture/ops path.",
        ),
    ] = None,
) -> None:
    """Phase 13B: retrain frozen A/B as production_forward (after sealed test)."""
    from jevlab.experiments.acceptance import current_git_commit
    from jevlab.experiments.artifacts import ArtifactError
    from jevlab.experiments.forward_registry import read_deployment
    from jevlab.experiments.production_models import (
        PRODUCTION_RATIONALE,
        build_production_models,
    )

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    paths.ensure_runtime_dirs()
    registry = paths.registry / "experiments.sqlite"
    deployment = read_deployment(registry)
    if deployment is None:
        typer.echo("forward_registry empty; run forward-capture first", err=True)
        raise typer.Exit(code=1)
    if train_parquet is None:
        typer.echo(
            "Provide --train-parquet with pre-deployment labeled rows "
            "(features+labels; date < forward_start).",
            err=True,
        )
        raise typer.Exit(code=2)
    frame = pd.read_parquet(train_parquet)
    try:
        result = build_production_models(
            frame,
            registry_path=registry,
            artifact_dir=paths.data_models / "production_forward",
            forward_root=paths.data_forward,
            deployment=deployment,
            git_commit=current_git_commit(cwd=paths.root),
        )
    except ArtifactError as exc:
        typer.echo(f"build-production-models failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"train_end={result.train_end.isoformat()}")
    typer.echo(f"n_train_rows={result.n_train_rows}")
    typer.echo(f"artifact_ids={json.dumps(result.artifact_ids)}")
    typer.echo(f"hashes={result.hashes_path}")
    typer.echo(PRODUCTION_RATIONALE)


@app.command("forward-score")
def forward_score_cmd(
    session: Annotated[
        str | None,
        typer.Option("--session", help="NYSE session YYYY-MM-DD (default: all complete)."),
    ] = None,
    scores_parquet: Annotated[
        Path | None,
        typer.Option(
            "--scores-parquet",
            help="Optional feature frame to score (fixture path); else load from data/forward/.",
        ),
    ] = None,
) -> None:
    """Phase 13B: score forward inputs with production A/B (no orders)."""
    from jevlab.experiments.artifacts import ArtifactError
    from jevlab.experiments.forward_eval import (
        list_complete_forward_sessions,
        load_session_score_frame,
        score_forward_frame,
    )

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    paths.ensure_runtime_dirs()
    registry = paths.registry / "experiments.sqlite"
    artifact_dir = paths.data_models / "production_forward"
    scenario = next(c for c in configs.backtest.cost_scenarios if c.name == "normal")

    try:
        if scores_parquet is not None:
            frame = pd.read_parquet(scores_parquet)
            day = date.fromisoformat(session) if session else None
            result = score_forward_frame(
                frame,
                registry_path=registry,
                artifact_dir=artifact_dir,
                forward_root=paths.data_forward,
                session_date=day,
                top_k=configs.strategy.top_k,
                cost_bps_per_side=scenario.cost_bps_per_side,
                sell_fee_bps=configs.backtest.sell_fee_bps,
            )
            typer.echo(
                json.dumps(
                    {
                        "session_date": result.session_date.isoformat(),
                        "n_scored": result.n_scored,
                        "artifact_ids": result.artifact_ids,
                        "scores_path": str(result.scores_path),
                        "selected_counts": result.selected_counts,
                    },
                    indent=2,
                )
            )
            return

        sessions = (
            [date.fromisoformat(session)]
            if session
            else list_complete_forward_sessions(paths.data_forward)
        )
        if not sessions:
            typer.echo("no complete forward sessions to score", err=True)
            raise typer.Exit(code=1)
        for d in sessions:
            frame = load_session_score_frame(paths.data_forward, d)
            result = score_forward_frame(
                frame,
                registry_path=registry,
                artifact_dir=artifact_dir,
                forward_root=paths.data_forward,
                session_date=d,
                top_k=configs.strategy.top_k,
                cost_bps_per_side=scenario.cost_bps_per_side,
                sell_fee_bps=configs.backtest.sell_fee_bps,
            )
            typer.echo(
                f"{d.isoformat()} n={result.n_scored} selected={result.selected_counts} "
                f"path={result.scores_path}"
            )
    except ArtifactError as exc:
        typer.echo(f"forward-score failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("forward-report")
def forward_report_cmd(
    scores_parquet: Annotated[
        Path,
        typer.Option("--scores-parquet", help="Scored forward rows (with model_id, score)."),
    ],
    labels_parquet: Annotated[
        Path,
        typer.Option("--labels-parquet", help="Labels with realized y when windows complete."),
    ],
    baseline_jev_parquet: Annotated[
        Path,
        typer.Option("--baseline-jev", help="2019-2023 Jev feature baseline for KS drift."),
    ],
    forward_jev_parquet: Annotated[
        Path,
        typer.Option("--forward-jev", help="Forward-window Jev features for KS drift."),
    ],
) -> None:
    """Phase 13B: join labels, rolling A-vs-B, KS drift, 60-session checkpoint."""
    from jevlab.experiments.forward_eval import (
        ForwardLabelJoinError,
        run_forward_report,
    )

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    paths.ensure_runtime_dirs()
    try:
        result = run_forward_report(
            forward_root=paths.data_forward,
            scores=pd.read_parquet(scores_parquet),
            labels=pd.read_parquet(labels_parquet),
            baseline_jev=pd.read_parquet(baseline_jev_parquet),
            forward_jev=pd.read_parquet(forward_jev_parquet),
            report_dir=paths.reports / "forward",
        )
    except ForwardLabelJoinError as exc:
        typer.echo(f"forward-report refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"n_labeled={result.n_sessions_labeled}")
    typer.echo(f"report={result.report_path}")
    if result.checkpoint_60_path is not None:
        typer.echo(f"checkpoint_60={result.checkpoint_60_path} (informational only)")
    typer.echo(json.dumps(result.rolling, indent=2, default=str))


@app.command("evaluate-scale")
def evaluate_scale_cmd(
    fixture_metrics: Annotated[
        Path | None,
        typer.Option(
            "--fixture-metrics",
            help="JSON ScaleGateMetrics for fixture evaluation.",
        ),
    ] = None,
) -> None:
    """Phase 13B: evaluate S1-S8 -> SCALE_ACCEPTANCE.json only if all hold."""
    from jevlab.constants import TINY_LIVE_CEILING_USD
    from jevlab.experiments.acceptance import current_git_commit
    from jevlab.experiments.artifacts import ArtifactError, verify_acceptance
    from jevlab.experiments.forward_eval import (
        SCALE_DOES_NOT_RAISE_CEILING,
        ScaleGateMetrics,
        evaluate_scale,
    )

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    paths.ensure_runtime_dirs()
    tiny_path = paths.reports_acceptance / "TINY_LIVE_ACCEPTANCE.json"
    if not tiny_path.is_file():
        typer.echo("TINY_LIVE_ACCEPTANCE.json missing; SCALE requires it as parent", err=True)
        raise typer.Exit(code=1)
    if fixture_metrics is None:
        typer.echo(
            "Provide --fixture-metrics JSON for scale evaluation "
            "(live forward metrics assembly is ops-driven).",
            err=True,
        )
        raise typer.Exit(code=2)
    raw = json.loads(Path(fixture_metrics).read_text(encoding="utf-8"))
    metrics = ScaleGateMetrics(**{k: v for k, v in raw.items() if hasattr(ScaleGateMetrics, k)})
    try:
        result = evaluate_scale(
            metrics=metrics,
            acceptance_dir=paths.reports_acceptance,
            registry_path=paths.registry / "experiments.sqlite",
            tiny_live_path=tiny_path,
            git_commit=current_git_commit(cwd=paths.root),
            research_config_hash=configs.research_config_hash,
            qset_hash=str(configs.jev.qset_hash or ""),
            model_resolved=str(configs.jev.model_id),
        )
    except ArtifactError as exc:
        typer.echo(f"evaluate-scale failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"all_passed={result.gate_report.get('all_passed')}")
    typer.echo(f"gate_report={result.gate_report_path}")
    typer.echo(SCALE_DOES_NOT_RAISE_CEILING)
    typer.echo(f"TINY_LIVE_CEILING_USD={TINY_LIVE_CEILING_USD}")
    if result.acceptance is not None:
        body = verify_acceptance(result.acceptance.path)
        typer.echo(f"acceptance={result.acceptance.path}")
        typer.echo(f"integrity_digest={body['integrity_digest']}")
        typer.echo(f"parent_digest={body.get('parent_artifact_digest')}")
        typer.echo("note=integrity_digest is an integrity digest, not a signature")
    elif result.rejection_path is not None:
        typer.echo(f"rejection={result.rejection_path}")
        raise typer.Exit(code=1)
