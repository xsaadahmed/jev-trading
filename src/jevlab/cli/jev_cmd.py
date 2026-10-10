"""CLI: ``jev smoke|audit|freeze-qset|batch`` (Phases 5-7)."""

from __future__ import annotations

import json
from datetime import date

import typer

from jevlab.config.loader import load_all_configs
from jevlab.cv.sealed_guard import SealedDataGuard
from jevlab.features.columns import FEATURE_VERSION
from jevlab.features.pipeline import read_feature_store
from jevlab.jev.audit import (
    DEFAULT_BUDGET_USD,
    BudgetExceededError,
    BudgetGuard,
    SealedPartitionError,
    plan_audit_calls,
    refuse_sealed_read,
    run_audit_dry,
    write_audit_report,
)
from jevlab.jev.audit import freeze_qset as freeze_qset_impl
from jevlab.jev.batch_runner import (
    UPPER_BOUND_STATES,
    read_forward_start_date,
    run_bulk_inference,
)
from jevlab.jev.cache import JevCache, request_hash
from jevlab.jev.client import JevClient
from jevlab.jev.parse import parse_response
from jevlab.jev.preconditions import PreconditionError, assert_bulk_preconditions
from jevlab.jev.qset import load_qset
from jevlab.jev.state_builder import build_state_from_percentiles
from jevlab.jev.vocabulary import DEFAULT_BUCKET_EDGES
from jevlab.utils.logging import configure_logging, get_logger
from jevlab.utils.paths import ProjectPaths, get_project_root

app = typer.Typer(help="Jev API smoke / audit / freeze / batch commands.")
log = get_logger(__name__)


@app.command("smoke")
def smoke_cmd(
    live: bool = typer.Option(
        False,
        "--live",
        help="Make ONE real API call. Default is dry-run (no network).",
    ),
) -> None:
    """Validate qset + state + privacy; optionally call System One once."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    qset = load_qset(
        paths.root / configs.jev.qset_path,
        expected_hash=configs.jev.qset_hash,
        expected_purpose="research",
    )
    # Synthetic mid-bucket percentiles — no market identity.
    mid = 0.5
    percentiles = {
        "px_vs_sma200": mid,
        "px_vs_sma50": mid,
        "px_vs_sma20": mid,
        "ret_1d": mid,
        "ret_3d": mid,
        "ret_5d": mid,
        "ret_20d": mid,
        "rsi_3": mid,
        "rsi_14": mid,
        "rv_pct_252": mid,
        "vol_ratio_20": mid,
        "rel_ret_5d": mid,
        "spy_ret_5d": mid,
        "spy_px_vs_sma200": mid,
        "vix_lag1_pct_252": mid,
    }
    state = build_state_from_percentiles(
        percentiles,
        down_streak=2,
        edges=tuple(configs.jev.bucket_edges) or DEFAULT_BUCKET_EDGES,
    )
    typer.echo(json.dumps({"qset_hash": qset.content_hash, "state": state}, indent=2))
    if not live:
        typer.echo("dry-run: pass --live to make exactly one real API call")
        return

    cache = JevCache(paths.root / configs.jev.cache_dir)
    client = JevClient(config=configs.jev)
    questions = qset.as_request_questions()
    req_h = request_hash(
        route=configs.jev.route,
        model_requested=configs.jev.model_id,
        state=state,
        questions=qset.to_cacheable_dict(),
    )
    hit = cache.get(req_h)
    if hit is not None:
        typer.echo(f"cache hit attempt={hit.attempt}")
        parsed = parse_response(hit.response)
        typer.echo(json.dumps(parsed, indent=2))
        return

    response = client.infer(state, questions)
    record = cache.put(
        req_hash=req_h,
        route=configs.jev.route,
        model_requested=configs.jev.model_id,
        model_resolved=response.model,
        purpose="research",
        qset_version=qset.qset_version,
        state_schema_version=configs.jev.state_schema_version,
        response=response.model_dump(mode="json"),
    )
    parsed = parse_response(response)
    out = paths.reports / "phase5_smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_response.json").write_text(
        json.dumps(response.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    (out / "smoke_features.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps({"request_hash": req_h, "attempt": record.attempt, "features": parsed}, indent=2)
    )
    typer.echo(f"wrote {out}")


@app.command("audit")
def audit_cmd(
    live: bool = typer.Option(
        False,
        "--live",
        help="Permit real audit_qset API calls (default: dry-run, zero calls).",
    ),
    budget_usd: float = typer.Option(
        DEFAULT_BUDGET_USD,
        "--budget-usd",
        help="USD cap for live audit spend (input tokens x $0.042/M).",
    ),
) -> None:
    """Run §11.4 diagnostics (warning signals only). Uses audit_qset, never qset_v1."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    # Confirm sealed reads are refused (path under data/sealed/).
    try:
        refuse_sealed_read(paths.data_sealed / "__no_read__", paths.data_sealed)
        raise RuntimeError("expected sealed-read refusal")
    except SealedPartitionError:
        pass

    # Confirm audit qset loads; research qset is NOT used for diagnostics.
    audit_qset = load_qset(
        paths.root / configs.jev.audit_qset_path,
        expected_hash=configs.jev.audit_qset_hash,
        expected_purpose="audit",
    )
    plan = plan_audit_calls()
    typer.echo(json.dumps(plan, indent=2))

    guard = BudgetGuard(budget_usd=budget_usd)
    try:
        guard.plan(int(plan["total_calls"]))
    except BudgetExceededError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    if live:
        typer.echo(
            "LIVE audit mode is implemented as a guarded scaffold in Phase 6: "
            "full bulk diagnostic inference is not auto-executed here. "
            "Use dry-run for planned cost; Phase 6 acceptance is the AUDIT_REPORT "
            "plus human contamination disposition. Real per-state calls belong "
            "behind an explicit follow-up operator run with budget remaining.",
            err=True,
        )
        # Still refuse to silently burn budget: require dry-run for automated CI.
        # Live path records the plan and exits without calling the API in V1 Phase 6
        # automation (operator must use a dedicated script later if needed).
        typer.echo("Refusing automated live fan-out; write dry-run report instead.")

    features = None
    feat_path = paths.data_features / FEATURE_VERSION
    if feat_path.exists():
        guard_sealed = SealedDataGuard(
            sealed_root=paths.data_sealed,
            registry_path=paths.registry / "experiments.sqlite",
        )
        features = read_feature_store(feat_path)
        features = guard_sealed.filter_dev_frame(features)

    report = run_audit_dry(features, sealed_root=paths.data_sealed)
    report.meta["audit_qset_version"] = audit_qset.qset_version
    report.meta["audit_qset_hash"] = audit_qset.content_hash
    report.meta["budget_usd"] = budget_usd
    report.meta["live_requested"] = live
    out = paths.reports / "audit"
    md = write_audit_report(report, out)
    typer.echo(f"api_calls={report.meta.get('api_calls', 0)}")
    typer.echo(f"wrote {md} (hash={report.content_hash()})")
    typer.echo(
        "Complete docs/contamination_disposition.md as a HUMAN "
        "(code never auto-fills the decision)."
    )


@app.command("freeze-qset")
def freeze_qset_cmd() -> None:
    """Freeze qset_v1: write hash to config, ledger row, print git tag command."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    result = freeze_qset_impl(
        qset_path=paths.root / configs.jev.qset_path,
        config_path=paths.config / "jev.yaml",
        registry_path=paths.registry / "experiments.sqlite",
        state_schema_version=configs.jev.state_schema_version,
    )
    typer.echo(json.dumps(result, indent=2))
    typer.echo(f"Run manually (not executed): {result['git_tag_command']}")


@app.command("batch")
def batch_cmd(
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Run live bulk inference. Default is dry-run (plan + cost only).",
    ),
    budget_usd: float = typer.Option(
        DEFAULT_BUDGET_USD,
        "--budget-usd",
        help="USD cap; abort if projected spend exceeds this.",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        help="Bounded worker count for live calls.",
    ),
) -> None:
    """Resumable bulk historical Jev inference (Phase 7). Dry-run by default."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)

    try:
        report = assert_bulk_preconditions(
            root=paths.root,
            qset_path=paths.root / configs.jev.qset_path,
            frozen_hash=configs.jev.qset_hash,
            disposition_path=paths.docs / "contamination_disposition.md",
        )
    except PreconditionError as exc:
        typer.echo(f"precondition failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    qset = load_qset(
        paths.root / configs.jev.qset_path,
        expected_hash=configs.jev.qset_hash,
        expected_purpose="research",
    )
    cache = JevCache(paths.root / configs.jev.cache_dir)
    registry_path = paths.registry / "experiments.sqlite"
    forward_start = read_forward_start_date(registry_path)

    feat_path = paths.data_features / FEATURE_VERSION
    if not feat_path.exists():
        typer.echo(f"feature store missing: {feat_path}", err=True)
        raise typer.Exit(code=2)
    guard_sealed = SealedDataGuard(
        sealed_root=paths.data_sealed,
        registry_path=registry_path,
    )
    features = guard_sealed.filter_dev_frame(read_feature_store(feat_path))
    # Latest session in the (non-sealed) feature frame for interim end when no registry.
    dates = sorted(pd_to_dates(features["date"]))
    latest = dates[-1] if dates else None

    out_dir = paths.data_features / "jev" / configs.jev.qset_version
    checkpoint_path = paths.data_jev_cache / "batch_checkpoint.sqlite"
    dry_run = not execute

    client = None
    if execute:
        client = JevClient(config=configs.jev)

    try:
        result = run_bulk_inference(
            features,
            qset=qset,
            cache=cache,
            client=client,
            route=configs.jev.route,
            model_id=configs.jev.model_id,
            state_schema_version=configs.jev.state_schema_version,
            bucket_edges=configs.jev.bucket_edges or list(DEFAULT_BUCKET_EDGES),
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            forward_start=forward_start,
            latest_completed_session=latest,
            dry_run=dry_run,
            budget_usd=budget_usd,
            concurrency=concurrency,
        )
    except BudgetExceededError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    plan = result.plan.to_dict()
    plan["preconditions"] = report.checks
    plan["disposition"] = report.disposition_decision
    plan["dry_run"] = dry_run
    plan["upper_bound_states_note"] = (
        f"spec upper bound ≈ {UPPER_BOUND_STATES}; "
        "enumerated n_total is after warm-up drops and sealed exclusion"
    )
    typer.echo(json.dumps(plan, indent=2))
    if dry_run:
        typer.echo(
            f"dry-run: n_total={result.plan.n_total} "
            f"unique={result.plan.n_unique_hashes} "
            f"cache_hits={result.plan.n_cache_hits} "
            f"to_call={result.plan.n_to_call} "
            f"est_usd=${result.plan.estimated_cost_usd:.4f} "
            f"(upper_bound={UPPER_BOUND_STATES})"
        )
        typer.echo("pass --execute to run live bulk inference (budget-guarded)")
        return

    typer.echo(
        json.dumps(
            {
                "n_ok": result.n_ok,
                "n_error": result.n_error,
                "n_skipped_cache": result.n_skipped_cache,
                "n_missing": result.n_missing,
                "features_path": str(result.features_path),
                "meta": result.meta,
            },
            indent=2,
        )
    )


def pd_to_dates(series: object) -> list[date]:
    """Normalize a pandas date column to ``date`` objects."""
    import pandas as pd

    if not isinstance(series, pd.Series):
        raise TypeError("expected pandas Series")
    return [pd.Timestamp(str(x)).date() for x in series.tolist()]
