"""CLI: ``jev live`` — hard-disabled by default (MASTER_SPEC Phase 14).

Dry-run is the default. Reads (account/positions/orders) are allowed in dry run;
order submission is not. Never arms or enables trading unless the operator
explicitly runs ``arm`` / ``--execute`` under all L1-L14 gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from jevlab.brokers.live_gate import EXECUTION_EXPERIMENT_SENTENCE
from jevlab.utils.logging import configure_logging
from jevlab.utils.paths import ProjectPaths, get_project_root

app = typer.Typer(
    help="Live trading (hard-disabled). Dry-run default; no orders unless all gates pass.",
)


def _banner() -> None:
    typer.echo(EXECUTION_EXPERIMENT_SENTENCE)


@app.callback()
def live_callback() -> None:
    """Print mandatory execution-experiment sentence at every live CLI entry."""
    _banner()


@app.command("arm")
def arm_cmd(
    phrase: Annotated[
        str,
        typer.Option("--phrase", prompt=True, help="Must equal the configured arm phrase."),
    ],
) -> None:
    """Write `.live_arm` with ≤ 24 h TTL (does not enable LIVE_TRADING_ENABLED)."""
    from jevlab.brokers.arm import write_arm
    from jevlab.config.loader import load_all_configs

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    path = write_arm(
        paths.root,
        phrase=phrase,
        expected_phrase=configs.live.arm_phrase,
        ttl_hours=configs.live.arm_token_ttl_hours,
    )
    typer.echo(f"armed={path} ttl_hours={configs.live.arm_token_ttl_hours}")
    typer.echo("note=arm token alone does not enable live trading")


@app.command("disarm")
def disarm_cmd() -> None:
    """Remove `.live_arm`."""
    from jevlab.brokers.arm import disarm

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    disarm(paths.root)
    typer.echo("disarmed")


@app.command("status")
def status_cmd() -> None:
    """Show live config flags, arm state, and ceiling (read-only)."""
    from jevlab.brokers.arm import is_armed, read_arm
    from jevlab.config.loader import load_all_configs
    from jevlab.constants import TINY_LIVE_CEILING_USD

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    payload: dict[str, Any] = {
        "LIVE_TRADING_ENABLED": configs.live.LIVE_TRADING_ENABLED,
        "BROKER_ENV": configs.live.BROKER_ENV,
        "MAX_STRATEGY_CAPITAL_USD": configs.live.MAX_STRATEGY_CAPITAL_USD,
        "TINY_LIVE_CEILING_USD": TINY_LIVE_CEILING_USD,
        "armed": is_armed(paths.root),
        "arm": read_arm(paths.root),
        "dry_run_default": configs.live.dry_run_default,
        "execution_experiment_sentence": EXECUTION_EXPERIMENT_SENTENCE,
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("dry-run")
def dry_run_cmd() -> None:
    """Compute intended orders + read-only reconcile; never submit.

    Order endpoints are not called. Under default config, LiveBroker cannot
    even be constructed.
    """
    from jevlab.brokers.live import LiveBrokerDisabledError, build_live_broker
    from jevlab.brokers.live_gate import check_live_gates, live_context_from_config
    from jevlab.config.loader import load_all_configs

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    ctx = live_context_from_config(
        configs.live,
        project_root=paths.root,
        registry_path=paths.registry / "experiments.sqlite",
        research_config_hash=configs.research_config_hash,
        execute=False,
    )
    report = check_live_gates(ctx)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    # Attempt construct only to prove default refuses — never used for submit.
    try:
        build_live_broker(configs.live)
        typer.echo("warning=LiveBroker unexpectedly constructible", err=True)
    except LiveBrokerDisabledError as exc:
        typer.echo(f"live_broker_constructible=false reason={exc}")
    typer.echo("submitted=false")


@app.command("evaluate-tiny")
def evaluate_tiny_cmd(
    fixture_metrics: Annotated[
        Path | None,
        typer.Option(
            "--fixture-metrics",
            help="JSON TinyLiveMetrics for fixture evaluation (recommended).",
        ),
    ] = None,
) -> None:
    """Evaluate T1-T9 → TINY_LIVE_ACCEPTANCE.json only if all hold.

    No forward-session count. integrity_digest is not a signature.
    """
    from jevlab.brokers.evaluate_tiny import TinyLiveMetrics, evaluate_tiny
    from jevlab.config.loader import load_all_configs
    from jevlab.constants import TINY_LIVE_CEILING_USD
    from jevlab.experiments.acceptance import current_git_commit
    from jevlab.experiments.artifacts import ArtifactError, verify_acceptance

    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    paths.ensure_runtime_dirs()
    hist = paths.reports_acceptance / "HISTORICAL_ACCEPTANCE.json"
    if not hist.is_file():
        typer.echo("HISTORICAL_ACCEPTANCE.json missing", err=True)
        raise typer.Exit(code=1)
    if fixture_metrics is None:
        typer.echo(
            "Provide --fixture-metrics JSON for evaluate-tiny "
            "(ops evidence assembly is operator-driven).",
            err=True,
        )
        raise typer.Exit(code=2)
    raw = json.loads(Path(fixture_metrics).read_text(encoding="utf-8"))
    fields = {k: v for k, v in raw.items() if k in TinyLiveMetrics.__dataclass_fields__}
    metrics = TinyLiveMetrics(**fields)
    try:
        result = evaluate_tiny(
            metrics=metrics,
            acceptance_dir=paths.reports_acceptance,
            registry_path=paths.registry / "experiments.sqlite",
            historical_path=hist,
            git_commit=current_git_commit(cwd=paths.root),
            research_config_hash=configs.research_config_hash,
            qset_hash=str(configs.jev.qset_hash or ""),
            model_resolved=str(configs.jev.model_id),
        )
    except ArtifactError as exc:
        typer.echo(f"evaluate-tiny failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"all_passed={result.gate_report.get('all_passed')}")
    typer.echo(f"gate_report={result.gate_report_path}")
    typer.echo("forward_session_count_required=false")
    typer.echo(f"TINY_LIVE_CEILING_USD={TINY_LIVE_CEILING_USD}")
    typer.echo("note=integrity_digest is an integrity digest, not a signature")
    if result.acceptance is not None:
        body = verify_acceptance(result.acceptance.path)
        typer.echo(f"acceptance={result.acceptance.path}")
        typer.echo(f"integrity_digest={body['integrity_digest']}")
        typer.echo(f"parent_digest={body.get('parent_artifact_digest')}")
    elif result.rejection_path is not None:
        typer.echo(f"rejection={result.rejection_path}")
        raise typer.Exit(code=1)
