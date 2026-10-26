"""CLI: ``jev v2 final-test|preseal-report``."""

from __future__ import annotations

import json

import typer

from jevlab.cv.sealed_guard import SealedAccessError
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.logging import configure_logging
from jevlab.utils.paths import ProjectPaths, get_project_root
from jevlab.v2.report import write_preseal_report
from jevlab.v2.sealed_run import (
    REQUIRED_AUTHORIZATION_PHRASE,
    VOID_AUTHORIZATION_PHRASES,
    run_v2_final_test,
    run_v2_preflight,
)
from jevlab.v2.sealed_source import ProductionSealedSource
from jevlab.v2.statistics import FROZEN_N
from jevlab.v2.training import SEALED_T0, assert_sealed_training_cutoff

app = typer.Typer(help="V2 baseline-validation commands (pre-seal implementation).")


@app.command("preseal-report")
def preseal_report_cmd() -> None:
    """Write reports/v2_preseal reproducibility artifacts (no sealed access)."""
    configure_logging()
    out = write_preseal_report()
    typer.echo(json.dumps(out, indent=2))


@app.command("training-cutoff")
def training_cutoff_cmd() -> None:
    """Print the frozen embargo training cutoff (NYSE calendar; no market rows)."""
    cutoff = assert_sealed_training_cutoff()
    typer.echo(str(cutoff))


@app.command("final-test")
def final_test_cmd(
    preflight: bool = typer.Option(
        False, "--preflight", help="Verify guards then exit before SEAL/read."
    ),
    allow_second_look: bool = typer.Option(False, help="Diagnostics only; breaks seal."),
    seal_token: str = typer.Option("", envvar="JEV_SEAL_TOKEN"),
    authorization: str = typer.Option(
        "",
        "--authorization",
        help="Fresh r2 human authorization phrase; prior r1 authorization is void.",
    ),
) -> None:
    """One-shot V2 sealed evaluation.

    ``--preflight`` terminates before SEAL authorization and before any sealed
    read. The production source is constructed but not loaded.

    Production gate metrics are computed from the sealed result only. This
    command cannot accept caller-supplied V2GateMetrics. Candidate performance
    is not displayed until candidate computation, all 2000 nulls, and hard
    gates have completed.
    """
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    source = ProductionSealedSource(paths.data_sealed)
    if preflight:
        try:
            out = run_v2_preflight(
                root=paths.root,
                registry_path=paths.registry / "experiments.sqlite",
                configured_sealed_path=paths.data_sealed,
                require_clean_tree=True,
                require_git_tags=True,
                require_preseal_tag=True,
                require_hashes=True,
            )
        except (PreconditionError, SealedAccessError) as exc:
            out = {
                "preflight": True,
                "sealed_read": False,
                "filesystem_reads": source.filesystem_reads,
                "configured_path": source.configured_path_string(),
                "terminated_before_seal_authorization": True,
                "error": str(exc),
            }
        typer.echo(json.dumps(out, indent=2, default=str))
        raise typer.Exit(code=0 if "error" not in out else 1)

    if authorization.strip() in VOID_AUTHORIZATION_PHRASES:
        typer.echo(
            json.dumps(
                {
                    "error": "prior human authorization is VOID/CLOSED",
                    "sealed_read": False,
                    "acceptance_written": False,
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)
    if authorization.strip() != REQUIRED_AUTHORIZATION_PHRASE:
        typer.echo(
            json.dumps(
                {
                    "error": "fresh implementation-r2 authorization required",
                    "sealed_read": False,
                    "acceptance_written": False,
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)

    result = run_v2_final_test(
        root=paths.root,
        registry_path=paths.registry / "experiments.sqlite",
        acceptance_dir=paths.reports_acceptance,
        seal_dir=paths.reports / "v2_sealed",
        seal_token=seal_token,
        allow_second_look=allow_second_look,
        require_clean_tree=True,
        require_git_tags=True,
        require_preseal_tag=True,
        require_hashes=True,
        require_authorization=True,
        authorization=authorization,
        sealed_source=source,
        record_sealed_access=True,
        n=FROZEN_N,
        allow_reduced_n=False,
        allow_sealed_dates=True,
        result_label="confirmatory",
        t0=SEALED_T0,
    )
    typer.echo(
        json.dumps(
            {
                "acceptance_written": result.acceptance_written,
                "all_passed": result.gate_report.all_passed,
                "seal_broken": result.seal_broken,
            },
            indent=2,
        )
    )


@app.command("production-rehearsal")
def production_rehearsal_cmd(
    n: int = typer.Option(FROZEN_N, help="Null draws; frozen N=2000 for the canonical rehearsal."),
    repeat: bool = typer.Option(True, help="Repeat the production path for digest equality."),
) -> None:
    """Run the production computational path on a PRE-2024 pseudo-sealed source."""
    configure_logging()
    from jevlab.v2.rehearsal import run_production_path_rehearsal

    out = run_production_path_rehearsal(n=n, repeat=repeat)
    typer.echo(json.dumps(out.to_dict(), indent=2, default=str))
    raise typer.Exit(code=0 if out.ok else 1)
