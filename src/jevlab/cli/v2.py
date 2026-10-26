"""CLI: ``jev v2 final-test|preseal-report`` - V2 pre-seal tools.

``final-test`` is implemented but must not open 2024-2025 in this phase:
it refuses the real sealed source. V2-7 will replace that wiring.
"""

from __future__ import annotations

import json

import typer

from jevlab.utils.logging import configure_logging
from jevlab.utils.paths import ProjectPaths, get_project_root
from jevlab.v2.report import write_preseal_report
from jevlab.v2.sealed_run import ForbiddenRealSealedSource, run_v2_final_test
from jevlab.v2.training import assert_sealed_training_cutoff

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
    allow_second_look: bool = typer.Option(False, help="Diagnostics only; breaks seal."),
    seal_token: str = typer.Option("", envvar="JEV_SEAL_TOKEN"),
) -> None:
    """One-shot V2 sealed evaluation.

    Pre-seal implementation: real 2024-2025 source is forbidden. Preconditions
    are still enforced. Do not use this to open the seal in V2-1...V2-4.
    """
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    from jevlab.v2.acceptance import V2GateMetrics

    result = run_v2_final_test(
        root=paths.root,
        registry_path=paths.registry / "experiments.sqlite",
        acceptance_dir=paths.reports_acceptance,
        seal_dir=paths.reports / "v2_sealed",
        seal_token=seal_token,
        metrics=V2GateMetrics(),
        allow_second_look=allow_second_look,
        require_clean_tree=True,
        require_git_tags=True,
        require_preseal_tag=True,
        require_hashes=True,
        sealed_source=ForbiddenRealSealedSource(),
        record_sealed_access=True,
    )
    typer.echo(json.dumps({"acceptance_written": result.acceptance_written}, indent=2))
