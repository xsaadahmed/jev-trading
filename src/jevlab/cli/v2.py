"""CLI: ``jev v2 final-test|preseal-report|build-sealed-model``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from jevlab.cv.sealed_guard import SealedAccessError
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.logging import configure_logging
from jevlab.utils.paths import ProjectPaths, get_project_root
from jevlab.v2.hashes import FROZEN_IMPLEMENTATION_REVISION, FROZEN_PRESEAL_IMPL_TAG
from jevlab.v2.model_artifact import (
    DEFAULT_MANIFEST_REL,
    fit_presealed_model,
    load_model_freeze,
    write_sealed_test_model,
)
from jevlab.v2.report import write_preseal_report
from jevlab.v2.sealed_run import (
    REQUIRED_AUTHORIZATION_PHRASE,
    VOID_AUTHORIZATION_PHRASES,
    run_v2_final_test,
    run_v2_preflight,
)
from jevlab.v2.sealed_source import PreSealedTrainingSource, ProductionSealedSource
from jevlab.v2.statistics import FROZEN_N
from jevlab.v2.training import SEALED_T0, assert_sealed_training_cutoff

app = typer.Typer(help="V2 baseline-validation commands (pre-seal implementation).")


def _model_path(paths: ProjectPaths) -> Path:
    return paths.root / DEFAULT_MANIFEST_REL


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


@app.command("build-sealed-model")
def build_sealed_model_cmd() -> None:
    """Fit the final sealed-test Model A from pre-2024 stores only."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    source = PreSealedTrainingSource(use_dev_stores=True)
    payload = source.load()
    fitted, meta = fit_presealed_model(
        features=payload["features"],
        labels=payload["labels"],
        bars=payload["bars"],
        t0=SEALED_T0,
        source_manifest_digest=source.source_manifest_digest(root=paths.root),
    )
    from jevlab.v2.snapshot import git_head

    artifact = write_sealed_test_model(
        fitted=fitted,
        meta=meta,
        out_dir=paths.root / "reports" / "v2_preseal",
        git_commit=git_head(paths.root),
    )
    typer.echo(
        json.dumps(
            {
                "path": str(artifact.path),
                "integrity_digest": artifact.integrity_digest,
                "chosen_C": artifact.chosen_C,
                "final_training_decision_date": artifact.manifest["final_training_decision_date"],
                "training_observation_count": artifact.manifest["training_observation_count"],
                "sealed_underlying_reads": source.sealed_underlying_reads,
                "training_context_reads": source.training_context_reads,
                "implementation_revision": FROZEN_IMPLEMENTATION_REVISION,
            },
            indent=2,
            default=str,
        )
    )


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
        help="Fresh r3 human authorization phrase; prior r1/r2 authorizations are void.",
    ),
) -> None:
    """One-shot V2 sealed evaluation.

    ``--preflight`` terminates before SEAL authorization and before any sealed
    read. The production source is constructed but not loaded. Preflight may
    read the pre-sealed training source and the frozen model artifact.

    Production gate metrics are computed from the sealed result only. This
    command cannot accept caller-supplied V2GateMetrics. Candidate performance
    is not displayed until candidate computation, all 2000 nulls, and hard
    gates have completed. The runner loads the prebuilt Model A and does not
    fit after sealed access.
    """
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    source = ProductionSealedSource(paths.data_sealed)
    training = PreSealedTrainingSource(use_dev_stores=True)
    model_path = _model_path(paths)
    freeze = load_model_freeze(paths.root)
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
                training_source=training,
                model_artifact_path=model_path if model_path.is_file() else None,
                require_prebuilt_model=True,
                model_freeze=freeze or None,
            )
            out["production_sealed_source_reads"] = source.filesystem_reads
        except (PreconditionError, SealedAccessError) as exc:
            out = {
                "preflight": True,
                "sealed_read": False,
                "filesystem_reads": source.filesystem_reads,
                "sealed_underlying_reads": source.filesystem_reads,
                "training_context_reads": training.training_context_reads,
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
                    "error": "fresh implementation-r3 authorization required",
                    "sealed_read": False,
                    "acceptance_written": False,
                    "required_tag": FROZEN_PRESEAL_IMPL_TAG,
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
        training_source=training,
        model_artifact_path=model_path,
        require_prebuilt_model=True,
        model_freeze=freeze or None,
        record_sealed_access=True,
        n=FROZEN_N,
        allow_reduced_n=False,
        allow_sealed_dates=True,
        result_label="confirmatory",
        t0=SEALED_T0,
        implementation_tag=FROZEN_PRESEAL_IMPL_TAG,
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
    """Run the r3 production computational path on a PRE-2024 two-source rehearsal."""
    configure_logging()
    from jevlab.v2.rehearsal import run_production_path_rehearsal

    out = run_production_path_rehearsal(n=n, repeat=repeat)
    typer.echo(json.dumps(out.to_dict(), indent=2, default=str))
    raise typer.Exit(code=0 if out.ok else 1)
