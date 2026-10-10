"""CLI: ``jev report build`` - HTML/Markdown from experiment manifest (Phase 11)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jevlab.reporting.build_report import build_report
from jevlab.reporting.statements import CONTAMINATION_LABEL
from jevlab.utils.logging import configure_logging
from jevlab.utils.paths import ProjectPaths, get_project_root

app = typer.Typer(help="Build experiment reports from manifests (no recomputation).")

ManifestArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, help="Path to manifest.json"),
]
OutDirOpt = Annotated[
    Path | None,
    typer.Option("--out-dir", help="Output directory (default: reports/<experiment_id>/)."),
]


@app.command("build")
def build_cmd(
    manifest: ManifestArg,
    out_dir: OutDirOpt = None,
) -> None:
    """Build report.html and report.md from a manifest only."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    report = build_report(manifest)
    dest = out_dir or (paths.reports / report.experiment_id)
    written = report.write(dest)
    typer.echo(f"experiment_id={report.experiment_id}")
    typer.echo(f"verdict={report.conclusion.kind}")
    typer.echo(f"conclusion={report.conclusion.text}")
    typer.echo(f"contamination_label={CONTAMINATION_LABEL}")
    typer.echo(f"html={written['html']}")
    typer.echo(f"markdown={written['markdown']}")
