"""Typer app shell: version + data commands (later phases add more)."""

from __future__ import annotations

import typer

from jevlab.cli.data import app as data_app
from jevlab.cli.features import app as features_app
from jevlab.cli.jev_cmd import app as jev_app
from jevlab.cli.live import app as live_app
from jevlab.cli.report import app as report_app
from jevlab.cli.run import app as run_app
from jevlab.cli.v2 import app as v2_app

app = typer.Typer(
    name="jev",
    help="Jev trading research CLI. Live trading is hard-disabled by default.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(run_app, name="run")
app.add_typer(jev_app, name="jev")
app.add_typer(report_app, name="report")
app.add_typer(live_app, name="live")
app.add_typer(v2_app, name="v2")


@app.callback()
def main() -> None:
    """jevlab CLI entry point."""


@app.command("version")
def version() -> None:
    """Print package version."""
    from jevlab import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
