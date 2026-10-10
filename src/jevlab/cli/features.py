"""CLI: ``jev features build``."""

from __future__ import annotations

import json
from datetime import date

import typer

from jevlab.config.loader import load_all_configs
from jevlab.data.downloader import default_store
from jevlab.features.columns import FEATURE_VERSION
from jevlab.features.pipeline import (
    build_feature_store,
    feature_store_digest,
    write_feature_store,
)
from jevlab.utils.logging import configure_logging
from jevlab.utils.paths import ProjectPaths, get_project_root

app = typer.Typer(help="Feature store build (causal, as_of-filtered).")


@app.command("build")
def build_cmd(
    start: str | None = typer.Option(None, help="First session YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="Last session YYYY-MM-DD"),
    vendor: str = typer.Option("tiingo", help="Bar vendor to read"),
) -> None:
    """Build ``data/features/v1`` from stored bars + VIX via ``as_of``."""
    configure_logging()
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    bars = store.read_bars(vendor=vendor)
    vix = store.read_external(series_id=configs.data.vix_series_id)
    if bars.empty or vix.empty:
        typer.echo("No bars/VIX found; run `jev data download` first.", err=True)
        raise typer.Exit(code=1)

    sessions = sorted({d for d in bars["date"].tolist()})
    if start:
        sessions = [d for d in sessions if d >= date.fromisoformat(start)]
    if end:
        sessions = [d for d in sessions if d <= date.fromisoformat(end)]

    features = build_feature_store(
        bars,
        vix,
        symbols=list(configs.universe.symbols),
        sessions=sessions,
    )
    out_dir = paths.data_features / FEATURE_VERSION
    write_feature_store(features, out_dir)
    digest = feature_store_digest(features)
    typer.echo(
        json.dumps(
            {
                "rows": len(features),
                "feature_version": FEATURE_VERSION,
                "output": str(out_dir),
                "digest": digest,
            },
            indent=2,
        )
    )
