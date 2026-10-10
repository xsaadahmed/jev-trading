"""CLI: ``jev data download|validate|reconcile``."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import Annotated, Any

import typer

from jevlab.config.loader import config_hash, load_all_configs
from jevlab.data.downloader import MarketDataDownloader, default_store, write_quality_report
from jevlab.data.validators import (
    reconcile_closes,
    validate_bars,
    validate_external_series,
)
from jevlab.utils.logging import configure_logging, get_logger
from jevlab.utils.paths import ProjectPaths, get_project_root

app = typer.Typer(help="Market data download, validation, and reconciliation.")
log = get_logger(__name__)


def _paths() -> ProjectPaths:
    return ProjectPaths.from_root(get_project_root())


@app.command("download")
def download_cmd(
    start: str | None = typer.Option(None, help="Override start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="Override end date YYYY-MM-DD"),
    force: bool = typer.Option(False, help="Re-download even if manifest has the job"),
    no_cross_check: bool = typer.Option(False, help="Skip Alpaca cross-check download"),
    symbol: Annotated[
        list[str] | None,
        typer.Option(help="Limit to symbol(s); repeatable"),
    ] = None,
) -> None:
    """Download Tiingo bars + FRED VIX (+ Alpaca cross-check by default)."""
    configure_logging()
    paths = _paths()
    paths.ensure_runtime_dirs()
    downloader = MarketDataDownloader.from_config(
        paths=paths,
        include_cross_check=not no_cross_check,
        require_secrets=True,
    )
    result = downloader.run(
        start=date.fromisoformat(start) if start else None,
        end=date.fromisoformat(end) if end else None,
        symbols=symbol,
        include_cross_check=not no_cross_check,
        force=force,
    )
    typer.echo(json.dumps(asdict(result), indent=2))
    if result.failed:
        raise typer.Exit(code=1)


@app.command("validate")
def validate_cmd(
    vendor: str = typer.Option("tiingo", help="Bar vendor to validate"),
    start: str | None = typer.Option(None, help="Session window start"),
    end: str | None = typer.Option(None, help="Session window end"),
) -> None:
    """Validate stored bars and VIX; write reports/data_quality.json."""
    configure_logging()
    paths = _paths()
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    start_d = date.fromisoformat(start or configs.data.start_date)
    end_d = date.fromisoformat(end) if end else date.today()

    bars = store.read_bars(vendor=vendor)
    ext = store.read_external(series_id=configs.data.vix_series_id)
    bar_report = validate_bars(bars, start=start_d, end=end_d)
    ext_report = validate_external_series(ext)

    payload: dict[str, Any] = {
        "universe_hash": config_hash(configs.universe),
        "vendor": vendor,
        "bars_ok": bar_report.ok,
        "bars_issues": [asdict(i) for i in bar_report.issues],
        "external_ok": ext_report.ok,
        "external_issues": [asdict(i) for i in ext_report.issues],
        "n_bar_rows": len(bars),
        "n_external_rows": len(ext),
    }
    out = paths.reports / "data_quality.json"
    write_quality_report(out, validation=payload)
    typer.echo(json.dumps(payload, indent=2, default=str))
    if not bar_report.ok or not ext_report.ok:
        raise typer.Exit(code=1)


@app.command("reconcile")
def reconcile_cmd(
    primary_vendor: str = typer.Option("tiingo"),
    other_vendor: str = typer.Option("alpaca"),
) -> None:
    """Reconcile close prices across vendors; list >50bp disagreements."""
    configure_logging()
    paths = _paths()
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    primary = store.read_bars(vendor=primary_vendor)
    other = store.read_bars(vendor=other_vendor)
    report = reconcile_closes(primary, other, price_col="adj_close")
    payload = {
        "universe_hash": config_hash(configs.universe),
        "primary_vendor": primary_vendor,
        "other_vendor": other_vendor,
        "n_overlapping_days": report.n_overlapping_days,
        "share_diff_gt_5bp": report.share_diff_gt_5bp,
        "days_gt_50bp": report.days_gt_50bp,
        "issues": [asdict(i) for i in report.issues],
    }
    out = paths.reports / "reconcile_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("universe-hash")
def universe_hash_cmd() -> None:
    """Print the canonical hash of config/universe.yaml."""
    paths = _paths()
    configs = load_all_configs(paths.config)
    typer.echo(config_hash(configs.universe))
