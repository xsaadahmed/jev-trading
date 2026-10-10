"""Offline mocked HTTP tests for providers, resume, and manifest."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from jevlab.config.loader import config_hash, load_all_configs
from jevlab.data.downloader import MarketDataDownloader
from jevlab.data.providers.alpaca import AlpacaBarsProvider
from jevlab.data.providers.fred import FredSeriesProvider
from jevlab.data.providers.rate_limit import RateLimiter
from jevlab.data.providers.tiingo import TiingoBarsProvider
from jevlab.data.store import DataStore
from jevlab.utils.paths import ProjectPaths


def _tiingo_payload(symbol: str, days: list[str]) -> list[dict[str, Any]]:
    rows = []
    for i, d in enumerate(days):
        px = 100.0 + i
        rows.append(
            {
                "date": f"{d}T00:00:00.000Z",
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px + 0.5,
                "volume": 1_000_000 + i,
                "adjOpen": px,
                "adjHigh": px + 1,
                "adjLow": px - 1,
                "adjClose": px + 0.5,
                "adjVolume": 1_000_000 + i,
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
        )
    return rows


def _fred_payload(days: list[str]) -> dict[str, Any]:
    return {"observations": [{"date": d, "value": str(10.0 + i)} for i, d in enumerate(days)]}


def _alpaca_payload(days: list[str]) -> dict[str, Any]:
    bars = []
    for i, d in enumerate(days):
        px = 100.0 + i
        bars.append(
            {
                "t": f"{d}T05:00:00Z",
                "o": px,
                "h": px + 1,
                "l": px - 1,
                "c": px + 0.5,
                "v": 500_000 + i,
            }
        )
    return {"bars": bars, "next_page_token": None}


@pytest.fixture
def mock_clients() -> dict[str, httpx.Client]:
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]

    def tiingo_handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.path.rstrip("/").split("/")[-2]
        return httpx.Response(200, json=_tiingo_payload(symbol, days))

    def fred_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fred_payload(days))

    def alpaca_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_alpaca_payload(days))

    return {
        "tiingo": httpx.Client(transport=httpx.MockTransport(tiingo_handler)),
        "fred": httpx.Client(transport=httpx.MockTransport(fred_handler)),
        "alpaca": httpx.Client(transport=httpx.MockTransport(alpaca_handler)),
    }


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return DataStore(
        bars_dir=tmp_path / "bars",
        external_dir=tmp_path / "external",
        raw_dir=tmp_path / "raw",
        registry_path=tmp_path / "experiments.sqlite",
    )


def test_tiingo_provider_parses_mocked_http(mock_clients: dict[str, httpx.Client]) -> None:
    provider = TiingoBarsProvider(
        api_key="k",
        client=mock_clients["tiingo"],
        rate_limiter=RateLimiter(per_hour=1000, per_day=1000),
    )
    frame = provider.fetch_daily_bars("SPY", date(2024, 1, 2), date(2024, 1, 4))
    assert len(frame) == 3
    assert list(frame["symbol"].unique()) == ["SPY"]
    assert frame.loc[0, "adj_close"] == pytest.approx(100.5)


def test_fred_and_alpaca_providers_parse(
    mock_clients: dict[str, httpx.Client],
) -> None:
    fred = FredSeriesProvider(
        api_key="k",
        client=mock_clients["fred"],
        rate_limiter=RateLimiter(per_hour=1000, per_day=1000),
    )
    vix = fred.fetch_series("VIXCLS", date(2024, 1, 2), date(2024, 1, 4))
    assert len(vix) == 3
    assert vix.loc[0, "series_id"] == "VIXCLS"

    alpaca = AlpacaBarsProvider(
        api_key="k",
        api_secret="s",
        client=mock_clients["alpaca"],
        rate_limiter=RateLimiter(per_hour=1000, per_day=1000),
    )
    bars = alpaca.fetch_daily_bars("QQQ", date(2024, 1, 2), date(2024, 1, 4))
    assert len(bars) == 3


def test_download_resume_and_manifest(
    store: DataStore,
    mock_clients: dict[str, httpx.Client],
    tmp_path: Path,
) -> None:
    # Point config loader at real repo configs but write data into tmp store.
    paths = ProjectPaths.from_root()
    configs = load_all_configs(paths.config)
    downloader = MarketDataDownloader(
        store=store,
        data_cfg=configs.data,
        universe=configs.universe,
        primary=TiingoBarsProvider(
            api_key="k",
            client=mock_clients["tiingo"],
            rate_limiter=RateLimiter(per_hour=1000, per_day=1000),
        ),
        cross_check=AlpacaBarsProvider(
            api_key="k",
            api_secret="s",
            client=mock_clients["alpaca"],
            rate_limiter=RateLimiter(per_hour=1000, per_day=1000),
        ),
        vix=FredSeriesProvider(
            api_key="k",
            client=mock_clients["fred"],
            rate_limiter=RateLimiter(per_hour=1000, per_day=1000),
        ),
    )

    start, end = date(2024, 1, 2), date(2024, 1, 4)
    first = downloader.run(
        start=start,
        end=end,
        symbols=["SPY"],
        include_cross_check=True,
        force=False,
    )
    assert "tiingo:SPY" in first.completed
    assert "alpaca:SPY" in first.completed
    assert "fred:VIXCLS" in first.completed
    assert first.skipped == []
    assert first.failed == []
    assert store.manifest_path.is_file()
    manifest_lines = store.manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 3
    record = json.loads(manifest_lines[0])
    assert record["row_count"] == 3
    assert len(record["file_sha256"]) == 64
    assert len(record["logical_digest"]) == 64
    assert record["params"]["universe_hash"] == config_hash(configs.universe)

    bars = store.read_bars(symbols=["SPY"], vendor="tiingo")
    assert len(bars) == 3
    assert bars["available_at"].notna().all()
    assert bars["retrieved_at"].notna().all()
    ext = store.read_external(series_id="VIXCLS")
    assert len(ext) == 3
    assert ext["available_at"].notna().all()

    # Resume: second run skips all three jobs.
    second = downloader.run(
        start=start,
        end=end,
        symbols=["SPY"],
        include_cross_check=True,
        force=False,
    )
    assert sorted(second.skipped) == sorted(["tiingo:SPY", "alpaca:SPY", "fred:VIXCLS"])
    assert second.completed == []
    # Manifest unchanged length (no duplicate appends on skip).
    assert len(store.manifest_path.read_text(encoding="utf-8").strip().splitlines()) == 3

    # Hive layout exists.
    hive = list((tmp_path / "bars").glob("symbol=SPY/year=2024/vendor=tiingo/part.parquet"))
    assert len(hive) == 1
    loaded = pd.read_parquet(hive[0])
    assert len(loaded) == 3


def test_universe_hash_stable() -> None:
    configs = load_all_configs()
    assert configs.universe.symbols[0] == "SPY"
    assert len(configs.universe.symbols) == 13
    h1 = config_hash(configs.universe)
    h2 = config_hash(load_all_configs().universe)
    assert h1 == h2
    assert len(h1) == 64
