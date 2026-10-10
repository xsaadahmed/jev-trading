# Phase 1 Report — Market data pipeline with point-in-time availability

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§5, 6.1, 18 (downloads/bars/external), 19 (validators), 21 Phase 1, 21A  
**Preserved:** Phase 0 scaffold and utilities

---

## What was built

### Providers (`src/jevlab/data/providers/`)

- Shared `BarsProvider` / `SeriesProvider` protocols + `HttpProviderBase` (httpx, tenacity retries on transport/429/5xx, injectable `RateLimiter`).
- **Tiingo** EOD raw+adjusted (default 50/hour, 1000/day).
- **FRED** `VIXCLS` observations.
- **Alpaca** daily bars for cross-check only (uses `BROKER_API_KEY` / `BROKER_API_SECRET`; volume stored but not used for research).

### Availability (`data/availability.py`)

- Bars: `available_at = session_date 18:00 America/New_York`.
- FRED: `available_at = next NYSE session after obs_date at 08:00 ET`.
- `retrieved_at` stamped at download time.
- Single query helper `as_of(table, decision_time)` — timezone-aware; keeps `available_at <= decision_time` only.
- `decision_time(t) = t 18:30 ET`.

### Store / resume (`data/store.py`, `data/downloader.py`)

- Hive parquet: `symbol=/year=/vendor=/part.parquet` for bars; `series_id=/year=/` for external.
- Downloads manifest: JSONL under `data/raw/downloads_manifest.jsonl` + SQLite `downloads` (params, time, file sha256, logical digest, row count).
- Resume skips jobs whose `download_id` (hash of vendor+params including universe hash) already exists.
- Universe config remains the §21A 13 ETFs starting `2015-01-01`; hash via `jev data universe-hash`.

### Validators + CLI

- Duplicate dates, missing NYSE sessions, OHLC consistency, non-positive prices, adj/raw ratio jumps, volume anomalies, cross-vendor reconcile (>5bp share, list >50bp days on `adj_close`).
- CLI: `jev data download|validate|reconcile|universe-hash`.

### Docs

- `docs/DATA_NOTES.md` — availability policy, FRED/ALFRED vintage findings, storage, reconcile thresholds.

---

## Decisions

1. Stooq is listed in `data.yaml` but **not** implemented as a provider in Phase 1; Alpaca satisfies the cross-check requirement. Documented in DATA_NOTES.
2. Reconciliation compares **`adj_close`** (Tiingo adjusted vs Alpaca `adjustment=all`).
3. Half-days still get bar `available_at` at 18:00 ET (policy timestamp, not exchange close).
4. Empty vendor payloads are recorded as failures (no silent success); resume continues other symbols after per-symbol errors.

---

## Deviations / ambiguities

| Item | Choice |
| ---- | ------ |
| Stooq in `cross_check_vendors` | Deferred; Alpaca only for Phase 1 |
| Alpaca free-feed volume understatement | Ignored for validation beyond non-positive checks; closes only for reconcile |
| Labels PK / features | Untouched (Phase 0 only) |

No §21A conflicts.

---

## How to run

```bash
uv sync --all-extras
# set TIINGO_API_KEY, FRED_API_KEY, BROKER_API_KEY, BROKER_API_SECRET in .env
jev data download
jev data validate
jev data reconcile
make check
```

Offline tests never hit the network (httpx `MockTransport`).

---

## Test results

| Check | Result |
| ----- | ------ |
| `ruff check` / `ruff format --check` | pass |
| `mypy src/jevlab` | pass |
| `pytest -q` | 31 passed |

Coverage includes: mocked Tiingo/FRED/Alpaca HTTP; each validator with injected faults; resume + manifest fields; availability on weekends/holidays/half-days; `as_of` excluding future `available_at` and same-date VIX at `decision_time(t)`.

---

## Known limitations

- Live download against real vendors is not executed in this phase (keys/network not required for acceptance tests).
- Full-history acceptance (“all 13 symbols + VIX, zero missing sessions”) requires a real `jev data download` run by the operator after keys are configured.
- Stooq cross-check and ALFRED vintage automation remain future optional work (see DATA_NOTES).

---

## Stop

Phase 1 is complete. Do not proceed to Phase 2 until explicitly requested.
