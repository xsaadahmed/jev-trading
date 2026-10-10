# Data Notes — Market data & point-in-time availability

**Phase:** 1  
**Universe:** 13 ETFs in `config/universe.yaml` (hashed via `jev data universe-hash`)  
**Primary bars:** Tiingo EOD raw + adjusted  
**VIX:** FRED `VIXCLS`  
**Cross-check:** Alpaca daily bars (close prices only; volume not used)  
**Stooq:** listed in `data.yaml` cross_check_vendors for future optional use; not required for Phase 1 acceptance (Alpaca implements the cross-check path).

---

## Availability policy (MASTER_SPEC §6.1)

| Input | `available_at` rule | Notes |
| ----- | ------------------- | ----- |
| Daily OHLCV for session t | `t 18:00 America/New_York` | Conservative; live runs also store `retrieved_at` |
| FRED VIX observation for date d | `08:00 ET` on the **next NYSE session after d** | Enables VIX lag-1 at decision_time(t)=t 18:30 ET |
| Decision time | `t 18:30 America/New_York` | Features/states must use `as_of(table, decision_time)` only |

Half-days remain NYSE sessions; bars still receive `available_at = session 18:00 ET` (policy timestamp, not exchange close clock).

Weekends/holidays: bar calendar is NYSE sessions only (missing sessions are validation errors, never forward-filled). FRED may emit observations on calendar dates that are not NYSE sessions; availability still maps to the next NYSE session 08:00 ET.

---

## FRED / ALFRED vintage information

**What we looked for.** ALFRED (Archival FRED) provides vintage snapshots of series as they existed on past dates. Ideal verification would confirm when each `VIXCLS` observation first appeared in the public API relative to the observation date.

**What is available without paid/enterprise tooling.**

1. **FRED observations API** (`/fred/series/observations`) returns `date` + `value` for `VIXCLS`. It does **not** return a per-observation publication timestamp in the free observations payload used by this project.
2. **ALFRED vintage dates** (`/fred/series/vintagedates` and realtime-start/realtime-end parameters on observations) can list when the series vintage changed. For `VIXCLS`, vintages typically advance on business days, but the vintage date is a **series revision snapshot date**, not a guaranteed wall-clock publication time for the prior session's close.
3. **Live `retrieved_at`** (stored on every download row) is the project's empirical check: if a live pull after session t ever retrieves VIX(t) earlier than the policy `available_at`, log `availability.violation` (Phase 13A/live) and fail closed.

**V1 conclusion (conservative).** Because free FRED/ALFRED metadata does not prove that official VIX close for date d is always available by d 18:30 ET historically, Phase 1 keeps the §6.1 rule: VIX(d) becomes available at the next NYSE session 08:00 ET. Features must use lag-1 VIX (`vix_lag1_pct_252` in Phase 2). Same-day VIX would require a source with verified publication timestamps and would be a new `feature_version`, never an in-place change.

**Optional later verification steps (not blocking Phase 1).**

- Pull ALFRED vintage dates for `VIXCLS` and compare vintage date vs observation date distribution; document any same-day vintages.
- During live shadow capture, compare `retrieved_at` of VIX(d) against policy `available_at` and against decision_time(d).

---

## Storage layout

- Bars: hive parquet `data/processed/bars/symbol=SYM/year=YYYY/vendor=VENDOR/part.parquet`
- External: `data/processed/external/series_id=VIXCLS/year=YYYY/part.parquet`
- Manifest: `data/raw/downloads_manifest.jsonl` + SQLite `downloads` table (`registry/experiments.sqlite`)
- Each download records: params JSON, downloaded_at, file sha256 (corruption detection), logical content digest, row count

---

## Reconciliation thresholds (§5.2)

- Report share of overlapping symbol-days with `|close_other − close_primary| / close_primary > 5 bp`
- List all symbol-days with difference `> 50 bp` (flagged, not silently corrected)
- Comparison column: `adj_close` (Tiingo adjusted vs Alpaca `adjustment=all`)

---

## Commands

```bash
jev data download[--start YYYY-MM-DD] [--end YYYY-MM-DD] [--force] [--no-cross-check]
jev data validate [--vendor tiingo]
jev data reconcile [--primary-vendor tiingo] [--other-vendor alpaca]
jev data universe-hash
```

Secrets: `TIINGO_API_KEY`, `FRED_API_KEY`, `BROKER_API_KEY`, `BROKER_API_SECRET` (env only).
