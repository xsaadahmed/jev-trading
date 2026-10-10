# Phase 2 Report — Feature pipeline

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§6, 6.1, 21 Phase 2, 21A  
**Preserved:** Phases 0–1 (scaffold, market data, availability)

---

## What was built

### Feature modules (`src/jevlab/features/`)

| Module | Responsibility |
| ------ | -------------- |
| `columns.py` | `FEATURE_VERSION="v1"`, `MODEL_A_COLUMNS` (17), `STORED_ONLY_COLUMNS` (3), `STATE_V1_SOURCE_COLUMNS` mapping for information parity |
| `rolling.py` | Log returns, SMA, Wilder RSI, down streak, realized vol, own-252 trailing percentile |
| `technical.py` | Per-symbol technicals from as-of bars (returns, gap, trend, RSI, RV, volume) |
| `market_context.py` | `spy_ret_5d`, `spy_px_vs_sma200`, `vix_lag1_pct_252` (previous session only) |
| `relative.py` | `rel_ret_5d`, `xs_rank_ret_5d` (cross-sectional pct rank) |
| `pipeline.py` | Session builder via `as_of(decision_time)`; hive write/read; logical digest |

### Eligibility (`src/jevlab/targets/eligibility.py`)

- `eligible = 1[adj_close > SMA200]` (warm-up → not eligible).

### Output

- 20 stored features + `eligible`, every row carries `decision_time = t 18:30 America/New_York` (stored UTC).
- Hive parquet: `data/features/v1/symbol=/year=/part.parquet`.
- CLI: `jev features build`.

### MODEL_A_COLUMNS (exactly 17)

`ret_1d, ret_3d, ret_5d, ret_20d, down_streak, px_vs_sma20, px_vs_sma50, px_vs_sma200, rsi_3, rsi_14, rv_pct_252, vol_ratio_20, vix_lag1_pct_252, spy_ret_5d, spy_px_vs_sma200, rel_ret_5d, xs_rank_ret_5d`

Stored-only (not Model A): `gap_1d`, `rv_20`, `dollar_vol_20`.

---

## Decisions

1. **Every session build calls `as_of(decision_time(t))`** on bars and VIX before any feature math (availability leaks cannot influence features).
2. **`vix_lag1_pct_252`** looks up VIX at `previous_nyse_session(t)` and its own-252 percentile in the as-of VIX history — never same-day VIX.
3. **`dollar_vol_20`** = `log(mean(raw_close * raw_volume, 20))` (raw dollars per §5.2).
4. **`xs_rank_ret_5d`** = pandas `rank(pct=True, method='average')` across symbols on the date.
5. **Own percentile** = `mean(window <= x_t)` over a full 252-length window (NaN while warming up).
6. Information-parity scaffold: every `state_v1` source column is asserted ⊆ `MODEL_A_COLUMNS` (text state builder remains Phase 5).

---

## Deviations / ambiguities

| Item | Choice |
| ---- | ------ |
| Session-loop performance | Correctness-first O(sessions × history) rebuild; fine for V1 ETF panel; may optimize later without changing digests |
| Partial windows for percentiles | Require full 252 (warm-up NaNs only) |
| Cross-section when a symbol is missing that day | Rank among symbols present |

No §21A conflicts.

---

## How to run

```bash
# after Phase 1 data is on disk
jev features build [--start YYYY-MM-DD] [--end YYYY-MM-DD]
make check
```

---

## Test results

| Check | Result |
| ----- | ------ |
| ruff / mypy | pass |
| pytest | all pass (Phase 0–2 suite) |

Covered: known RSI/SMA/percentile values; truncate-and-compare causality; perturb-future; availability (`available_at > decision_time` ignored); VIX(t) mutation leaves features at t unchanged; warm-up NaNs only; hypothesis price scale-invariance (excl. `dollar_vol_20`); adj-price scale invariance; determinism; identical `logical_table_digest` across rebuild/parquet encodings (not byte equality).

---

## Known limitations

- Full-history feature build over ~10y × 13 symbols is intentionally simple (per-session `as_of`); operator runtime may be minutes on a laptop.
- No labels, models, or Jev code in this phase.

---

## Stop

Phase 2 is complete. Do not proceed to Phase 3 until explicitly requested.
