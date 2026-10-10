# Phase 3 Report — Labels, walk-forward CV, sealed guard, baselines R0/A/D

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§8, 9, 10.3–10.4, 21 Phase 3, 21A  
**Preserved:** Phases 0–2 (scaffold, market data, features)

---

## What was built

### Labels (`src/jevlab/targets/labels.py`)

Single module for all label logic:

| Symbol | Definition |
| ------ | ---------- |
| `R` | `adj_open[t+4] / adj_open[t+1] - 1` |
| `Rbar` | Equal-weight mean of `R` over the 13 ETFs (skip incomplete dates) |
| `X` | `R - Rbar` |
| `y` | `1[X > 0]` on eligible rows only |
| `y_cost` | `1[R > roundtrip_cost_dec]` (eligible-gated) |
| `roundtrip_cost_dec` | `2 * cost_bps_per_side/10_000 + sell_fee_bps/10_000` (decimal return) |

- Exact check: 2.5 bps/side + 0.3 bp fee → `0.00053`; two-sided cost counted once.
- Per-symbol multiplier from config scales `cost_bps_per_side` only (not sell fee).
- Names carry units (`*_bps`, `*_dec`). Label window starts at open `t+1` (after feature time close `t`).

### Walk-forward CV (`src/jevlab/cv/`)

- `walkforward.py`: expanding train from 2016-01..2018-12; 20 quarterly OOS folds 2019Q1–2023Q4; embargo 5 sessions (`t <= T0 - 5`); inner expanding CV (3 splits, same purge).
- `purge.py`: training label windows must not intersect the test window.
- `sealed_guard.py`: data after 2023-12-29 under `data/sealed/`; load requires `JEV_SEAL_TOKEN` and appends `sealed_access_log`; `filter_dev_frame` drops 2024+ for all dev pipelines.

### Models (`src/jevlab/models/`)

| Model | Behavior |
| ----- | -------- |
| R0 | Score = `-ret_5d` (lowest 5d return ranks highest among eligible) |
| A | `StandardScaler` + L2 logistic on `MODEL_A_COLUMNS` (17); `C ∈ {0.01, 0.1, 1}` by inner-CV log-loss |
| D | LightGBM with fixed small hyperparameters from config; `deterministic=True`, `n_jobs=1` |

All sklearn transforms fit **in-fold** via `Pipeline`. OOS metrics: log-loss, AUC, rank-IC.

### Selection (`src/jevlab/signals/selection.py`)

Top-K eligible per rebalance date; ties broken by symbol order.

### CLI

```bash
jev run baselines [--cost-scenario normal]
```

Writes `reports/phase3_baselines/oos_{R0,A,D}.parquet`, `signals_*.parquet`, and `metrics.json`.

---

## Decisions

1. **Rbar** requires all 13 universe symbols present that decision date; incomplete cross-sections yield NaN Rbar/X/y for that date.
2. **Embargo** is session-count based on the NYSE session list shared with purge checks (label needs open `t+4`).
3. **Dev max date** = 2023-12-29; any pipeline frame is filtered before fit/score; sealed path is opt-in only.
4. **sklearn typing**: mypy overrides for model modules (`ignore_missing_imports` / `follow_untyped_imports`) — sklearn ships without stubs usable under `no-untyped-call`.

No §21A conflicts. Backtesting and Jev were not implemented.

---

## How to run

```bash
# after Phase 1 data + Phase 2 features on disk
jev run baselines --cost-scenario normal
make check
```

---

## Test results

| Check | Result |
| ----- | ------ |
| ruff / format | pass |
| mypy | pass (59 source files) |
| pytest | 67 passed |
| `make check` | pass |

Covered: label window after feature time; train label windows vs test purge; deterministic 20 folds + inner splits; global-fit leakage (scaler differs from full-sample); sealed guard blocks/logs; cost formula `0.00053` and one-time round-trip; no 2024+ rows in dev frames; top-K symbol-order ties; R0 lowest-`ret_5d` ranking.

---

## Known limitations

- Full walk-forward on real features requires Phase 1–2 artifacts on disk; unit tests use synthetic panels.
- No backtest ledger, risk module, or Jev integration in this phase.

---

## Stop

Phase 3 is complete. Do not proceed to Phase 4 until explicitly requested.
