# Phase 9 Report — Statistical evaluation

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§12, 16.3 S4, 21 Phase 9, 21A  
**Preserved:** Phases 0–8, 13A

---

## What was built

### `evaluation/bootstrap.py`

Stationary (Politis–Romano) block bootstrap over **dates** (not rows):

| Parameter | Value |
| --------- | ----- |
| Mean block length | 10 |
| Draws | 10,000 (seeded) |
| Targets | paired Δlog-loss, Δrank-IC, ΔSharpe, ΔCAGR (B − A) |
| Interval | percentile 95% CI |

### `evaluation/tests.py`

| Routine | Spec role |
| ------- | --------- |
| `diebold_mariano` | per-date mean log-loss difference; Newey–West HAC lag 5 |
| `deflated_sharpe_ratio` / `expected_max_sharpe` | Bailey–López de Prado DSR with trial-ledger N |
| `holm_adjust` / `benjamini_hochberg_adjust` | confirmatory + secondary multiplicity |
| `minimum_detectable_effect` | MDE for ΔSharpe and ΔIC given n and A–B correlation |
| `forward_consistency_z` | §16.3 S4: z = (ΔIC_hist − ΔIC_fwd)/SE_fwd; pass if z < 1.28 |

### `evaluation/predictive.py`

Log-loss, Brier, AUC, 10 equal-count calibration bins, reliability table, daily Spearman rank-IC with Newey–West t-stat (lag 5).

### `evaluation/stats_summary.py` + CLI

`jev run stats` builds `reports/phase9_stats/stats_summary.json` for A vs B and appends a **confirmatory** `trial_ledger` row (`feature_set=A_vs_B`). Outputs carry `potentially_training_contaminated`.

---

## Acceptance artifact

| Item | Location |
| ---- | -------- |
| Stats summary | `reports/phase9_stats/stats_summary.json` |
| Bootstrap config | mean_block=10, n_draws=10000, seed=42 |
| Confirmatory trial | `registry/experiments.sqlite` (`purpose=confirmatory`, `A_vs_B`) |

Primary hypothesis packaging: Holm/BH on the two confirmatory tests (Δlog-loss, ΔIC). Point estimates in the committed summary use a **synthetic A/B OOS panel** (Phase 8 lineup parquet not present locally); re-run `jev run lineup` then `jev run stats` against real artifacts before any gate decision.

---

## Tests

`tests/unit/test_phase9_stats.py`:

- Planted-effect bootstrap CI coverage ~95%
- Null DM Type I ~5% over 200 sims; p-values ~uniform (KS)
- DSR identity (N=1) and trial-count deflation
- MDE monotonicity in n and correlation
- Holm/BH + `forward_consistency_z` S4 threshold
- Predictive metrics / calibration / confirmatory ledger write

---

## Checks

```text
make check  →  ruff + mypy + pytest  (all passed)
```

---

## Decisions

1. Bootstrap is in-house (no `arch`); date resampling only.
2. Confirmatory multiplicity is exactly two Holm-adjusted tests (Δlog-loss, ΔIC); DSR uses ledger trial count for secondaries.
3. `forward_consistency_z` is exported for Phase 13B scale evaluation; not a Phase 9 gate.
4. Stop after Phase 9 — no Phase 10 robustness/placebo work.
