# Phase 10 Report — Robustness and placebo suite

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§13, 21 Phase 10, 21A  
**Preserved:** Phases 0–9, 13A

---

## What was built

### `evaluation/placebo.py`

| Id | Transform | N (spec) |
| -- | --------- | -------- |
| P1 | Within-date Jev shuffle | 1000 |
| P2 | Global Jev shuffle | 500 |
| P3 | Per-asset circular shift (lag ≥ 60) | 500 |
| P4 | AR(1)-matched Gaussian noise | 500 |
| P5 | In-fold LightGBM MODEL_A → each Jev feature; B with predicted-only and residual-only | 1 |
| P6 | State-shuffle runner (see `jev/audit.py`) | 1 |

- Seeded loops retrain B on placebo Jev columns (market cols / folds / pipeline fixed).
- Optional economic leg via `run_fast` (ΔSharpe).
- Empirical p = `(1 + #{placebo ≥ observed}) / (1 + N)` on ΔlogLoss gain (= logloss(A) − logloss(B)).

### `jev/audit.py` (P6)

`plan_p6_state_shuffle`, `build_p6_shuffled_states`, `run_p6_state_shuffle`: 3,000 mismatched states; dry by default; `--live-p6` + USD budget cap arms the work queue (no silent network spend).

### `evaluation/robustness.py`

Data-only tables (no narrative conclusions):

- Per-feature ablation (drop each of 17 MODEL_A cols)
- Per-Jev-question ablation (drop each of 8 Jev cols)
- Regime splits: SPY vs SMA200, lagged-VIX terciles, 2020, 2022
- Leave-one-ETF-out (+ best-contributor flag)
- Phases 0/1/2, K ∈ {2,3,4}
- Regularization ±1 C-grid step
- Cost sweep (4 scenarios), delayed execution (`open_t1` / `delayed`)
- Choice-order sensitivity (Jev-column permutation proxy)
- Paraphrase plan: 2 paraphrases × 3,000 states (logged as trial; live calls gated)

### CLI

```bash
jev run robustness
jev run robustness --n-p1 1000 --n-p2 500 --n-p3 500 --n-p4 500
jev run robustness --skip-placebos --skip-loo
jev run robustness --live-p6 --budget-usd 25
```

Outputs under `reports/robustness_report/` (`placebo_results.json`, `robustness_data.json`, per-section JSON). Placebo / paraphrase rows written to `trial_ledger` with `purpose=placebo` or `exploratory`.

---

## Acceptance artifact

| Item | Location |
| ---- | -------- |
| Robustness data | `reports/robustness_report/` |
| Fixture note | Synthetic panel with reduced placebo N; full N via CLI when features+Jev parquet exist |
| Contamination label | `potentially_training_contaminated` |
| Narrative conclusions | **null** (data only) |

---

## Tests

`tests/unit/test_phase10_robustness.py`:

- Marginal preservation (P1/P2)
- Circular-shift autocorrelation + marginals (P3)
- Seed determinism (P1/P4)
- Empirical p-value formula
- Small-N placebo loop runtime budget
- P5 / P6 dry / robustness writer + ledger `purpose=placebo`

---

## Checks

```text
make check  →  ruff + mypy + pytest  (all passed)
```

---

## Decisions

1. Observed placebo statistic is ΔlogLoss **gain** (A − B) so the ≥ empirical-p convention matches “larger = more extreme under H1.”
2. P6 and paraphrase live calls stay operator-gated (`--live-p6` / budget); dry mode writes plans only.
3. Choice-order sensitivity uses Jev-column value permutation as a zero-API proxy until paraphrase calls run.
4. Stop after Phase 10 — no Phase 11 reporting work.
