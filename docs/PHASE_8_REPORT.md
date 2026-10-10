# Phase 8 Report — Jev-enhanced models and attribution lineup

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§4.3, 9, 21 Phase 8, 21A  
**Preserved:** Phases 0–7 (and existing Phase 13A forward-capture scaffolding)

---

## What was built

### Column-set registry (`models/column_sets.py`)

| Model | Columns | n | Learner | Purpose |
| ----- | ------- | - | ------- | ------- |
| A | `MODEL_A_COLUMNS` | **17** | logistic | confirmatory |
| B | A + 8 Jev | **25** | logistic (same grid/folds/seed as A) | confirmatory |
| C | 8 Jev | **8** | logistic | exploratory |
| D | A | **17** | LightGBM | exploratory |
| E | A + 8 Jev | **25** | LightGBM | exploratory |
| F | A + 8 placebo slots | **25** | logistic | exploratory; **interface only** (Phase 10) |
| G | A + drop-first bucket one-hots | **17 + N** (N logged) | logistic | exploratory |

Asserted in code: `B = A + JEV_FEATURE_COLUMNS`, sizes above, A/B share pipeline/grid/folds/seed.

### Transforms (`models/transforms.py`)

In-fold `SelectiveLogitClip` (clip 0.01..0.99 → logit) on Jev probability columns, then `StandardScaler` + learner. `j_downside_E` (score expectation) is not logit-transformed.

### Controls (`models/controls.py`)

- Model F: `placebo_*` column names (NaN placeholders).
- Model G: drop-first one-hots of state buckets; dummy count logged in metadata.

### Runner (`experiments/runner.py`)

`run_model_set`: folds → OOS predictions → top-K selection → backtests under all cost scenarios → comparison table → `manifest.json` → `trial_ledger` (confirmatory for A vs B only; else exploratory). All outputs labeled **potentially_training_contaminated**.

### CLI

```bash
jev run lineup
jev run lineup --models A,B,C --skip-backtests
```

---

## Comparison table

Produced at `reports/phase8_lineup/comparison_table.md` (and `.csv`) when the operator runs `jev run lineup` against local features + optional `jev_features`. The table reports metrics only — **no conclusion language**.

Synthetic fixture run (tests) confirms the table header and contamination label; live numbers require the feature store + Phase 7 parquet.

---

## Tests

`tests/unit/test_phase8_lineup.py`:

- A/B column-diff + asserted sizes
- identical fold indices / shared OOS keys
- logit transform
- ledger confirmatory + exploratory entries
- manifest completeness
- sealed-date refusal
- end-to-end `run_model_set` (no conclusion wording)

---

## Checks

```text
make check  →  ruff + mypy + pytest
```

---

## Decisions

1. Confirmatory purpose is reserved for the A vs B paired ledger row; C/D/E/G are exploratory; F is skipped until Phase 10 fills placebo values.
2. Comparison artifacts contain metrics and the contamination label only — no pass/fail or narrative conclusions.
3. Existing Phase 13A `forward-capture` CLI/registry left intact.

---

## Stop

Phase 8 complete. Do **not** auto-continue to Phase 9.
