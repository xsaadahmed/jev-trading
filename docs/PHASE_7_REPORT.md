# Phase 7 Report — Bulk historical Jev inference

**Date:** 2026-10-08  
**Status:** Complete (implementation + dry-run; no live bulk spend)  
**Spec:** MASTER_SPEC §§7, 18, 21 Phase 7, 21A  
**Preserved:** Phases 0–6

---

## What was built

### Preconditions (`src/jevlab/jev/preconditions.py`)

Enforced before `jev jev batch` (dry-run or `--execute`):

| Check | Rule |
| ----- | ---- |
| qset hash | `load_qset(..., expected_hash=jev.yaml qset_hash)`; mismatch aborts |
| Tags | `qset-v1-frozen`, `prereg-v1` must exist |
| Working tree | `git status --porcelain` empty |
| Disposition | `docs/contamination_disposition.md` has exactly one checked decision ≠ `stop` |

### Batch runner (`src/jevlab/jev/batch_runner.py`)

| Piece | Behavior |
| ----- | -------- |
| Windows | Dev `2016-01-04..2023-12-29` + interim `2026-01-02..` day before `FORWARD_START` (or latest completed session if `forward_registry` empty) |
| Sealed | `2024-01-02..2025-12-31` excluded; `SealedBulkError` if present — Phase 7 never uses a seal token |
| Dedupe | By `request_hash`; identical states share one call |
| Cache | Hits skip the API; responses immutable |
| Concurrency | Bounded `ThreadPoolExecutor` (default 4) |
| Checkpoint | SQLite status / latency / tokens / model_resolved; safe resume |
| Budget | Abort if projected spend for uncached calls exceeds cap |
| Model pin | Abort if `model_resolved !=` pinned id |
| Missing | Logged as errors; **never imputed** into `jev_features` |
| Output | `data/features/jev/qset_v1/jev_features.parquet` + `metadata.json` tagged `potentially_training_contaminated` |

### CLI

```bash
jev jev batch                 # dry-run: counts + cost estimate, zero API calls
jev jev batch --execute       # live bulk (preconditions + budget guard)
jev jev batch --budget-usd 25 --concurrency 4
```

---

## Dry run (upper bound)

No live API calls. Planning estimate at 800 tokens/call × $0.042/M input tokens:

| Metric | Value |
| ------ | ----- |
| upper_bound_states | **28,600** |
| sealed_excluded | 2024-01-02 .. 2025-12-31 |
| estimated_cost_usd | **$0.9610** |
| default_budget_usd | $25.00 |
| api_calls | **0** |
| contamination_tag | potentially_training_contaminated |

Operator CLI dry-run against the feature store additionally requires a frozen `qset_hash`, tags, clean tree, and a human disposition ≠ `stop`. This phase did **not** run `--execute`.

---

## Tests

`tests/unit/test_jev_batch.py`:

- kill-and-resume (checkpoint skips completed hashes)
- dedupe (identical states → one API call; both rows get features)
- budget guard (projected spend aborts)
- sealed refusal
- wrong-hash / disposition-`stop` refusal
- row-count reconciliation
- missing responses stay missing (no impute)
- model_resolved mismatch abort
- cache-hit skip
- dry-run zero calls
- upper-bound cost under $25

---

## Checks

```text
make check  →  ruff + mypy + pytest  (see session log)
```

---

## Decisions

1. Dry-run is the default; `--execute` is required for live spend.
2. Sealed years are hard-refused; no seal-token path exists in Phase 7.
3. Features metadata always carries `potentially_training_contaminated`.
4. Live bulk is left for a human operator after freeze + disposition + tags.

---

## Stop

Phase 7 implementation complete. Do **not** auto-continue to Phase 13A.
