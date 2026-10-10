# Phase 13B Report — Production-forward models, forward scoring, SCALE evaluator

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§10.7, 16.3, 21 Phase 13B, 21A  
**Preserved:** Phases 12 (sealed final-test) and 13A (forward INPUT capture) required and present  
**Non-goals:** No orders; no `jevlab.brokers` imports; V1 capital ceiling unchanged at $10.00

---

## What was built

### `experiments/production_models.py`

`jev run build-production-models`:

1. Requires existing `sealed_test` A/B artifacts (untouched).
2. Retrains the **exact frozen procedure** (same columns, logistic class, C/seed, target, embargo) on all pre-deployment rows with labels known by deployment (embargo applied).
3. Refuses any training row with `date >= forward_start_date`.
4. Archives `kind='production_forward'` with hash, `train_end`, column lists, git commit, and pickled pipelines.
5. Records hashes in `trial_ledger` and `data/forward/production_model_hashes.json` **before** any forward-label join.

**Documented rationale (§10.7):** The sealed evaluation measured the ex-ante frozen procedure trained <= 2023 and scored on held-out sealed data; that artifact stays untouched. Refitting the same procedure on more pre-deployment data after the seal is opened does not alter that evidence and is evaluated only on forward data after the training cutoff.

### `experiments/forward_eval.py`

| Command | Role |
| ------- | ---- |
| `jev run forward-score` | Score captured forward inputs with production A/B; store scores, top-K, hypothetical trades under normal costs |
| `jev run forward-report` | Join realized labels when windows complete (hash guard); rolling A-vs-B IC/log-loss/net returns; KS drift of 8 Jev features vs 2019-2023; automatic 60-session checkpoint (**informational only**) |
| `jev run evaluate-scale` | Evaluate S1-S8; write `SCALE_ACCEPTANCE.json` (integrity digest, parent = TINY_LIVE digest, expires 90 days) only when all hold |

**Label-join guard:** refuses if production hashes are absent or newer than the first label join.

**Ceiling invariant:** `SCALE_ACCEPTANCE` does not modify `TINY_LIVE_CEILING_USD` (remains 10.00); V1 live code gains no ability to exceed $10.00.

### CLI

```bash
jev run build-production-models --train-parquet PATH
jev run forward-score [--session YYYY-MM-DD] [--scores-parquet PATH]
jev run forward-report --scores-parquet ... --labels-parquet ... --baseline-jev ... --forward-jev ...
jev run evaluate-scale --fixture-metrics PATH
```

---

## Tests (`tests/unit/test_phase13b_forward_eval.py`)

| Invariant | Coverage |
| --------- | -------- |
| Production vs sealed differ only by training sample | Columns/C/seed identical; digests and `train_end` differ |
| No forward row in training | Frame with `date >= forward_start` refused |
| Label join refused before hashes | `ForwardLabelJoinError` when hashes absent |
| Hashes newer than first join refused | Rewrite after join → refuse |
| S1-S8 each blocking independently | Parametrized sole-failure per gate |
| Digest verification | Tamper fails `verify_acceptance` |
| 60-session checkpoint cannot grant status | `grants_scale_ready=false`; no acceptance write |
| No broker imports | AST check on production_models / forward_eval |
| Sealed required before production | Missing sealed_test blocks build |

---

## Checks

```text
ruff check          OK
ruff format --check OK
mypy src/jevlab     OK (103 source files)
pytest -q           OK (full suite)
make check          OK
```

---

## Acceptance mapping

| Spec item | Status |
| --------- | ------ |
| Production-forward artifacts built (fixture) | Pass |
| Sealed_test untouched | Pass |
| Hashes before label join | Pass |
| Forward report + KS + 60-session informational | Pass |
| SCALE evaluator per-condition blocking + digest | Pass |
| No orders / no broker / no $10 exceed path | Pass |

**Stop:** Phase 13B only. Phase 14 (live infrastructure) not started.
