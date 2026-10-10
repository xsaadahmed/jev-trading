# Phase 12 Report — Sealed final test / HISTORICAL_RESEARCH_PASS

**Date:** 2026-10-08  
**Status:** Complete (implemented + fixture-tested; **not** executed for real)  
**Spec:** MASTER_SPEC §§16.1, 10.7, 21 Phase 12, 21A  
**Preserved:** Phases 0–11, 13A

---

## What was built

### `experiments/artifacts.py`

- `write_acceptance(kind, payload, ...)` → integrity-digested JSON under `reports/acceptance/`
- `verify_acceptance(path, ...)` — digest, expiry, config lineage
- `archive_model_artifact(...)` for `kind='sealed_test'` (**never overwritten**)
- Fields: `status`, `created_at`, `expires_at` (180d historical), `git_commit`, `research_config_hash`, `qset_hash`, `model_resolved`, `gate_report_sha256`, `payload`, `integrity_digest = sha256(canonical_json(all other fields))`
- Explicitly an **integrity digest, not a signature** (never called “signed”)
- Historical payload includes: *“Historical Jev evidence is potentially training-contaminated; this status does not establish that the edge is genuine.”*

### `experiments/acceptance.py`

- `HistoricalGateMetrics` + evaluators for **G1–G10** (§16.1)
- Non-evaluable (`None` inputs) blocks the artifact
- G10 includes sealed placebo P1 protocol (N=500) and **exactly one** sealed access
- **No forward-session count** in the evaluation
- `fit_sealed_test_models` — train A/B on rows `date <= 2023-12-29` only; archive permanently
- `run_final_test` — preconditions, seal state, gate report, acceptance or rejection
- Second look: requires `--allow-second-look`, stamps **SEAL BROKEN**, blocks `HISTORICAL_ACCEPTANCE.json`

### Preconditions (`jev run final-test`)

Clean git (optional skip for fixtures), tags `prereg-v1` + `qset-v1-frozen`, trial-ledger snapshot hash, human SEAL token (`JEV_SEAL_TOKEN`), contamination disposition ≠ `stop`.

### CLI

```bash
jev run final-test --seal-token <token> --fixture-metrics tests/fixtures/phase12_metrics_pass.json --skip-git-checks
jev run final-test --seal-token <token> --allow-second-look --fixture-metrics ...
```

Live sealed Jev (~6,500 states) is **not** auto-executed; fixture metrics drive gate evaluation for this phase.

---

## Tests

`tests/unit/test_phase12_final_test.py`:

- Each of G1–G10 failing independently blocks the artifact  
- Non-evaluable gate blocks  
- Second-run / SEAL BROKEN behavior  
- Tampered acceptance fails `verify_acceptance`  
- Tampered gate report sha mismatch  
- Sealed access count exactly once  
- Sealed-test models refuse dates after 2023-12-29 and refuse overwrite  
- Disposition `stop` blocks  
- Gate report asserts no forward-session dependency  

---

## Checks

```text
make check  →  ruff + mypy + pytest  (all passed)
```

---

## Decisions

1. Fixture-first: real sealed spend is out of scope for this phase’s automated path.  
2. Seal state file + access log jointly enforce one-shot / second-look rules.  
3. Stop after Phase 12 — no Phase 13B production-forward / SCALE work.
