# Phase 6 Report — Jev diagnostics + question freeze

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§11.4, 7.4, 21 Phase 6, 21A  
**Preserved:** Phases 0–5

---

## What was built

### Audit module (`src/jevlab/jev/audit.py`)

Warning-signal diagnostics only — **not proof**, **not pass/fail gates**. Every section states the banner and that historical Jev results are **potentially training-contaminated**.

| Piece | Behavior |
| ----- | -------- |
| Sealed refusal | `assert_audit_dates_allowed` (≤ 2023-12-29); `refuse_sealed_read` for paths under `data/sealed/` |
| Dev sample | 300 rows from 2016–2018; **no labels**; degeneracy check (Noul sd, choice peak fraction) |
| Determinism stats | Per-feature sd / max abs diff across repeat attempts |
| D1 | Era E1/E2/E3 bounds; exact one-sided binomial vs 1/3; multinomial logistic benchmark on `MODEL_A_COLUMNS` |
| D2 | Outcome AUC + rank-IC vs logistic benchmark; date-block bootstrap CI helper; **no threshold** |
| D3 | Rank-IC of Jev features vs X by era (scaffold for post–Phase 7) |
| D5 | `state_v1c` coarse 3-bucket phrases; fine vs coarse IC/AUC comparison |
| Budget | `BudgetGuard` at $0.042/M input tokens; default cap $25 |
| Dry-run | Default path: plan + cost estimate, **zero API calls**, writes `AUDIT_REPORT.md` |

Diagnostic questions load from `config/audit_qset.json` only (never `qset_v1`).

### Freeze tooling

```bash
jev jev freeze-qset
```

Writes `qset_hash` into `config/jev.yaml`, inserts a `trial_ledger` row, prints `git tag qset-v1-frozen` (**does not tag automatically**).

### Human docs

- `docs/contamination_disposition.md` — **TEMPLATE**; decision checkboxes unchecked; code never auto-fills.
- `docs/jev_cutoff.md` — D6 vendor-query log template.

### CLI

```bash
jev jev audit                 # dry-run (default)
jev jev audit --live --budget-usd 25
jev jev freeze-qset
```

Live flag does **not** fan out bulk audit calls automatically (budget-preserving); dry-run report is the Phase 6 artifact.

---

## Decisions

1. Dry-run is the default and the CI-safe path; planned call counts + USD estimate are always printed.
2. Automated `--live` fan-out is refused in the CLI to avoid accidental spend; operators use the plan + budget guard for any later manual live run.
3. `state_v1c` reuses fine vocabulary phrases at low/mid/high indices (3 buckets via edges 1/3, 2/3).

No bulk historical inference. No §21A conflicts.

---

## Test results

| Check | Result |
| ----- | ------ |
| ruff / format | pass |
| mypy | pass (80 source files) |
| pytest | all pass (Phase 0–6 suite) |
| `make check` | pass |

Covered: exact binomial orientation (~227/600 for p<0.01); era boundaries; benchmark chance vs strong signal; sealed-read refusal + post-2023 dates; dry-run zero calls; budget cap; degeneracy flags; freeze-qset hash/ledger; disposition template has no auto-filled decision.

---

## Stop

Phase 6 is complete. Do not proceed to Phase 7 until explicitly requested.
