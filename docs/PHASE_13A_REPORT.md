# Phase 13A Report — Forward INPUT capture (shadow logging; no orders)

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§11.4, 16.3, 21 Phase 13A, 21A  
**Preserved:** Phases 0–7 (1, 2, 5, 6, 7 required and present)

---

## What was built

### `experiments/forward_registry.py`

Immutable single-row deployment registry:

| Field | Role |
| ----- | ---- |
| `deployment_timestamp` | UTC wall clock of first successful complete capture |
| `jev_model_id` | Pinned research model id |
| `qset_hash` | Research qset content hash |
| `feature_version` / `state_schema_version` | `v1` / `state_v1` |
| `git_commit` | `git rev-parse HEAD` at registration |
| `forward_start_date` | First NYSE session with `decision_time` **after** deployment |

`register_deployment(...)` raises `ForwardRegistryImmutableError` on a second write.

### `experiments/forward_capture.py`

`run_forward_capture` / CLI `jev run forward-capture`:

1. Calendar-aware session resolve; holidays/weekends → `skipped_holiday`
2. Gate at `decision_time` = session **18:30 ET** (`too_early` before)
3. Idempotent per date (`skipped_idempotent` when `status.json` is `complete`)
4. Download bars + VIX with `available_at` / `retrieved_at` (Tiingo + FRED; Alpaca off)
5. Features via `as_of(decision_time)`; `state_v1` + privacy validation
6. Jev with pinned id; **abort Jev part** on resolved-id mismatch; **never impute**
7. Store under `data/forward/YYYY-MM-DD/` (features, states, raw responses, failures, status)
8. First complete success writes `forward_registry`

**Explicit non-goals (enforced):** no orders; no `jevlab.brokers` imports; no model scoring; no label joins.

### Scheduling docs

`docs/FORWARD_CAPTURE.md` — cron, launchd, and Windows Task Scheduler notes.

### Supporting change

`targets/__init__.py` lazy-loads label constructors so feature/eligibility imports (used by capture) do not pull `targets.labels` into memory. Eligibility remains eager.

### CLI

```bash
jev run forward-capture
jev run forward-capture --session YYYY-MM-DD
jev run forward-capture --no-download
```

---

## Tests (`tests/unit/test_forward_capture.py`)

| Invariant | Coverage |
| --------- | -------- |
| Idempotent per date | Second run → `skipped_idempotent`, zero Jev calls |
| Holiday skip | Thanksgiving / non-session → `skipped_holiday` |
| Jev failure → incomplete | No registry write; empty responses; failures logged; never imputed |
| Model-id mismatch | Aborts after first bad resolve; incomplete; no success rows |
| Registry immutability | Second `register_deployment` raises |
| Import boundary | AST + runtime: no `jevlab.brokers` |
| No scoring/labels reachable | AST forbids models/labels/signals/…; runtime forbids those modules |

---

## Checks

```text
ruff check          OK
ruff format --check OK
mypy src/jevlab     OK (88 source files)
pytest -q           OK (full suite)
```

---

## How to run (ops)

See `docs/FORWARD_CAPTURE.md`. Typical weekday job at **18:45 America/New_York**:

```bash
uv run jev run forward-capture
```

Requires `JEV_API_KEY`, `TIINGO_API_KEY`, `FRED_API_KEY` in the environment. Acceptance target (ops, not gated in CI): **10 consecutive clean sessions** with registry populated.

---

## Decisions / notes

1. **`FORWARD_START_DATE`** follows the spec literally: first session whose 18:30 ET decision time is strictly after `deployment_timestamp`. The first captured session may therefore precede the clean-forward window if deployment is stamped after that session’s decision time.
2. Forward evidence is valid **only** for the pinned Jev version in the registry; a new version restarts the window (§11.4 / §16.3 S8).
3. Historical Jev results remain **potentially training-contaminated**; forward data is the only clean evidence.
4. Incomplete days are retriable; complete days are immutable for that date.

---

## Known limitations

- Live acceptance (10 consecutive clean sessions) is an ops milestone, not asserted in CI.
- Warm-up for state percentiles still requires substantial feature history in the store for production runs.
- Phase 13B (scoring / labels / scale evaluator) is **not** implemented.

---

## Stop

Phase 13A complete. Did not proceed to Phase 8 or 13B.
