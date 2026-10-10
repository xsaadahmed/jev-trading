# Phase 5 Report — Jev integration (mock-first)

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§2, 7, 21 Phase 5, 21A  
**Preserved:** Phases 0–4

---

## What was built

### Client (`src/jevlab/jev/client.py`)

- `httpx` POST to `https://api.typesafe.ai/v1/systemone` with `Bearer` from `JEV_API_KEY`.
- Token-bucket limiter (`rate_limit_per_minute`); tenacity retries on 429/529/5xx honoring `Retry-After`.
- Research mode **refuses** aliases `jev-latest` / `jev-preview` and requires pinned `jev-1.13.0`.
- Structured logs via structlog with secret redaction; no bulk calls.

### Schemas (`jev/schemas.py`)

Pydantic models for §2.3 request / §2.4 response (Noul, Choice, Score) plus documented example contract tests.

### State (`vocabulary.py`, `state_builder.py`, `privacy.py`)

- Closed vocabulary of every allowed `state_v1` key/leaf (bucket phrases + streak words + `fund_type`).
- Builder: 16 feature-derived fields via own-trailing-252 percentile buckets + spelled `down_streak`; constant fund descriptor; **no JSON numbers**.
- `validate_state_privacy(state, vocab)`: closed vocab; ticker rejection; ISO dates / months / years 1990–2100; currency / % / floats / price-like; prohibited keys; no numeric JSON types.

### Questions + cache

- `config/qset_v1.json` — 4 concepts / 5 evaluations (§7.2); `setup_type_b` reverses option order; canonical hash; mismatch aborts when frozen hash configured.
- `config/audit_qset.json` — audit-only (§11.4); `purpose=audit`; never enters `jev_features`.
- `jev/cache.py` — SQLite index + zstd JSONL shards; `request_hash = sha256(canonical_json({route, model_requested, state, questions}))`; immutable; repeats as new attempts.

### Parser (`jev/parse.py`)

Noul p; score expectation + P(level≥3); choice probs averaged across a/b; recomputed Choice/Score/Noul confidence; `j_order_disagree` = total variation distance.

### CLI

```bash
jev jev smoke          # dry-run (no network)
jev jev smoke --live   # exactly ONE real call when explicitly passed
```

`batch_runner.run_bulk_inference` raises `NotImplementedError` (Phase 7).

---

## Decisions

1. Phrase banks are fixed English strings matching §7.1 semantics (seven percentile bins from `bucket_edges`).
2. Score confidence uses the documented ordinal MAD-from-mode formula; Choice uses `(p_max-1/n)/(1-1/n)`.
3. Audit qset responses are tagged `purpose='audit'` in the cache layer; research features come only from `qset_v1`.

No §21A conflicts. No bulk inference or audits run.

---

## Test results

| Check | Result |
| ----- | ------ |
| ruff / format | pass |
| mypy | pass (79 source files) |
| pytest | all pass (Phase 0–5 suite) |
| `make check` | pass |

Covered: mock 200/401/422/429/529/timeout/malformed; retry; cache hit/miss + immutability; hash stability; alias refusal; §2.4 schema example; privacy property test (5,000 random rows) + injected violation types; Choice order-averaging / TV / confidence math; qset reverse-order + hash mismatch.

---

## How to run

```bash
jev jev smoke
# optional one live call (requires JEV_API_KEY):
jev jev smoke --live
make check
```

---

## Stop

Phase 5 is complete. Do not proceed to Phase 6 until explicitly requested.
