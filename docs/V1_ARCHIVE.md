# V1 Study Archive — Closed

**Final conclusion:** Jev does not demonstrate useful predictive value.

**Phase 11 reporting commit:** `7bef0e5`  
**Archival note commit:** `4bd6f29`  
**Study-close commit / tag:** `v1-study-complete` (does **not** move `prereg-v1` / `qset-v1-frozen`)

---

## Shutdown actions

1. **Windows scheduled task** `jevlab-forward-capture` set to **Disabled** (definition retained for audit).
2. Existing V1 forward rows under `data/forward/` and the immutable `forward_registry` row **preserved**.
3. No Phase 12; sealed 2024–2025 **not accessed**; no `HISTORICAL_ACCEPTANCE.json`.

---

## Forward capture (V1)

| Field | Value |
| ----- | ----- |
| deployment_timestamp | `2026-10-16T04:20:04.946315+00:00` |
| FORWARD_START_DATE | `2026-10-16` |
| Sessions captured | 2 (`2026-10-15` shadow-only; `2026-10-16` on/after FORWARD_START) |
| Complete Jev rows | 26 (13 × 2); failures 0 |
| Outcomes scored | **No** |

V1 forward observations must **not** be reused as evidence for a future changed study.

---

## Sealed holdout

> 2024–2025 remains unopened and may only be used by a separately preregistered future study or other explicitly defined one-shot evaluation.

- `sealed_access_log` count = **0**
- Phase 12 never run
- `HISTORICAL_ACCEPTANCE.json` absent

---

## Statuses

| Status | Value |
| ------ | ----- |
| HISTORICAL_RESEARCH_PASS | NO |
| TINY_LIVE_READY | NO |
| SCALE_READY | NO |

---

## Integrity digests

See `docs/archive/V1_ARCHIVAL_MANIFEST.json`. Digests are **integrity digests**, not cryptographic signatures.

Frozen tags remain:

- `prereg-v1` → `eb1a1ad27dcd67df298fb96eb14c7f3dee843137`
- `qset-v1-frozen` → `eb1a1ad27dcd67df298fb96eb14c7f3dee843137`

---

## Local-only backup note

Gitignored research artifacts (e.g. `data/`, `reports/**`, `registry/*.sqlite`) are **not** in git and require separate backup. Never back up or upload `.env` / API keys.
