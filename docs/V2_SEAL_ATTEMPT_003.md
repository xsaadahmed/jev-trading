# V2 seal authorization attempt 003 (VOID / CLOSED for implementation r3)

Status: **NOT CONSUMED / CLOSED FOR IMPLEMENTATION R3**. This attempt did not open 2024–2025.

## What was authorized

Implementation revision **r3** was authorized for a one-shot sealed holdout.

| Field | Value |
| ----- | ----- |
| Protocol revision | 1 |
| Implementation revision | 3 (attempted; not executed) |
| Exact execution commit | `1b2648813f81d16c53dbf4ee47ba565177bef35f` |
| Tag | `v2-preseal-implementation-r3` (immutable audit artifact; not moved) |
| Authorized phrase supplied | `OPEN V2 R1 IMPLEMENTATION R3 SEALED HOLDOUT ONCE` |

## Preflight

All listed integrity checks passed. Official `jev v2 final-test --preflight` completed.

| Check | Result |
| ----- | ------ |
| Working tree | clean |
| HEAD | `1b2648813f81d16c53dbf4ee47ba565177bef35f` |
| Design tags | peel to `1cd7e8a6107313a30385134197eeb0dcdb9c7324` |
| Research hashes | match |
| Frozen Model A | match (C=0.01, cutoff=2023-12-22, 19376 rows) |
| Sealed bytes read | **0** |
| Sealed access count | **0** |
| Production sealed source filesystem reads | **0** |
| Prior V2 result / `V2_SEALED_PASS` | absent |
| Seal broken / second look | NO |

## Why the runner was not started

Two structural gaps were found **before** any sealed load:

**A. Sealed `frames` were not attached to the scoring path.**  
`ProductionSealedSource.load()` returns `{configured_path, n_files, frames}`. The r3 runner passed that payload into `construct_combined_study_frames()`, which only consumes `features` / `bars` / `labels` and does not classify `frames`. Sealed 2024–2025 observations would have been silently omitted; evaluation would have continued on pre-2024 history only.

**B. Causal sealed feature construction was not invoked.**  
The real production runner did not call `build_causal_score_features()` (or the equivalent frozen V1 `as_of` pipeline). Even classified sealed bars would not necessarily become frozen Model A feature rows.

Either defect would consume the one-shot seal without a valid confirmatory result.

## What was not observed

- the atomic sealed runner was not started
- no candidate result
- no null result
- no gate was evaluated
- no acceptance artifact
- no `V2_SEALED_PASS`
- 2024–2025 remains unopened

## Runtime seal token vs human phrase

The human authorization phrase is not a secret and is not `JEV_SEAL_TOKEN`.

`JEV_SEAL_TOKEN` is a local runtime environment token required by `SealedDataGuard` before the first access log row. It must be set in the process environment (for example via a gitignored `.env`). It must never be committed, printed, or logged. Preflight reports only present/missing. A missing token must fail before sealed access.

## Disposition

- r3 authorization is **CLOSED for implementation r3** (not consumed; seal unused)
- r1 and r2 authorizations remain VOID/CLOSED
- r3 implementation tag remains an immutable audit artifact
- a **fresh** human authorization will be required for implementation **r4**
- r4 is an implementation correction only; protocol revision remains **1**
- research hashes and the frozen Model A logical digest must not change
