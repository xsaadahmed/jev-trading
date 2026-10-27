# V2 r4 implementation review

Protocol revision remains **1**. This is implementation revision **4**.

`v2-preseal-implementation-r1`, `v2-preseal-implementation-r2`, and `v2-preseal-implementation-r3` remain immutable audit artifacts and are not execution-capable.

This packet is **not** V2 confirmatory evidence and does **not** authorize sealed access.

## Why implementation r1 was rejected

`v2-preseal-implementation-r1` (`05a9f816c950473a5ed9851aa17555dcfd0619c4`) discarded or ignored the sealed payload and evaluated caller/default passing `V2GateMetrics`. It could write `V2_SEALED_PASS` without computing Model A or the N=2000 random null.

The authorized r1 sealed attempt correctly stopped before any 2024-2025 read. Sealed access count stayed **0**.

## Why implementation r2 was rejected

`v2-preseal-implementation-r2` (`8197cd906752c40f18129fa372e56b6013fc1079`) implemented a real computational path, but two blockers stopped the authorized attempt **before sealed access**:

**A.** Authorization phrase mismatch.

**B.** The r2 production path attached only `ProductionSealedSource` (2024-2025) and did not attach allowed pre-2024 training/context.

Recorded in `docs/V2_SEAL_ATTEMPT_002.md`. r2 authorization is VOID/CLOSED. Zero sealed bytes were read. Access count stayed **0**.

## Why implementation r3 was rejected

`v2-preseal-implementation-r3` (`1b2648813f81d16c53dbf4ee47ba565177bef35f`) correctly split sources and prebuilt Model A, but two structural gaps were found **before sealed access**:

**A.** `ProductionSealedSource.load()` returns `{configured_path, n_files, frames}`. The r3 runner passed that bag into `construct_combined_study_frames()`, which only consumes `features` / `bars` / `labels`. Sealed 2024-2025 observations would have been silently omitted.

**B.** The real production runner never called `build_causal_score_features()`. Even classified sealed bars would not necessarily become frozen Model A feature rows.

Either defect would consume the one-shot seal without a valid confirmatory result. The r3 authorization attempt correctly STOPPED BEFORE SEALED ACCESS. Recorded in `docs/V2_SEAL_ATTEMPT_003.md`. r3 authorization is NOT CONSUMED / CLOSED FOR IMPLEMENTATION R3. Zero sealed bytes were read. Access count stayed **0**.

## Proof all three attempts had zero sealed access

| Attempt | Commit | Preflight filesystem reads | Sealed access count | 2024-2025 opened |
| ------- | ------ | -------------------------- | ------------------- | ---------------- |
| r1 | `05a9f81` | 0 | 0 | NO |
| r2 | `8197cd9` | 0 | 0 | NO |
| r3 | `1b26488` | 0 | 0 | NO |
| r4 freeze (this packet) | tag `v2-preseal-implementation-r4` | 0 | 0 | NO |

Real registry `sealed_access_log` count at r4 freeze: **0**.
No `V2_SEALED_PASS`. No confirmatory `SEALED_RESULT`. Seal broken = NO. Second look = NO.

## Implementation r4

- Tag: `v2-preseal-implementation-r4`
- Implementation commit: see freeze JSON `r4_commit`
- Branch: `v2-baseline-validation`
- Protocol revision: **1**
- Implementation revision: **4**
- r1, r2, and r3 tags were **not** moved

Architectural correction only. No research semantic changed.

1. Typed `ProductionSealedPayload`. After sealed read, every loaded frame is classified into bars or VIX. Anonymous `"frames"` bags never enter analytics.
2. Unknown / unexpected / duplicate / wrong-symbol / schema-violating frames fail closed.
3. Once the seal is opened, zero sealed scoring rows is an execution-integrity failure. No historical-only fallback. No gates. No `V2_SEALED_PASS`.
4. Combined causal market history = allowed pre-2024 context plus sealed observations. Duplicate keys keep sealed. Deterministic sort. Audit `row_origin` is not a model feature.
5. The real runner calls `build_causal_score_features()` (frozen V1 `as_of` / `available_at <= decision_time(t)`).
6. Scoring rows are sealed-period only. Historical context warms trailing windows and is not itself scored.
7. Targets/returns come from the same combined store. Incomplete t+4 windows, including late-2025 rows that would need 2026, are dropped as frozen. No 2026 access.
8. Exact prebuilt Model A is loaded. No `.fit()`, no `fit_transform`, no inner CV, no C search after sealed read.
9. Source provenance is audit metadata only.
10. Strong two-source pseudo-sealed rehearsal uses the same payload class as `ProductionSealedSource`.

## Explicit production data-flow

```
human r4 authorization + JEV_SEAL_TOKEN present
→ sealed payload load
→ classify_production_sealed_payload → ProductionSealedPayload
→ merge historical context + sealed market (sealed wins duplicates)
→ build_causal_score_features()
→ sealed eligibility / feature rows
→ labels and execution returns from combined bars
→ frozen prebuilt Model A predict / predict_proba
→ phase 0 / 1 / 2 candidate backtests
→ N=2000 random null
→ gates
→ diagnostics
→ artifacts
→ optional acceptance
```

No sealed payload may be discarded. No silent historical-only fallback exists.

## Unchanged frozen research hashes

| Item | Hash |
| ---- | ---- |
| research config | `e436ed977a02a39a0a280d78dcad74fa67e3ff1276f18a3430f249a9aeaf13ce` |
| candidate | `aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28` |
| random null | `9365e46d2ffc657d2b078cd851b90bac8f8ace3357313743769efdabf6498b2e` |
| acceptance gates | `5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be` |
| seed mapping | `5c773811984b2fea232889295560f3ded3e6db79ca8e949ee5518933f88e58f9` |

Live hashes match. Protocol revision = 1.

## Frozen prebuilt Model A (unchanged)

Vintage implementation revision of the artifact remains **3**. r4 loads it; r4 does not rebuild or refit it.

| Field | Value |
| ----- | ----- |
| artifact kind | `v2_sealed_test_model` |
| integrity digest | `98c77b0e270be8ef11d3c8d29bef9b8d645e86a2e1d86a5bf43a4fa59609ca6d` |
| model logical digest | `74e301458559a511229b92235628db17cca80e4ebdff016a792784cab14ccc6c` |
| preprocessing digest | `2ea88b000718f053e94fcd07bdc742220b90b3263f039d9756999901716598e9` |
| training-row digest | `d397b9f3ac093ad30e04de982208072db69b449fd62766a4fd28929fac30881d` |
| training/context manifest digest | `79ddc0caf350bc816a509433dd43baf5c70339a7000433026eb275dadb07266f` |
| chosen C | **0.01** |
| final training decision date | **2023-12-22** |
| training observation count | **19376** |
| fit count | **1** |
| 2024/2025 training rows | **0 / 0** |
| contains 2024-2025 | **false** |

## Model A equivalence (2019-2023, not V2 evidence)

- Eligible rows: **11470**
- Folds: **20**, identical IDs
- C = **0.01** every fold
- Prediction max abs diff: **0.0**
- Round trips: **1187 / 1182 / 1188**
- Sharpe: `0.5827386645170073`
- CAGR: `0.07775255992272519`
- Ledger-fast equal: **true**

## Pseudo-sealed data-flow oracle

Same production payload type, combiner, causal builder, prebuilt-model load, scoring, candidate backtest, N=2000, and gate derivation. Only the underlying date range differs.

| Count | Value |
| ----- | ----- |
| raw pseudo-sealed rows | 80 |
| classified rows | 80 |
| rows in combined history (sealed origin) | 64 |
| rows reaching feature builder | 64 |
| valid pseudo-sealed feature rows | 64 |
| candidate scoring rows | 64 |
| first candidate date | 2021-02-03 |
| last candidate date | 2021-02-25 |
| predictions | 48 |
| completed trades | 72 |
| unknown frames | 0 |
| canary seen combined / features / scoring | true / true / true |
| future perturbation | earlier features unchanged |
| truncate-and-compare | features ≤ t match full build |
| historical-only fallback | false |
| causal feature builder invoked | true |

If the pseudo-sealed source is removed or contributes only historical dates, candidate scoring fails closed.

## N=2000 / RNG / oracle / gates

- N: **2000 / 2000** complete; dropped draws **0**
- Null-vector digest: `463a6e9b7d59464199663aa36d519af7af3a75c62457f9aaa573dfbe98e9aa0e`
- Deterministic repeat: **true**
- Oracle gate comparison: **true**
- 6000/6000 unique revision-1 child-seed streams
- Same primary statistic, p-value formula, E1-E5 / S1-S3 / Z1, diagnostic-only robustness
- Fit after sealed read: **false**
- `V2_SEALED_PASS` written: **false**
- Preflight sealed underlying reads: **0**
- Preflight training/context reads: **1**

The throwaway rehearsal model may select a synthetic C (here 0.1). That is not the frozen production Model A, whose chosen C remains **0.01** and whose logical digest is unchanged.

## Authorization

Frozen r4 phrase (exact): `OPEN V2 R1 IMPLEMENTATION R4 SEALED HOLDOUT ONCE`

- exact phrase accepted only for r4
- r1 phrase rejected (VOID)
- both r2 phrases rejected (VOID)
- r3 phrase rejected (VOID)
- typo / case-change rejected
- r4 phrase cannot authorize r1, r2, or r3 tags/commits

Prior r1/r2/r3 authorizations remain VOID/CLOSED. A **fresh** human authorization is required for r4.

## Runtime seal-token policy

The human authorization phrase is not a secret and is not `JEV_SEAL_TOKEN`.

`JEV_SEAL_TOKEN` is a local runtime environment token required by `SealedDataGuard` before the first access-log row.

- set in the process environment (for example via a gitignored `.env`)
- never committed
- never printed or logged
- preflight reports only present/missing
- missing token fails before sealed access

## Preflight zero sealed reads

Preflight may inspect code, tags, hashes, the frozen model, pre-2024 context, and path configuration. It must not open sealed files, enumerate sealed data, classify real sealed frames, or count real sealed rows.

Expected real sealed underlying reads during preflight: **0**.

## Atomicity / second-look

Fault injection after access record, sealed payload load, payload classification, historical/sealed merge, causal feature build, label build, model load, candidate prediction, candidate backtests, random null, gates, report, and acceptance rename: seal consumed, no acceptance, restart is second-look only.

## Remaining known issues

- Adjacent-C and horizon-sensitivity robustness variants remain diagnostic-only; they do not refit a second canonical model after sealed access.
- Late sealed-window labels/trades still follow frozen V1 boundary semantics (no 2026 completion).
- Pseudo-seal rehearsal uses a synthetic pre-2024 panel for N=2000 software validation; the production Model A artifact was fit on the real pre-2024 hive and is unchanged.
- Real 2024-2025 remains unopened. No V2 result exists. No `V2_SEALED_PASS`.

integrity_digest is an integrity digest, not a cryptographic signature.
