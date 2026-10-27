# V2 r3 implementation review

Protocol revision remains **1**. This is implementation revision **3**.

`v2-preseal-implementation-r1` and `v2-preseal-implementation-r2` remain immutable audit artifacts and are not execution-capable.

This packet is **not** V2 confirmatory evidence and does **not** authorize sealed access.

## Why implementation r1 was rejected

`v2-preseal-implementation-r1` (`05a9f816c950473a5ed9851aa17555dcfd0619c4`) discarded or ignored the sealed payload and evaluated caller/default passing `V2GateMetrics`. It could write `V2_SEALED_PASS` without computing Model A or the N=2000 random null.

The authorized r1 sealed attempt correctly stopped before any 2024-2025 read. Sealed access count stayed **0**.

## Why implementation r2 was rejected

`v2-preseal-implementation-r2` (`8197cd906752c40f18129fa372e56b6013fc1079`) implemented a real computational path, but two blockers stopped the authorized attempt **before sealed access**:

**A.** The supplied authorization phrase (`OPEN V2 R1 IMPLEMENTATION R2 SEALED HOLDOUT ONCE`) did not match the exact phrase frozen into r2 (`OPEN V2 REVISION 1 SEALED HOLDOUT ONCE WITH IMPLEMENTATION R2`).

**B.** The r2 production path attached only `ProductionSealedSource` (2024-2025). It did not attach the already-allowed pre-2024 training/context stores required to fit frozen Model A. Executing would have consumed the one-shot seal and then failed at training.

Recorded in `docs/V2_SEAL_ATTEMPT_002.md`. r2 authorization is VOID/CLOSED. Zero sealed bytes were read. Access count stayed **0**.

## Proof both authorization attempts had zero sealed reads

| Attempt | Commit | Preflight filesystem reads | Sealed access count | 2024-2025 opened |
| ------- | ------ | -------------------------- | ------------------- | ---------------- |
| r1 | `05a9f81` | 0 | 0 | NO |
| r2 | `8197cd9` | 0 | 0 | NO |
| r3 freeze (this packet) | tag `v2-preseal-implementation-r3` | 0 | 0 | NO |

Real registry `sealed_access_log` count at r3 freeze: **0**.
No `V2_SEALED_PASS`. No confirmatory `SEALED_RESULT`. Seal broken = NO. Second look = NO.

## Implementation r3

- Tag: `v2-preseal-implementation-r3`
- Branch: `v2-baseline-validation`
- Protocol revision: **1**
- Implementation revision: **3**
- r1 and r2 tags were **not** moved

Architectural correction only:

1. `PreSealedTrainingSource` may read pre-2024 training/context without consuming the seal.
2. `ProductionSealedSource` remains unread until a fresh r3 authorization is consumed.
3. `jev v2 build-sealed-model` fits exactly one Model A before any 2024-2025 access.
4. The sealed runner loads that immutable artifact and does not `.fit()` after sealed read.
5. Causal scoring combines allowed historical context with opened sealed observations (`available_at <= decision_time(t)`).
6. End-2025 decisions are evaluable only when t+1 entry and t+4 exit exist inside 2025-12-31. No 2026 completion.

## Unchanged frozen research hashes

| Item | Hash |
| ---- | ---- |
| research config | `e436ed977a02a39a0a280d78dcad74fa67e3ff1276f18a3430f249a9aeaf13ce` |
| candidate | `aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28` |
| random null | `9365e46d2ffc657d2b078cd851b90bac8f8ace3357313743769efdabf6498b2e` |
| acceptance gates | `5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be` |
| seed mapping | `5c773811984b2fea232889295560f3ded3e6db79ca8e949ee5518933f88e58f9` |

Live hashes match. Protocol revision = 1.

## Prebuilt sealed-test Model A

Built from allowed pre-2024 development stores only. Contains **no** 2024-2025 information.

| Field | Value |
| ----- | ----- |
| artifact kind | `v2_sealed_test_model` |
| integrity digest | `74e188001985d879135c0e99b5417235d5659a9b104d8b2be57b4286b896105c` |
| model logical digest | `74e301458559a511229b92235628db17cca80e4ebdff016a792784cab14ccc6c` |
| preprocessing digest | `2ea88b000718f053e94fcd07bdc742220b90b3263f039d9756999901716598e9` |
| training-row digest | `d397b9f3ac093ad30e04de982208072db69b449fd62766a4fd28929fac30881d` |
| training/context manifest digest | `79ddc0caf350bc816a509433dd43baf5c70339a7000433026eb275dadb07266f` |
| chosen C | **0.01** |
| final training decision date | **2023-12-22** |
| training observation count | **19376** |
| fit count | **1** |
| 2024/2025 training rows | **0 / 0** |
| deterministic rebuild | identical C, row digest, preprocessing, coefficients, logical digest |

Frozen in `config/v2/SEALED_TEST_MODEL_FREEZE.json`. If the artifact changes, r3 authorization fails closed.

## Model A equivalence (2019-2023, not V2 evidence)

- Eligible rows: **11470**
- Folds: **20**, identical IDs
- C = **0.01** every fold
- Prediction max abs diff: **0.0**
- Selections identical
- Trades identical
- Round trips: **1187 / 1182 / 1188**
- Sharpe: `0.5827386645170073`
- CAGR: `0.07775255992272519`
- Ledger-fast equal: **true**

## Production-path pseudo-seal rehearsal

Two real source classes: pseudo pre-sealed training/context and pseudo sealed scoring. Prebuild model, freeze/hash, preflight with zero pseudo-seal reads, authorize r3 phrase, open pseudo seal once, score with prebuilt model, N=2000, derive gates, oracle check.

- Label: `development_rehearsal_only`
- Pre-seal period: 2021-06-01 to 2021-06-24
- Pseudo-sealed period: 2021-06-25 to 2021-07-21
- N: **2000 / 2000** complete; dropped draws **0**
- Null-vector digest: `cd357551102529816662f14214499a24e9fdbf422969e722d901d476d77915ef`
- Deterministic repeat: **true**
- Oracle gate comparison: **true**
- Fit count: **1**; fit after sealed read: **false**
- Training rows from 2024/2025: **0 / 0**
- Preflight training/context reads: **1**
- Preflight sealed underlying reads: **0**
- `V2_SEALED_PASS` written: **false**

## N=2000 / RNG / oracle / E5 / concentration

- 6000/6000 unique revision-1 child-seed streams (frozen invariant)
- Independent oracle recomputed candidate phase Sharpes, mean Sharpe, null median, empirical p, Sharpe advantage, CAGRs, MDD magnitude, phase wins, S2/S3 from saved artifacts; equal within 1e-12
- E5: `-0.35` PASS; `-0.3500001` FAIL; `-0.80` FAIL
- S2/S3 use total positive completed-trade P&L; missing metrics fail closed

## Placeholder-metric regression

- Production `V2GateMetrics` defaults are `None` (fail closed)
- `run_v2_final_test` has **no** `metrics` parameter
- Production CLI does not import or construct `V2GateMetrics`
- Gate values derive only from canonical result artifacts

## Authorization phrase tests

Frozen r3 phrase (exact): `OPEN V2 R1 IMPLEMENTATION R3 SEALED HOLDOUT ONCE`

- exact phrase accepted
- r1 phrase rejected (VOID)
- both r2 phrases rejected (VOID)
- typo / case-change rejected
- r3 phrase cannot authorize r1 or r2 tags/commits

Prior r1 and r2 authorizations remain VOID/CLOSED. Fresh human authorization is required for r3.

## Atomicity / second-look

Fault injection after sealed-access record, sealed load, feature construction, prebuilt-model load, prediction, each candidate phase, early/mid/final null, gate derivation, report writing, and before acceptance write: seal consumed, no acceptance artifact, restart is second-look only.

## Preflight zero sealed reads

The complete r3 preflight may read:

- pre-2024 training/context
- frozen model artifact
- manifests/hashes

It must not read 2024-2025 sealed contents.

Expected real sealed underlying reads during preflight: **0**.

## Remaining known issues

- Adjacent-C and horizon-sensitivity robustness variants remain diagnostic-only; they do not refit a second canonical model after sealed access.
- Late sealed-window labels/trades still follow frozen V1 boundary semantics (no 2026 completion).
- Pseudo-seal rehearsal uses a short pre-2024 synthetic panel for N=2000 software validation; the production Model A artifact was fit on the real pre-2024 hive.
- Real 2024-2025 remains unopened. No V2 result exists. No `V2_SEALED_PASS`.

integrity_digest is an integrity digest, not a cryptographic signature.
