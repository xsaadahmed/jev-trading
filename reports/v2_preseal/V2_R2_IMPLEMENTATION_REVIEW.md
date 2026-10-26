# V2 r2 real-runner implementation review

Protocol revision remains **1**. This is implementation revision **2**.
`v2-preseal-implementation-r1` is preserved as an incomplete audit artifact and is not execution-capable.

This packet is **not** V2 confirmatory evidence and does **not** authorize sealed access.

## Why implementation r1 was rejected

`v2-preseal-implementation-r1` (`05a9f816c950473a5ed9851aa17555dcfd0619c4`) loaded or discarded the sealed payload, logged placeholder step names, and evaluated caller/default `V2GateMetrics` (including passing defaults `p_value=0.01`, `sharpe_advantage=0.20`, `cagr_normal=0.05`). It could theoretically write `V2_SEALED_PASS` without computing Model A or the N=2000 random null.

The authorized r1 sealed attempt correctly stopped before any 2024–2025 read.

## Proof r1 never accessed the seal

- Real registry `sealed_access_log` count at r2 freeze: **0**
- `V2_SEALED_PASS.json` absent
- `reports/v2_sealed/SEALED_RESULT.json` absent
- Preflight unit tests: production source construction and `jev v2 final-test --preflight` perform **0** filesystem reads of sealed contents

## Implementation r2

- Tag: `v2-preseal-implementation-r2`
- Implementation commit: `1e9fa8629421bfe323a7c52be442147ca77d61b3`
- Branch: `v2-baseline-validation`
- r1 tag was **not** moved

## Unchanged frozen research hashes

| Item | Hash |
| ---- | ---- |
| research config | `e436ed977a02a39a0a280d78dcad74fa67e3ff1276f18a3430f249a9aeaf13ce` |
| candidate | `aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28` |
| random null | `9365e46d2ffc657d2b078cd851b90bac8f8ace3357313743769efdabf6498b2e` |
| acceptance gates | `5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be` |
| seed mapping | `5c773811984b2fea232889295560f3ded3e6db79ca8e949ee5518933f88e58f9` |

Live hashes match. Protocol revision = 1.

## Model A equivalence (2019–2023, not V2 evidence)

- Eligible rows: **11470**
- Folds: **20**, identical IDs
- C = **0.01** every fold
- Prediction max abs diff: **0.0**
- Selections identical
- Trades identical
- Round trips: **1187 / 1182 / 1188**
- Canonical metrics unchanged (mean Sharpe `0.5827386645170073`)
- Ledger-fast equal: **true**

## Production-path pseudo-seal rehearsal

- Label: `development_rehearsal_only`
- Data period: t0=`2021-06-25`, cutoff=`2021-06-18` (pre-2024 synthetic source)
- N: **2000 / 2000** complete; dropped draws **0**
- Null-vector digest: `cd357551102529816662f14214499a24e9fdbf422969e722d901d476d77915ef`
- Deterministic repeat: **true**
- Oracle gate comparison: **true**
- Fit count: **1**; C selected on training only
- Training rows from 2024/2025: **0 / 0**
- `V2_SEALED_PASS` written: **false**
- Performance does not influence frozen thresholds

## N=2000 digest / oracle / concentration / E5

- Null digest as above; 6000/6000 child-seed uniqueness remains the frozen r1 invariant
- Independent oracle recomputed p, Sharpe advantage, CAGRs, MDD magnitude, phase wins, S2/S3 from saved artifacts; equal within 1e-12
- S2/S3 use total **positive completed-trade P&L**; zero P+ fails closed
- E5: `-0.35` PASS; `-0.3500001` FAIL; `-0.80` FAIL (magnitude `abs(MDD) <= 0.35`)

## Placeholder-metric regression

- Production `V2GateMetrics` defaults are `None` (fail closed)
- `run_v2_final_test` has **no** `metrics` parameter
- Production CLI does not import or construct `V2GateMetrics`
- Fixture stub payload consumes the seal and **cannot** write `V2_SEALED_PASS`
- Unit tests use `passing_fixture_metrics()` only

## Atomic fault-injection

Crashes after sealed read, feature construction, training, prediction, candidate backtest, first/mid/final null draw, gate evaluation, report writing, and immediately before acceptance write: seal remains consumed, no `V2_SEALED_PASS`, restart is a second look.

## Preflight zero-read

`jev v2 final-test --preflight` and production-source construction: underlying real sealed reads = **0**.

Importing the complete V2-7 pipeline does not access sealed contents.

## Authorization

Prior phrase `OPEN V2 REVISION 1 SEALED HOLDOUT ONCE` is **VOID/CLOSED**.
r2 requires a fresh phrase: `OPEN V2 REVISION 1 SEALED HOLDOUT ONCE WITH IMPLEMENTATION R2`.
r1 implementation tag is not sufficient for execution.

## Remaining known issues

- Pseudo-seal rehearsal uses a short pre-2024 synthetic panel (same N=2000 mechanics) rather than the full 2015–2023 development hive; that is software validation only.
- Adjacent-C and horizon-sensitivity robustness variants are recorded as diagnostic spec/neighbors; they do not refit a second canonical model.
- Late sealed-window labels/trades still follow frozen V1 boundary semantics (no 2026 completion).
- Real 2024–2025 remains unopened.

integrity_digest is an integrity digest, not a cryptographic signature.
