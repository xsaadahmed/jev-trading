# V2 final pre-seal human-review packet

**Disposition: V2 PRE-SEAL FREEZE BLOCKED**

The frozen child-seed formula was not changed. `v2-preseal-implementation` was not created. The real 2024-2025 test was not executed.

## Frozen design

- Frozen design commit: `6f4f815a7551244673cb8f0a853f238ad11b8544`
- Implementation freeze commit: **not created**
- Latest implementation commit: `450ecf3fe4dd837f9918b67c767544f807284f1e`
- prereg-v2-baseline: `6f4f815a7551244673cb8f0a853f238ad11b8544`
- v2-design-frozen: `6f4f815a7551244673cb8f0a853f238ad11b8544`
- Research config hash: `96947d9901b5366246ab1073af7ec17a83cf2bd074ecb0f32f66fcad21a8fe1c`
- Candidate hash: `aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28`
- Random-null hash: `d8661aa1fcaf42ce239865049a9707d41dee2c598faf4de06de7630cfead977e`
- Acceptance-gate hash: `5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be`

## Model A / training

- Model A equivalence status: PASS (2019-2023 implementation equivalence only)
- Training cutoff: 2023-12-22
- Fit count: 1
- 2024 training rows: 0

## Random-null full rehearsal (software validation only)

- Label: `development_rehearsal_only`
- N: 2000
- n_executed: 2000
- Null-vector digest: `954d004379ae89f8e23061cb476a6d20210b329676dbb422ac00c0f654cf5c84`
- Deterministic repeat: true
- Runtime: ~438.5 s per 2000 draws (acceptable)
- NaN draws: 0
- Thresholds were not modified from this result

## Seed collision audit

- Formula (unchanged): `(base_seed + draw_id + phase) mod 2**64`
- Encoding: plain integer addition; no extra mixing
- Requested streams: 6000 (2000 x 3)
- Unique child seeds: 2002
- Collision seed count: 2000
- Colliding pair count: 5998
- Unique pairs: `(draw=1, phase=0)` and `(draw=2000, phase=2)` only
- Example: `child(1,1) == child(2,0) == 15224036858585574917`
- Same-phase draws are distinct; same-draw phases are distinct
- Material violation of distinct `(draw_id, phase)` streams: **yes**
- Disposition: **BLOCK_FREEZE** (human must decide whether an amended preregistration / new study version is required)

## Tests

- Gate independent-failure / all-pass / tamper: PASS
- Atomicity / crash-after-read / partial-result protection: PASS
- Second-look: PASS (SEAL BROKEN; never writes V2_SEALED_PASS)
- Training cutoff: PASS
- Sealed preflight no-read: PASS (`filesystem_reads=0`)
- Production sealed source adapter present: yes (`ProductionSealedSource`; construction does not read)
- Preflight performed: yes (`jev v2 final-test --preflight`)
- Underlying real sealed reads: 0
- Preflight error (expected until freeze tag exists): missing `v2-preseal-implementation`

## Integrity / environment

- Pre-seal snapshot digest: `fcf57c242861b1d42dfde4c08464f27ace3240723aaa093788432d8fe0af0d77`
- Sealed access log count: 0
- Prior V2 sealed results: none
- V2_SEALED_PASS existing: no
- V2 forward registry: none
- Secrets scan: PASS
- Working tree after this packet: committed separately if reports remain

## Remaining known issues

1. **BLOCKER:** frozen additive child-seed rule reuses PCG64 streams across adjacent `(draw_id, phase)` pairs. Formula was not changed.
2. Rehearsal uses synthetic pre-2024 panels; frozen N=2000 is validated, but the rehearsal statistic is not V2 evidence.
3. Real `jev v2 final-test` still requires `v2-preseal-implementation` (intentionally not created).
4. Production `load()` is implemented but was never authorized against 2024-2025.
5. No V2 forward collection.

This packet contains no 2024-2025 result or metadata derived by reading the sealed dataset.

Final review report digest: `2bc0b4d9bf21b9a5e35880bd096e0358f36e4496e233d6d1ce330c4150695b8f`

integrity_digest is an integrity digest, not a cryptographic signature.
