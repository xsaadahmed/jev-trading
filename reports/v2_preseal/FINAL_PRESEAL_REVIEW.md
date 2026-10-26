# V2 revision-1 final pre-seal human-review packet

**Protocol revision:** 1  
**Disposition recorded after implementation tag:** see JSON `disposition`

This amendment was made before any 2024-2025 sealed outcome, feature, label, return, metric, or summary statistic was accessed.

The original frozen protocol remains preserved as an audit artifact and is superseded for execution by protocol revision 1.

## Protocol

- Amendment: `docs/V2_AMENDMENT_001_SEED_STREAMS.md`
- Amendment reason: original additive child-seed formula produced cross-phase collisions (6000 requested streams, 2002 unique)
- Amendment document SHA-256: `aa7f2f5197a584a3ac5fcf6be1df26dc7daf03d414fe30e695ce87910beb0cc8`
- Original design commit: `6f4f815a7551244673cb8f0a853f238ad11b8544`
- Amended design commit: `1cd7e8a6107313a30385134197eeb0dcdb9c7324`
- Implementation commit: `32c63e0fd4d04542cf9b5705d213c350a1b0e82c`
- Original tags (not moved): `prereg-v2-baseline`, `v2-design-frozen` both peel to `6f4f815a7551244673cb8f0a853f238ad11b8544`
- Revision-1 design tags: `prereg-v2-baseline-r1`, `v2-design-frozen-r1` both peel to `1cd7e8a6107313a30385134197eeb0dcdb9c7324`

## Hashes

| Item | Original | Revision 1 |
| ---- | -------- | ---------- |
| research_config | `96947d9901b5366246ab1073af7ec17a83cf2bd074ecb0f32f66fcad21a8fe1c` | `e436ed977a02a39a0a280d78dcad74fa67e3ff1276f18a3430f249a9aeaf13ce` |
| random_null | `d8661aa1fcaf42ce239865049a9707d41dee2c598faf4de06de7630cfead977e` | `9365e46d2ffc657d2b078cd851b90bac8f8ace3357313743769efdabf6498b2e` |
| candidate | `aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28` | unchanged |
| acceptance_gates | `5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be` | unchanged |
| universe / feature_set / target / strategy / costs / eligibility / training | unchanged | unchanged |

## Seed policy (revision 1)

- `base_seed = 15224036858585574915` (preserved)
- Generator: `numpy.random.Generator(numpy.random.PCG64(child_seed))`
- Namespace: `v2-baseline-validation-random-null-r1`
- `draw_id in [0, 1999]`, `phase in {0,1,2}`
- Child seed: first 16 bytes of SHA-256 of UTF-8 `f"{namespace}|{base_seed}|{draw_id}|{phase}"` as unsigned big-endian int
- Requested streams: 6000
- Unique streams: 6000
- Collision count: 0
- Seed-mapping digest: `5c773811984b2fea232889295560f3ded3e6db79ca8e949ee5518933f88e58f9`

## Model A equivalence (2019-2023; not V2 confirmation)

- Status: PASS
- Eligible rows: 11470
- Folds: 20 identical
- C: 0.01 every fold
- Prediction max abs diff: 0.0 (tolerance 1e-9)
- Selections identical
- Trade identities identical
- Round trips: 1187 / 1182 / 1188
- Sharpe: 0.5827386645170073
- CAGR: 0.07775255992272519
- Ledger-fast equality: true

## Full-N rehearsal (software validation only)

- Label: `development_rehearsal_only`
- N: 2000
- n_executed: 2000
- Dropped/NaN draws: 0
- Fallback N: no
- Null-vector digest: `cd357551102529816662f14214499a24e9fdbf422969e722d901d476d77915ef`
- Deterministic repeat: true
- Rehearsal artifact digest: `ded29402c6ac1c71013264b10a85705c1b34676a024f67d2d658528ea8396b67`
- Does not alter thresholds or research design

## Tests

- All V2 unit tests: PASS (67 including secret scan)
- Model A equivalence: PASS
- Seed uniqueness / mapping determinism / process-restart: PASS
- Random-null selection invariants / uniformity: PASS
- Gate independent-failure / all-pass / tamper: PASS
- Production sealed-source no-read: PASS
- Preflight no-read: PASS
- Original design tags not sufficient for execution: PASS
- Atomicity fault injection: PASS
- Second-look invalidation: PASS
- Training cutoff: PASS
- Predictive/robustness isolation: PASS
- Secret scan: PASS
- Ruff (V2 paths): PASS
- Mypy (V2 paths): PASS

## Seal / access

- Production sealed source: `ProductionSealedSource` (construction does not read)
- Preflight no-read: yes
- Real sealed underlying reads: 0
- Sealed access log count: 0
- 2024-2025 accessed: NO
- Prior V2 result: none
- `V2_SEALED_PASS.json`: absent
- V2 forward registry: none
- Real sealed test executed: NO

## Integrity

- Pre-seal snapshot digest: `c357109236bd58093c1ef5ae07d564f86045fd14bbfacd75e6ac45be6558dcff`
- `.env` ignored and untracked
- Atomicity: crash/fault injection covered; no pause in atomic path
- Second-look: `--allow-second-look` => SEAL BROKEN; never writes `V2_SEALED_PASS`

## Remaining known issues

1. Rehearsal uses synthetic pre-2024 panels; the rehearsal statistic is not V2 evidence.
2. Real `jev v2 final-test` must not be executed until a human authorizes the seal.
3. Production `load()` is implemented but was never authorized against 2024-2025.
4. No V2 forward collection.
5. Repo-wide `ruff check src tests` and `mypy src/jevlab` still report pre-existing non-V2 findings (for example `cli/run.py` line length and `evaluation/phase10_completion.py` types). V2 paths are clean.

This packet contains no 2024-2025 result or metadata derived by reading the sealed dataset.

Final review report digest: `18c4210cc9952255ae8074085e95992cd1046e4756ed7fc92d5c1739e28557c7`

integrity_digest is an integrity digest, not a cryptographic signature.
