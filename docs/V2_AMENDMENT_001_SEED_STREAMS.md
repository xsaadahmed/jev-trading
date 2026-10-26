# V2 Protocol Amendment 001 — Random-Null RNG Stream Derivation

**Protocol revision:** 1  
**Status:** PRE-SEAL AMENDMENT (no sealed data accessed)  
**Supersedes for execution:** original V2 design commit `6f4f815a7551244673cb8f0a853f238ad11b8544`  
**Original tags (preserved, not moved):** `prereg-v2-baseline`, `v2-design-frozen`

This amendment was made before any 2024–2025 sealed outcome, feature, label, return, metric, or summary statistic was accessed.

The original frozen protocol remains preserved as an audit artifact and is superseded for execution by protocol revision 1.

---

## Reason

The originally frozen additive formula produced cross-phase child-seed collisions.

```
child = (base_seed + draw_id + phase) mod 2^64
```

## Discovery time

Before any sealed access.

## Sealed access count at amendment

0

## Observed original behavior

6000 requested `(draw_id, phase)` streams produced only 2002 unique child seeds.

Example collision under the original formula:

```
child(r, 1) == child(r+1, 0)
```

Same-phase draws were distinct; same-draw phases were distinct; adjacent draws reused PCG64 streams across phases.

## Scientific impact

The primary candidate, null definition, N, statistic, hypotheses, gates, thresholds, and sealed period are unchanged.

Nature of amendment: **only** deterministic child RNG stream derivation changes.

Unchanged:

- Model A / universe / features / target / eligibility
- K, holding/execution, costs
- N = 2000
- primary statistic (mean-over-phases annualized net Sharpe at normal costs)
- p ≤ 0.05
- Sharpe advantage ≥ +0.10
- E1–E5, S1–S3, Z1
- sealed period 2024-01-02 through 2025-12-31
- robustness definitions (still non-gating)

## Preserved constants

- `base_seed = 15224036858585574915`
- Generator: `numpy.random.PCG64`
- Canonical string that produced `base_seed`: `"v2-baseline-validation-random-null-v1"`

## Revision-1 child derivation

```
namespace = "v2-baseline-validation-random-null-r1"
draw_id in [0, 1999]
phase in {0, 1, 2}

canonical_string = f"{namespace}|{base_seed}|{draw_id}|{phase}"   # exact UTF-8
digest = SHA256(canonical_string.encode("utf-8"))
child_seed = int.from_bytes(digest[:16], byteorder="big", signed=False)  # 128-bit
rng = numpy.random.Generator(numpy.random.PCG64(child_seed))
```

Do not use Python’s built-in `hash()`. Do not use process-dependent hashing. Do not truncate to 32 bits.

Frozen invariant: `unique_child_seeds == 6000`.

## Hash consequences

Only random-null semantics changed. Aggregate research-config hash therefore changes. Candidate, universe, feature-set, target, strategy, cost, and acceptance-gate hashes are unchanged.

| Hash | Original (revision 0) | Revision 1 |
| ---- | --------------------- | ---------- |
| random_null | `d8661aa1fcaf42ce239865049a9707d41dee2c598faf4de06de7630cfead977e` | `9365e46d2ffc657d2b078cd851b90bac8f8ace3357313743769efdabf6498b2e` |
| research_config | `96947d9901b5366246ab1073af7ec17a83cf2bd074ecb0f32f66fcad21a8fe1c` | `e436ed977a02a39a0a280d78dcad74fa67e3ff1276f18a3430f249a9aeaf13ce` |
| candidate | `aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28` | unchanged |
| acceptance_gates | `5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be` | unchanged |

6000-stream mapping digest (revision 1): `5c773811984b2fea232889295560f3ded3e6db79ca8e949ee5518933f88e58f9`
