# V2 Known Issues and Design Limitations (FROZEN WITH DESIGN)

**Study:** V2 — Ranking-value validation of exact V1 Model A  
**Branch:** `v2-baseline-validation`  
**Protocol revision:** 1  
**Original freeze tags (preserved):** `prereg-v2-baseline`, `v2-design-frozen`  
**Execution-authoritative tags:** `prereg-v2-baseline-r1`, `v2-design-frozen-r1`  
**Amendment:** `docs/V2_AMENDMENT_001_SEED_STREAMS.md`  
**Sealed 2024–2025 accessed:** **NO**  
**Sealed access count:** must remain **0** until authorized V2-7

This file records inherent limitations and resolved design conflicts. It must not be used to quietly revise gates after sealed results are seen.

---

## 1. Inherent study limitations

1. **Post-selection of Model A.** Candidate chosen after V1 2019–2023 results. Sealed 2024–2025 is the primary fresh confirmatory evidence; 2019–2023 is prior/development only.  
2. **Ranking-value null, not full signal-free strategy null.** Random control keeps V1 eligibility/SMA200. Incremental claim is about **ranking among eligible names**, not about the eligibility filter itself.  
3. **No Jev.** Cannot speak to Jev; V1’s negative Jev conclusion remains archived.  
4. **Long/cash equity exposure.** Absolute returns can reflect market risk premia; E1/E2 exist to require ranking advantage.  
5. **Two-year sealed window.** Limited power; fail ≠ never works; pass ≠ perpetual edge.  
6. **V1 forward unusable for V2.** Pre-V2 deployment rows cannot become V2 clean-forward.  
7. **Integrity digests ≠ signatures.** Local tamper evidence only.

---

## 2. Resolved V1 ambiguities (do not reopen after seal)

### 2.1 Shortage / cash rule

**Resolved from code (authoritative):**

- `src/jevlab/signals/selection.py` → `select_top_k`: `selected = (rank <= K)` among eligible ⇒ `min(K, n_eligible)`.  
- `src/jevlab/portfolio/sizing.py` → `equal_weight_notionals`: each selected gets `equity/K`; unused slots stay cash.  
- Ledger calls these at `src/jevlab/backtest/ledger.py`.

**Docs alignment:** MASTER_SPEC “if fewer than 3 eligible, hold the rest in cash” matches code. Ambiguous shorthand “cash if fewer than 3” must **not** be read as forced 100% cash when 1–2 names are eligible.

### 2.2 Sealed training cutoff vs `DEV_MAX_DATE`

**Conflict found in V1:**

- Walk-forward science / `embargo_cutoff_date`: training decisions for T0 must satisfy `s(t) ≤ s(T0)−5`. For `T0=2024-01-02` this yields **`2023-12-22`**.  
- `fit_sealed_test_models` filtered `date ≤ 2023-12-29` without applying that embargo relative to T0.  
- MASTER_SPEC text claimed sealed-test training used `t ≤ 2023-12-29 (embargo applied)` — inconsistent with the helper’s correct cutoff.

**V2 resolution (human design decision §3):** use embargo-correct cutoff **`2023-12-22`**, assert it from the NYSE calendar at runtime, fit once, score sealed with no quarterly refits. This is a **protocol correction**, not a retuning of Model A ranking features/parameters.

### 2.3 Primary claim narrowed

Broad “does Model A have an edge?” language removed. Primary claim is ranking value vs mechanics-matched random selection **conditional on the same eligibility filter**.

### 2.4 Multiplicity / soft flags

Holm co-primary family removed. Soft adverse AUC/IC status discretion removed. Robustness cannot rescue/fail the canonical candidate except via explicitly listed hard gates.

### 2.5 Concentration math

V1 G7 used different share definitions. V2 uses predeclared positive-P&L ratios (top-5 profitable / ETF shares of `P+`) with fail-if-`P+≤0`.

---

## 3. Remaining non-blocking concerns (acceptable at freeze)

1. **Eligibility is not randomized.** A pass supports ranking value given SMA200 eligibility; it does not prove the eligibility rule itself is optimal.  
2. **E2 (+0.10 Sharpe advantage)** is a pre-seal practical threshold, not estimated from sealed data; it may be conservative or lenient relative to unknown sealed variance — that is intentional.  
3. **E5 worst-phase DD ≤ 35%** is a pathology cap, not fitted to V1’s historical DD.  
4. **Context equal-weight benchmark** uses daily close-to-close mean returns (`benchmark_returns`), which is **not** mechanics-matched to the 3-session top-K book — acceptable because it is context-only.  
5. **Phase-pooled concentration** (canonical) vs per-phase tables (exploratory): hard gate uses the pooled sealed Model A normal-cost round trips as specified in `V2_MASTER_SPEC.md`.  
6. **Original child-seed formula collided (amended pre-seal).** The additive rule `(base_seed + r + φ) mod 2^64` produced 2002 unique seeds among 6000 requested `(draw_id, phase)` streams. Protocol amendment 001 replaces child derivation with SHA-256 mixing; uniqueness `6000/6000` is now a frozen invariant. The original formula remains an audit artifact. See `docs/V2_AMENDMENT_001_SEED_STREAMS.md`. Revision-1 streams are distinct by construction of the 128-bit digest prefix; they are not claimed to be cryptographically independent beyond that uniqueness invariant.

None of these require sealed peeking to resolve.

---

## 4. Leakage / sealed-touch risks (mitigations)

| Risk | Mitigation |
| ---- | ---------- |
| Read `data/sealed/` during V2-0…V2-6 | Fixtures / pre-2024 only; access count must stay 0 |
| Train including 2024/2025 | Cutoff assert `≤ 2023-12-22` |
| Quarterly sealed refit | Forbidden |
| Pause after seeing A before nulls | One-shot sealed command |
| Second look | `--allow-second-look` ⇒ BROKEN; no `V2_SEALED_PASS` |
| V1 forward reuse | Explicitly forbidden for scale/forward gates |
| Jev import in V2 path | No Jev features/API/qset in candidate |

---

## 5. Status-model conflicts to avoid

1. `V2_SEALED_PASS` ≠ capital scale license.  
2. `V2_TINY_LIVE_READY` requires sealed pass **plus** operational gates.  
3. `V2_SCALE_READY` requires **V2’s own** post-deployment forward evidence.  
4. Do not invent V1 `HISTORICAL_ACCEPTANCE` for V2.

---

## 6. What this draft does **not** authorize

- Freezing or tagging V2  
- Accessing 2024–2025  
- Running Model A on the sealed period  
- Parameter search / retuning Model A  
- Starting V2 forward collection  
- Broker orders / live trading  
- Claiming any V2 acceptance status

---

## 7. Anti-overfitting reminder

> No V2 rule, feature, threshold, model, target, universe, strategy parameter, acceptance gate, or null construction may be changed based on the observed 2024–2025 sealed result. A failed V2 requires a new preregistered study.

2019–2023 remains prior/development evidence only.
