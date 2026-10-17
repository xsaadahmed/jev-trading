# V2 Preregistration (FROZEN)

**Study:** V2 — Ranking-value validation of exact V1 Model A  
**Parent tip:** `e5724df` (`v1-study-complete`)  
**Branch:** `v2-baseline-validation`  
**Freeze tags:** `prereg-v2-baseline`, `v2-design-frozen`  
**Config:** `config/v2/research.yaml`  
**Sealed access:** must remain **0** until authorized V2-7 run

This document is the frozen analysis plan. It is binding at the freeze tags. It does **not** authorize sealed execution by itself.

---

## 1. Hypotheses

### Primary confirmatory hypothesis (one test)

On the sealed period 2024-01-02 → 2025-12-31, the exact frozen V1 Model A **ranking** procedure’s mean-over-phases annualized **net Sharpe under normal costs** occupies a favorable upper tail of the mechanics-matched **random ranking** null (N=2000, same eligibility), with empirical one-sided

\[
p = (1 + \#\{S_r \ge S_A\}) / (1 + N) \le 0.05.
\]

### Practical-effect companion (deterministic gate, not a second hypothesis test)

\[
S_A - \mathrm{median}(\{S_r\}) \ge +0.10.
\]

### Absolute economic companions (deterministic gates)

- Mean-over-phases net CAGR(A) > 0 at normal costs.  
- Mean-over-phases net CAGR(A) > 0 at pessimistic costs.  
- Worst-phase max drawdown(A) ≤ 35% at normal costs.

### Null interpretation

Failure means: under the preregistered sealed evaluation, Model A does not demonstrate economically meaningful **ranking value** beyond random selection among the same eligible ETFs (and/or fails absolute economic/risk gates).

---

## 2. Selection bias acknowledgement

Model A was chosen after observing V1’s 2019–2023 walk-forward results.

Therefore:

- 2019–2023 is **prior/development evidence only** (reproducibility, power/planning).  
- It is **not** fresh V2 confirmation.  
- No gate threshold may be tailored to force that path to pass.  
- No V2 claim may treat 2019–2023 as confirmatory.

---

## 3. Anti-overfitting

> No V2 rule, feature, threshold, model, target, universe, strategy parameter, acceptance gate, or null construction may be changed based on the observed 2024–2025 sealed result. A failed V2 requires a new preregistered study.

---

## 4. Frozen analysis objects

| Object | Source of truth |
| ------ | --------------- |
| Universe | `config/universe.yaml` @ V1 tip |
| Features | `MODEL_A_COLUMNS` (17) |
| Eligibility | `adj_close > sma200` |
| Labels | `build_labels` (`y=1[X>0]`) |
| Model | Logistic + scaler; C ∈ {0.01, 0.1, 1.0}; seed 42 |
| C selection | `_select_C_inner` inside pre-sealed train only |
| Shortage / sizing | `select_top_k` + `equal_weight_notionals` → `min(K,n_elig)` at `equity/K`, rest cash |
| Phases | 0/1/2; primary stats mean across phases |
| Costs | `backtest.yaml` |
| Ranking null | §6 below |
| Jev / qset | **None** |

---

## 5. Periods and training cutoff

| Period | Role |
| ------ | ---- |
| Feature history from 2015– | Warm-up as allowed |
| Training decision dates `≤ 2023-12-22` | Pre-sealed Model A fit (embargo vs T0) |
| 2019–2023 WF | Dev/prior only |
| **2024-01-02 – 2025-12-31** | **Primary sealed confirmatory score window** |
| V1 forward | **Forbidden** as V2 confirmation |
| V2 forward | Later; disabled until after `V2_SEALED_PASS` + new deploy |

### Locked sealed training rule

- `T0 = 2024-01-02`.  
- Training rows: `s(t) ≤ s(T0) − 5` via `embargo_cutoff_date`.  
- **Stored final training decision-date cutoff: `2023-12-22`.**  
- Inner CV only inside that set; select C; fit **one** artifact; score entire sealed window; **no** quarterly sealed refits; **no** 2024/2025 training rows.

V1 note (resolved for V2): `fit_sealed_test_models` used `date ≤ 2023-12-29` without embargo relative to T0. V2 uses the embargo-correct cutoff `2023-12-22` per human design decision.

---

## 6. Random ranking null (exact)

`N = 2000`.

`base_seed = int.from_bytes(SHA256(b"v2-baseline-validation-random-null-v1").digest()[:8], "big")`  
**= `15224036858585574915`**.

For draw `r ∈ {1..2000}` and phase `φ ∈ {0,1,2}`:

`child_seed = (base_seed + r + φ) mod 2^64` → `Generator(PCG64(child_seed))`.

Each decision date (calendar order): random permutation of that day’s **Model A eligible set**; take first `min(3, n_eligible)`; size with `equal_weight_notionals(..., k=3)`; same engine/costs/capital as Model A.

Randomize **only** ranking among eligible names.

---

## 7. Primary statistic and p-value

- `S` = mean_φ annualized **net Sharpe**, normal costs.  
- `p = (1 + #{S_r ≥ S_A}) / (1 + 2000)`.  
- Pass E1 iff `p ≤ 0.05`.  
- **Single** primary confirmatory statistical test. No Holm family.

---

## 8. Hard gates (complete list)

| ID | Rule |
| -- | ---- |
| R1 | Mechanics / no leakage (ledger≡fast, accounting, embargo) |
| E1 | `p ≤ 0.05` |
| E2 | `S_A − median(S_r) ≥ +0.10` |
| E3 | mean_φ net CAGR(A) > 0, normal costs |
| E4 | mean_φ net CAGR(A) > 0, pessimistic costs |
| E5 | worst-phase max DD(A) ≤ 35%, normal costs |
| PS1 | ≥ 2/3 phases: Sharpe_A(φ) > median Sharpe_random(φ) (normal) |
| C1 | Concentration: top-5 profitable round trips < 25% of `P+`; no ETF > 35% of `P+`; if `P+ ≤ 0` → fail |
| Z1 | One-shot seal integrity |

`V2_SEALED_PASS` requires **all** of the above.

---

## 9. Secondary / non-gating analyses

**Predictive (report only):** log-loss, Brier, AUC, daily rank-IC, date-block bootstrap CI, calibration.  
No soft adverse status flags.

**Robustness (diagnostic only):** K∈{2,4}, C±1 grid step, LOO ETF, phase tables, cost scenarios, delayed execution, horizon±1.  
Cannot rescue or independently fail the canonical candidate except where hard gates already reference the same quantity.

**Context (report only):** SPY buy-and-hold; equal-weight 13-ETF daily mean returns (`benchmark_returns`).  
Not substitutes for the ranking null.

---

## 10. Multiplicity

One primary statistical test (E1).  
E2–E5, PS1, C1, R1, Z1 are deterministic acceptance gates, not a multiple-testing family requiring Holm/BH.

---

## 11. Seal protocol

Single authorized run performs fit → score → 2000 nulls → all hard gates → all diagnostics → immutable write → close seal.  
No pause between candidate metrics and null/acceptance.  
`--allow-second-look` ⇒ SEAL BROKEN ⇒ never `V2_SEALED_PASS`.

---

## 12. Status artifacts (names)

| Status | Artifact (draft path) |
| ------ | --------------------- |
| `V2_SEALED_PASS` | `reports/acceptance/V2_SEALED_ACCEPTANCE.json` |
| `V2_TINY_LIVE_READY` | `reports/acceptance/V2_TINY_LIVE_ACCEPTANCE.json` |
| `V2_SCALE_READY` | `reports/acceptance/V2_SCALE_ACCEPTANCE.json` |

Each carries `integrity_digest` over canonical JSON of other fields.

`V2_TINY_LIVE_READY` = sealed pass + operational safety + tiny capital + dry-run/reconciliation.  
`V2_SCALE_READY` = prior statuses + **new** V2 deploy + post-deploy forward under separately frozen gates.  
V1 forward rows never count.

---

## 13. Forward (future; disabled now)

Clean-forward begins only after `V2_SEALED_PASS`, production-forward refit, artifact hash stored, and new V2 deployment timestamp.  
Until then, forward capture stays disabled.

---

## 14. Failure conclusion language

On hard-gate fail, emit:

> The frozen Model A ranking procedure does not demonstrate robust economically meaningful ranking value over mechanics-matched random selection under the V2 preregistered sealed evaluation.

Do not claim markets are untradeable or that no future study could succeed.

---

## 15. Implementation phases

V2-0 spec → V2-1 reconstruction → V2-2 null engine → V2-3 acceptance/guards on fixtures → V2-4 pre-seal repro → V2-5 freeze/tag → V2-6 human review → **V2-7 one-shot sealed** → V2-8 report → V2-9 forward/tiny-live only if sealed pass.

**V2-7 is not authorized by this draft.**

---

## 16. Pre-freeze consistency review (hardened)

Checked and fixed in this revision:

| Risk | Resolution |
| ---- | ---------- |
| Holm / multiple primary tests | Removed; one primary test (E1) |
| Quarterly sealed refitting | Forbidden; fit-once policy |
| Train on 2024/2025 | Forbidden; cutoff 2023-12-22 |
| Ambiguous random control | Ranking-only among same eligible set; mechanics identical |
| Unspecified hard thresholds | E1–E5, PS1, C1 fully numeric |
| Subjective soft AUC/IC flag | Removed |
| Robustness rescues candidate | Explicitly forbidden |
| V1 forward as V2 evidence | Explicitly forbidden |
| Jev dependency | None |
| Sealed touch in V2-0…V2-6 | Fixtures / pre-2024 only |
| Shortage ambiguity | Cited `select_top_k` + `equal_weight_notionals` |
| Train ≤ 2023-12-29 vs embargo | V2 locks embargo cutoff 2023-12-22 |

Remaining non-blocking notes: see `docs/V2_KNOWN_ISSUES.md`.
