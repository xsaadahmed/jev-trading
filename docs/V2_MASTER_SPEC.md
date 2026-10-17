# V2 Master Specification — Baseline Ranking-Value Validation of V1 Model A

**Study ID:** V2  
**Branch:** `v2-baseline-validation` (forked from V1 archival tip `e5724df`)  
**Status:** RESEARCH DESIGN FROZEN (`prereg-v2-baseline` / `v2-design-frozen`)  
**Parent study:** V1 (archived; conclusion: *Jev does not demonstrate useful predictive value.*)  
**Freeze date:** 2026-10-17  
**Config:** `config/v2/research.yaml` + `config/v2/FREEZE_MANIFEST.json`

This document is the frozen scientific and engineering specification for V2. Freeze authorizes pre-sealed implementation only. It does **not** authorize sealed 2024–2025 access, sealed evaluation, forward capture, live trading, or acceptance artifacts.

---

## 0. Relationship to V1

| Item | Rule |
| ---- | ---- |
| V1 artifacts / tags / qset / conclusions | **Immutable.** Do not modify. |
| V1 `prereg-v1` / `qset-v1-frozen` | Untouched. |
| V1 forward registry / forward rows | **Never** V2 clean-forward or scale evidence. |
| Jev / qset / Jev features / Jev API | **Absent from V2.** |
| Contamination (Jev) | **Not applicable** (no Jev inputs). |

V2 is a **new study**, not a rescue or retuning of V1’s Jev hypothesis.

---

## 1. Research question (precise)

**Primary confirmatory question.**

> Does the exact frozen V1 Model A ranking procedure add economically meaningful out-of-sample value over mechanics-matched random selection, conditional on the same V1 eligibility filter, during the untouched 2024–2025 period?

**What the primary null tests.**  
The primary random control retains V1’s eligibility / SMA200 filter and all trading mechanics. Therefore it is a **ranking-value null**, not a completely signal-free full-strategy null (eligibility timing is held fixed).

**What it does not claim.**  
- That long/cash ETF exposure beats cash merely because markets rise.  
- That Model A beats SPY or an equal-weight universe (those are **context only**).  
- That Jev has value (V1 already closed that claim).

**Anti-overfitting (mandatory):**

> No V2 rule, feature, threshold, model, target, universe, strategy parameter, acceptance gate, or null construction may be changed based on the observed 2024–2025 sealed result. A failed V2 requires a new preregistered study.

**Prior/development acknowledgement:**  
Model A was selected after observing V1’s 2019–2023 walk-forward results. That interval is **prior/development evidence only**. No V2 claim may treat 2019–2023 as fresh confirmation.

---

## 2. Exact confirmatory candidate — V1 Model A

Sole confirmatory candidate: **exact V1 Model A** and its associated frozen trading procedure.  
If any semantic below cannot be reconstructed from committed V1 artifacts at `e5724df`, implementation **STOP**s.

### 2.1 Source-of-truth map

| Semantic | Committed source |
| -------- | ---------------- |
| Universe (13 ETFs) | `config/universe.yaml` (`universe_version: v1`) |
| Model A columns (17) | `src/jevlab/features/columns.py` → `MODEL_A_COLUMNS` |
| Eligibility | `config/strategy.yaml` → `eligibility_rule: "adj_close > sma200"` |
| Target / labels | `src/jevlab/targets/labels.py` → `build_labels` (`y = 1[X>0]`, `X=R−R̄`) |
| Logistic + pipeline | `src/jevlab/models/pipelines.py` / `logistic.py` (StandardScaler + L2 logistic) |
| C grid | `config/strategy.yaml` → `logistic_C_grid: [0.01, 0.1, 1.0]` |
| Inner CV C pick | `pipelines._select_C_inner` (mean inner-fold log-loss; 3 expanding splits) |
| Walk-forward embargo helper | `src/jevlab/cv/purge.py` → `embargo_cutoff_date` (5 sessions) |
| Top-K selection | `src/jevlab/signals/selection.py` → `select_top_k` |
| Equal-weight sizing / shortage cash | `src/jevlab/portfolio/sizing.py` → `equal_weight_notionals` |
| Ledger / fast engines | `src/jevlab/backtest/ledger.py`, `fast.py` |
| Costs | `config/backtest.yaml` |
| Phases / K / horizon / entry lag | `config/strategy.yaml` |
| Risk caps (live path) | `config/risk.yaml` (research sealed path uses V1 ledger accounting) |

### 2.2 Universe

`SPY, QQQ, IWM, DIA, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY`.

### 2.3 Features (17)

`ret_1d, ret_3d, ret_5d, ret_20d, down_streak, px_vs_sma20, px_vs_sma50, px_vs_sma200, rsi_3, rsi_14, rv_pct_252, vol_ratio_20, vix_lag1_pct_252, spy_ret_5d, spy_px_vs_sma200, rel_ret_5d, xs_rank_ret_5d`

No new features. Feature construction remains the V1 Phase-2 pipeline (`as_of` decision close).

### 2.4 Eligibility

`adj_close > sma200` on the decision date.

### 2.5 Target

- Decision session `t`.  
- Entry: adjusted open at `t+1` (`entry_lag_sessions = 1`).  
- Exit: adjusted open at `t+4` (`horizon_sessions = 3`).  
- `R = aO[t+4]/aO[t+1] − 1`.  
- `R̄` = equal-weight mean of `R` over the 13 ETFs.  
- `X = R − R̄`; primary label `y = 1[X > 0]` on eligible rows.  
- Secondary `y_cost` may be reported; **primary predictive diagnostics use `y`**.

### 2.6 Model class and hyperparameter procedure

- Learner: L2 logistic regression with train-fold `StandardScaler` (V1 lineup pipeline).  
- Grid: `C ∈ {0.01, 0.1, 1.0}`.  
- Seed: `42`.  
- Inner expanding-window CV on the **pre-sealed training set only** (`make_inner_folds`, 3 splits, same 5-session embargo).  
- Choose `C` by minimum mean inner-fold log-loss (`_select_C_inner`).  
- Fit **one** final sealed-test Model A on the full pre-sealed training set with that `C`.  
- **No quarterly refitting inside 2024–2025.**

### 2.7 Selection, shortage, and sizing (resolved from V1 code)

**Selection** (`select_top_k`): among eligible symbols on a rebalance date, sort by score descending, symbol ascending for ties; mark `selected = 1` iff `rank <= K`.  
⇒ selects **`min(K, n_eligible)`** names. Never selects ineligible names.

**Sizing** (`equal_weight_notionals`): each selected name receives notional `equity_usd / K` (not `/ n_selected`).  
⇒ when `n_eligible < K`, unused slots remain **cash** at weight `(K − n_selected)/K`.

**Authoritative wording (MASTER_SPEC §4.2 + code):** hold top-K eligible equal-weighted at `1/K` each; if fewer than K eligible, hold those selected and **the rest in cash**.

Resolved ambiguity: prose “cash if fewer than 3 eligible” means **partial cash for unused slots**, **not** forced 100% cash when `n_eligible ∈ {1,2}`. Implementation must match `select_top_k` + `equal_weight_notionals`.

### 2.8 Portfolio / trading mechanics

| Item | Value |
| ---- | ----- |
| `top_k` | 3 |
| Weighting | equal `equity/K` per selected name |
| Rebalance | every 3 sessions |
| Phases | 0, 1, 2 (`rebalance_dates`: session index `% 3 == phase`) |
| Primary aggregation | **mean across phases 0/1/2** unless a gate says otherwise |
| Entry | decision after close `t`; fill adjusted open `t+1` (+ slippage convention) |
| Exit | open at `t+4` |
| `fill_convention` | `adjusted_open_plus_slippage` |
| Research capital | `initial_capital_usd = 100000.0` |
| Fractional shares | true |
| Cost scenarios | optimistic 1.0 / **normal 2.5** / pessimistic 6.0 / stress 12.0 bps/side |
| `sell_fee_bps` | 0.3 |
| Per-symbol multipliers | `config/backtest.yaml` |
| Cash interest | 0 |

### 2.9 Explicit exclusions

- No Jev features, qset, or API calls.  
- No Models B/C/D/E/F/G.  
- No retuning of C grid, K, horizon, eligibility, universe, costs, or ranking rule based on sealed outcomes.  
- No use of V1 forward rows as V2 evidence.

---

## 3. Sealed training procedure (locked)

Primary sealed score window: **T0 = 2024-01-02** through **2025-12-31** (NYSE sessions).

### 3.1 Embargo / training decision-date cutoff

Training decision rows must satisfy the exact V1 five-session embargo:

\[
s(t) \le s(T_0) - 5
\]

Implemented by `jevlab.cv.purge.embargo_cutoff_date(sessions, T0, embargo_sessions=5)`.

**Derived cutoff (asserted from NYSE calendar, not guessed):**

| Quantity | Value |
| -------- | ----- |
| T0 | 2024-01-02 |
| Embargo | 5 sessions |
| **Final training decision date** | **2023-12-22** |

Sessions immediately before T0:  
`2023-12-22, 2023-12-26, 2023-12-27, 2023-12-28, 2023-12-29, 2024-01-02(=T0)`.

Therefore labels used for training are fully known before T0 (exit open for a 2023-12-22 decision is on/before the last pre-T0 session under the embargo definition).

**Do not write “train ≤ 2023-12-29” as the V2 sealed training rule.**  
That date is V1’s `DEV_MAX_DATE` / sealed partition boundary. V1’s `fit_sealed_test_models` filtered `date <= 2023-12-29` without applying `embargo_cutoff_date` relative to T0 — conflicting with MASTER_SPEC’s walk-forward embargo and with the parenthetical “embargo applied” on sealed-test training. **V2 resolves this conflict in favor of the embargo rule** (human design decision §3). Implementation must derive and assert cutoff `2023-12-22` (or the calendar-equivalent if the session calendar changes, which it must not for historical NYSE dates).

### 3.2 Fit-once sealed policy

1. Restrict training rows to decision dates `≤ 2023-12-22` (embargo-correct).  
2. Run inner expanding-window CV **only** inside that set; select `C`.  
3. Fit **one** final Model A artifact on the full pre-sealed training set.  
4. Freeze/archive that artifact (hash + metadata including training cutoff and chosen `C`).  
5. Score the **entire** sealed period 2024-01-02 → 2025-12-31 with that artifact.  
6. **No** quarterly refitting.  
7. **No** incorporation of 2024 or 2025 observations into training.  
8. **No** second model after seeing partial sealed results.

---

## 4. Primary random ranking null

### 4.1 Construction

For each randomization draw `r = 1..N` and each rebalance phase `φ ∈ {0,1,2}`:

On every Model A decision date in that phase:

- Use **exactly** the same eligible ETF set as Model A that day.  
- Use the same rebalance phase calendar.  
- Select `min(3, n_eligible)` by a uniform random ranking among eligible names.  
- Apply the same shortage/cash sizing (`equity/K` per selected; unused slots cash).  
- Same entry/exit/hold, costs, capital, missing-data behavior, and ledger/fast engine.

**Only** the ranking/selection among eligible ETFs is randomized.

**Do not randomize:** eligibility, dates, K, exposure policy, costs, holding period, execution, or risk logic.

### 4.2 N and seeds (frozen before sealed access)

- `N = 2000` independent draws.  
- Canonical string: `"v2-baseline-validation-random-null-v1"`.  
- `base_seed = int.from_bytes(SHA256(canonical_string).digest()[:8], byteorder="big")`  
  (unsigned 64-bit integer from the first 8 digest bytes)  
  **= `15224036858585574915`**.  
- For draw `r` (1-based) and phase `φ ∈ {0,1,2}`:  
  `child_seed = (base_seed + r + φ) mod 2^64`  
  used to seed `numpy.random.Generator(numpy.random.PCG64(child_seed))`.  
- Within a (draw, phase), advance the generator once per decision date in calendar order; at each date draw a random permutation of the eligible symbol list and take the first `min(3, n_eligible)`.

Seeds are locked in this document **before** sealed access. Do not re-choose seeds after outcomes are seen.

### 4.3 Interpretation

This is a **ranking-value** null conditional on V1 eligibility. Positive absolute returns vs cash do not establish ranking value. SPY / equal-weight context benchmarks do not substitute for this null.

---

## 5. One primary confirmatory statistical test

**Primary statistic `S`:**  
mean across phases 0/1/2 of Model A’s **annualized net Sharpe** under **normal** transaction costs.

Compute the same `S_r` for each of the 2000 random draws.

**Empirical one-sided p-value:**

\[
p = \frac{1 + \#\{S_r \ge S_A\}}{1 + N}
\]

**Primary statistical threshold:** `p ≤ 0.05`.

There is **exactly one** primary confirmatory hypothesis test.  
No Holm / BH correction across economic gates.  
Do not invent co-primary tests after seeing sealed data.

---

## 6. Hard acceptance gates for `V2_SEALED_PASS`

All of the following must pass. Thresholds are fixed **before** sealed access.

### 6.1 Mechanics / seal integrity (hard)

| ID | Gate |
| -- | ---- |
| R1 | Ledger ≡ fast (V1 tolerances); accounting identity; costs correctly applied; no train/test leakage under the locked embargo |
| Z1 | One-shot sealed execution integrity (§9); second look ⇒ seal BROKEN ⇒ cannot grant `V2_SEALED_PASS` |

### 6.2 Confirmatory economic / ranking gates (hard)

| ID | Gate |
| -- | ---- |
| **E1** | Primary randomization `p ≤ 0.05` |
| **E2** | Minimum practical Sharpe advantage: `S_A − median({S_r}) ≥ +0.10` (normal costs, mean-over-phases). Chosen pre-seal as a minimum practical effect, **not** from 2024–2025. |
| **E3** | Mean-over-phases Model A net CAGR > 0 at **normal** costs |
| **E4** | Mean-over-phases Model A net CAGR > 0 at **pessimistic** costs |
| **E5** | Worst-phase Model A max drawdown ≤ 35% at **normal** costs |

E2–E5 are deterministic thresholds (no multiplicity adjustment).

### 6.3 Phase stability (hard) — PS1

For **at least 2 of 3** rebalance phases φ:

`Sharpe_A(φ; normal) > median({Sharpe_random(φ; normal)})`

Report all three phases. No cherry-picking.

### 6.4 Concentration (hard; unambiguous) — C1

Using completed round trips from the sealed Model A ledger under **normal** costs, pooled across phases for the canonical report (also report per-phase tables exploratorily):

Define round-trip P&L via V1 `_round_trip_pnl` semantics (`cohort_id` buy/sell pair; `pnl_usd = sell_net − buy_cost`).

Let `P+ = sum of pnl_usd over round trips with pnl_usd > 0`.

**Only evaluate ratios when `P+ > 0`.** Because E3 requires positive net CAGR, a non-profitable strategy cannot pass merely because concentration is unevaluable — if `P+ ≤ 0`, concentration **fails**.

When `P+ > 0`:

1. Sum `pnl_usd` of the top 5 **profitable** completed round trips (largest positive `pnl_usd`); require that sum `< 0.25 · P+`.  
2. For each ETF, sum its positive round-trip `pnl_usd`; require every ETF’s share `≤ 0.35 · P+` (strict: no ETF `> 35%`).

No manual judgment.

### 6.5 What does **not** gate `V2_SEALED_PASS`

- Predictive metrics (log-loss, Brier, AUC, rank-IC, bootstrap CIs, calibration) — **secondary diagnostics only**.  
- Robustness variants (K=2/4, C neighbors, LOO ETF, delayed execution, horizon sensitivity, stress costs beyond E4) — **diagnostic only**.  
- SPY / equal-weight context benchmarks — **context only**.

No favorable robustness variant may rescue a failed canonical candidate.  
No unfavorable noncanonical variant alone may invalidate a canonical candidate that passes all frozen hard gates.  
No subjective “soft adverse AUC/IC” status override.

---

## 7. Predictive diagnostics (secondary)

On sealed scored rows for Model A vs `y`, **report**:

- log-loss, Brier, AUC  
- daily cross-sectional rank-IC (mean)  
- date-block bootstrap 95% CI for mean daily rank-IC (stationary bootstrap, mean block length 10, 10_000 draws, seed 42)  
- calibration / reliability diagram  

The final report may narrate weak or adverse predictive metrics. **Status remains rule-based** from §6 only.

---

## 8. Robustness variants (diagnostic)

Preregistered diagnostics (pre-2024 fixtures for development; sealed run computes them in the same one-shot pass):

- K ∈ {2, 4} around canonical K=3  
- C one grid step around the chosen canonical C (where defined on `{0.01,0.1,1.0}`)  
- leave-one-ETF-out  
- phase breakdown (also used by the hard phase-stability gate)  
- cost scenarios (optimistic / normal / pessimistic / stress)  
- delayed execution (t+1 close; t+2 open)  
- horizon ±1 (relabel exploratory; diagnostic only)

These **do not** redefine the candidate and **do not** grant or revoke `V2_SEALED_PASS` except where a hard gate above explicitly references the same quantity (phase medians; pessimistic CAGR; drawdown).

---

## 9. Context benchmarks (not primary)

Report for the sealed window (CAGR, Sharpe, max DD, cumulative return):

1. **SPY buy-and-hold** — daily adj-close-to-close (as in `benchmark_returns`).  
2. **Equal-weight 13-ETF universe** — daily mean of members’ adj-close-to-close returns (`benchmark_returns` with the V1 universe list).

**Explicit:** a positive return versus cash does not establish ranking value. The random-control test is the primary test of incremental ranking value. Context benchmarks are not substitutes for that null.

---

## 10. One-shot sealed execution

The real sealed command must, in a **single non-interactive run**:

1. Verify clean working tree.  
2. Verify V2 frozen tag + config / code hashes.  
3. Verify sealed-access count = 0.  
4. Verify V2 code/tests frozen.  
5. Require explicit human SEAL token.  
6. Access 2024–2025 exactly once.  
7. Fit the pre-sealed Model A artifact (embargo cutoff asserted).  
8. Score all sealed observations.  
9. Generate all 2000 null draws.  
10. Calculate **all** hard gates.  
11. Calculate all preregistered diagnostics/robustness/context tables.  
12. Write the complete immutable result bundle.  
13. Close the seal.

**No manual pause** after seeing candidate performance but before null/acceptance results.

**Second invocation:**

- Requires explicit `--allow-second-look`.  
- Marks V2 seal **BROKEN**.  
- May produce diagnostics.  
- Can **never** produce `V2_SEALED_PASS`.

Development/CI before the authorized run: fixtures and pre-2024 data only. Sealed access count must remain 0 through V2-0…V2-6.

---

## 11. Status model

| Status | Requires | Meaning |
| ------ | -------- | ------- |
| `V2_SEALED_PASS` | Every frozen hard gate in §6 | Exact Model A passed preregistered 2024–2025 ranking-value validation |
| `V2_TINY_LIVE_READY` | Valid `V2_SEALED_PASS` + separate operational safety gates + tiny capital cap + dry-run/reconciliation | Execution-readiness for a tiny-dollar experiment — **not** proof of future profitability |
| `V2_SCALE_READY` | Prior statuses + **new** V2 production deployment + clean post-deployment forward evidence under separately frozen forward gates | Consider scaling capital |

Never conflate these. Sealed pass ≠ scale. Tiny live ≠ edge proof.

Artifact integrity: JSON + `integrity_digest = sha256(canonical_json(other fields))` (digest, not a cryptographic signature).

---

## 12. Forward data policy (disabled now)

Do **not** start V2 forward evidence collection yet.  
Do **not** reuse V1 forward data (`deployment_timestamp = 2026-10-16T04:20:04.946315+00:00`).

V2 clean-forward evidence begins only after:

1. `V2_SEALED_PASS` exists,  
2. the exact production-forward refit procedure is executed,  
3. the production artifact hash is stored,  
4. a new V2 deployment timestamp is recorded.

Any input capture before that may be stored operationally later if desired, but can **never** count toward `V2_SCALE_READY`.

**For now: V2 forward capture remains disabled.**

---

## 13. Implementation phases

| Phase | Work | Sealed 2024–2025 |
| ----- | ---- | ---------------- |
| **V2-0** | Spec / preregistration (this draft → freeze) | NO |
| **V2-1** | Exact Model A reconstruction + equivalence tests | NO (fixtures / pre-2024) |
| **V2-2** | Random-null engine + synthetic/pre-2024 tests | NO |
| **V2-3** | Acceptance evaluator + sealed guard on fixtures | NO |
| **V2-4** | Pre-seal reproducibility report | NO |
| **V2-5** | Freeze / tag V2 | NO |
| **V2-6** | Human pre-seal review | NO |
| **V2-7** | **ONE-SHOT** real 2024–2025 evaluation | YES (once) |
| **V2-8** | Final sealed report | after V2-7 |
| **V2-9** | Only if `V2_SEALED_PASS`: production-forward / tiny-live work | n/a |

**Do not run V2-7 until freeze + human authorization.**

---

## 14. Reproducibility and hashing

- Manifest: git commit, V2 freeze tag, config hashes, feature version, training cutoff, chosen C, model artifact hash, `base_seed`, N, cost scenarios, metric tables.  
- Logical table digests for features/labels/predictions/selections (§20).  
- Dirty working tree cannot support confirmatory sealed runs.

---

## 15. Failure rules

- Any hard-gate fail ⇒ publish that the frozen Model A ranking procedure does **not** demonstrate robust economically meaningful ranking value under the V2 preregistered sealed evaluation; stop.  
- No tuning on sealed outcomes.  
- New idea ⇒ new preregistered study (V3+).
