# Pre-registration — Jev Trading Research V1

**Status:** Draft committed at Phase 0. Tag `prereg-v1` is applied **before** bulk Jev inference (Phase 7), not at scaffold time.

**Source:** MASTER_SPEC.md Sections 3, 4, 8, and 16. Section 21A is the canonical invariant list; if this document conflicts with §21A, §21A wins.

---

## 1. Hypotheses (from §3)

**Market behavior.** Short-term mean reversion in liquid US equity ETFs: after a multi-day pullback inside a long-term uptrend, the fund tends to outperform the rest of the universe over the next three sessions, *unless* the pullback is the beginning of a regime break. Deterministic features measure pullback depth, trend, volatility and volume. The question is whether Jev's reading of the pattern as a whole helps distinguish temporary dislocation from breakdown beyond what a flexible model fitted on the same inputs can.

**Null hypothesis H0.** The cross-sectional out-of-sample predictive loss of Model B (baseline + Jev features) equals that of Model A (baseline only).

**Primary hypothesis H1 (pre-registered, one-sided).** Out-of-sample log-loss(B) < log-loss(A) and mean daily rank-IC(B) > rank-IC(A), over the walk-forward period 2019-01 → 2023-12, with the placebo-permutation p-value < 0.05.

**Secondary hypothesis H2.** Net-of-cost Sharpe and CAGR of the strategy driven by B exceed that of A under normal costs.

**Evidence labeling.** Every historical result involving Jev features is *potentially training-contaminated*. Only forward data collected after the recorded deployment timestamp is clean; historical evidence can justify continuing (`HISTORICAL_RESEARCH_PASS`) but cannot by itself justify scaling capital.

---

## 2. Experiment design (from §4)

### 2.1 Universe (fixed ex ante)

`SPY, QQQ, IWM, DIA, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY` (13 ETFs).

- Committed in `config/universe.yaml` before any data is downloaded.
- Context series (not traded): SPY (market) and VIX close from FRED `VIXCLS`, with VIX lagged one session.

### 2.2 Strategy family

**"Pullback in uptrend", long/cash, 3-session horizon.**

- Eligibility: adjusted close(t) > SMA200(t).
- Ranking: each model produces score s_{i,t}; baseline rule ranks by lowest 5-day-return percentile.
- Selection: top K = 3 eligible ETFs equal-weighted; cash if fewer than 3 eligible.
- Rebalance: every 3 sessions on fixed calendar phase (phase 0). Phases 1 and 2 are robustness; acceptance uses the mean of the three phases.
- Entry at open(t+1), exit at open(t+4). No stops, no leverage, no shorting, no margin, no averaging down. Cash earns 0 in the backtest.

### 2.3 Model lineup

| Model | Inputs | Learner | Role |
| ----- | ------ | ------- | ---- |
| R0 | none | rule: lowest 5d return among eligible | Deterministic reference |
| A | 17 deterministic columns | L2 logistic | Primary baseline |
| B | 17 + 8 Jev features | same logistic / grid / folds as A | Primary treatment |
| C | 8 Jev only | logistic | Attribution |
| D | 17 deterministic | LightGBM (fixed small hyperparams) | Flexible baseline |
| E | 17 + 8 Jev | LightGBM as D | Flexible + Jev |
| F | 17 + 8 placebo Jev | logistic | Control |
| G | 17 + one-hot of semantic buckets | logistic | Discretization control |

Only A→B is confirmatory. Everything else is exploratory attribution.

---

## 3. Target definition (from §8)

- Gross return: **R_{i,t} = aO_{i,t+4} / aO_{i,t+1} − 1** (adjusted opens).
- Universe benchmark: **R̄_t** = equal-weight mean of R over the 13 ETFs.
- Excess return: **X_{i,t} = R_{i,t} − R̄_t**.
- **Primary label: y_{i,t} = 1[X_{i,t} > 0]**, eligible rows only (adj close > SMA200 at t).
- Secondary label: **y_cost = 1[R_{i,t} > roundtrip_cost_dec]** where
  `roundtrip_cost_dec = 2 × cost_bps_per_side / 10,000 + sell_fee_bps / 10,000` (decimal return).
  Example (normal): 2 × 2.5 bp + 0.3 bp = 0.00053.

Horizon H = 3 sessions. Label join lives only in `targets.build_labels`.

---

## 4. Readiness statuses and acceptance gates (from §16)

Three statuses, three artifacts — never merged:

| Status | Question | Artifact |
| ------ | -------- | -------- |
| `HISTORICAL_RESEARCH_PASS` | Is historical evidence strong enough to continue? | `HISTORICAL_ACCEPTANCE.json` |
| `TINY_LIVE_READY` | Is software safe for a ≈$10 operational shakedown? | `TINY_LIVE_ACCEPTANCE.json` |
| `SCALE_READY` | Is there enough post-deployment evidence to consider more capital? | `SCALE_ACCEPTANCE.json` |

Dependency: `TINY_LIVE_READY` requires `HISTORICAL_RESEARCH_PASS`; `SCALE_READY` requires `TINY_LIVE_READY`, executed live round trips, and clean forward evidence.

Artifacts carry an **integrity digest** (not a cryptographic signature):
`integrity_digest = sha256(canonical_json(all other fields))`.

### 4.1 HISTORICAL_RESEARCH_PASS (G1–G10)

Fixed before any Jev result exists. Summary (full table in MASTER_SPEC §16.1):

- G1 Contamination diagnostics procedural (D1–D6 + disposition ≠ stop); no numeric proof claim.
- G2 Reproducibility within §20 tolerances.
- G3 Mechanics / sample sufficiency (**not** baseline profitability).
- G4 Predictive increment: ΔlogLoss(B−A) < 0 with CI excluding 0; ΔIC > 0; placebo P1/P3 p < 0.05; P5 residual retains ≥ 50% of gain.
- G5 Stability across folds and years.
- G6 Economic increment and **Model B viability** (net CAGR(B) > 0 at normal costs).
- G7 Concentration limits.
- G8 Parameter stability.
- G9 Drawdown bound relative to A.
- G10 Sealed test 2024-01-02 → 2025-12-31 opened once.

Pass writes `HISTORICAL_ACCEPTANCE.json`. Fail on any gate → publish "no demonstrated value" and stop; tuning to pass is prohibited.

Mandatory report statement: *"Historical Jev evidence is potentially training-contaminated; this status does not establish that the edge is genuine."*

### 4.2 TINY_LIVE_READY (T1–T10)

Operational only — no forward-session requirement. Includes valid historical artifact, production-forward models, safety tests, dry-run, kill-switch drill, pinned model, `MAX_STRATEGY_CAPITAL_USD ≤ 10.00`, cash account, no open critical issues, and typed arming.

### 4.3 SCALE_READY (S1–S8)

Requires ≥ 120 clean forward sessions, ≥ 20 live round trips, forward ΔIC/ΔlogLoss consistent with history, no major drift, execution realism, human sign-off. V1 code cannot exceed $10.00 even after this status.

### 4.4 Mandatory report statements (§16.4)

- Historical: *"Results involving Jev features are potentially training-contaminated."*
- Live: *"This live phase is an execution experiment, not evidence of a profitable trading edge."*
- Possible failure conclusion: *"Jev does not demonstrate useful predictive value."*

---

## 5. Analysis plan (pointers)

- Walk-forward: 20 quarterly folds 2019Q1–2023Q4, embargo 5 sessions (§10.4).
- Costs: 1 / 2.5 / 6 / 12 bps per side + ≈0.3 bp sell fee (§10.2).
- Primary stats: date-block stationary bootstrap (mean block 10), HAC DM lag 5, placebo suite (§12, §13).
- Confirmatory tests: exactly two (ΔlogLoss, ΔIC for A→B), Holm-adjusted.
- Qset hash and frozen tag recorded before bulk inference; deviations require a new tag and an explicit deviation entry in the final report.

---

## 6. Deviations log

| Date | Tag | Deviation | Rationale |
| ---- | --- | --------- | --------- |
| _(none yet)_ | | | |
