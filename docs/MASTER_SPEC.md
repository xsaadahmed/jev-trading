# JEV TRADING RESEARCH PROJECT — MASTER SPEC 

Date of research: 2026-10-07. Status of every external fact is tagged **[V]** (verified against an official or primary page during this research), **[3P]** (reported by a third-party page; re-verify before relying on it) or **[?]** (not documented anywhere I could find; must be tested empirically in Phase 0/6).

---



## 1. EXECUTIVE OVERVIEW

**What this is.** A controlled experiment to measure whether Jev's probabilistic judgments add out-of-sample predictive and economic value to a deliberately simple, long/cash, daily-bar ETF mean-reversion strategy. The deliverable is a *verdict with uncertainty*, not a trading bot. "Jev adds nothing" is a first-class, successful outcome.

**My prior (stated so it can be audited later).** I put roughly 85–90% on "no reliable incremental value after costs." Reasons: (1) Jev is a text-native "System One" gut-check model, not a market model; (2) its own docs say it is weak on numbers and literal in reading [V]; (3) anything it "knows" about markets is general text knowledge that the deterministic features already capture; (4) the real danger is not that Jev fails, but that it *appears* to succeed through leakage, training contamination, or luck. The design spends most of its effort on that second danger.

**The findings that shaped the design:**

1. **Training contamination cannot be ruled out by historical testing.** Jev was released in September 2026 and its training-data cutoff is not documented [?]. Any historical market state before ~2026 may have been seen (with its outcome) during training. No historical diagnostic can prove the absence of such exposure: a model can classify eras or predict outcomes for legitimate reasons (volatility regime, trend structure, genuine skill) or for illegitimate ones (memorization), and the two are observationally entangled. Therefore: (a) states are anonymized (closed-vocabulary ordinal text; no ticker, date, year, price level or raw measurement); (b) a set of *diagnostics* (§11.4) is run on data through 2023 only and treated as **warning signals, never as proof or as hard pass/fail thresholds**; (c) **all historical Jev-based results are labeled "potentially training-contaminated"** everywhere they appear; (d) the **only clean evidence is post-deployment forward data** — observations whose decision time is after the shadow pipeline's recorded deployment timestamp, which cannot have been in Jev's training set. Forward evidence is therefore the gate for scaling capital (§16.3), while historical evidence only decides whether the project continues and whether a tiny operational test is permitted.
2. **Jev is bad at numbers [V].** Its jaggedness page says: keep arithmetic in code, prefer semantic over numeric representations, don't use Score outputs to recover numeric magnitudes. So raw feature values are never sent. Code converts each feature to an ordinal English bucket using trailing-window percentiles, and Jev judges the *pattern*.
3. **Cost is irrelevant; statistical power is the binding constraint.** The whole historical Jev dataset costs about $2–4 (§25). What is scarce is independent observations: 13 highly correlated ETFs over ~~10 years with overlapping 3-day targets (~~420 non-overlapping 3-session blocks in the 2019–2023 walk-forward period). Paired P&L comparisons will be underpowered (SE of a Sharpe *difference* ≈ 0.25 even at 0.8 correlation between strategies), so the **primary hypothesis is on predictive loss / rank-IC (more power), with P&L as the secondary economic check.**
4. **Three different questions must not be conflated** (§16): *Is the historical evidence strong enough to continue?* (`HISTORICAL_RESEARCH_PASS`); *is the software safe enough for a ≈$10 execution shakedown?* (`TINY_LIVE_READY`); *is there enough genuinely post-deployment evidence to consider more capital?* (`SCALE_READY`). The $10 experiment is an operational test; it is never evidence of an edge.

**Answers to your 24 questions (pointers to detail):**


| #   | Question                          | Short answer                                                                                                                                                                                                  | §           |
| --- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| 1   | Behavior predicted                | Whether an ETF in a long-term uptrend that has pulled back will outperform the 13-ETF universe over the next 3 sessions (open-to-open).                                                                       | 3, 8        |
| 2   | Simplest credible baseline        | Deterministic rule (uptrend + weakest 5-day return) and a regularized logistic regression on 17 deterministic features. Baseline profitability is **not** a prerequisite for testing Jev.                     | 4, 9, 16    |
| 3   | What Jev could add                | Fuzzy contextual pattern classification ("temporary pullback vs. breakdown", regime fit) as a nonlinear, semantically-informed re-encoding of the same inputs. Possibly nothing.                              | 3, 7        |
| 4   | What Jev receives                 | One anonymized JSON object: 16 feature-derived closed-vocabulary text fields + 1 constant descriptor. No ticker, date, year, price level or raw number.                                                       | 7           |
| 5   | Questions                         | 4 conceptual judgments, 5 evaluations: 2 Noul (`dislocation`, `regime_supportive`) + 1 Score (`downside_risk`) + 2 Choice evaluations (`setup_type_a`, `setup_type_b`: same question, reversed option order). | 7           |
| 6   | Probabilities → features          | Noul p; Score expectation and P(level≥3); Choice class probs averaged over both orders; every raw distribution stored; 8 model-input features.                                                                | 7, 9        |
| 7   | Prevent overfitting               | Questions frozen and hashed *before* bulk inference; tiny dev sample from the train period only, no labels; trial ledger; sealed test; deflated Sharpe.                                                       | 7.4, 11, 12 |
| 8   | Number of Jev calls               | ≈35,100 feature states (≈28,600 before the seal is opened + ≈6,500 sealed-period) + ≈16,000 for audits, placebos, paraphrase checks.                                                                          | 25          |
| 9   | Cost                              | ≈$2–4 for the feature dataset; ≈$25 total budget cap including audits and variants. Input pricing $0.042 per million tokens [V].                                                                              | 25          |
| 10  | Caching                           | Content-addressed (sha256 of canonical request), raw JSONL shards + SQLite index, immutable, versioned.                                                                                                       | 18          |
| 11  | Model updates                     | Always request pinned `jev-1.13.0`, never the alias; a new version is a separate dataset and restarts forward evidence.                                                                                       | 20          |
| 12  | Data provider                     | Tiingo free (primary), FRED (VIX, lagged one session), Alpaca/Stooq (cross-check/fallback).                                                                                                                   | 5           |
| 13  | Universe                          | 13 fixed liquid ETFs listed before 2015 (SPY, QQQ, IWM, DIA + 9 sector SPDRs).                                                                                                                                | 4           |
| 14  | Horizon                           | 3 sessions: entry open(t+1), exit open(t+4); rebalance every 3 sessions.                                                                                                                                      | 4, 8        |
| 15  | Split methodology                 | Expanding walk-forward 2019–2023 (dev), sealed 2024–2025 (one shot), interim 2026 (exploratory, potentially contaminated), clean forward from the recorded deployment timestamp.                              | 10          |
| 16  | Walk-forward                      | 20 quarterly folds, 5-session purge/embargo, inner time-series CV.                                                                                                                                            | 10          |
| 17  | Costs                             | 1 / 2.5 / 6 / 12 bps per side (optimistic/normal/pessimistic/stress) plus ≈0.3 bp sell-side regulatory fees; round trip = 2 × per-side + fees.                                                                | 10          |
| 18  | Success metrics                   | Primary: Δ out-of-sample log-loss and rank-IC (B vs A). Secondary: net Sharpe, CAGR, drawdown.                                                                                                                | 12, 16      |
| 19  | Statistics                        | Date-block bootstrap, Diebold–Mariano with HAC, placebo permutation p-values, deflated Sharpe.                                                                                                                | 12          |
| 20  | Placebos                          | Six controls incl. a "Jev explained by deterministic features" test.                                                                                                                                          | 13          |
| 21  | Conditions for $10 live           | `HISTORICAL_RESEARCH_PASS` + operational safety conditions → `TINY_LIVE_READY` (no forward-session count needed). Scaling needs `SCALE_READY` (≥120 clean forward sessions).                                  | 15, 16      |
| 22  | Broker                            | Alpaca (commission-free, $1 fractional minimum, no minimum deposit) [V].                                                                                                                                      | 15          |
| 23  | Catastrophic failures             | Section 24.                                                                                                                                                                                                   | 24          |
| 24  | What convinces me Jev has no edge | Section 24, "Disconfirmation criteria".                                                                                                                                                                       | 24          |


---



## 2. JEV TECHNICAL RESEARCH — VERIFIED CURRENT API

Source of truth: `https://docs.typesafe.ai` (introduction, API reference, models, confidence, state, jaggedness pages) [V]. Retrieved 2026-10-07. The jaggedness page was last reviewed by TypeSafe on 2026-10-02.

### 2.1 What Jev is

A "System One" model: it returns typed decisions with probabilities instead of generated text. Questions are evaluated *in parallel and in isolation* against one shared state. Trained with "RLCD" (reinforcement learning for calibrated decisions). The same weights serve every account; **no fine-tuning, no per-customer adaptation** [V]. Not trained on customer requests [V].

### 2.2 Endpoint, auth, access

- `POST https://api.typesafe.ai/v1/systemone`, header `Authorization: Bearer <API_KEY>`, `Content-Type: application/json` [V].
- `GET https://api.typesafe.ai/v1/models` lists accounts' accepted names (aliases only; versioned IDs are accepted whether or not listed) [V].
- Official Python SDK `typesafe_sdk` (class `TypeSafeClient` / async variant) and JS SDK `@typesafe-ai/sdk` [V]. **This project uses plain** `httpx` **against the HTTP API** (see §19) so the exact bytes sent are what we hash and store; the SDK is optional.
- **Direct-access availability [3P/?]:** community pages say direct TypeSafe access was waitlist/early-access while Jev is also reachable through OpenRouter (`typesafe/jev-1.13`, `~typesafe/jev-latest`), Cloudflare Workers AI (`typesafe/jev`), Vercel AI Gateway (`typesafe-ai/jev`) and AIML API. These routes have *different endpoints, schemas, credentials and prices*. **Decision:** the client is route-agnostic behind an interface, but V1 uses the official endpoint; if you only have OpenRouter access, set `route: openrouter` and re-run the Phase 5 schema contract tests. A different route is a different dataset (route is part of the cache key). Ignore any site claiming a different "official" endpoint (one search result named `thejevai.com`; that is not TypeSafe's documented API).



### 2.3 Request schema [V]

```json
{
  "model": "jev-1.13.0",
  "state": "string | object | array",
  "questions": {
    "<your_question_id>": { "type": "noul|choice|score", "instructions": "string|object|array", "criteria": "..." }
  }
}
```

- `questions` keys are yours; they are **not sent to the model and not used in inference** [V].
- `noul`: `criteria` optional `{ "true": ..., "false": ... }`.
- `choice`: `criteria` required map option→description (max 255 options).
- `score`: `criteria` required ordered array of level descriptions (2–10 levels).
- `instructions` may be a structured object that references other fields by name in backticks [V].



### 2.4 Response schema [V]

```json
{ "model": "jev-1.13.0",
  "answers": {
    "q_noul":   {"type":"noul",   "noul": 0.95},
    "q_choice": {"type":"choice", "choice":"billing", "probabilities":{"billing":0.88,"technical":0.12}, "confidence":0.81},
    "q_score":  {"type":"score",  "score":1.05, "legend":{"0":"Calm","1":"Frustrated"}, "probabilities":{"0":0.0,"1":0.95}, "confidence":0.92}
  },
  "usage": {"input_tokens":296, "output_tokens":20} }
```

- `model` in the response is the **resolved versioned ID** — we persist it per inference.
- Noul returns a single probability, **no confidence field**; the distance from 0.5 plays that role (`|2p−1|`) [V].
- Choice confidence = (p_max − 1/n)/(1 − 1/n); Score confidence uses the ordinal mean-absolute-deviation formula (documented exactly) [V]. We store raw probabilities and recompute confidence ourselves.



### 2.5 Limits, pricing, versioning [V unless tagged]


| Item                             | Value                                                                                                                                                                                                                                              |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current model                    | `jev-1.13.0`                                                                                                                                                                                                                                       |
| Aliases                          | `jev-latest` → `jev-1.13.0`; `jev-preview` → `jev-1.13.0` (no preview build exists now)                                                                                                                                                            |
| Price                            | $42 / billion input tokens = **$0.042 / million**; output tokens free                                                                                                                                                                              |
| Rate limits                      | 100K tokens/s and 80 requests/s, **"adjusting dynamically… can change without notice"**                                                                                                                                                            |
| Context                          | 64k tokens/request; 32k for state + longest single question                                                                                                                                                                                        |
| Input                            | Text only (string / object / array of text)                                                                                                                                                                                                        |
| Errors                           | 401, 422, 429, 529; exponential backoff recommended; SDKs honor `retry-after`                                                                                                                                                                      |
| Retention                        | Zero data retention only for enterprise (see Legal page)                                                                                                                                                                                           |
| Determinism / seed / temperature | **[?] Not documented.** Test empirically: repeat identical requests, measure variance (Phase 6).                                                                                                                                                   |
| Training-data cutoff             | **[?] Not documented.** Ask TypeSafe (Discord / support) and record the answer; treat historical data as potentially contaminated (historical testing cannot show otherwise).                                                                      |
| Can alias responses change?      | **Yes.** "An alias moves when a new release ships, so the answers behind it can change." [V] Pin the versioned ID. Whether *old versioned IDs are retained forever* is **[?]** — snapshot every answer locally because a retired ID may disappear. |
| Historical responses changing    | Cached responses never change. Re-querying the same pinned ID may still differ if serving is nondeterministic [?].                                                                                                                                 |




### 2.6 Recommended usage and known failure modes (from the vendor's own jaggedness page) [V]

1. Reads questions literally → state exact conditions, put boundary cases in criteria.
2. **Weak with math, counting, numeric representations; Score magnitudes are not numerically calibrated** → code does all arithmetic; we use Score only as an *ordinal* feature.
3. Poor date/time comparison → we send no dates.
4. Indirection and double negatives degrade accuracy → write direct, positively-phrased questions.
5. **Context rot:** accuracy drops as irrelevant state grows → compact state (~14 fields).
6. Adversarial text can steer it → not applicable to numeric-derived states, but injection-style content from news is a reason **not** to feed news in V1.
7. Contradictory instructions vs. criteria hurt → align them.
8. **Choice option order bias toward the first option** → every Choice is asked in two opposite orders and averaged.
9. Not a generator.

Best-practice patterns we adopt: *atomic questions*, *speculative fan-out* (many questions, one request; their cookbook claims ~12x cheaper and 10x faster than separate calls with no change in answers), *composite scoring in code*, and *confidence-gated use*. They also publish a cookbook that trains a downstream classical model (CatBoost) on Jev probabilities — structurally the same as our Model B, which is reassuring about intended usage but is **vendor marketing, not evidence of market skill**.

### 2.7 Things I could not verify and what the project does about each


| Unknown                                | Handling                                                                                                                                                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Training cutoff / market-data exposure | Contamination *diagnostics* (§11.4: warning signals, not proof); all historical Jev results labeled "potentially training-contaminated"; post-deployment forward data is the only clean evidence; scale gate §16.3 |
| Serving determinism                    | Repeat-call audit on 300 states × 3 repeats; if sd of any Noul > 0.02 we store the mean of 3 calls for all states                                                                                                  |
| Direct-access status for your account  | Phase 5 smoke test; route-agnostic client                                                                                                                                                                          |
| Long-term retention of `jev-1.13.0`    | Local immutable response archive is the dataset of record                                                                                                                                                          |
| Whether rate limits tighten            | Token-bucket limiter, resumable job, no deadline coupling                                                                                                                                                          |


---



## 3. TRADING HYPOTHESIS

**Market behavior.** Short-term mean reversion in liquid US equity ETFs: after a multi-day pullback inside a long-term uptrend, the fund tends to outperform the rest of the universe over the next three sessions, *unless* the pullback is the beginning of a regime break. The deterministic features measure pullback depth, trend, volatility and volume. The question is whether Jev's reading of the *pattern as a whole* (is this a temporary dislocation or a breakdown? is the regime supportive?) helps distinguish the two cases beyond what a flexible model fitted on the same inputs can.

**Null hypothesis H0.** The cross-sectional out-of-sample predictive loss of Model B (baseline + Jev features) equals that of Model A (baseline only).

**Primary hypothesis H1 (pre-registered, one-sided).** Out-of-sample log-loss(B) < log-loss(A) and mean daily rank-IC(B) > rank-IC(A), over the walk-forward period 2019-01 → 2023-12, with the placebo-permutation p-value < 0.05.

**Secondary hypothesis H2.** Net-of-cost Sharpe and CAGR of the strategy driven by B exceed that of A under normal costs.

**What Jev can contribute that code cannot (honest version).** Not new information — every input is derived from the same prices. It can only add a *different functional form* (a semantic prior over patterns learned from text). That is exactly why Placebo P5 (§13) tests whether Jev outputs are just a re-encoding that a gradient-boosted model fed the same inputs reproduces.

**Evidence labeling.** Every historical result involving Jev features is *potentially training-contaminated* (§11.4). Only forward data collected after the recorded deployment timestamp is clean; historical evidence can justify continuing (`HISTORICAL_RESEARCH_PASS`) but cannot by itself justify scaling capital (§16).

---



## 4. VERSION 1 EXPERIMENT



### 4.1 Universe (fixed ex ante, committed to git before any data is downloaded)

`SPY, QQQ, IWM, DIA, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY` (13 ETFs).

- All listed before 2015, so full history from 2016 with a 252-session feature warm-up; no survivorship problem because none were delisted or merged in the window.
- Excluded on purpose: XLRE (2015 launch, thin early history), XLC (2018), single stocks (survivorship; needs a paid point-in-time constituent history; earnings-event contamination), leveraged/inverse ETFs.
- Honest limitation: 13 ETFs are heavily correlated (a single market factor dominates). Effective independent bets per day ≈ 3–5. Hence the emphasis on cross-sectional ranking (market factor differenced out) and on date-clustered inference.
- Extensibility: `universe.yaml` is versioned; adding names creates a new dataset version. A stock universe is a Phase-2 research project and needs point-in-time membership data.
- Context series (not traded): SPY (market) and VIX close from FRED `VIXCLS`. **VIX is lagged one session** (`vix_lag1_pct_252` uses the previous session's close) because same-day publication by the day's 18:30 ET decision time cannot be guaranteed and cannot be reproduced historically (§6.1).



### 4.2 Strategy family (single, to keep the experiment controlled)

**"Pullback in uptrend", long/cash, 3-session horizon.**

- *Eligibility* (deterministic code): adjusted close(t) > SMA200(t).
- *Ranking*: each model produces a score s_{i,t} for each eligible ETF; baseline rule ranks by lowest 5-day-return percentile.
- *Selection*: hold the top K = 3 eligible ETFs equal-weighted (33.3% each, max); if fewer than 3 eligible, hold the rest in cash.
- *Rebalance*: every 3 sessions on a fixed calendar phase (phase 0). Phases 1 and 2 are reported as robustness; acceptance uses the mean of the three phases (removes start-date luck).
- *Entry* at open(t+1), *exit* at open(t+4). No stops in V1 (adds a tunable parameter; the max-drawdown kill switch is a portfolio-level safety, not a signal).
- No leverage, no shorting, no margin, no averaging down. Cash earns 0 in the backtest (conservative).

**Why 3 sessions and mean reversion:** it is the best-documented, simplest effect on liquid ETFs at daily frequency; 3 sessions balances sample size (2,514 sessions in 2016–2025 ⇒ ≈ 838 cohorts) against cost drag (1 round trip per 3 days at 1–3 bps/side is manageable); Jev latency is irrelevant (signals computed after the close, orders before the next open); the 2–5 day horizon family is the natural setting for the "temporary dislocation vs. breakdown" question.

### 4.3 Model lineup (the experimental design)


| Model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Inputs                                                                      | Learner                                                 | Purpose                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------- |
| **R0**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | none                                                                        | rule: lowest 5d return among eligible                   | Deterministic reference                                  |
| **A (primary baseline)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 17 deterministic columns                                                    | L2 logistic regression                                  | The comparison anchor                                    |
| **B (primary treatment)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | the same 17 + 8 Jev features (25 columns)                                   | **same** logistic, same regularization grid, same folds | Incremental value of Jev, linear                         |
| **C**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 8 Jev features only                                                         | logistic                                                | Standalone information content                           |
| **D**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 17 deterministic                                                            | LightGBM (small, fixed hyperparameters)                 | Does a flexible learner already capture what Jev "adds"? |
| **E**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 17 + 8 Jev (25 columns)                                                     | LightGBM, same settings as D                            | Jev's value under a flexible learner                     |
| **F (control)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 17 + 8 *placebo* Jev features (25 columns)                                  | logistic                                                | Null distribution (see §13)                              |
| **G (control)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 17 + one-hot of the semantic buckets sent to Jev (drop-first; count logged) | logistic                                                | "Is the gain just from discretization?"                  |
| Only A→B is the confirmatory test. Everything else is attribution/diagnosis and is labeled exploratory in the report. **Information parity:** every field of the Jev state is a deterministic function of columns inside Model A's 17, so any Jev gain cannot come from extra raw information (A additionally has `xs_rank_ret_5d`, which Jev never sees). **Baseline profitability is not a precondition** for running A→B; Model B itself must be net-positive after normal costs for `HISTORICAL_RESEARCH_PASS` (§16.1 G6). |                                                                             |                                                         |                                                          |


---



## 5. MARKET / DATA ARCHITECTURE



### 5.1 Provider comparison (free-first)


| Provider                 | History                                                                                                                           | Adjusted/raw                   | Limits / price                                                                         | Verdict                                                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tiingo**               | 30+ years of US EOD [3P]; adjusted + raw OHLCV; includes delisted tickers via flag [3P]                                           | Both (`adjOpen/adjClose`, raw) | Free: ~50 req/hour, 1,000 req/day, ~500 unique symbols/month [3P]; no per-second limit | **V1 primary.** 16 symbols × 1 call each fits in a day. Dividends/splits endpoints may need a paid plan [3P] — we derive from adj/raw ratios instead. |
| **FRED** (St. Louis Fed) | VIX (`VIXCLS`) daily                                                                                                              | n/a                            | Free key                                                                               | **V1 primary for VIX/macro** (observations carry `available_at`; VIX lagged one session, §6.1).                                                       |
| **Alpaca Market Data**   | ~2016+ bars                                                                                                                       | adjusted options               | Free plan historically IEX-only feed [?]; IEX volume understates consolidated volume   | **Fallback / cross-check for prices; do not use volume.** Verify current free-plan feed.                                                              |
| **Massive (ex-Polygon)** | Free "Stocks Basic": 5 calls/min, **2 years** EOD only; Starter $29/mo (5y), Developer $79/mo (10y), Advanced $199/mo (20y+) [3P] | Adjusted + raw                 | see left                                                                               | **Optional paid upgrade** if Tiingo quality fails. Free tier too short.                                                                               |
| **Stooq**                | Long EOD CSV                                                                                                                      | Adjusted only                  | Free, no key                                                                           | Cross-check only (no raw, license unclear).                                                                                                           |
| **Yahoo Finance**        | Long                                                                                                                              | Adjusted only                  | Unofficial                                                                             | Prototyping/cross-check only; never source of record.                                                                                                 |
| **Alpha Vantage**        | –                                                                                                                                 | –                              | Very low free caps (25/day) [3P]                                                       | Not suitable.                                                                                                                                         |
| **Nasdaq Data Link**     | Varies                                                                                                                            | –                              | Mostly paid                                                                            | Not needed.                                                                                                                                           |


**Recommendation:** Tiingo (primary) + FRED (VIX) for the whole V1 research; Alpaca for the *broker* and as a price cross-check; Massive Developer ($79/mo) only if cross-source reconciliation (below) finds material disagreement. Free-tier requirements: Tiingo key, FRED key, Alpaca account, TypeSafe key. Optional paid improvements: Massive/Tiingo paid plans, Wharton/CRSP-style point-in-time constituents (only needed for the later stock universe), news vendors (explicitly out of scope V1).

### 5.2 Data-quality protocol

- Download raw + adjusted OHLCV + volume for each symbol once; store immutable parquet with a manifest (symbol, vendor, request params, download time, sha256, row count).
- Reconcile Tiingo vs. Alpaca (and Stooq) on close prices: report the share of days with |diff| > 5 bps; any symbol-day with > 50 bps disagreement is flagged and listed in the data report (not silently corrected).
- Corporate actions: ETFs rarely split; detect by raw close jumps with unchanged adjusted close, and by adj/raw ratio steps (distributions).
- Adjusted-price leakage: vendor adjusted series are back-adjusted with *future* dividends. Mitigation: features use only scale-invariant returns/ratios computed on the adjusted series (the back-adjustment multiplies a window by a constant factor except across ex-dates, where it is economically correct); anything requiring an absolute level or dollar value (dollar volume, fills, live sizing) uses raw prices. Residual tiny leakage (dividend-timing) is documented and bounded by a unit test that recomputes features from raw+dividend data on SPY.
- Calendar: NYSE calendar from `exchange_calendars`; missing sessions are errors, not forward-filled.
- Timestamp semantics: EOD bars are labeled by session date; we assume a bar is *final* only after 18:00 ET on that date (feature job runs ≥ 18:30 ET live; full availability policy in §6.1).

---



## 6. FEATURE ARCHITECTURE (compact V1 set, all as-of close of t)

All rolling windows are trailing and inclusive of t only. Percentile ranks use the asset's *own* trailing 252-session window **excluding the current observation's future** (rank of today's value within the last 252 values including today). Cross-sectional ranks are computed across the 13 ETFs on the same date.


| Group                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Feature (id)                                                    | Definition                                                    | Why                          |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------- |
| Return                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `ret_1d`, `ret_3d`, `ret_5d`, `ret_20d`                         | log adj close returns                                         | pullback depth/length        |
| Gap                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `gap_1d`                                                        | open(t)/close(t−1) − 1 (adjusted)                             | overnight sentiment          |
| Trend                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `px_vs_sma20`, `px_vs_sma50`, `px_vs_sma200`                    | close / SMA − 1                                               | uptrend integrity, extension |
| Momentum                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `rsi_3`, `rsi_14` (Wilder)                                      | standard                                                      | short-term oversold          |
| Volatility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `rv_20` (std of 1d log ret × √252), `rv_pct_252`                | level and own-percentile                                      | risk regime                  |
| Volume                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `vol_ratio_20` (vol/20d mean raw volume), `dollar_vol_20` (log) | raw volume                                                    | capitulation / liquidity     |
| Market ctx                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `spy_ret_5d`, `vix_lag1_pct_252`, `spy_px_vs_sma200`            | SPY; FRED VIX of the **previous** session, own-252 percentile | regime                       |
| Relative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `rel_ret_5d` (ETF 5d − SPY 5d), `xs_rank_ret_5d`                | cross-sectional                                               | idiosyncratic pullback       |
| Streak                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `down_streak`                                                   | consecutive down closes                                       | pattern                      |
| That is **20 stored features** (plus the eligibility flag). **Model A uses a fixed 17** (`ret_1d, ret_3d, ret_5d, ret_20d, down_streak, px_vs_sma20, px_vs_sma50, px_vs_sma200, rsi_3, rsi_14, rv_pct_252, vol_ratio_20, vix_lag1_pct_252, spy_ret_5d, spy_px_vs_sma200, rel_ret_5d, xs_rank_ret_5d`) chosen *now*, on conceptual grounds, before seeing any result, so that every field shown to Jev is a function of Model A's inputs (information parity, §4.3). The remaining three (`gap_1d`, `rv_20`, `dollar_vol_20`) stay in the feature store but are used only in the explicit ablation list. Excluded from V1 (with reasons): earnings/news (events need point-in-time timestamps and are irrelevant for ETFs; news text is an injection/leak risk), breadth (needs constituent data), intraday features (daily-only). |                                                                 |                                                               |                              |


Leakage tests for features (see §19): every feature is recomputed on a truncated series ending at t and must equal the full-series value at t; a "future perturbation" test mutates all data after t and asserts feature(t) is unchanged.

### 6.1 Point-in-time availability policy (all non-price and price inputs)

Every row of every input table carries `available_at` (UTC timestamp). The decision time for session t is `decision_time(t) = t 18:30 America/New_York`. **Rule: an observation may be used only if** `available_at ≤ decision_time`**.** A single helper `as_of(table, decision_time)` enforces it; features, states and backtests all read through it, so a backtest sees exactly the information that would exist live.


| Input                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Source        | `available_at` rule (V1)                                                                                                               | How used                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Daily OHLCV for session t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Tiingo        | t 18:00 ET (conservative; live runs record `retrieved_at`)                                                                             | features at t                                                               |
| VIX close for date d                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | FRED `VIXCLS` | **08:00 ET of the next NYSE session after d** [?: FRED publishes with a lag; verify with ALFRED vintage dates and live `retrieved_at`] | **lagged one session**: features at t use VIX(t−1), available at t 08:00 ET |
| Chosen method and why: same-day VIX at 18:30 ET cannot be guaranteed from the free provider and its historical publication time cannot be reconstructed for backtests; a **one-session lag** is available both live and historically, costs little information (VIX is highly persistent; `rv_pct_252` carries same-day volatility), and is the safest V1 choice. Using same-day VIX would require a source that proves the official close is available by 18:30 ET *and* provides historical publication timestamps; if later adopted it must be a new `feature_version` (a new trial), never an in-place change. |               |                                                                                                                                        |                                                                             |
| Live monitoring: record actual `retrieved_at`; if any source arrives later than its policy `available_at`, log `availability.violation` and the day fails closed (no new entries). Earnings/news/event features are out of V1 scope; if ever added they require a verified publication timestamp.                                                                                                                                                                                                                                                                                                                  |               |                                                                                                                                        |                                                                             |


---



## 7. JEV QUESTION ARCHITECTURE



### 7.1 What Jev receives: `state_v1` (anonymized, closed vocabulary, no numbers)

The state builder maps each underlying feature to one of seven ordered buckets using the asset's own trailing 252-session percentile (≤5%, ≤20%, ≤40%, 40–60%, ≤80%, ≤95%, >95%) and renders it as a fixed English phrase from a **closed vocabulary** defined in `config/jev.yaml` and enumerated in `jev/vocabulary.py`. Integers are spelled as words (e.g., the down-streak is `none/one/two/three/four/five_or_more`). The state has **16 feature-derived fields plus one constant descriptor** and contains no JSON numbers.

```json
{
  "schema": "state_v1",
  "fund_type": "a diversified US equity exchange-traded fund",
  "long_term_trend": "price is moderately above its long-term average",
  "medium_term_trend": "price is slightly below its medium-term average",
  "short_term_position": "price is well below its short-term average",
  "one_day_return": "small decline",
  "three_day_return": "large decline",
  "five_day_return": "very large decline",
  "twenty_day_return": "moderate decline",
  "consecutive_down_days": "four",
  "short_term_momentum_oscillator": "deeply oversold",
  "medium_term_momentum_oscillator": "oversold",
  "volatility": "above its own one-year norm",
  "volume_vs_recent_average": "heavy",
  "versus_broad_market_five_day": "weaker than the broad market",
  "broad_market_five_day_return": "moderate decline",
  "broad_market_trend": "price is above its long-term average",
  "market_volatility_index_previous_close": "elevated compared with the past year"
}
```

(Field-to-feature map: `long/medium/short_term_*` ← `px_vs_sma200/50/20`; returns ← `ret_1d/3d/5d/20d`; streak ← `down_streak`; oscillators ← `rsi_3/rsi_14`; volatility ← `rv_pct_252`; volume ← `vol_ratio_20`; versus market ← `rel_ret_5d`; broad market ← `spy_ret_5d`, `spy_px_vs_sma200`; VIX ← `vix_lag1_pct_252`.)

**What must never be sent:** ticker/fund identity, calendar date or year, absolute price or index level, raw returns, raw RSI, raw VIX, raw volume, raw percentile values, or any other exact market measurement that could trivially identify an observation. **What is allowed:** fixed feature-semantics descriptors in field names and phrases ("one_day", "five_day", "long-term", "short-term", "medium-term"), because they describe the feature, not the observation.

**Semantic privacy validation (**`jev/privacy.py`**, enforced before every request and in tests):** (1) every key and leaf string must belong to the closed vocabulary generated from config; (2) no JSON numbers, floats, percentages, currency symbols or price-like tokens anywhere; (3) no strings matching any universe ticker or ticker-like uppercase token outside an allowlist; (4) no ISO dates, month names, four-digit years 1990–2100, or date-like patterns; (5) no prohibited metadata keys (`symbol`, `ticker`, `date`, `timestamp`, `price`, `volume`, `vix_level`, `rsi`, etc.); (6) schema/version fields must equal the configured values. Tests inject each violation type and assert detection, and property-test 5,000 random feature rows for zero violations. A coarser variant `state_v1c` (3 buckets per field) exists only for diagnostic D5. The fund-identity ablation (adding a generic sector descriptor) is a later, separately registered experiment.

### 7.2 Questions `qset_v1` (frozen before bulk inference)

Design choice: smallest set that covers the hypothesis from distinct angles; each is a gut-check, positively phrased, with aligned criteria (per the jaggedness guidance).


| Id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Type             | Exact instructions                                                                                                                                         | Criteria                                                                                                                                                                                                                         | Feature(s) produced                                                                                                                                                                   | Why it might add information                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `dislocation`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | noul             | "Is this fund's recent decline more likely a short-lived pullback that partly reverses within a few trading days than the start of a sustained downtrend?" | true: "The decline is shallow or moderate, the longer-term trend is intact, and market conditions look orderly." false: "The decline is deep or widespread, the longer-term trend is damaged, or market conditions show stress." | `j_dislocation` = p(yes)                                                                                                                                                              | holistic judgment of conflicting trend/depth/stress cues |
| `downside_risk`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | score (5 levels) | "How elevated is the risk of a further large decline over the next few trading days?"                                                                      | ["Very low","Low","Normal","Elevated","Extreme"]                                                                                                                                                                                 | `j_downside_E` = expectation 0–4; `j_downside_p3` = P(level ≥ 3)                                                                                                                      | non-linear risk reading; distribution tail               |
| `regime_supportive`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | noul             | "Is the broader market environment supportive of buying a short-term pullback in a fund that is in an uptrend?"                                            | true: "The broad market trend is intact and volatility is not extreme." false: "The broad market is in a downtrend or volatility is under stress."                                                                               | `j_regime` = p(yes)                                                                                                                                                                   | context interaction                                      |
| `setup_type_a` / `setup_type_b`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | choice           | "Which description best fits the recent price behavior?"                                                                                                   | options: `pullback_in_uptrend`, `trend_breakdown`, `sideways_noise`, `extended_rally_pause`, each with a one-line definition. `_b` uses the reverse option order.                                                                | `j_pullback_p`, `j_breakdown_p`, `j_sideways_p` (average of a/b); `j_setup_conf` = confidence of averaged distribution; `j_order_disagree` = total variation distance between a and b | taxonomy of pattern; order-bias diagnostic               |
| The frozen set is **4 conceptual judgments evaluated as 5 questions** (2 Noul: `dislocation`, `regime_supportive`; 1 Score: `downside_risk`; 2 Choice evaluations: `setup_type_a`, `setup_type_b`, the same question with reversed option order). So 8 Jev features: `j_dislocation, j_downside_E, j_downside_p3, j_regime, j_pullback_p, j_breakdown_p, j_sideways_p, j_setup_conf`. (`j_order_disagree` is stored as a quality metric, not a model input.) Dropped on purpose from your candidate list: "relative strength vs similar setups" (requires numeric comparison, Jev's weakness) and "internal contradiction" (computed in code from the distribution spread if ever needed). |                  |                                                                                                                                                            |                                                                                                                                                                                                                                  |                                                                                                                                                                                       |                                                          |




### 7.3 Storage and transformation

Each response is stored raw, then parsed into a wide table (`jev_features`) keyed by `(symbol, date, qset_version, state_schema_version, model_id)`. Transformations (rank-normalization, logit of noul probabilities with clipping at [0.01, 0.99]) are *fit on training folds only* inside the model pipeline.

### 7.4 Question-development protocol (anti-overfitting)

1. Wording above is **already written** here; there is no tuning loop.
2. Phase 5 may use a **dev sample of 300 states drawn randomly from 2016–2018 (train period only)** solely to check format validity, non-degeneracy (each Noul has sd > 0.05; no option has p > 0.98 in more than 90% of rows), and API errors. *Outcome labels are not joined to the dev sample.* A wording change prompted by degeneracy is allowed once, recorded in the trial ledger as `qset_v1.1`, and the dev sample is discarded.
3. The question set is then frozen: `qset_v1.json` canonical-hashed, git-tagged `qset-v1-frozen`, hash written into the trial ledger. Bulk inference refuses to run if the hash differs.
4. Any later question change = a new qset version = a new row in the trial ledger, counts toward multiple-testing correction, and **cannot** be used for the acceptance gate (only qset_v1 can).

---



## 8. TARGET DEFINITION

Notation: universe U (13 ETFs), decision date t (after the close), session index s(t). Entry open O_{i,t+1}; exit open O_{i,t+4} (H = 3 sessions). Adjusted opens `aO` are used for returns so dividends are included.

- Gross return: **R_{i,t} = aO_{i,t+4} / aO_{i,t+1} − 1**
- Universe benchmark: **R̄*t = (1/|U|) Σ_j R*{j,t}** (equal-weight, all 13, same window)
- Excess return: **X_{i,t} = R_{i,t} − R̄_t**
- **Primary label: y_{i,t} = 1[ X_{i,t} > 0 ]**, defined only for eligible rows (close > SMA200 at t).
- Secondary label (reported, not primary): **y_cost = 1[ R_{i,t} > roundtrip_cost_dec ]**, with `roundtrip_cost_dec = 2 × cost_bps_per_side / 10,000 + sell_fee_bps / 10,000`, a *decimal return*. `cost_bps_per_side` is the one-way all-in cost (half-spread + slippage), so the round trip counts it exactly once on each leg (2×) and the sell-side regulatory fee once. Example (normal scenario): 2 × 2.5 bp + 0.3 bp = 5.3 bp = 0.00053. Per-symbol multipliers scale `cost_bps_per_side`. Defined on eligible rows only.

Why excess and not raw: equity ETFs rise on ~58% of 3-day windows, so a raw label mostly learns drift; the excess label removes the market factor, and the long-only portfolio still captures drift passively (R0/benchmark comparisons make this visible).

Leakage controls: the label uses data through t+4 open; a row at date t is therefore *unknown* until session t+4 begins. Hence the purge/embargo in §10; features at t use only data ≤ close t; the label join is done in exactly one function (`targets.build_labels`) with a test that every label's window starts strictly after the feature timestamp.

---



## 9. MODELING / SIGNAL APPROACH

Evaluated methods: (A) deterministic gating on a Jev threshold (rejected: arbitrary cut-offs, the thing you said to avoid), (B) hand-weighted score (rejected: free weights = overfitting), (C) **Jev probabilities as features in an L2 logistic regression** (chosen), (D) cross-sectional ranking (used for selection *on top of* C), (E) calibrated expected-return model (deferred: only if C shows value).

**V1 recommendation:** logistic regression with L2 (`C ∈ {0.01, 0.1, 1}` picked by inner time-series CV on log-loss), features standardized with train-fold statistics, Jev probabilities logit-transformed. Selection: among eligible ETFs on each rebalance date take the top K=3 by predicted P(y=1). Same pipeline, same folds, same grid for A and B — the only difference is the column set. Complexity (LightGBM, models D/E) is run in parallel as *diagnostic*, not as the headline, and only matters if the simple pair is inconclusive.

No fixed "buy if Jev > 0.8" threshold anywhere. Jev outputs influence trades only through coefficients learned on past data.

---



## 10. BACKTESTING METHODOLOGY



### 10.1 Timing (no look-ahead)

1. Session t closes 16:00 ET. Bars have `available_at = t 18:00 ET`; `decision_time(t) = t 18:30 ET`.
2. At `decision_time(t)` compute features(t) from rows with `available_at ≤ decision_time` only (§6.1). VIX for date t is **not** available (and is not used); VIX(t−1) is. Build states, call Jev (cached for history; live for new days).
3. Score → select targets on rebalance dates (phase 0; phases 1, 2 as robustness).
4. Orders are created after the close for the **t+1 open**. Backtest fill: `aO_{t+1} × (1 ± slippage)`; exit fill at `aO_{t+4} × (1 ∓ slippage)`. Live: submitted 08:30–09:25 ET as fractional market orders that execute at the open.
5. No intraday information, no same-bar fills, no fills at the close that generated the signal.
6. The backtest uses the same `as_of` mechanism and `available_at` policy as live, so it reproduces live information availability (the policy timestamps are at least as conservative as real publication times; live monitoring flags any violation).



### 10.2 Costs (per side, all-in half-spread + slippage, plus sell-side regulatory fees ≈0.3 bp [verify current SEC/FINRA rates])


| Scenario                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | bps/side | Meaning                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----------------------------------- |
| optimistic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 1.0      | SPY-like spreads, benign open       |
| **normal**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 2.5      | headline case                       |
| pessimistic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 6.0      | wide sector-ETF spreads / bad opens |
| stress                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 12.0     | crisis opens; shows break-even      |
| Commission = 0 (Alpaca, long-only cash account). A round trip costs `roundtrip_cost_dec = 2 × cost_bps_per_side/10,000 + sell_fee_bps/10,000` (normal: 5.3 bp); one round trip per 3-session cohort at 100% turnover ⇒ ≈ 5.3 bp per cohort ≈ 4.4% per year (84 cohorts). Every model must overcome this drag, but a loss-making baseline does not by itself prevent testing Jev (§16.1 G3, G6). Per-symbol cost multipliers (SPY/QQQ ×0.6, thin sectors ×1.5) live in `config/backtest.yaml` and are fixed in advance. |          |                                     |




### 10.3 Data splits


| Block                  | Dates                                                                                         | Use                                                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Warm-up                | 2015-01-02 → 2015-12-31                                                                       | features only (252-session windows)                                                                                        |
| Train/dev              | 2016-01-04 → 2023-12-29                                                                       | everything, via walk-forward                                                                                               |
| Walk-forward OOS (dev) | 2019-01-02 → 2023-12-29                                                                       | development scoring; allowed to be inspected repeatedly **but each inspection that changes anything is logged as a trial** |
| **Sealed final test**  | 2024-01-02 → 2025-12-31                                                                       | opened exactly once (Phase 12)                                                                                             |
| Interim (exploratory)  | 2026-01-02 → day before `FORWARD_START_DATE`                                                  | informational only; potentially inside Jev's training window; potentially training-contaminated                            |
| **Clean forward**      | sessions whose decision time is after the immutable deployment timestamp (Phase 13A registry) | the only clean evidence; shadow-logged, no orders; basis of `SCALE_READY`                                                  |


Because Jev's cutoff is unknown, 2024–2025 cannot be assumed uncontaminated even though it is "unseen by us". It is sealed against *our* snooping; the forward period is the only evidence against *Jev's* prior exposure.

### 10.4 Walk-forward

- Expanding window, first training window 2016-01 → 2018-12 (3 years).
- Test folds are calendar quarters from 2019-Q1 to 2023-Q4 (20 folds).
- For each fold: fit on rows with session index `s(t) ≤ s(T0) − 5` (T0 = first decision date of the test fold): a training label uses open(t+4), so `t + 4 < T0` requires `t ≤ T0 − 5` (embargo = 5 sessions: entry lag 1 + horizon 3 + 1 buffer), hence no training label window overlaps the test period; hyperparameter choice via inner expanding-window CV on the training rows only (3 inner splits, same purge); freeze the model; score the whole test quarter; **never refit inside a quarter**.
- Standardization/rank transforms are fit inside the fold.
- Predictions are concatenated into one OOS series per model; all metrics are computed from OOS series only.
- Every model (A, B, C, D, E, F, G) uses identical folds, row sets, seeds and calendars.



### 10.5 Engine design (transparent custom, no framework)

Chosen: **small custom daily-bar engine**, because the strategy is daily, long-only, tiny; correctness and auditability matter more than features; frameworks (vectorbt, backtesting.py, Backtrader, Zipline) hide fill timing, adjust data internally, and make look-ahead audits harder.

- `engine/ledger.py`: an event loop over sessions with explicit cash, positions, orders, fills; every fill written to a trades table. Invariant checked each day: `cash + Σ shares × mark = equity`.
- `engine/fast.py`: vectorized equivalent used for placebo/bootstrap loops (thousands of runs). **A regression test proves ledger and fast paths produce identical equity curves (relative tolerance 1e-9, same environment) on random data.**
- Fractional shares allowed in backtest (matches Alpaca); position values rounded to cents.



### 10.6 Portfolio rules (V1)

K=3, equal weight 1/3, max 34% per position, max 100% gross, no leverage/margin, cash when fewer than 3 eligible, rebalance every 3 sessions, hold exactly 3 sessions unless the kill-switch triggers, no averaging down, no re-entry logic.

### 10.7 Model artifacts: sealed-test vs production-forward

- **Sealed-test model (**`kind='sealed_test'`**)**: trained only on rows with `t ≤ 2023-12-29` (embargo applied), used to score 2024–2025 exactly once in Phase 12, archived permanently and never overwritten (needed to reproduce the historical verdict).
- **Production-forward model (**`kind='production_forward'`**)**: after the sealed test is complete the 2024–2025 observations are no longer sealed. Phase 13B retrains the **exact frozen procedure** (same feature definitions, qset, model class, target, universe, horizon, strategy rules, hyperparameter grid and inner-CV method) on all pre-deployment rows whose labels are known by the deployment timestamp, as a **separate artifact** with its own hash and training-end date. It is the model used for shadow scoring and any live trading.
- **Why this does not invalidate the sealed evaluation:** the sealed evidence measured the ex-ante-frozen *procedure* trained on ≤ 2023 and scored on data it had never seen; that artifact and its result are preserved. The refit changes only the amount of training data (a standard refit after validation), introduces no new design choice, and is evaluated only on forward data that post-dates its training cutoff. Any change beyond the training sample is a new trial and cannot inherit the historical acceptance.
- Production-model hashes are written to the ledger **before** any forward-label join (Phase 13B guard), so forward outcomes cannot influence the model.

---



## 11. BIAS / LEAKAGE PROTECTIONS (explicit mapping)


| Risk                                 | Control                                                                                                                                                         | Verified by                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Look-ahead in features               | trailing windows only; features recomputed on truncated data                                                                                                    | `test_feature_causality` (truncate + perturb-future)         |
| Look-ahead via adjusted prices       | scale-invariant features; raw for dollars; SPY dividend recompute test                                                                                          | `test_adjustment_invariance`                                 |
| Label leakage / overlapping targets  | purge/embargo; labels built in one function                                                                                                                     | `test_label_window_after_features`, `test_walkforward_purge` |
| Survivorship                         | fixed pre-2015 ETFs; no delistings in sample; universe file is committed                                                                                        | `test_universe_hash`                                         |
| Data leakage via normalization       | fit-in-fold pipelines (sklearn `Pipeline`)                                                                                                                      | `test_no_global_fit` (scaler fit stats ≠ full-sample stats)  |
| Jev training contamination           | anonymized closed-vocabulary states; diagnostics as warning signals only; "potentially training-contaminated" labeling; forward data as the only clean evidence | §11.4, §16.3                                                 |
| Prompt/question overfitting          | frozen hashed qset; dev sample from train only, no labels; trial ledger                                                                                         | ledger check at bulk run                                     |
| Repeated testing of final test       | `SealedDataGuard` + access log; one-shot protocol                                                                                                               | `test_sealed_guard`                                          |
| Unrealistic fills                    | open+slippage; cost scenarios; delayed-execution test                                                                                                           | robustness suite                                             |
| Ignoring costs                       | costs mandatory in engine (no zero-cost default for reports)                                                                                                    | `test_costs_required`                                        |
| Cherry-picked universe/dates         | universe and dates fixed in config before data download; config hash in experiment manifest                                                                     | manifest diff                                                |
| Retroactively changing Jev questions | hash mismatch aborts runs; changes counted as trials                                                                                                            | ledger                                                       |
| Data snooping across many variants   | trial ledger + deflated Sharpe + Holm for secondaries                                                                                                           | §12                                                          |




### 11.4 Training-contamination diagnostics (warning signals, NOT proof, NOT hard gates)

**The limitation, stated plainly.** Jev's training data and cutoff are undocumented [?]. If Jev was trained on text that described or tabulated 2016–2025 market behavior, its outputs on historical states could embed knowledge of outcomes. Historical testing cannot establish that this did not happen:

- A model can legitimately infer *which era* a state comes from (volatility regimes, trend structure, the 2020 crash profile) without having memorized anything.
- A model can legitimately have *predictive skill* on a state; a high AUC is exactly what a real edge would also produce.
- Conversely, memorization could be subtle enough to evade any diagnostic.

Therefore the diagnostics below produce **warning signals that require human review**; they do not produce a numeric pass/fail. **Every historical result involving Jev features is labeled "potentially training-contaminated" in all reports, regardless of diagnostic outcome.** The **only clean evidence** is **forward data**: sessions whose decision time is after the deployment timestamp recorded by the shadow-capture pipeline (Phase 13A), using a pinned Jev version released before that timestamp. Such observations did not exist when that Jev version was trained. (A *later* Jev version could in principle be trained on them; forward evidence is therefore valid only for the version that produced it. A new Jev version restarts forward evidence.)

**Data restriction.** All diagnostics use only sessions ≤ 2023-12-29 and never touch `data/sealed/`. Diagnostic questions live in a separate `audit_qset` (own hash, own version); their responses are stored with `purpose='audit'`, never merged into `jev_features`, and never used as model inputs.

**D1 — Era-classification diagnostic (3 eras, chance = 1/3).**

- Eras (roughly equal in sessions, ≈ 630–690 each): **E1** 2016-01-04 → 2018-06-29 (628 sessions), **E2** 2018-07-02 → 2021-03-31 (≈693), **E3** 2021-04-01 → 2023-12-29 (≈691).
- Sample: **n = 600** anonymized states, 200 per era, random across ETFs, dates ≥ 5 sessions apart within an era.
- Question: Choice over `early`, `middle`, `late` period with one-line neutral definitions (calendar dates appear *only* in this audit question; they are never in any feature state).
- Null: accuracy ~ Binomial(600, 1/3); exact one-sided binomial p-value; mean = 200, sd ≈ 11.5, so p < 0.01 corresponds to about ≥ 227 correct (≈ 37.8%). The code computes the exact test; the number is for orientation.
- **Benchmark:** a multinomial logistic regression on the 17 deterministic features of the same 600 states, blocked cross-validation. This answers "how well can the *legitimate* information in the state identify the era?"
- **Reading:** Jev ≈ chance → no signal. Jev above chance but ≤ benchmark → consistent with legitimate regime inference. Jev clearly **above** the benchmark (bootstrap CI of the difference excludes 0) → *warning* (information beyond what the state's features support). None of these outcomes is proof of either conclusion.

**D2 — Outcome diagnostic with benchmark.** On **2,000** train-period states drawn from 2016-01-04 → 2018-12-31 (no overlap with the 300-state dev sample), Jev answers an audit-only Noul *"Will this fund outperform the broad group of similar funds over the next three trading days?"* Report AUC and rank-IC with date-block bootstrap CIs, next to the same statistics from a logistic regression on the 17 deterministic features (blocked CV). **No threshold.** A Jev AUC far above the benchmark is a *warning* (real skill or leakage; cannot be distinguished here); a Jev AUC ≈ benchmark suggests Jev is re-expressing deterministic information.

**D3 — Era-stability of the Jev→outcome relationship (descriptive).** Using the *frozen-qset features* on dev data (2016–2023), compute rank-IC of each Jev feature versus X by era E1/E2/E3. A relationship concentrated in older eras and fading toward recent ones is a *possible* memorization signature but also looks like ordinary alpha decay; it is reported, not gated. Because Jev's cutoff is unknown, the same pattern in the 2026 interim period (likely inside or after Jev's training window) is reported separately.

**D4 — Attribution cross-check (cross-reference).** Placebo P5 (§13): does the Jev gain survive when Jev is replaced by its projection onto the deterministic features? If yes, Jev adds nothing beyond re-encoding, whether or not it is contaminated; if the gain lives in the residual, that residual is exactly where leaked knowledge would also live, so P5 *cannot* clear contamination — it only characterizes the gain.

**D5 — Granularity sensitivity.** On a 3,000-state subset re-run the frozen questions with a coarser state (3 buckets per field instead of 7, `state_v1c`). Compare the Jev→outcome relationship fine vs. coarse. An effect that exists only at the finest granularity (where states are most "fingerprintable") is a *warning*. Logged as an exploratory trial.

**D6 — Vendor query.** Ask TypeSafe for the training-data cutoff and whether financial time series or market-history text were included; record the answer or non-answer in `docs/jev_cutoff.md`.

**Disposition (procedural, not numeric).** After Phase 6 a human writes `docs/contamination_disposition.md` stating: diagnostics run (with report hashes), observed warnings, decision (`proceed`, `proceed_with_caveats`, `stop`), name/date. A `stop` decision blocks the project. A `proceed` decision means only that the project continues; the "potentially training-contaminated" label remains.

**What is deliberately *not* done:** no gate of the form "outcome AUC ≤ x" and no claim that era-classification success proves memorization.

---



## 12. STATISTICAL EVALUATION



### 12.1 Predictive quality (primary)

Log-loss, Brier, ROC-AUC, precision@K, calibration curve with isotonic/bin reliability diagram (10 equal-count bins), **daily cross-sectional Spearman rank-IC** (score vs X among eligible ETFs; mean IC, IC t-stat with Newey–West lag 5, hit rate of IC>0).

### 12.2 Economic metrics (secondary)

Total return, CAGR, annualized vol, Sharpe, Sortino, max drawdown, Calmar, hit rate, average and median trade, profit factor, turnover, exposure, trade count, average holding period, 5%/1% tail losses (CVaR), skewness, beta and alpha to SPY (daily OLS, HAC SEs), contribution by symbol, by year, by regime, top-5-trade share of P&L.

### 12.3 Inference

1. **Paired block bootstrap** of (B − A): resample *dates* in stationary blocks (mean length 10) from the OOS series, recompute ΔlogLoss, ΔIC, ΔSharpe, ΔCAGR; 10,000 draws; report percentile 95% CIs. Resampling dates (not rows) respects cross-sectional correlation.
2. **Diebold–Mariano** on the per-date mean log-loss difference with Newey–West HAC (lag ≥ 5, overlapping 3-day targets).
3. **Placebo permutation test** (§13): empirical p = (1 + #{placebo Δ ≥ observed Δ}) / (1 + N), N = 1,000 (P1), 500 (P2), 500 (P3), 500 (P4); P5 and P6 are single runs.
4. **Multiple testing:** a **trial ledger** (SQLite table, §18) records every distinct (qset, state schema, feature set, model class, hyperparameter grid, universe, horizon) ever scored. Confirmatory tests: exactly two (ΔlogLoss and ΔIC for A→B on walk-forward OOS), Holm-adjusted. Secondary/exploratory results are reported with *deflated Sharpe ratio* (Bailey–López de Prado) using the ledger trial count and Holm/BH adjusted p-values; Hansen's SPA is optional if many variants accumulate.
5. **Power statement printed in every report** (walk-forward OOS has 1,258 sessions ≈ 419 non-overlapping 3-session blocks, so the effective number of independent date-blocks is of that order, not 13 × 1,258): minimum detectable ΔSharpe and ΔIC given sample length and measured correlation between A and B return streams. If the study is underpowered, the report says "inconclusive," not "no effect."



### 12.4 Interpretation rule

"Real rather than random" requires all of: bootstrap CI excluding zero for ΔlogLoss; placebo percentile ≥ 95; consistent sign in ≥ 65% of walk-forward folds; same-sign result in the sealed test; and contamination diagnostics (§11.4) reviewed and dispositioned. This test is about chance, not about training contamination; only forward data addresses the latter (§16.3).

---



## 13. ROBUSTNESS / PLACEBO EXPERIMENTS

**Placebos (Jev columns replaced, market features untouched, identical folds/pipeline):**

- **P1 Within-date shuffle:** permute Jev vectors across ETFs on the same date. Keeps date-level regime content, destroys asset-specific content. Tests whether the gain is cross-sectional. N = 1,000.
- **P2 Global shuffle:** permute Jev vectors across all rows. Destroys everything; sanity null (B should not beat A here). N = 500.
- **P3 Circular-shift within asset:** shift each asset's Jev series by a random lag ≥ 60 sessions (N = 500). Preserves autocorrelation and marginal distribution — the toughest "spurious persistence" null.
- **P4 Matched noise:** Gaussian features with matching dimension and autocorrelation (AR(1) fit), N = 500. Checks model-capacity artifacts.
- **P5 Deterministic-projection test:** fit a LightGBM mapping the 17 deterministic columns → each Jev feature (in-fold). Run B with (i) predicted-Jev only, (ii) residual-Jev only. If the gain survives with predicted-Jev and vanishes with residuals, Jev is a re-encoding of what the baseline already knows.
- **P6 State-shuffle placebo (new Jev calls):** send Jev states built from other rows' features (rows permuted before bucketing) on a 3,000-state subset; outputs on mismatched states should show **no** relationship to the row's outcomes; any relationship indicates a join/leakage bug or non-state information in the response (a pipeline-integrity check; single run).
Reporting: histogram of placebo ΔlogLoss/ΔSharpe with the real value marked; empirical p-values.

**Robustness suite:** per-feature ablation (drop each deterministic feature); per-question ablation (drop each Jev question); regimes (bull/bear via SPY vs SMA200, high/low (lagged) VIX tercile, 2020 crash, 2022 bear); per-ETF leave-one-out (remove best contributor); per-phase offsets 0/1/2; K ∈ {2,3,4}; horizon ±1 session (with relabeling, exploratory); cost scenarios; **delayed execution** (fill at t+1 close, t+2 open); threshold-free perturbation of regularization ±1 grid step; Choice-order and question-paraphrase sensitivity (2 paraphrases per question on a 3,000-state subset, new calls, logged as trials); model-version comparison only when a new Jev version exists (separate dataset, never overwrites).

---



## 14. RISK MANAGEMENT (deterministic code, separate from signals)

`risk/` is a pure module: `(proposed_orders, portfolio_state, market_state, config) → (approved_orders, rejections)`. Jev and models never see it and cannot override it.


| Rule                              | Value (V1)                                                                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Max position                      | 34% of **strategy** capital (live: each position ≥ $1.00 notional)                                                                                                       |
| Max gross exposure                | 100% of strategy capital (no margin/leverage); live: strategy market value + open buy notional + proposed buy ≤ min(`MAX_STRATEGY_CAPITAL_USD`, strategy equity) (§15.1) |
| Max new orders per day            | 6                                                                                                                                                                        |
| Max daily loss                    | 5% of start-of-day strategy equity → no new entries; flag                                                                                                                |
| Max drawdown from high-water mark | 20% of strategy equity → kill switch (flatten at next open, require manual reset)                                                                                        |
| Duplicate-order prevention        | deterministic `client_order_id = sha1(strategy_id, date, symbol, side)`; ledger lookup before submit                                                                     |
| Stale data                        | last bar must equal the last *completed* NYSE session and every input must satisfy `available_at ≤ decision_time`; otherwise no trading                                  |
| Missing data                      | any NaN in required features for a symbol → symbol skipped; > 2 symbols missing → day skipped                                                                            |
| Jev failure / malformed / timeout | **fail closed:** no new entries from Jev-dependent signals that day; open positions exit on schedule; never silently substitute a different model in live mode           |
| Holidays/market hours             | `exchange_calendars`; no submissions on non-sessions; submit window 08:30–09:25 ET                                                                                       |
| Kill switch                       | file `KILL_SWITCH` present, or env `TRADING_HALT=1`, or drawdown breach → all new orders blocked                                                                         |
| Model-ID verification             | resolved Jev model ID ≠ pinned ID, or production model hash ≠ acceptance artifact → no Jev-dependent entries                                                             |
| Reconciliation                    | ledger vs. broker positions/open orders must reconcile before any order (live)                                                                                           |
| Account type                      | non-cash account, margin multiplier > 1, or shorting enabled → fail closed                                                                                               |


---



## 15. LIVE-TRADING ARCHITECTURE (designed in Phase 14; hard-disabled by default)

**Terminology.** *Shadow logging* generates signals and records hypothetical trades and outcomes; it places **no broker orders** and calls no order endpoint. *Paper trading* submits simulated orders to a broker's paper environment. **Paper trading is not part of the mandatory workflow.** A `PaperBroker` class may exist because the interface is reusable, but nothing in the acceptance gates requires running it. *Dry run* = the live executor computes the orders it would send and performs read-only broker calls (account, positions, assets) but sends no orders.

**Broker:** Alpaca [V]: commission-free US-listed cash-account trading via API; fractional trading from **$1 notional** in 2,000+ eligible stocks/ETFs; **fractional orders are market orders only**, long-only, filled at the NBBO at submission with no price improvement; orders entered while the market is closed queue for the open; no account minimum; fractional trades are subject to the same fees as whole shares; day-trade counting applies (we hold ≥ 3 sessions so it is moot). Verify `fractionable=true` for each ETF via the assets endpoint before trading. Alternatives behind the interface: Interactive Brokers (fractional via API, more complexity). Funding and account-approval timing are not verified here; start onboarding early.

**$10 reality check.** With $10.00 of strategy capital and K = 3, each position ≈ $3.33. Expected edge ≈ 10–20 bps per cohort ⇒ well under 1 cent per trade. The $10 phase is an **operational shakedown (order flow, fills, reconciliation, kill switch), not a statistical test and not evidence of an edge.** Every live-mode CLI banner, run summary and report must print exactly: **"This live phase is an execution experiment, not evidence of a profitable trading edge."**

### 15.1 Strategy-capital isolation (the cap limits the *strategy*, not the account)

The user may hold other cash or securities in the same broker account. The bot must never trade them.

- `MAX_STRATEGY_CAPITAL_USD` (default `10.00`). The source defines a constant `TINY_LIVE_CEILING_USD = 10.00`; configuration above it raises at startup. V1 contains **no code path that permits strategy capital above $10.00** (a scale-up allocation schedule is explicitly out of V1 scope even after `SCALE_READY`).
- **Attribution:** every order uses `client_order_id = f"{strategy_id}-{session_date:%Y%m%d}-{symbol}-{side}-{sha1(...)[:8]}"` (length ≤ 128), with `strategy_id = "jev-pullback-v1"`. An append-only `strategy_ledger` records every strategy order and fill. **The ledger is the source of truth for which shares belong to the strategy**; broker position quantities are never used as "the strategy's" holdings.
- **Cap rule, checked before every order:**  
`strategy_market_value + outstanding_strategy_buy_notional + proposed_buy_notional ≤ allocatable`, where `allocatable = min(MAX_STRATEGY_CAPITAL_USD, strategy_equity)` and `strategy_equity = MAX_STRATEGY_CAPITAL_USD + cumulative strategy P&L from the ledger`. Profits never increase deployment above the cap; losses reduce it. Also require broker cash ≥ proposed notional (no margin use).
- **Sells never exceed ledger quantity.** Sells are submitted with explicit `qty ≤ strategy_qty[symbol]`; the endpoints that close an entire position or all positions are never called.
- **External holdings:** at arming and at every session start compute `external_qty[symbol] = broker_qty − strategy_qty`. If `external_qty < 0` (broker holds *less* than the ledger says) → halt, fail closed. Positive `external_qty` is permitted and untouched.
- **Isolation limits (documented):** the broker aggregates positions per symbol, so isolation relies on our ledger. If the ledger and broker disagree beyond tolerance, if open orders exist that match our prefix but are absent from the ledger, or if the account is not a plain cash account (see L14) → **fail closed** and require manual reconciliation.



### 15.2 Interfaces

```python
class BrokerInterface(Protocol):
    def get_account(self) -> AccountState: ...
    def get_positions(self) -> list[Position]: ...
    def list_open_orders(self, client_order_prefix: str | None = None) -> list[OrderStatus]: ...
    def submit_order(self, order: OrderRequest) -> OrderAck: ...
    def get_order(self, client_order_id: str) -> OrderStatus: ...
    def cancel_order(self, client_order_id: str) -> None: ...
    def is_market_open(self) -> bool: ...
```

Implementations: `BacktestBroker` (simulated fills), `PaperBroker` (same Alpaca code, paper base URL; optional, never required), `LiveBroker` (hard-gated).

### 15.3 Independent live gates (ALL must hold; any failure ⇒ **zero** order submissions)


| Id                                                                                                                                              | Gate                               | Default / rule                                                                                                                                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L1                                                                                                                                              | `LIVE_TRADING_ENABLED=true`        | default `false`                                                                                                                                                                                                                                                               |
| L2                                                                                                                                              | `BROKER_ENV=live`                  | default unset/`paper`; live base URL used only if this is `live`                                                                                                                                                                                                              |
| L3                                                                                                                                              | Strategy-capital cap               | `MAX_STRATEGY_CAPITAL_USD ≤ 10.00`, cap rule of §15.1 satisfied for the proposed order                                                                                                                                                                                        |
| L4                                                                                                                                              | `TINY_LIVE_ACCEPTANCE.json` valid  | integrity digest verifies; status `TINY_LIVE_READY`; unexpired (≤ 90 days); research-config hash matches current config; parent `HISTORICAL_ACCEPTANCE.json` digest matches and is unexpired                                                                                  |
| L5                                                                                                                                              | Operator arm token                 | `jev live arm` requires typing `I-ACCEPT-REAL-MONEY-RISK-USD-10`; writes `.live_arm` with a ≤ 24 h TTL; absent/expired ⇒ blocked                                                                                                                                              |
| L6                                                                                                                                              | `--execute` flag                   | absent ⇒ dry run only                                                                                                                                                                                                                                                         |
| L7                                                                                                                                              | Kill switch clear                  | no `KILL_SWITCH` file, `TRADING_HALT` unset, drawdown breach not latched                                                                                                                                                                                                      |
| L8                                                                                                                                              | Fresh data                         | last bar equals last completed NYSE session; feature `available_at ≤ decision_time`                                                                                                                                                                                           |
| L9                                                                                                                                              | Jev data present                   | all 5 evaluations present, schema-valid, for every symbol needed; otherwise **no new Jev-dependent entries** (fail closed)                                                                                                                                                    |
| L10                                                                                                                                             | Model-ID verification              | resolved model in the response equals pinned `jev-1.13.0`; production-forward model artifact hash matches the artifact named in the acceptance file                                                                                                                           |
| L11                                                                                                                                             | Duplicate-order protection         | `client_order_id` unique in `strategy_ledger`; broker lookup before submit                                                                                                                                                                                                    |
| L12                                                                                                                                             | Reconciliation before trading      | ledger vs. broker positions/open orders reconcile (§15.1); mismatch ⇒ halt                                                                                                                                                                                                    |
| L13                                                                                                                                             | Deterministic risk module approval | §14 rules approve each order                                                                                                                                                                                                                                                  |
| L14                                                                                                                                             | No leverage / margin / shorting    | account verified as a cash-type account with no margin multiplier and shorting disabled (via broker account fields; names verified at implementation time; unknown/ambiguous ⇒ fail closed); only `buy` notional orders and `sell` qty ≤ ledger holdings are ever constructed |
| Each gate is checked by an independent function; a **safety test per gate** proves its failure alone yields zero calls to `submit_order` (§19). |                                    |                                                                                                                                                                                                                                                                               |




### 15.4 Pre-trade sequence (each session, dry run or live)

1. Verify gates L1–L2, L4–L7 → 2. read-only reconcile (L12) → 3. data freshness and Jev/model checks (L8–L10) → 4. generate intended orders from the production-forward model → 5. risk module (L13) → 6. cap check (L3) and account-type check (L14) → 7. duplicate check (L11) → 8. only if `--execute` and armed: submit; otherwise log the intended order and stop. Orders are submitted 08:30–09:25 ET for the open; sells for the 3-session exit are submitted the same way.



### 15.5 Audit trail

Every order (including dry-run intended orders) logs to append-only `live_orders`: signal, feature snapshot hash, Jev inference IDs and outputs, resolved model ID, model artifact hash, portfolio and ledger state, intended order, risk decision, cap computation, broker request/response, fill, estimated vs. realized cost, reconciliation result, and the mandatory execution-experiment sentence.

---



## 16. READINESS STATUSES AND ACCEPTANCE GATES



### 16.0 Three statuses, three artifacts

These answer three different questions and are never merged.


| Status                                                                                                                                                              | Question it answers                                                            | Artifact (in `reports/acceptance/`) | Produced by                                                             | What it authorizes                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HISTORICAL_RESEARCH_PASS**                                                                                                                                        | Is the *historical* evidence strong enough to justify continuing?              | `HISTORICAL_ACCEPTANCE.json`        | Phase 12 (one-shot sealed test)                                         | Continue the project; build production-forward models; run live infrastructure in dry-run. **Not** live money. Does **not** assert Jev's edge is free of training contamination. |
| **TINY_LIVE_READY**                                                                                                                                                 | Is the software safe and complete enough for a ≈$10 *operational shakedown*?   | `TINY_LIVE_ACCEPTANCE.json`         | Phase 14 (`jev live evaluate-tiny`)                                     | Live orders with strategy capital ≤ $10.00 only. No forward-session requirement.                                                                                                 |
| **SCALE_READY** (a.k.a. SCIENTIFIC_FORWARD_PASS)                                                                                                                    | Is there enough genuinely post-deployment evidence to *consider* more capital? | `SCALE_ACCEPTANCE.json`             | Phase 13B (`jev forward evaluate-scale`), after enough forward sessions | Permission to consider an allocation above $10 (the allocation schedule and code to support it are **out of V1 scope**; V1 live code cannot exceed $10.00).                      |
| Dependency: `TINY_LIVE_READY` requires `HISTORICAL_RESEARCH_PASS`; `SCALE_READY` requires `TINY_LIVE_READY`, executed live round trips, and clean forward evidence. |                                                                                |                                     |                                                                         |                                                                                                                                                                                  |


**Artifact integrity.** Each artifact is JSON with fields: `status`, `created_at`, `expires_at`, `git_commit`, `research_config_hash`, `qset_hash`, `model_resolved`, `gate_report_sha256`, `parent_artifact_digest` (for Tiny and Scale), `payload`, and `integrity_digest = sha256(canonical_json(all other fields))`. This is an **integrity digest, not a cryptographic signature**: it detects accidental edits, stale artifacts and mismatched configs; it does not defend against a malicious local user (the threat model is accidental activation, backed by a manual typed arming step). A real HMAC would need a separate secret and adds operational burden without addressing the actual threat, so it is not used. Loading code verifies the digest, expiry (Historical: 180 days; Tiny: 90 days; Scale: 90 days) and config lineage; any mismatch ⇒ the artifact is invalid.

### 16.1 HISTORICAL_RESEARCH_PASS gates (Phase 12)

All thresholds are fixed now, before any Jev result exists, and are meant to be demanding but achievable if a modest real effect exists. Always: qset_v1, model `jev-1.13.0`, normal costs unless stated; "mean over phases" means the mean over rebalance phases 0/1/2. Every result in this table is **potentially training-contaminated**.


| #                                                                                                                                                                                                                                                                                                                                         | Criterion                                                         | Threshold                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G1                                                                                                                                                                                                                                                                                                                                        | Contamination diagnostics (procedural)                            | D1–D6 of §11.4 executed with saved reports; `docs/contamination_disposition.md` present with a decision other than `stop`. **No numeric pass/fail; no claim of proof either way.**                                                                                                                                                                                                                                                                      |
| G2                                                                                                                                                                                                                                                                                                                                        | Reproducibility                                                   | The final experiment re-run from its manifest in the same environment reproduces trade lists and selections exactly and metrics within the tolerances of §20; Jev determinism audit complete (mitigated per §2.5 if needed)                                                                                                                                                                                                                             |
| G3                                                                                                                                                                                                                                                                                                                                        | Mechanics and sample sufficiency (**not baseline profitability**) | Accounting invariants hold every day; ledger ≡ fast path; cost scenarios applied as configured; label base rate on eligible rows in [40%, 60%]; for A and B each: ≥ 300 trades per phase in walk-forward OOS, mean exposure ≥ 30% of strategy capital; A's max drawdown at normal costs ≤ 40% (pathology check). Failure here means an implementation or design error: repair or halt. **A may have zero or negative CAGR and testing still proceeds.** |
| G4                                                                                                                                                                                                                                                                                                                                        | Predictive increment (walk-forward 2019–2023)                     | ΔlogLoss(B−A) < 0 with 95% date-block bootstrap CI excluding 0; ΔIC > 0; placebo P1 and P3 empirical p < 0.05 (percentile ≥ 95); P5 residual-Jev retains ≥ 50% of the gain                                                                                                                                                                                                                                                                              |
| G5                                                                                                                                                                                                                                                                                                                                        | Stability                                                         | ΔlogLoss ≤ 0 in ≥ 13 of the 20 folds (65%); sign consistent in ≥ 4 of the 5 calendar years 2019–2023                                                                                                                                                                                                                                                                                                                                                    |
| G6                                                                                                                                                                                                                                                                                                                                        | Economic increment and **Model B viability**                      | Mean over phases: net Sharpe(B) − net Sharpe(A) ≥ +0.20; net CAGR(B) > net CAGR(A); **net CAGR(B) > 0 at normal costs** (mean over phases, and ≥ 2 of 3 phases individually); net CAGR(B) > 0 at pessimistic costs; sign of ΔSharpe unchanged at stress costs                                                                                                                                                                                           |
| G7                                                                                                                                                                                                                                                                                                                                        | Concentration                                                     | Top 5 trades < 20% of total net P&L of B; no ETF > 35% of B's net P&L; ΔSharpe still > 0 after removing the best ETF                                                                                                                                                                                                                                                                                                                                    |
| G8                                                                                                                                                                                                                                                                                                                                        | Parameter stability                                               | ≥ 80% of one-step-perturbed configurations (regularization, K ∈ {2,3,4}, phase) keep ΔSharpe > 0                                                                                                                                                                                                                                                                                                                                                        |
| G9                                                                                                                                                                                                                                                                                                                                        | Drawdown                                                          | B max drawdown ≤ min(max(1.2 × A's, 25%), 35%)                                                                                                                                                                                                                                                                                                                                                                                                          |
| G10                                                                                                                                                                                                                                                                                                                                       | Sealed test (2024-01-02 → 2025-12-31, opened once)                | Models trained only on data ≤ 2023-12-29; ≥ 150 trades per phase; ΔlogLoss < 0 and ΔIC > 0 (same sign as G4); placebo P1 one-sided p < 0.10; net CAGR(B) > 0 at normal costs (mean over phases); G7 and G9 conditions hold on the sealed period; sealed partition accessed exactly once (access log)                                                                                                                                                    |
| Pass ⇒ write `HISTORICAL_ACCEPTANCE.json` (status `HISTORICAL_RESEARCH_PASS`). Fail on any ⇒ publish the "no demonstrated value" report and stop; tuning to pass is prohibited by protocol. The report states: *"Historical Jev evidence is potentially training-contaminated; this status does not establish that the edge is genuine."* |                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                         |




### 16.2 TINY_LIVE_READY conditions (Phase 14; operational, no forward-session requirement)


| #                                                                                                                                                                                                                                                                                                                                                                                                  | Condition                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| T1                                                                                                                                                                                                                                                                                                                                                                                                 | Valid `HISTORICAL_ACCEPTANCE.json` (digest verifies, unexpired, research-config hash identical to the current research configs)                              |
| T2                                                                                                                                                                                                                                                                                                                                                                                                 | Production-forward model artifacts (A and B) exist, built by the frozen procedure (§10.7) and hashed **before** any forward-label analysis (Phase 13B guard) |
| T3                                                                                                                                                                                                                                                                                                                                                                                                 | Live infrastructure complete; all safety tests (§19) pass at the current commit, recorded in `safety_test_report.json`                                       |
| T4                                                                                                                                                                                                                                                                                                                                                                                                 | Dry run for ≥ 10 consecutive sessions: intended orders generated, read-only broker reconciliation, zero discrepancies, zero gate bypasses                    |
| T5                                                                                                                                                                                                                                                                                                                                                                                                 | Kill-switch drill executed and logged (file, env var, drawdown trigger)                                                                                      |
| T6                                                                                                                                                                                                                                                                                                                                                                                                 | Pinned Jev model confirmed: config pins `jev-1.13.0`; every dry-run session's resolved model ID equals it                                                    |
| T7                                                                                                                                                                                                                                                                                                                                                                                                 | `MAX_STRATEGY_CAPITAL_USD ≤ 10.00` and strategy-capital isolation tests pass (including external-holdings cases)                                             |
| T8                                                                                                                                                                                                                                                                                                                                                                                                 | Account verified as cash-type, no margin, no shorting (L14)                                                                                                  |
| T9                                                                                                                                                                                                                                                                                                                                                                                                 | `docs/KNOWN_ISSUES.md` contains no open issue of severity `critical`                                                                                         |
| T10                                                                                                                                                                                                                                                                                                                                                                                                | Explicit operator arming at runtime (`jev live arm` typed phrase, ≤ 24 h token) — not stored in the artifact                                                 |
| Notes: (a) the 60/120-session forward requirements are **not** conditions here; (b) shadow data may already exist but is not required; (c) T-conditions are all about plumbing and safety — they carry no evidence about profitability. Weak, unstable, placebo-like, single-ETF, single-year, or cost-fragile historical evidence fails G4–G10 and therefore blocks `TINY_LIVE_READY` through T1. |                                                                                                                                                              |




### 16.3 SCALE_READY conditions (Phase 13B evaluator)

Scaling beyond the tiny allocation is **prohibited** until this status exists. This is the gate that raises confidence the historical result was not a training-contamination artifact, because forward observations post-date Jev's training. All forward statistics use the production-forward models (§10.7), sessions with decision time after the recorded deployment timestamp, labels fully realized, normal costs.


| #                                                                                                                                                                                                                                                                                                                                                                           | Condition                                                                                                                                                                                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S1                                                                                                                                                                                                                                                                                                                                                                          | `TINY_LIVE_READY` was granted and ≥ 20 completed live round trips exist in `strategy_ledger`                                                                                                                             |
| S2                                                                                                                                                                                                                                                                                                                                                                          | ≥ **120** completed forward sessions with realized labels (a **60-session** interim checkpoint report is generated automatically; it is informational and cannot grant status)                                           |
| S3                                                                                                                                                                                                                                                                                                                                                                          | Forward ΔIC(B−A) > 0 and ΔlogLoss(B−A) < 0 (point estimates)                                                                                                                                                             |
| S4                                                                                                                                                                                                                                                                                                                                                                          | Consistency with history: forward ΔIC is not significantly worse than the walk-forward ΔIC: z = (ΔIC_hist − ΔIC_fwd)/SE_fwd < 1.28 (one-sided 10%), SE from the date-block bootstrap                                     |
| S5                                                                                                                                                                                                                                                                                                                                                                          | No major drift: KS statistic < 0.15 between forward and 2019–2023 distributions for each of the 8 Jev model inputs; resolved model ID constant on every session; fraction of sessions with missing/invalid Jev data ≤ 2% |
| S6                                                                                                                                                                                                                                                                                                                                                                          | No major degradation after costs: hypothetical forward cumulative net return of B at normal costs > −5%, forward max drawdown ≤ 15%, and cumulative net return of B ≥ that of A − 2.5 percentage points                  |
| S7                                                                                                                                                                                                                                                                                                                                                                          | Execution realism: mean realized per-side cost on live round trips ≤ 2 × the normal assumption (≤ 5 bps); zero unreconciled orders; zero risk-limit breaches; no incidents                                               |
| S8                                                                                                                                                                                                                                                                                                                                                                          | Human sign-off, and no Jev version change during the forward window (a change restarts the window)                                                                                                                       |
| Power statement: 120 sessions is *weak* statistical evidence; `SCALE_READY` means "nothing contradicts the historical result and nothing broke," not "the edge is proven." If the 60-session checkpoint shows ΔIC < 0, a mandatory human review is triggered; if S2–S6 fail at 120 sessions, scaling remains prohibited and the report's conclusion is updated accordingly. |                                                                                                                                                                                                                          |




### 16.4 Statements that must appear in reports

- Historical: *"Results involving Jev features are potentially training-contaminated."*
- Live: *"This live phase is an execution experiment, not evidence of a profitable trading edge."*
- Possible conclusion on failure: *"Jev does not demonstrate useful predictive value."*

---



## 17. REPOSITORY ARCHITECTURE

```
jev-research/
  README.md  pyproject.toml  uv.lock  .env.example  .gitignore  .pre-commit-config.yaml  Makefile
  docs/                MASTER_SPEC.md, PREREGISTRATION.md, KNOWN_ISSUES.md, jev_cutoff.md, contamination_disposition.md, DATA_NOTES.md, LIVE_RUNBOOK.md, PHASE_<N>_REPORT.md
  config/              data.yaml features.yaml jev.yaml strategy.yaml backtest.yaml live.yaml universe.yaml risk.yaml
  data/                (git-ignored; DVC-style manifests committed)
    raw/  processed/  features/  jev_cache/  sealed/  forward/  models/
  registry/            experiments.sqlite (git-ignored), trial_ledger.csv (committed, append-only)
  src/jevlab/
    config/            pydantic settings, hashing of config, env loading
    utils/             hashing, canonical_json, time/calendar, logging, seeds, paths
    data/              providers (tiingo, fred, alpaca), availability.py (available_at, as_of), downloader, validators, store (parquet+duckdb)
    features/          technical.py, market_context.py, relative.py, pipeline.py, registry
    targets/           labels.py, eligibility.py
    jev/               client.py (httpx), schemas.py, state_builder.py, vocabulary.py, privacy.py, qset.py, cache.py, batch_runner.py, audit.py
    models/            logistic.py, gbm.py, pipelines.py, calibration.py, controls.py
    cv/                walkforward.py, purge.py, sealed_guard.py
    signals/           selection.py (top-K), aggregation.py
    portfolio/         state.py, sizing.py
    risk/              rules.py, killswitch.py
    backtest/          ledger.py, fast.py, costs.py, calendar.py
    evaluation/        metrics.py, predictive.py, bootstrap.py, tests.py (DM, DSR), placebo.py, robustness.py
    reporting/         templates/, build_report.py, plots.py
    brokers/           base.py, backtest_broker.py, paper_broker.py (optional, never required), live_broker.py, live_gate.py, strategy_ledger.py
    experiments/       manifest.py, runner.py, ledger.py (trial ledger), acceptance.py, artifacts.py (integrity-digested acceptance files), forward_capture.py, forward_registry.py, production_models.py, forward_eval.py
    cli/               typer apps: data, features, jev, run, report, live
  tests/ unit/ integration/ regression/ safety/
  scripts/  reports/ (reports/acceptance/ holds the three acceptance artifacts)  notebooks/ (exploration only; never source of record)
```

Responsibilities are one-per-package; `risk`, `brokers`, `jev` never import each other; `signals` never import `brokers`.

---



## 18. STORAGE / DATABASE SCHEMAS

**Strategy:** Parquet (hive-partitioned by symbol/year) for bars, features, Jev features; DuckDB for analytics over Parquet; SQLite for experiment/trial metadata, Jev cache index and live audit (single-user, transactional, trivially backed up); raw Jev request/response archive as zstd-compressed JSONL shards (append-only, immutable, hash-named).

```
market_bars(symbol TEXT, date DATE, vendor TEXT, raw_open, raw_high, raw_low, raw_close DOUBLE, raw_volume BIGINT,
            adj_open, adj_high, adj_low, adj_close DOUBLE, adj_volume DOUBLE, div_cash DOUBLE, split_factor DOUBLE,
            download_id TEXT, available_at TIMESTAMP, PRIMARY KEY(symbol, date, vendor))
downloads(download_id PK, vendor, params_json, downloaded_at, file_sha256, logical_digest, row_count)
external_series(series_id TEXT, obs_date DATE, value DOUBLE, available_at TIMESTAMP, retrieved_at TIMESTAMP, download_id TEXT, PRIMARY KEY(series_id, obs_date))
features(symbol, date, feature_version, decision_time TIMESTAMP, <20 feature columns + eligible>, PRIMARY KEY(symbol,date,feature_version))
states(state_hash PK, symbol, date, state_schema_version, state_json, features_hash)
jev_requests(request_hash PK, route, model_requested, state_hash, qset_version, purpose TEXT CHECK(purpose IN ('feature','audit','placebo','forward')), request_json_sha, created_at)
jev_inferences(inference_id PK, request_hash FK, attempt INT, model_resolved TEXT, response_json_sha, shard_file, shard_offset,
               input_tokens INT, output_tokens INT, latency_ms INT, http_status INT, created_at, UNIQUE(request_hash, attempt))
jev_features(symbol, date, qset_version, state_schema_version, model_resolved, <8 feature cols>, j_order_disagree,
             inference_ids TEXT, PRIMARY KEY(symbol,date,qset_version,state_schema_version,model_resolved))
labels(symbol, date, horizon INT, label_version, R DOUBLE, Rbar DOUBLE, X DOUBLE, y INT, roundtrip_cost_dec DOUBLE, y_cost INT, eligible INT, PRIMARY KEY(...))
signals(experiment_id, model_id, symbol, date, score DOUBLE, rank INT, selected INT, PRIMARY KEY(experiment_id, model_id, symbol, date))
trades(experiment_id, trade_id PK, symbol, entry_date, exit_date, side, qty, entry_px, exit_px, cost_entry, cost_exit, pnl_gross, pnl_net, phase INT)
equity(experiment_id, date, cash, positions_value, equity, PRIMARY KEY(experiment_id, date))
experiments(experiment_id PK, created_at, git_commit, git_dirty INT, data_hash, feature_version, state_schema_version, qset_version,
            qset_hash, model_resolved, strategy_config_json, backtest_config_json, seed INT, date_start, date_end, costs_json, status, notes)
metrics(experiment_id, model_id, scope TEXT, metric TEXT, value DOUBLE, ci_low, ci_high, PRIMARY KEY(experiment_id,model_id,scope,metric))
trial_ledger(trial_id PK, created_at, qset_version, state_schema_version, feature_set, model_class, grid_json, universe_version,
             horizon, purpose TEXT CHECK(purpose IN ('confirmatory','exploratory','audit','placebo')), experiment_id, result_summary)
sealed_access_log(access_id PK, at, user, experiment_hash, reason, partition)
live_orders(order_id PK, client_order_id UNIQUE, strategy_id, mode TEXT CHECK(mode IN ('dry_run','live')), cap_calc_json, resolved_model, model_artifact_hash, at, signal_json, feature_hash, inference_ids, portfolio_json, intended_order_json, risk_decision_json, request_json, response_json, fill_px, fill_qty, est_cost, realized_cost, reconcile_status)
strategy_ledger(ledger_id PK, client_order_id UNIQUE, strategy_id, symbol, side, qty DOUBLE, notional DOUBLE, status, fill_px, fill_qty, at, mode)   -- append-only; source of truth for strategy holdings
forward_registry(registry_id PK, deployment_timestamp TIMESTAMP, jev_model_id, qset_hash, feature_version, state_schema_version, git_commit, forward_start_date DATE)   -- immutable once written
forward_scores(session_date, symbol, model_artifact_id, score DOUBLE, selected INT, jev_inference_ids, PRIMARY KEY(session_date, symbol, model_artifact_id))
model_artifacts(model_artifact_id PK, kind TEXT CHECK(kind IN ('sealed_test','production_forward')), model_id, columns_json, train_end DATE, hyperparams_json, content_digest, git_commit, created_at)
acceptance_artifacts(artifact_id PK, status TEXT CHECK(status IN ('HISTORICAL_RESEARCH_PASS','TINY_LIVE_READY','SCALE_READY')), path, integrity_digest, parent_digest, gate_report_sha256, research_config_hash, git_commit, created_at, expires_at)
```

**Jev cache key:** `request_hash = sha256(canonical_json({route, model_requested, state, questions}))` with sorted keys, no whitespace, floats formatted by a fixed rule. Hits never call the API. Repeats (determinism audit) are extra `attempt` rows, not overwrites. Raw shards are never modified; a nightly job re-hashes shards against the index.

---



## 19. TECHNOLOGY STACK, TESTING AND CONFIG

**Stack:** Python 3.12, `uv` (env + lock), pandas + NumPy (sklearn-friendly; Polars unnecessary at this size), scikit-learn, LightGBM, SciPy, statsmodels (HAC, OLS), `arch` (stationary bootstrap) or a ≤40-line in-house implementation with tests, DuckDB, PyArrow, SQLite via SQLAlchemy 2, httpx + tenacity (retry/backoff), pydantic v2 (+ pydantic-settings), structlog, typer, Jinja2 + Plotly (HTML report), `exchange_calendars`, pytest + hypothesis, ruff, mypy (strict on `jev`, `backtest`, `risk`, `brokers`), pre-commit. Not used: vectorbt/backtesting.py/Backtrader/Zipline (see §10.5).

**Config files** (`pydantic` models validate each; every config is hashed into the experiment manifest; no magic constants in code): `data.yaml` (vendors, symbols file, date range, paths), `features.yaml` (windows, feature set ids), `jev.yaml` (route, endpoint, pinned model id, rate limit, retries, timeouts, bucket edges, state schema, qset file path+hash), `strategy.yaml` (eligibility, K, horizon, phases, model grid, seeds), `backtest.yaml` (cost scenarios, per-symbol multipliers, fill convention), `risk.yaml`, `live.yaml` (all flags default off; `MAX_STRATEGY_CAPITAL_USD: 10.00`; arm-token TTL; artifact expiry days). Secrets: env only (`JEV_API_KEY`, `TIINGO_API_KEY`, `FRED_API_KEY`, `BROKER_API_KEY`, `BROKER_API_SECRET`); `.env.example` has empty placeholders; a pre-commit hook + test scan the repo for key-like strings; logs redact `Authorization` and any env secret value.

**Test architecture**

- *Feature:* causality (truncate / perturb-future), known-value cases (RSI, SMA on toy series), NaN handling at warm-up, adjustment invariance, `available_at` filtering, VIX-lag test, information-parity test (every state field derivable from Model A's 17 columns).
- *Data:* duplicate dates, missing sessions vs calendar, OHLC consistency (L ≤ O,C ≤ H), non-positive prices, split/dividend anomaly detection, vendor-reconciliation thresholds.
- *Labels/CV:* round-trip cost label formula (2.5 bp/side + 0.3 bp fee ⇒ exactly 0.00053; round trip counted once); label window starts after feature time; purge gap ≥ embargo; no train row's label window intersects test period; fold boundaries deterministic.
- *Backtest:* hand-computed toy market (3 assets, 10 days) yields exactly predicted trades/P&L; ledger vs fast path equivalence (≤ 1e-9 relative); accounting invariant every day; cost sign tests (costs never increase equity); delayed-execution shifts fills correctly.
- *Jev:* mock transport for 200/401/422/429/529/timeout/malformed JSON; retry with backoff (jittered, honor `retry-after`); cache hit/miss correctness; hash stability across key order; model_resolved stored; refuses alias in research mode; refuses to run with qset hash mismatch; confidence recomputation matches vendor formula; Choice order-averaging math; **semantic privacy tests** (closed vocabulary; injected tickers, ISO dates, years, prices, floats, prohibited keys all detected; fixed horizon descriptors allowed).
- *Stats:* bootstrap on a known-effect simulation recovers the effect; under the null the permutation p-values are ~uniform (Type-I error ≈ 5% over 200 sims).
- *Risk:* each rule has a rejecting case; stale-data and holiday cases.
- *Live safety (*`tests/safety/`*):* with default config no order-capable `LiveBroker` can be built and a transport mock asserts **zero HTTP requests to order endpoints**; **one parametrized test per independent gate L1–L14 (§15.3) in which only that gate fails and a mock broker asserts** `submit_order` **was called zero times**; cap tests (external holdings present, profits never raise the cap, `TINY_LIVE_CEILING_USD`, cap > 10 raises); sells never exceed ledger quantity and close-position endpoints are never called; expired/tampered/lineage-mismatched acceptance artifacts block; duplicate `client_order_id` blocked; reconciliation mismatch halts; secrets never logged.
- *Regression:* golden-file metrics for a tiny fixture experiment, compared within the tolerances of §20 (never by parquet byte equality).

**Observability:** structlog JSON; events: `data.download`, `features.built`, `jev.request`, `jev.cache_hit`, `jev.cache_miss`, `jev.error`, `experiment.start/finish`, `trade.generated`, `trade.executed`, `risk.rejected`, `broker.error`, `sealed.access`, `availability.violation`, `forward.capture`, `acceptance.written`, `live.gate_failed`; every record carries `experiment_id`, `git_commit`, `run_id`; secret redaction processor.

---



## 20. REPRODUCIBILITY / VERSIONING / MODEL DRIFT

- **Experiment manifest** (`reports/<experiment_id>/manifest.json`, also in SQLite): git commit (+ dirty flag; dirty runs cannot support confirmatory results), `uv.lock` hash, **logical data digest** (see below), feature version, state schema version, qset version+hash, Jev route + requested model + resolved model, strategy/backtest/risk config hashes, seed(s), date ranges, cost scenarios, model artifact hashes, resulting metrics. `jev reproduce <experiment_id>` checks out the commit, verifies digests, reruns offline from cached data and compares outputs under the tolerances below.
- **Logical content digests (not file-byte hashes).** Parquet files can legitimately differ in bytes (writer version, row-group layout, compression, metadata). Tables are therefore hashed by *logical content*: load → select schema columns in sorted order → normalize dtypes (float64, int64, UTC timestamps, UTF-8 strings) → sort by primary key → serialize rows canonically (floats via `repr` of float64) → sha256, together with a hash of the schema. File-level sha256 is still recorded in the `downloads` manifest to detect corruption of raw vendor downloads, but it is **not** a cross-environment equality criterion.
- **Dependency pinning:** `uv.lock` committed; Python minor version recorded; key library versions (numpy, pandas, scikit-learn, lightgbm, scipy, statsmodels, pyarrow) written to every manifest.
- **Tolerance policy (documented, tested):**

  | Quantity                                                                      | Same environment                                                                     | Different machine/OS/library build |
  | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------- |
  | Accounting identities (cash + holdings = equity), ledger ≡ fast-path equality | absolute ≤ 1e-9 × max(1, equity)                                                     | same                               |
  | Feature values                                                                | relative ≤ 1e-12                                                                     | relative ≤ 1e-9                    |
  | Logistic coefficients/predicted probabilities                                 | relative ≤ 1e-9                                                                      | absolute ≤ 1e-6                    |
  | LightGBM predicted probabilities (`deterministic=True`, single thread)        | absolute ≤ 1e-9                                                                      | absolute ≤ 1e-6                    |
  | Selected ETFs / trade lists                                                   | exactly equal (ties broken by symbol order; scores within 1e-9 of a tie are flagged) | equal unless flagged near-tie      |
  | Summary metrics (CAGR, Sharpe, log-loss, IC)                                  | relative ≤ 1e-8                                                                      | relative ≤ 1e-5                    |
  | Bootstrap/placebo p-values (fixed seeds)                                      | exactly equal                                                                        | absolute ≤ 1/N_draws               |

- **Versioned artifacts:** `feature_version`, `state_schema_version`, `qset_version`, `audit_qset_version`, `label_version`, `universe_version`, `model_artifact_id`; each change produces new rows/files, never in-place edits.
- **Drift policy:** research requests use `jev-1.13.0` only (the client refuses `jev-latest` in research mode). If `jev-latest` starts resolving to a newer ID, nothing changes: cached data is untouched. A new version is evaluated by re-running Jev on dev partitions *as a new dataset*, compared with the old one on (a) feature distribution shift, (b) agreement on identical states, (c) the full A-vs-B experiment (logged as new trials). Daily shadow/live runs log the resolved model ID and abort Jev-dependent steps if it differs from the pinned ID. **A new Jev version restarts forward evidence** (§11.4, §16.3 S8): it may have been trained on the earlier forward period.
- **Determinism:** seeds fixed for all stochastic procedures; LightGBM `deterministic=True`, single-threaded in confirmatory runs; bootstrap and placebo seeds recorded.
- **Pre-registration:** `docs/PREREGISTRATION.md` (hypotheses, universe, splits, qset hash, thresholds from §16, analysis plan) is committed and tagged `prereg-v1` before bulk Jev inference; later edits require a new tag and an explicit "deviation" entry in the final report.

---



## 21. DETAILED IMPLEMENTATION PHASES

**Execution order:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → **13A** → 8 → 9 → 10 → 11 → 12 → **13B** → 14.
Phase 13A (forward capture) is implemented right after Phase 7 so clean forward data starts accumulating as early as technically possible; Phase 13B (production-forward models, forward scoring/evaluation, scale evaluator) must wait for Phase 12 because the production-forward models train on the sealed years only after the seal has been opened.

### PHASE 0 — Scaffold, governance, utilities

- **GOAL:** Reproducible repo skeleton; hashing, config, logging, seeds, trial ledger, pre-registration docs.
- **FILES:** `pyproject.toml`, `.env.example`, `.pre-commit-config.yaml`, `Makefile`, `docs/MASTER_SPEC.md`, `docs/PREREGISTRATION.md`, `docs/KNOWN_ISSUES.md`, `src/jevlab/{config,utils}/`, `config/*.yaml`, `registry/trial_ledger.csv`, `tests/unit/test_utils.py`.
- **TASKS:** pydantic config loading + config hashing; `canonical_json`; `logical_table_digest` (§20); structlog with secret redaction; calendar helper; seed helper; SQLite registry init (tables of §18); `.gitignore`; secret-scan test; units helpers (explicit `bps`↔decimal conversion functions with unit-typed names).
- **TESTS:** hash stability across key order/whitespace; logical digest invariant to row order/parquet encoding differences and sensitive to value changes; bps↔decimal conversions; config validation; redaction; calendar holidays.
- **ACCEPTANCE:** `make check` green; no secrets in repo; spec present.
- **DEPENDENCIES:** none. **OUTPUT:** installable package.



### PHASE 1 — Market data pipeline with point-in-time availability

- **GOAL:** Immutable, validated, reconciled daily bars + VIX, each observation carrying `available_at`.
- **FILES:** `data/providers/{tiingo,fred,alpaca}.py`, `data/availability.py`, `data/downloader.py`, `data/validators.py`, `data/store.py`, `config/universe.yaml`, `cli/data.py`, `docs/DATA_NOTES.md`.
- **TASKS:** download raw+adjusted OHLCV (2015-01-01→today) with rate limiting and resume; FRED `VIXCLS`; store parquet + `downloads` manifest; **availability policy (§6.1):** bars `available_at = session_date 18:00 ET`; FRED VIX observation for date d `available_at = next NYSE session after d at 08:00 ET` (conservative; cross-checked against FRED/ALFRED vintage dates and live `retrieved_at` when obtainable, recorded in `DATA_NOTES.md`); validators (§19); Alpaca/Stooq cross-check report.
- **TESTS:** mocked HTTP; validator fault injection; resume; manifest; availability assignment around weekends/holidays/half-days; `test_unavailable_rows_rejected` (any query with `decision_time` earlier than `available_at` excludes the row).
- **ACCEPTANCE:** all 13 symbols + VIX validated, zero missing sessions, reconciliation report, `available_at` populated for every row.
- **DEPENDENCIES:** 0. **OUTPUT:** `data/processed/bars/`, `data/processed/external/`, data quality report.



### PHASE 2 — Feature pipeline

- **GOAL:** Causal, versioned feature store for **20 features** + eligibility flag; Model A's 17-column set; information-parity check.
- **FILES:** `features/{technical,market_context,relative,pipeline}.py`, `targets/eligibility.py`, `tests/unit/test_features_*.py`.
- **TASKS:** implement §6 exactly; `vix_lag1_pct_252` uses the VIX observation of the *previous* session (§6.1); every feature row carries `decision_time` and is built only from rows with `available_at ≤ decision_time`; own-percentile helper; cross-sectional ranks; `feature_version="v1"`.
- **TESTS:** known values; truncate-and-compare causality; perturb-future; availability filter; VIX-lag test (changing VIX(t) does not change any feature at t); adjustment invariance; hypothesis scale-invariance; **information-parity test:** each field of `state_v1` is a deterministic function of columns in Model A's 17-column set.
- **ACCEPTANCE:** causality and availability tests pass; rebuilt feature store matches the previous build's **logical digest** (§20) in the same environment.
- **DEPENDENCIES:** 1. **OUTPUT:** `data/features/v1/`.



### PHASE 3 — Targets, walk-forward machinery, baseline models R0/A/D

- **GOAL:** Labels (primary and cost-clearing), purged walk-forward, sealed guard, baseline predictions.
- **FILES:** `targets/labels.py`, `cv/{walkforward,purge,sealed_guard}.py`, `models/{logistic,gbm,pipelines}.py`, `signals/selection.py`.
- **TASKS:** labels per §8, including `y_cost = 1[R > roundtrip_cost_dec]` with `roundtrip_cost_dec = 2·cost_bps_per_side/10_000 + sell_fee_bps/10_000` (decimal return; per-symbol multiplier applies to `cost_bps_per_side`); 20 folds (2019Q1–2023Q4), embargo 5 sessions (`t ≤ T0 − 5`); inner CV; fit-in-fold pipelines; OOS prediction store; sealed partition guard + access log; top-K selection.
- **TESTS:** label-after-feature; purge invariants; deterministic folds; global-fit leakage; guard; **cost-label formula test** (with 2.5 bps/side and 0.3 bp fee, threshold is exactly 0.00053; round-trip is never double-counted; unit-typed names); no 2024+ rows read.
- **ACCEPTANCE:** OOS predictions for R0, A, D on 2019–2023 with log-loss/AUC/rank-IC.
- **DEPENDENCIES:** 2. **OUTPUT:** `signals` parquet for A, D; fold report.



### PHASE 4 — Backtest engine, costs, risk module, metrics

- **GOAL:** Auditable ledger engine + fast path + metrics; **validate the baseline mechanics and statistical meaningfulness, not its profitability.**
- **FILES:** `backtest/{ledger,fast,costs,calendar}.py`, `portfolio/{state,sizing}.py`, `risk/{rules,killswitch}.py`, `evaluation/metrics.py`.
- **TASKS:** event loop per §10; 4 cost scenarios; 3 phases; fractional shares; risk module; metrics §12.2; benchmarks (equal-weight universe, SPY); a `baseline_sanity_report` covering: accounting invariants, ledger≡fast, trade counts per phase, exposure, label base rate, realized costs vs configured, drawdown pathology check, power/MDE statement.
- **TESTS:** hand-computed toy market; accounting invariant; ledger≡fast; cost monotonicity; delayed execution; risk rejections.
- **ACCEPTANCE:** R0/A/D run under 4 cost scenarios; sanity report produced. **Negative or zero baseline CAGR does not stop the project**; only failed mechanics checks (G3) or an obviously unusable design (documented, human-decided) do.
- **DEPENDENCIES:** 3. **OUTPUT:** baseline results + sanity report.



### PHASE 5 — Jev client, state builder, question set, cache (mock-first)

- **GOAL:** Correct, safe Jev integration without bulk spending.
- **FILES:** `jev/{client,schemas,state_builder,vocabulary,privacy,qset,cache,batch_runner}.py`, `config/jev.yaml`, `config/qset_v1.json`, `config/audit_qset.json`, `cli/jev.py`.
- **TASKS:** httpx client (limiter, retries, redaction); pydantic response models; state builder producing the closed-vocabulary `state_v1` (§7.1) with bucket edges from config; `vocabulary.py` enumerates every allowed leaf string; `privacy.py` implements the semantic anonymization validator; qset loader with hash check; cache; parser to 8 features + `j_order_disagree`; `jev smoke --live`.
- **TESTS:** mock transport for every status; cache; hash stability; alias refusal; schema contract tests from §2.4; **semantic privacy tests (§7.1):** closed vocabulary, no tickers/ISO dates/years/prices/floats/prohibited keys, while allowing the fixed horizon descriptors; property test over 5,000 random feature rows.
- **ACCEPTANCE:** offline tests pass; one manual smoke call stored and parsed.
- **DEPENDENCIES:** 2. **OUTPUT:** draft `qset_v1.json` (not frozen).



### PHASE 6 — Determinism, degeneracy and contamination diagnostics; question freeze

- **GOAL:** Run §11.4 diagnostics D1–D6 on data ≤ 2023; record human disposition; freeze qset.
- **FILES:** `jev/audit.py`, `reports/audit/`*, `docs/jev_cutoff.md`, `docs/contamination_disposition.md` (template).
- **TASKS:** dev sample (300 states, 2016–2018, no labels), degeneracy check; determinism audit (300 × 3); D1 (n = 600, 3 eras, exact binomial vs 1/3, logistic benchmark); D2 (2,000 states, benchmark comparison, no threshold); D3 scaffold; D5 (3,000 states, coarse state); freeze (`jev freeze-qset`).
- **TESTS:** exact-binomial correctness (known values), benchmark pipeline on synthetic data, era assignment boundaries, **sealed-partition refusal**, dry-run makes zero calls, budget cap.
- **ACCEPTANCE:** `AUDIT_REPORT.md` produced with every result labeled "warning signal, not proof"; frozen hash recorded; human disposition file completed (decision recorded by a person).
- **DEPENDENCIES:** 5. **OUTPUT:** audit report, frozen qset.



### PHASE 7 — Bulk historical Jev inference (dev partitions only)

- **GOAL:** Cached Jev features for all (symbol, date) rows 2016-01-04 → 2023-12-29 and the interim period 2026-01-02 → day before `FORWARD_START`; **never 2024–2025**.
- **FILES:** `jev/batch_runner.py`, `cli/jev.py batch`, `jev_features` parquet.
- **TASKS:** enumerate/dedupe states; resumable concurrency-limited run; budget guard; model-ID check; build `jev_features`; label datasets "potentially training-contaminated" in metadata.
- **TESTS:** resume; dedupe; budget guard; sealed refusal; wrong-hash refusal; row-count reconciliation.
- **ACCEPTANCE:** ≥ 99.9% valid responses (rest logged, never imputed).
- **DEPENDENCIES:** 6. **OUTPUT:** `data/features/jev/qset_v1/` (≈ 28,600 states upper bound).



### PHASE 13A — Forward capture (shadow input logging; no orders)

- **GOAL:** Start accumulating clean forward data as early as possible.
- **FILES:** `experiments/forward_capture.py`, `experiments/forward_registry.py`, `cli/run.py forward-capture`, scheduler docs.
- **TASKS:** after each completed NYSE session at ≥ 18:30 ET: fetch bars/VIX (with `available_at`, `retrieved_at`), build features and states, call Jev with the pinned ID (abort Jev part on resolved-ID mismatch), store everything with timestamps. **First successful run writes the immutable** `forward_registry` **row** (`deployment_timestamp`, Jev model ID, qset hash, feature/state versions); `FORWARD_START_DATE` = first session whose decision time is after it. Captures **inputs only**; no model scoring and no labels at this stage.
- **TESTS:** idempotent per day; holiday skip; Jev failure → day marked incomplete (never imputed); registry immutability; no broker imports.
- **ACCEPTANCE:** 10 consecutive clean sessions captured; registry populated.
- **DEPENDENCIES:** 1, 2, 5, 6. **OUTPUT:** `data/forward/`.



### PHASE 8 — Jev-enhanced models and attribution lineup

- **GOAL:** Models B, C, E, G on identical folds; comparison tables.
- **FILES:** `models/pipelines.py` (extend), `models/controls.py`, `experiments/runner.py`.
- **TASKS:** column-set registry (§4.3: A 17, B 25, C 8, D 17, E 25, F 25, G 17 + one-hot); same grids/folds; in-fold logit transforms; automatic trial-ledger entries; predictive metrics and backtests.
- **TESTS:** A and B differ only in columns (assert: B = A + 8); identical fold indices; ledger written; column-count assertions.
- **ACCEPTANCE:** results for A–G on 2019–2023; no 2024+ access. Report labels Jev results "potentially training-contaminated".
- **DEPENDENCIES:** 3, 4, 7. **OUTPUT:** comparison v0 (no conclusion language).



### PHASE 9 — Statistical evaluation layer

- **GOAL:** Bootstrap CIs, DM test, IC, deflated Sharpe, power statement.
- **FILES:** `evaluation/{predictive,bootstrap,tests}.py`.
- **TASKS:** stationary block bootstrap over dates (mean block 10); HAC DM (lag 5); DSR; Holm; MDE calculator; the forward-consistency z-statistic (§16.3 S4) as a reusable function.
- **TESTS:** planted-effect recovery; null calibration (Type-I ≈ 5%); known DSR values.
- **ACCEPTANCE:** `stats_summary.json`; primary hypothesis logged as confirmatory trial.
- **DEPENDENCIES:** 8. **OUTPUT:** stats summary.



### PHASE 10 — Robustness and placebo suite

- **GOAL:** P1–P6 and the robustness list of §13.
- **FILES:** `evaluation/{placebo,robustness}.py`, `jev/audit.py` (P6).
- **TASKS:** seeded placebo loops with `run_fast` (N per §13); P5 projection; ablations; regime/ETF/phase breakdowns; cost and delay sweeps; P6 new calls (budget-guarded).
- **TESTS:** placebo marginal/autocorrelation preservation; seed determinism; p-value formula.
- **ACCEPTANCE:** robustness data complete; < 2 h on a laptop.
- **DEPENDENCIES:** 9. **OUTPUT:** `robustness_report/`.



### PHASE 11 — Reporting

- **GOAL:** One HTML (+MD) report per experiment.
- **FILES:** `reporting/{build_report.py,templates/,plots.py}`.
- **TASKS:** sections: strategy definition, dataset and splits, feature definitions, Jev question definitions (exact text + hash), equity and drawdown curves, monthly returns, trade distribution, baseline comparison, Jev comparison, calibration/reliability, placebo histograms, cost sensitivity, robustness, statistical uncertainty with power statement, contamination-diagnostics summary, status table (three statuses), conclusion. Mandatory statements of §16.4. Rule-based conclusion able to emit exactly "Jev does not demonstrate useful predictive value." and "Inconclusive: study underpowered."
- **TESTS:** golden render; one test per verdict path; mandatory-statement presence tests.
- **ACCEPTANCE:** report builds from manifest alone.
- **DEPENDENCIES:** 9, 10. **OUTPUT:** `reports/<exp>/report.html`.



### PHASE 12 — Sealed one-shot test and HISTORICAL_RESEARCH_PASS

- **GOAL:** Single evaluation on 2024–2025 and the §16.1 verdict. **Does not depend on any forward-session count.**
- **FILES:** `experiments/acceptance.py`, `experiments/artifacts.py`, `cli/run.py final-test`.
- **TASKS:** preconditions (clean git, tags `prereg-v1`, `qset-v1-frozen`, trial ledger snapshot, SEAL token, contamination disposition ≠ stop); run Jev inference for sealed states (≈ 6,500); refit the frozen configuration on rows ≤ 2023-12-29 and **store the sealed-test model artifacts permanently (**`kind='sealed_test'`**)**; score the sealed period once; evaluate G1–G10; write gate report; **write** `HISTORICAL_ACCEPTANCE.json` **(integrity digest) only if every gate passes**.
- **TESTS:** each gate failing independently blocks the artifact; second run requires `--allow-second-look`, stamps "SEAL BROKEN" and blocks artifact creation; tampered gate report fails digest verification; non-evaluable gate blocks.
- **ACCEPTANCE:** gate report and verdict recorded regardless of pass/fail.
- **DEPENDENCIES:** 11, 13A (recommended), human sign-off. **OUTPUT:** `HISTORICAL_ACCEPTANCE.json` or rejection report.



### PHASE 13B — Production-forward models, forward scoring, SCALE evaluator

- **GOAL:** Models for forward/live use; clean-forward evaluation; `SCALE_ACCEPTANCE.json` when warranted.
- **FILES:** `experiments/production_models.py`, `experiments/forward_eval.py`, `cli/run.py {build-production-models,forward-score,forward-report,evaluate-scale}`.
- **TASKS:** after Phase 12 (seal opened), retrain the **exact frozen procedure** (same features, qset, model class, target, universe, horizon, strategy rules, hyperparameter grid and inner-CV method) on all pre-deployment rows whose labels are known by the deployment timestamp (embargo applies); save as `kind='production_forward'` with hash and training-end date, **keeping the sealed-test artifact untouched**; record model hashes in the ledger **before** any forward-label join (guard enforces); score captured forward inputs retrospectively and going forward; fill realized labels when windows complete; rolling A-vs-B tables; 60-session checkpoint report; S1–S8 evaluation → `SCALE_ACCEPTANCE.json` (integrity digest) only when all hold.
- **TESTS:** production artifacts differ from sealed artifacts only by training sample; forward-label join refused before model hashes exist; no forward row in training; idempotency; scale evaluator per-condition blocking; digest verification.
- **ACCEPTANCE:** production-forward artifacts built; forward report generated; evaluator behaves correctly on fixtures.
- **DEPENDENCIES:** 12, 13A. **OUTPUT:** production artifacts; `data/forward/` scores; forward reports.



### PHASE 14 — Live infrastructure ($10 strategy capital), hard-disabled

- **GOAL:** Auditable execution stack that cannot place orders by accident, plus the `TINY_LIVE_READY` evaluator.
- **FILES:** `brokers/`*, `brokers/live_gate.py`, `brokers/strategy_ledger.py`, `risk/*` (live wiring), `cli/live.py`, `tests/safety/*`, `docs/LIVE_RUNBOOK.md`.
- **TASKS:** §15 in full: gates L1–L14, strategy-capital isolation, ledger-based reconciliation, idempotent IDs, dry-run default, arm/disarm, kill switch, audit table, mandatory execution-experiment sentence; `jev live evaluate-tiny` checks T1–T9 and writes `TINY_LIVE_ACCEPTANCE.json`; `PaperBroker` available but unused by any gate.
- **TESTS:** one safety test per gate (L1–L14) proving zero `submit_order` calls; cap tests (including external holdings, profits not raising the cap, ceiling constant); idempotency; reconciliation mismatch halts; secrets not logged; artifact digest/expiry/lineage tests.
- **ACCEPTANCE:** all safety tests pass; code-review checklist; **no real order is sent in this phase.**
- **DEPENDENCIES:** 4, 12, 13B (production-forward models). **OUTPUT:** disabled live stack + runbook.

---



## 21A. CONSISTENCY CHECKLIST (CANONICAL INVARIANTS — NEVER CONTRADICT)


| Item                     | Invariant                                                                                                                                                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Universe                 | `SPY, QQQ, IWM, DIA, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY` (13 ETFs); fixed in `universe.yaml`                                                                                                       |
| Target                   | y_{i,t} = 1[X_{i,t} > 0], X = R − R̄; R = aO_{t+4}/aO_{t+1} − 1 (adjusted opens); R̄ = equal-weight mean of R over the 13 ETFs; defined on eligible rows (adj close > SMA200)                               |
| Secondary label          | y_cost = 1[R > roundtrip_cost_dec], roundtrip_cost_dec = 2 × cost_bps_per_side/10,000 + sell_fee_bps/10,000 (decimal)                                                                                       |
| Horizon                  | 3 sessions: entry open(t+1), exit open(t+4); rebalance every 3 sessions on phase 0 (phases 1, 2 as robustness; gates use the mean over phases)                                                              |
| Decision timestamp       | after close of session t, at ≥ 18:30 ET; only inputs with `available_at ≤ decision_time`                                                                                                                    |
| Execution timestamp      | orders for the open of t+1 (backtest: adjusted open + slippage; live: submitted 08:30–09:25 ET)                                                                                                             |
| External-data timing     | bars `available_at` = session date 18:00 ET; VIX lagged one session (`vix_lag1_pct_252`)                                                                                                                    |
| Feature counts           | 20 stored features; **Model A = 17 columns**; 3 stored-only (`gap_1d`, `rv_20`, `dollar_vol_20`)                                                                                                            |
| Jev feature count        | **8 model-input features** (+ `j_order_disagree` quality metric, not a model input)                                                                                                                         |
| Frozen question count    | **4 concepts / 5 evaluations**: 2 Noul + 1 Score + 2 Choice evaluations (a/b orders)                                                                                                                        |
| State                    | 16 feature-derived closed-vocabulary text fields + constant `fund_type` (`state_v1`); no numbers/tickers/dates/years/prices/raw measurements                                                                |
| Model columns            | A 17 · B 25 (A + 8) · C 8 · D 17 · E 25 · F 25 (A + 8 placebo) · G 17 + one-hot buckets                                                                                                                     |
| Train / dev              | features from 2015-01-02 (warm-up); research 2016-01-04 → 2023-12-29; walk-forward OOS 2019-01-02 → 2023-12-29 (20 quarterly folds, embargo 5 sessions)                                                     |
| Sealed dates             | 2024-01-02 → 2025-12-31; opened exactly once in Phase 12                                                                                                                                                    |
| Interim                  | 2026-01-02 → day before `FORWARD_START_DATE`; exploratory; potentially training-contaminated                                                                                                                |
| Forward data             | sessions whose decision time is after the immutable deployment timestamp in `forward_registry`, with the pinned Jev version released earlier; the only clean evidence; valid only for that Jev version      |
| Historical Jev results   | always labeled "potentially training-contaminated"                                                                                                                                                          |
| Model A                  | L2 logistic on the 17 deterministic columns                                                                                                                                                                 |
| Model B                  | identical pipeline/grid/folds as A on A's 17 columns + 8 Jev features                                                                                                                                       |
| Sealed-test model        | trained on ≤ 2023-12-29; archived permanently                                                                                                                                                               |
| Production-forward model | same frozen procedure, trained on all pre-deployment data; separate artifact; used for shadow scoring and live                                                                                              |
| Baseline profitability   | **not** required for testing Jev; Model B net CAGR > 0 at normal costs **is** required for `HISTORICAL_RESEARCH_PASS`                                                                                       |
| Tiny live capital cap    | `MAX_STRATEGY_CAPITAL_USD ≤ 10.00` strategy capital (not account equity); V1 cannot exceed it                                                                                                               |
| Statuses                 | `HISTORICAL_RESEARCH_PASS` → `TINY_LIVE_READY` → `SCALE_READY`; artifacts are integrity-digested, not signed                                                                                                |
| Paper trading            | **not mandatory**; `PaperBroker` optional                                                                                                                                                                   |
| Forward shadow logging   | **mandatory** (Phase 13A capture, 13B scoring); places no orders                                                                                                                                            |
| $10 live                 | requires `TINY_LIVE_READY`; no forward-session count; is an execution experiment only                                                                                                                       |
| Scaling beyond $10       | prohibited until `SCALE_READY`: ≥ 120 clean forward sessions + ≥ 20 live round trips + S3–S8; V1 code cannot exceed $10                                                                                     |
| Statistics               | date-block stationary bootstrap (mean block 10), HAC/Newey–West lag 5, placebo N: P1 1,000 · P2 500 · P3 500 · P4 500 · P5 single run · P6 single run; confirmatory tests: 2 (ΔlogLoss, ΔIC), Holm-adjusted |
| Costs                    | per side 1 / 2.5 / 6 / 12 bps (+ ≈0.3 bp sell-side regulatory fees); commission 0                                                                                                                           |


---



## 22. CURSOR MASTER IMPLEMENTATION SPEC

**Repository:** create `jev-research/` per §17; package name `jevlab`; Python ≥ 3.12; `uv` for env; `src/` layout.

**Non-negotiable rules for the agent**

1. This file (`docs/MASTER_SPEC.md`) is authoritative; **Section 21A is the canonical invariant list**. If an instruction conflicts with it, stop and report; do not improvise.
2. Priorities in order: no data leakage → scientific validity → reproducibility → correctness → risk control → simplicity → explainability → maintainability → performance. Never trade the first five for convenience.
3. One phase at a time, in the execution order of §21 (note 13A before 8 and 13B after 12). Do **not** implement later phases, "helpful" extras, additional features, strategies, universes, models, or question variants. Specifically **do not**: call Jev with a real key except in explicitly marked `--live` commands; touch `data/sealed/` outside Phase 12; add options/margin/shorting/leverage; add news features; implement live order submission before Phase 14; use `jev-latest` for research; add a backtesting framework; add any code path allowing strategy capital above $10.00; make paper trading a requirement of any gate.
4. Coding standards: type hints everywhere; mypy strict on `jev`, `backtest`, `risk`, `brokers`, `evaluation`, `experiments`; ruff (lint+format); pydantic models for configs and API payloads; pure functions where possible; no global state; no magic numbers (config only); **docstrings state time semantics** ("as of close of t", "available_at"); **variable names carry units** (`cost_bps_per_side`, `roundtrip_cost_dec`); small modules; explicit `NotImplementedError` rather than silent stubs; no `print`; structlog only.
5. Time and leakage hygiene: every DataFrame documents its time semantics; any function touching labels lives in `targets/labels.py`; rolling ops are trailing; every external input has `available_at` and is filtered by `available_at ≤ decision_time`; transforms are fit only on training rows via sklearn pipelines; randomness uses explicit seeds.
6. Secrets: environment variables only; never log, print or commit them; tests never need real keys or network.
7. Terminology: *shadow logging* = signals + hypothetical outcomes, no orders; *paper trading* = simulated broker orders (optional, never required); *integrity digest* ≠ cryptographic signature; *forward data* = post-deployment-timestamp data only; historical Jev results are "potentially training-contaminated".
8. Every phase ends with: `ruff check`, `ruff format --check`, `mypy`, `pytest -q` green, docs updated, and `docs/PHASE_<N>_REPORT.md` (what was built, decisions, deviations, how to run, known limitations). Then **stop**.
9. If data, API behavior, or a requirement is ambiguous, write the question into the report and choose the most conservative option; never fabricate data, never impute missing Jev responses.
10. Tests are part of the deliverable; names describe invariants; no test depends on wall-clock time or network.
11. Reproducibility: compare outputs using logical content digests and the tolerance table of §20, never byte equality of parquet files.

**Key interfaces (signatures are binding)**

```python
# jev/schemas.py
class NoulAnswer(BaseModel): type: Literal["noul"]; noul: float
class ChoiceAnswer(BaseModel): type: Literal["choice"]; choice: str; probabilities: dict[str,float]; confidence: float
class ScoreAnswer(BaseModel): type: Literal["score"]; score: float; legend: dict[str,str]; probabilities: dict[str,float]; confidence: float
class SystemOneResponse(BaseModel): model: str; answers: dict[str, Answer]; usage: Usage
# jev/client.py
class JevClient: def evaluate(self, state: dict, questions: dict) -> RawInference
# jev/privacy.py
def validate_state_privacy(state: dict, vocab: Vocabulary) -> list[PrivacyViolation]   # empty list == OK
# jev/cache.py
class JevCache: def get(self, request_hash: str) -> RawInference | None; def put(self, inf: RawInference) -> str
# cv/walkforward.py
def make_folds(dates, first_train_end, test_freq="Q", embargo_sessions=5) -> list[Fold]
# targets/labels.py
def roundtrip_cost_dec(cost_bps_per_side: float, sell_fee_bps: float) -> float   # 2*c/1e4 + fee/1e4
# backtest
def run_ledger(signals, bars, cfg: BacktestConfig, risk: RiskConfig) -> BacktestResult
def run_fast(...) -> BacktestResult  # must equal run_ledger
# experiments/artifacts.py
def write_acceptance(kind: Literal["historical","tiny_live","scale"], payload: dict, parent: Artifact | None) -> Path
def verify_acceptance(path: Path, *, research_config_hash: str, now: datetime) -> Artifact   # raises on any mismatch
# experiments/forward_registry.py
def register_deployment(...) -> ForwardRegistryRow   # immutable once written
# brokers/base.py: BrokerInterface (see §15.2)
# brokers/live_gate.py
def check_live_gates(ctx: LiveContext) -> GateReport   # L1..L14, each independently evaluated
```

**Libraries:** pandas, numpy, scikit-learn, lightgbm, scipy, statsmodels, duckdb, pyarrow, sqlalchemy, pydantic, pydantic-settings, httpx, tenacity, structlog, typer, jinja2, plotly, exchange-calendars, pytest, hypothesis, ruff, mypy, pre-commit, pyyaml, zstandard. (No vectorbt/backtrader/zipline.)

**Test strategy:** unit → integration (fixtures only, offline) → regression (golden metrics within §20 tolerances) → safety. CI target: `make check` runs everything under 5 minutes offline.

---



## 23. INDIVIDUAL CURSOR PROMPTS

Run in this order: 0, 1, 2, 3, 4, 5, 6, 7, **13A**, 8, 9, 10, 11, 12, **13B**, 14. Common protocol (embedded in every prompt): inspect the repo first; preserve earlier functionality; implement only this phase; write tests; run ruff, mypy, pytest; update docs; write `docs/PHASE_<N>_REPORT.md`; stop. Section 21A of `docs/MASTER_SPEC.md` is canonical.

**CURSOR PROMPT — PHASE 0**

```
Read docs/MASTER_SPEC.md fully (create it from the attached text if missing); Section 21A is the canonical invariant list. Create the repository skeleton of Section 17 and implement Phase 0 of Section 21 only.
Do: pyproject (uv, Python 3.12, deps from Sec 22), src/jevlab package skeleton (each subpackage with __init__.py and a docstring stating its responsibility), pydantic config loading + config hashing, utils: canonical_json, sha256_of, logical_table_digest (Sec 20: load -> sorted columns -> normalized dtypes -> sort by primary key -> canonical row serialization -> sha256, plus schema hash), seeds, paths, NYSE calendar helper (exchange_calendars), unit helpers with unit-typed names (bps_to_decimal, decimal_to_bps), structlog with secret-redaction processor, SQLite registry init with ALL tables of Sec 18 (including external_series, forward_registry, model_artifacts, acceptance_artifacts, strategy_ledger), .env.example with empty placeholders (JEV_API_KEY, TIINGO_API_KEY, FRED_API_KEY, BROKER_API_KEY, BROKER_API_SECRET), .gitignore (data/, registry/*.sqlite, .env), pre-commit (ruff, mypy, secret scan), Makefile `check`, docs/PREREGISTRATION.md populated from Sections 3, 4, 8, 16, docs/KNOWN_ISSUES.md (empty template with severity column), placeholder YAML configs in config/ with all keys of Sec 19 (live.yaml: LIVE_TRADING_ENABLED false, MAX_STRATEGY_CAPITAL_USD 10.00, TINY_LIVE_CEILING_USD constant lives in source, not config).
Tests: hash stability across key order/whitespace; logical_table_digest invariant to row order and parquet encoding but sensitive to value change; unit conversions; config validation failures; secret redaction; calendar holidays; repo-wide secret-scan test.
Do NOT implement data download, features, Jev code, models, or backtesting. Run ruff, mypy, pytest, fix everything, write docs/PHASE_0_REPORT.md, stop.
```

**CURSOR PROMPT — PHASE 1**

```
Inspect the repo and docs/MASTER_SPEC.md (Sec 5, 6.1, 21A). Preserve Phase 0. Implement ONLY Phase 1: market data pipeline with point-in-time availability metadata.
Providers behind a Provider protocol (httpx, tenacity retries, rate limiter, resumable): Tiingo EOD raw+adjusted (50 req/hour, 1000/day), FRED VIXCLS, Alpaca bars for cross-check only. config/universe.yaml = the 13 ETFs of Sec 21A, start 2015-01-01 (committed, hashed). Store hive-partitioned parquet (symbol/year) plus a downloads manifest (params, time, file sha256 for corruption detection, row count, logical digest).
Availability (data/availability.py): every bar row gets available_at = session date 18:00 America/New_York; every FRED observation for date d gets available_at = 08:00 ET on the next NYSE session after d (conservative); store retrieved_at when downloading; record in docs/DATA_NOTES.md what FRED/ALFRED vintage information (if any) could be used to verify publication timing. Provide a single query helper `as_of(table, decision_time)` that returns only rows with available_at <= decision_time; all downstream code must use it.
Validators: duplicate dates, missing NYSE sessions, OHLC consistency, non-positive prices, adj/raw ratio jumps, volume anomalies, cross-vendor reconciliation (share of days with |close diff| > 5bp; list >50bp days). CLI: `jev data download|validate|reconcile`.
Tests (offline, mocked HTTP): each validator with injected faults; resume after interruption; manifest; availability around weekends/holidays/half-days; test that as_of() excludes rows whose available_at is after decision_time (including the VIX for the same date).
Do NOT compute features, labels or models. Run checks, write docs/DATA_NOTES.md and PHASE_1_REPORT.md, stop.
```

**CURSOR PROMPT — PHASE 2**

```
Inspect repo + docs/MASTER_SPEC.md (Sec 6, 6.1, 21A); preserve phases 0-1. Implement ONLY Phase 2: feature pipeline.
Implement the 20 features of Sec 6 (trailing windows only; own-252-session percentile inclusive of t; cross-sectional ranks), the eligibility flag (adj close > SMA200), feature_version="v1", parquet output with decision_time (18:30 ET of session t) on every row. All inputs must come through as_of(decision_time). vix_lag1_pct_252 uses the VIX observation of the PREVIOUS session (percentile in its own trailing 252 window); there is no same-day VIX anywhere. Define MODEL_A_COLUMNS as exactly these 17: ret_1d, ret_3d, ret_5d, ret_20d, down_streak, px_vs_sma20, px_vs_sma50, px_vs_sma200, rsi_3, rsi_14, rv_pct_252, vol_ratio_20, vix_lag1_pct_252, spy_ret_5d, spy_px_vs_sma200, rel_ret_5d, xs_rank_ret_5d. The other stored features (gap_1d, rv_20, dollar_vol_20) are not model inputs in V1.
Tests: known values (RSI, SMA, percentile); truncate-and-compare causality; perturb-future (randomize all data after t, features at t unchanged); availability (a value with available_at > decision_time never influences a feature); VIX-lag test (changing VIX on date t changes no feature at t); warm-up NaNs only; hypothesis scale-invariance under price scaling; adjustment invariance; determinism; rebuilt store has identical logical_table_digest in the same environment (do NOT compare parquet bytes).
Do NOT build labels, models or Jev code. Run checks, write report, stop.
```

**CURSOR PROMPT — PHASE 3**

```
Inspect repo + spec (Sec 8, 9, 10.3-10.4, 21A); preserve prior phases. Implement ONLY Phase 3: labels, walk-forward CV, sealed guard, baseline models R0/A/D.
targets/labels.py (single module for all label logic): R = aO[t+4]/aO[t+1]-1 (adjusted opens), Rbar = equal-weight mean over the 13 ETFs, X = R - Rbar, y = 1[X>0] on eligible rows only. Secondary label: y_cost = 1[R > roundtrip_cost_dec] where roundtrip_cost_dec(cost_bps_per_side, sell_fee_bps) = 2*cost_bps_per_side/10_000 + sell_fee_bps/10_000, returned as a DECIMAL return (so 2.5 bps/side + 0.3 bp fee = 0.00053). Never double-count the round trip. Use the per-symbol cost multiplier from config. Names must carry units.
cv/walkforward.py: expanding window, first train 2016-01..2018-12, quarterly test folds 2019Q1..2023Q4 (20 folds), embargo 5 sessions: training rows must satisfy t <= T0 - 5 sessions (label needs open t+4), inner expanding CV (3 splits, same purge). Models: R0 (lowest 5d return among eligible), A = StandardScaler + L2 logistic on MODEL_A_COLUMNS (17), C in {0.01,0.1,1} by inner-CV log-loss; D = LightGBM small fixed hyperparameters from config, deterministic, single thread. All transforms fit in-fold via sklearn Pipeline. cv/sealed_guard.py: data after 2023-12-29 lives in data/sealed/; loading needs an explicit token and writes sealed_access_log. signals/selection.py: top-K eligible per rebalance date, ties broken by symbol order.
Tests: label window starts after feature time; no training label window intersects the test window; deterministic folds; global-fit leakage; guard blocks/logs; cost-label formula (exact value 0.00053 for 2.5 bps + 0.3 bp; two-sided counted once); assert no 2024+ rows are read in any dev pipeline.
Do NOT implement backtesting or Jev. Output OOS predictions + metrics (log-loss, AUC, rank-IC). Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 4**

```
Inspect repo + spec (Sec 10, 12.2, 14, 16.1 G3, 21A); preserve prior phases. Implement ONLY Phase 4: backtest engine, costs, risk module, metrics, and a baseline SANITY report.
backtest/ledger.py: daily event loop with explicit cash/positions/orders/fills; decision after close t, fill at adjusted open t+1 (+slippage), exit at open t+4; K=3 equal weight, rebalance every 3 sessions with phase offsets 0/1/2; fractional shares; no leverage/shorts. backtest/costs.py: cost_bps_per_side scenarios optimistic 1.0 / normal 2.5 / pessimistic 6.0 / stress 12.0 + sell_fee_bps from config, per-symbol multipliers. backtest/fast.py: vectorized equivalent. risk/rules.py: pure function enforcing Sec 14 (position cap and gross cap relative to STRATEGY capital, max orders/day, daily-loss halt, drawdown kill, stale data). evaluation/metrics.py: all Sec 12.2 metrics incl. beta/alpha with HAC SEs. Benchmarks: equal-weight universe, SPY.
Produce a baseline_sanity_report for R0/A/D covering: accounting invariants, ledger==fast, trades per phase, exposure, label base rate, realized vs configured costs, drawdown pathology check, and a power/MDE statement. IMPORTANT: baseline PROFITABILITY is NOT a pass/fail criterion; a negative or zero CAGR must not stop the pipeline. Only failed mechanics checks (G3) or a documented fundamental design error should halt, and the halt decision is human.
Tests: hand-computed toy market (3 assets, 10 days) exact trades/P&L; cash+holdings=equity every day (tolerance 1e-9 relative); ledger vs fast equal; costs never raise equity; delayed execution; each risk rule has a rejecting case.
Run R0/A/D under 4 cost scenarios and save tables. Do NOT add Jev. Run checks, report, stop.
```



**CURSOR PROMPT — PHASE 5**

```
Inspect repo + spec (Sec 2, 7, 21A); preserve prior phases. Implement ONLY Phase 5: Jev integration, mock-first. No bulk calls.
jev/client.py: httpx POST https://api.typesafe.ai/v1/systemone, Bearer from env JEV_API_KEY, token-bucket limiter, tenacity retries on 429/529/5xx honoring retry-after, timeouts, structured logs without secrets. Research mode REFUSES aliases (jev-latest/jev-preview) and requires the pinned id from config (jev-1.13.0). jev/schemas.py: pydantic models per Sec 2.4.
State: jev/vocabulary.py enumerates the CLOSED vocabulary of every allowed leaf string of state_v1 (generated from bucket config); jev/state_builder.py builds state_v1 of Sec 7.1 (16 feature-derived fields + constant fund_type) from features via own-trailing-252 percentile buckets; field names/values may contain fixed horizon descriptors (one_day, three_day, five_day, twenty_day, short/medium/long-term average) but NO ticker, calendar date, year, absolute price, raw return, raw RSI, raw VIX, raw volume or any raw numeric feature value. jev/privacy.py: validate_state_privacy(state, vocab) -> list of violations. Replace any regex 'no digits' idea with semantic tests: (1) every leaf string must be in the closed vocabulary; (2) reject strings matching any universe ticker or generic ticker-like uppercase tokens outside an allowlist; (3) reject ISO dates, month names, 4-digit years 1990-2100, and date-like patterns; (4) reject currency symbols, price-like numbers, percentages, floats; (5) reject prohibited metadata keys (symbol, ticker, date, timestamp, price, volume, vix, level); (6) the state must contain no numeric JSON types at all. Property-test over 5,000 randomly generated feature rows and include negative tests that inject each violation type.
jev/qset.py + config/qset_v1.json: the 4 concepts / 5 evaluations of Sec 7.2 (dislocation, downside_risk, regime_supportive, setup_type_a, setup_type_b with reversed option order), canonical hash, loader aborts on hash mismatch once a frozen hash is configured. config/audit_qset.json (separate hash, separate version) holds the audit-only questions of Sec 11.4; its responses are stored with purpose='audit' and never enter jev_features. jev/cache.py: SQLite index + zstd JSONL shards, request_hash = sha256(canonical_json({route, model_requested, state, questions})), immutable, repeats as separate attempts. Parser: noul p; score expectation and P(level>=3); choice probabilities averaged across both orders; recompute confidence from the documented formulas; j_order_disagree = total variation distance. CLI `jev smoke --live` makes ONE real call only when explicitly passed.
Tests: mock transport for 200/401/422/429/529/timeout/malformed; retry/backoff; cache hit/miss; hash stability; alias refusal; schema contract tests from the documented examples; privacy tests above; Choice order-averaging math.
Do NOT run bulk inference or audits. Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 6**

```
Inspect repo + spec (Sec 11.4, 7.4, 21A); preserve prior phases. Implement ONLY Phase 6: Jev diagnostics (warning signals, NOT proof, NOT pass/fail gates) and question freeze tooling.
All diagnostics use only sessions <= 2023-12-29 and MUST NOT touch data/sealed/ (assert in code and test). Diagnostic questions come from config/audit_qset.json, never from qset_v1.
(1) Dev sample: 300 states randomly from 2016-2018, NO labels joined; degeneracy check (each Noul sd > 0.05; no choice option p > 0.98 in more than 90% of rows). (2) Determinism audit: 300 states x 3 repeats stored as attempts; report per-feature sd and max abs diff. (3) D1 era diagnostic: 3 eras E1 2016-01-04..2018-06-29, E2 2018-07-02..2021-03-31, E3 2021-04-01..2023-12-29; n=600 anonymized states (200 per era, dates >= 5 sessions apart); Choice early/middle/late; exact one-sided binomial test against chance 1/3 (about 227/600 correct for p<0.01 - compute exactly); benchmark = multinomial logistic on MODEL_A_COLUMNS with blocked CV; report Jev vs benchmark with bootstrap CI of the difference. (4) D2 outcome diagnostic: 2,000 states from 2016-2018 (disjoint from the dev sample), audit-only Noul 'will this fund outperform ... over the next three trading days?', AUC and rank-IC with date-block bootstrap CIs next to the deterministic-benchmark values; NO threshold. (5) D3 scaffold: rank-IC of frozen-qset features vs X by era (descriptive; used after Phase 7). (6) D5: 3,000-state subset with a coarser state_v1c (3 buckets/field), fine vs coarse comparison. (7) `jev freeze-qset`: writes the qset hash to config, creates a ledger row, prints the git tag command (do not tag automatically).
The report must state in every diagnostic section: 'warning signal, not proof of memorization or of its absence'; and that all historical Jev results are potentially training-contaminated. Create a docs/contamination_disposition.md TEMPLATE (diagnostics run + report hashes, warnings observed, decision proceed|proceed_with_caveats|stop, reviewer, date) for a HUMAN to complete; the code must never auto-fill the decision.
Real calls only behind --live and a budget cap flag; default is dry-run printing planned calls and cost estimate (input tokens x $0.042/M).
Tests: exact-binomial values; era boundary assignment; benchmark pipeline on synthetic data (chance and strong-signal cases); sealed-access refusal; dry-run makes zero calls; budget cap.
Do NOT run bulk inference. Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 7**

```
Inspect repo + spec (Sec 7, 18, 21A); preserve prior phases. Implement ONLY Phase 7: resumable bulk historical Jev inference.
Preconditions enforced in code: qset hash equals the frozen hash; git tag qset-v1-frozen exists; prereg tag exists; working tree clean; docs/contamination_disposition.md exists with a decision other than 'stop' written by a human. Enumerate states for (symbol,date) rows from 2016-01-04 to 2023-12-29 plus the interim 2026-01-02 up to the session before FORWARD_START (if forward_registry is empty, up to the latest completed session); EXCLUDE 2024-01-02..2025-12-31 (sealed) - refuse without the sealed token (which Phase 7 never uses). Dedupe by request_hash, skip cache hits, bounded concurrency, checkpointing, safe resume, record status/latency/tokens, never impute missing responses. Budget guard aborts if projected spend exceeds the cap. Build jev_features parquet keyed by (symbol,date,qset_version,state_schema_version,model_resolved); abort if model_resolved != the pinned id. Tag dataset metadata 'potentially_training_contaminated'.
Tests: kill-and-resume, dedupe, budget guard, sealed refusal, wrong-hash refusal, row-count reconciliation, missing responses stay missing.
Provide a dry run that prints counts (about 28,600 states upper bound) and cost estimate first. Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 13A**

```
Inspect repo + spec (Sec 11.4, 16.3, 21 Phase 13A, 21A); preserve prior phases (1, 2, 5, 6, 7 must exist). Implement ONLY Phase 13A: forward INPUT capture. This places NO orders, imports NO broker code, and does NOT score models or join labels.
`jev run forward-capture`: after the latest completed NYSE session (calendar-aware, idempotent per date), at or after 18:30 ET: download bars and VIX with available_at and retrieved_at, build features via as_of(decision_time), build state_v1, validate privacy, call Jev with the pinned model id (abort the Jev part if the resolved id differs; never impute), and store features, states, raw Jev responses, resolved model id and timestamps under data/forward/. The FIRST successful run writes an immutable forward_registry row: deployment_timestamp (UTC), pinned model id, qset hash, feature/state versions, git commit. FORWARD_START_DATE = first session whose decision time is after deployment_timestamp. Forward data is the only clean evidence against training contamination and is valid only for the Jev version that produced it.
Document scheduling (cron/launchd) in docs. Tests: idempotency; holiday skip; Jev failure marks the day incomplete; model-id mismatch handling; registry immutability (second write raises); an import-boundary test proving no broker module is imported; no scoring/label code reachable.
Run checks, report, stop.
```



**CURSOR PROMPT — PHASE 8**

```
Inspect repo + spec (Sec 4.3, 9, 21A); preserve prior phases. Implement ONLY Phase 8: Jev-enhanced models and attribution lineup.
Column-set registry with ASSERTED sizes: A = 17 (MODEL_A_COLUMNS); B = A + the 8 Jev features (25); C = the 8 Jev features; D = LightGBM on A's 17; E = LightGBM on the 25; F = A + 8 placebo columns (25; interface only, filled in Phase 10); G = A + one-hot of the bucket labels sent to Jev (drop-first; count logged). B uses the identical pipeline, grid, folds, row set and seeds as A; only columns differ (assert in code). Logit transform (clip 0.01..0.99) and scaling fit in-fold. experiments/runner.py runs a model set end to end (folds -> OOS predictions -> selection -> backtests under all cost scenarios -> metrics) and writes manifests + trial_ledger rows automatically (purpose=confirmatory only for A vs B; everything else exploratory). All outputs labeled potentially_training_contaminated.
Tests: A/B column-diff assertion, column counts, identical fold indices, ledger entries, manifest completeness, no sealed access.
Output a comparison table with NO conclusion language. Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 9**

```
Inspect repo + spec (Sec 12, 16.3 S4, 21A); preserve prior phases. Implement ONLY Phase 9: statistical evaluation.
evaluation/bootstrap.py: stationary block bootstrap over DATES (mean block 10, seeded, 10k draws) for paired differences of log-loss, rank-IC, Sharpe, CAGR. evaluation/tests.py: Diebold-Mariano on per-date mean log-loss differences with Newey-West HAC (lag 5), deflated Sharpe ratio using the trial-ledger count, Holm and BH adjustments, a minimum-detectable-effect calculator for dSharpe and dIC given sample length and measured A-B correlation, and forward_consistency_z(ic_hist, ic_fwd, se_fwd) used by the scale evaluator. evaluation/predictive.py: log-loss, Brier, AUC, calibration bins (10 equal-count), reliability data, daily Spearman rank-IC with Newey-West t-stat.
Tests: planted-effect recovery (CI covers truth ~95%); null simulations (Type I ~5% over 200 sims; p-values ~uniform); DSR on published example values; MDE monotonicity.
Produce stats_summary.json for A vs B (primary hypothesis) and log it as a confirmatory trial. Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 10**

```
Inspect repo + spec (Sec 13, 21A); preserve prior phases. Implement ONLY Phase 10: robustness and placebo suite.
evaluation/placebo.py with seeded draws using run_fast and these counts: P1 within-date shuffle N=1000; P2 global shuffle N=500; P3 per-asset circular shift (lag >= 60 sessions) N=500; P4 AR(1)-matched Gaussian noise N=500; P5 deterministic-projection test (in-fold LightGBM from MODEL_A_COLUMNS to each Jev feature; run B with predicted-only and residual-only; single deterministic run, no N); P6 state-shuffle runner (new Jev calls on 3,000 states built from permuted rows) behind --live and a budget cap (single run). Empirical p = (1+#{placebo>=observed})/(1+N). evaluation/robustness.py: per-feature and per-Jev-question ablation, regime splits (SPY vs SMA200, VIX terciles using the LAGGED VIX, 2020, 2022), leave-one-ETF-out, phases 0/1/2, K in {2,3,4}, regularization +-1 grid step, cost sweep, delayed execution, Choice-order sensitivity, paraphrase sensitivity (2 paraphrases x 3,000 states, logged as trials).
Tests: placebo preserves marginals; circular shift preserves autocorrelation; seed determinism; p-value formula; runtime budget on a small fixture.
Output robustness data only (no narrative conclusions). Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 11**

```
Inspect repo + spec (Sec 16, 16.4, 21 Phase 11); preserve prior phases. Implement ONLY Phase 11: reporting.
reporting/build_report.py builds HTML (and Markdown) from an experiment manifest only (no recomputation). Sections: strategy definition; dataset and splits (marking sealed/interim/forward and the contamination status of each); feature definitions; Jev question definitions (exact text + hash); equity and drawdown curves; monthly return table; trade distribution; baseline comparison; Jev comparison; calibration/reliability; placebo histograms with observed values; cost sensitivity; robustness; statistical uncertainty incl. power statement; contamination-diagnostics summary (each labeled 'warning signal, not proof'); a three-status table (HISTORICAL_RESEARCH_PASS / TINY_LIVE_READY / SCALE_READY with artifact digests when present); final conclusion. Mandatory statements (Sec 16.4) must appear. The conclusion generator is rule-based from gate results and MUST be able to output exactly: 'Jev does not demonstrate useful predictive value.' as well as 'Inconclusive: study underpowered.' and a pass verdict; never use pass language when any gate fails; never imply that live $10 trading is evidence of an edge.
Tests: golden render on fixture; one test per verdict path; mandatory-statement presence tests; report builds with missing optional sections.
Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 12**

```
Inspect repo + spec (Sec 16.1, 10.7, 21 Phase 12, 21A); preserve prior phases. Implement ONLY Phase 12: the one-shot sealed final test and HISTORICAL_RESEARCH_PASS evaluation. Do NOT execute it for real; implement and test on fixtures. This phase must NOT depend on any forward-session count.
`jev run final-test` preconditions: clean git, tags prereg-v1 and qset-v1-frozen, trial-ledger snapshot hash recorded, human-provided SEAL token, contamination disposition decision != stop. It then: (1) runs Jev inference for sealed-period states (about 6,500, budget-guarded); (2) refits the frozen configuration on all rows <= 2023-12-29 and ARCHIVES these as model artifacts kind='sealed_test' (never deleted or overwritten); (3) scores 2024-01-02..2025-12-31 once; (4) evaluates gates G1-G10 of Sec 16.1 automatically, including the sealed-period placebo P1 (N=500) required by G10 (G1 checks that diagnostics reports and the human disposition file exist, with no numeric thresholds); (5) writes the gate report; (6) only if every gate passes, writes reports/acceptance/HISTORICAL_ACCEPTANCE.json via write_acceptance('historical', ...) with fields status, created_at, expires_at (180 days), git_commit, research_config_hash, qset_hash, model_resolved, gate_report_sha256, payload, and integrity_digest = sha256 of the canonical JSON of all other fields. This is an INTEGRITY DIGEST, not a cryptographic signature; never call it 'signed'. The report must state: 'Historical Jev evidence is potentially training-contaminated; this status does not establish that the edge is genuine.'
A second run for the same seal requires --allow-second-look, stamps SEAL BROKEN, and blocks artifact creation.
Tests on synthetic fixtures: each of G1-G10 failing independently blocks the artifact; second-run behavior; tampered gate report or artifact fails verify_acceptance; non-evaluable gate blocks; sealed partition accessed exactly once (access log); sealed-test models trained only on rows <= 2023-12-29.
Run checks, report, stop.
```

**CURSOR PROMPT — PHASE 13B**

```
Inspect repo + spec (Sec 10.7, 16.3, 21 Phase 13B, 21A); preserve prior phases (13A and 12 must exist). Implement ONLY Phase 13B: production-forward models, forward scoring/evaluation, and the SCALE evaluator. No orders, no broker imports.
(1) `jev run build-production-models`: AFTER the sealed test has been run, retrain the EXACT frozen procedure (same features, qset, model class, target, universe, horizon, strategy rules, hyperparameter grid and inner-CV method; no new design choices) for models A and B on all pre-deployment rows whose labels are known by deployment_timestamp (embargo applies; no forward row may be in training). Save as model_artifacts kind='production_forward' with hash, train_end, column lists, git commit; the sealed_test artifacts stay untouched. Record the hashes in the trial ledger BEFORE any forward-label join; forward-label code must refuse to run if production hashes are absent or newer than the first label join. Rationale to document: the sealed evaluation measured the ex-ante frozen procedure trained <= 2023; refitting the same procedure on more data after the seal is opened does not alter that evidence and is evaluated only on data after the training cutoff.
(2) `jev run forward-score`: score captured forward inputs (retrospectively and daily) with the production A and B; store scores, top-K selections, hypothetical trades under normal costs; (3) `jev run forward-report`: fill realized labels when 4-session windows complete; rolling A-vs-B IC/log-loss/hypothetical net returns; KS drift of the 8 Jev features vs 2019-2023; automatic 60-session checkpoint report (informational only); (4) `jev run evaluate-scale`: evaluates S1-S8 of Sec 16.3 (S1 uses strategy_ledger live round trips) and writes SCALE_ACCEPTANCE.json (integrity digest, parent = TINY_LIVE digest, expires 90 days) only when all hold. V1 live code must NOT gain any ability to exceed $10.00 because of this artifact.
Tests: production vs sealed artifact differ only by training sample; forward-label join refused before model hashes exist; no forward row in training; S1-S8 each blocking independently; digest verification; 60-session report cannot grant status.
Run checks, report, stop.
```



**CURSOR PROMPT — PHASE 14**

```
Inspect repo + spec (Sec 15, 16.2, 19, 21 Phase 14, 21A); preserve prior phases. Implement ONLY Phase 14: live infrastructure, HARD-DISABLED. Do not send any real order or make any order-submission broker call in tests or by default.
Implement brokers/base.py BrokerInterface (Sec 15.2), BacktestBroker, PaperBroker (optional, never required by any gate) and LiveBroker (Alpaca REST via httpx; fractional notional market orders, day TIF; asset fractionable check). brokers/strategy_ledger.py: append-only ledger of strategy orders/fills identified by client_order_id = f"{strategy_id}-{YYYYMMDD}-{symbol}-{side}-{sha1[:8]}" (strategy_id 'jev-pullback-v1', length <= 128). Strategy-capital isolation (Sec 15.1): constant TINY_LIVE_CEILING_USD = 10.00 in source; MAX_STRATEGY_CAPITAL_USD (default 10.00) above the ceiling raises at startup; before each order require strategy_market_value + outstanding_strategy_buy_notional + proposed_buy_notional <= min(MAX_STRATEGY_CAPITAL_USD, strategy_equity) and broker cash >= proposed notional; sells use explicit qty <= ledger quantity; NEVER call close-position/close-all endpoints; compute external_qty = broker_qty - ledger_qty each session and halt if negative; fail closed on any ledger/broker mismatch or foreign-prefix open order not in ledger. Do NOT use total account equity for any cap.
brokers/live_gate.py: check_live_gates() evaluates L1-L14 of Sec 15.3 as independent functions: LIVE_TRADING_ENABLED (default false); BROKER_ENV=live; strategy cap; valid TINY_LIVE_ACCEPTANCE.json (integrity digest, status, <= 90 days, research_config_hash match, parent HISTORICAL digest valid); typed arm phrase 'I-ACCEPT-REAL-MONEY-RISK-USD-10' creating .live_arm with <= 24h TTL; --execute; no KILL_SWITCH/TRADING_HALT/latched drawdown; fresh data (last completed session, available_at <= decision_time); Jev data present (fail closed: no new Jev-dependent entries); resolved model id == pinned id and production model artifact hash == the one in the acceptance file; duplicate-order protection; reconciliation before trading; risk module approval; account verified cash-type with no margin and no shorting (fail closed if fields ambiguous). The CLI is dry-run by default; reads (account/positions/orders) are allowed in dry run, order submission is not. Print the sentence 'This live phase is an execution experiment, not evidence of a profitable trading edge.' at startup and in every audit record. `jev live evaluate-tiny` checks T1-T9 of Sec 16.2 (no forward-session count) and writes TINY_LIVE_ACCEPTANCE.json with an integrity digest (never call it signed).
Safety tests (tests/safety): for EACH gate L1-L14 a parametrized test where ONLY that gate fails and a mock broker asserts submit_order was called ZERO times; default config cannot construct an order-capable LiveBroker; a transport mock proves zero HTTP requests to order endpoints on default paths; cap tests (external holdings present, profits do not raise the cap, ceiling constant, cap > 10 raises); sells never exceed ledger qty; idempotent IDs; reconciliation mismatch halts; expired/tampered/mismatched-lineage artifacts block; secrets never logged.
Add docs/LIVE_RUNBOOK.md. Run checks, report, stop. Do not arm or enable anything.
```

---



## 24. MAJOR RISKS / FAILURE MODES

**Technical (catastrophic if unnoticed)**

1. Look-ahead via features, adjusted prices, labels, fold boundaries, **or external inputs whose publication time is after the decision time (VIX)** → fake edge. *Controls:* causality tests, `available_at` filtering, one-session VIX lag, purge, single label function.
2. **Jev training contamination** (era recognition, outcome recall) → fake Jev edge that historical testing cannot rule out. *Controls:* anonymized closed-vocabulary states, diagnostics treated as warning signals, "potentially training-contaminated" labeling, forward-only scale gate (§16.3). This remains the most likely way to fool ourselves.
3. Silent Jev model change (alias drift, route change) → inconsistent features; also a new version can invalidate forward evidence. *Controls:* pinned ID, resolved-ID check every session, cache keyed by route+model, forward window restart on version change.
4. Nondeterministic or degenerate Jev outputs (always 0.9, order bias). *Controls:* determinism and degeneracy audits, two-order Choices.
5. Cache corruption/duplication → silently different datasets. *Controls:* immutable shards, logical digests, row-count reconciliation.
6. Vendor data errors (bad adjusted series, missing days). *Controls:* validators, cross-vendor reconciliation.
7. Duplicate or runaway live orders; **bot trading the user's other holdings or exceeding $10 of strategy capital**. *Controls:* idempotent `client_order_id`, strategy ledger as source of truth, cap rule on strategy exposure (not account equity), sells ≤ ledger qty, no close-position endpoints, per-gate zero-order tests, kill switch, reconcile-before-trade.
8. Isolation imperfect because the broker aggregates positions per symbol → fail closed on any ledger/broker mismatch; documented limitation.
9. Config/secret leakage. *Controls:* env-only, redaction, scan tests.
10. Rate-limit changes mid-batch. *Controls:* resumable runner, token bucket.
11. Misreading `TINY_LIVE_READY` as proof of an edge. *Controls:* mandatory execution-experiment sentence, three separate artifacts, scale gate.

**Statistical**

1. Underpowered comparison mistaken for evidence of absence or presence → power statement, "inconclusive" verdict; forward 120 sessions is still weak evidence.
2. Forking paths (any look at dev results then a tweak) → trial ledger, deflated Sharpe, frozen qset, sealed test.
3. Overlapping 3-day labels inflate significance → date-block bootstrap, HAC (lag 5).
4. Regime dependence (one good year drives everything) → per-year/regime consistency gates.
5. Mean-reversion alpha decayed or net-negative after costs → does not by itself stop Jev testing, but Model B must be positive net of normal costs for `HISTORICAL_RESEARCH_PASS`.
6. Correlated universe: apparent breadth is illusory → cross-sectional label and date-level inference.
7. Survivorship-free universe is also *selection*-biased toward ETFs that remain popular; acceptable for ETFs launched 1998–2000 but noted.
8. Re-training the production-forward model after the seal is opened could be misused as a second look → hashes recorded before any forward-label join; changes to procedure are new trials.

**What would convince me Jev provides no edge (any of):**

- ΔlogLoss CI includes 0 and placebo percentile < 95 in walk-forward, **or** the same-sign effect fails in the sealed test.
- B ≈ G (a plain bucketed-feature logistic matches B), or B ≈ B with predicted-only Jev (P5): Jev is a re-encoding.
- Gain disappears with P1/P3 placebos matching the real effect.
- Forward ΔIC ≤ 0 over ≥ 120 sessions, or forward results significantly worse than history (S4), even if historical results were strong (the contamination-artifact reading).
- Net improvement vanishes at pessimistic costs.
In any of these cases the correct conclusion is published as-is and the project stops; adding questions or features after seeing the failure is prohibited (new qset versions can only start a *new* pre-registered study).

---



## 25. ESTIMATED API / DATA COSTS

Session counts (NYSE): 2016–2023 = 2,012; 2024–2025 = 502; 2016–2025 = 2,514; 2026 to date ≈ 190. Counts below are *upper bounds* before deduplication of identical states.


| Item                                                                                                                                                                                                                                                                                    | Quantity                                                            | Estimate                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------- |
| Feature states, dev partition (13 ETFs × 2,012 sessions)                                                                                                                                                                                                                                | ≈ 26,200                                                            |                                                         |
| Feature states, interim 2026 (13 × ≈190)                                                                                                                                                                                                                                                | ≈ 2,500                                                             |                                                         |
| **Phase 7 dataset calls**                                                                                                                                                                                                                                                               | **≈ 28,600** (1 request each, all 5 evaluations batched)            |                                                         |
| Feature states, sealed 2024–2025 (13 × 502) — Phase 12                                                                                                                                                                                                                                  | ≈ 6,500                                                             |                                                         |
| **Total feature-state calls**                                                                                                                                                                                                                                                           | **≈ 35,100**                                                        |                                                         |
| Tokens per request (state ≈ 450 + 5 evaluations ≈ 100–150 each)                                                                                                                                                                                                                         | ≈ 1,100–1,500 (plan on 2,500 worst case)                            |                                                         |
| Input tokens, all feature states                                                                                                                                                                                                                                                        | ≈ 39–53 M (worst case ≈ 88 M)                                       | **≈ $1.7–$2.2** (worst case ≈ $3.7) at $0.042/M [V]     |
| Diagnostics: dev sample 300 + determinism 900 + D1 600 + D2 2,000 + D5 3,000                                                                                                                                                                                                            | ≈ 6,800 calls                                                       | < $0.60                                                 |
| P6 state-shuffle 3,000 + paraphrase sensitivity 2 × 3,000                                                                                                                                                                                                                               | ≈ 9,000 calls                                                       | < $1.00                                                 |
| **Audit/placebo/variant calls total**                                                                                                                                                                                                                                                   | **≈ 15,800 (≈ 16,000)**                                             | < $1.60                                                 |
| Forward capture / live                                                                                                                                                                                                                                                                  | ≈ 13 calls/day                                                      | < $0.01/day                                             |
| **Recommended Jev budget cap**                                                                                                                                                                                                                                                          |                                                                     | **$25** (large headroom; budget guards abort beyond it) |
| Wall-clock for Phase 7                                                                                                                                                                                                                                                                  | ≈ 28,600 calls at 10–20 req/s                                       | ≈ 25–50 min (limits are dynamic; resumable)             |
| Market data                                                                                                                                                                                                                                                                             | Tiingo free, FRED free                                              | $0                                                      |
| Optional                                                                                                                                                                                                                                                                                | Massive Developer $79/mo if reconciliation fails; Tiingo paid plans | optional                                                |
| Broker                                                                                                                                                                                                                                                                                  | Alpaca commission-free; no minimum deposit                          | $0 + the ≤ $10 strategy capital at risk                 |
| Caveats: pricing is for the direct TypeSafe route [V]; OpenRouter/Cloudflare/Vercel/AIML routes are billed separately at their own rates [3P]. Token counts are estimates — Phase 6 measures the real mean from `usage.input_tokens` and the Phase 7 dry run recomputes the projection. |                                                                     |                                                         |


---



## 26. FINAL RECOMMENDATION

1. **Do it, but as a falsification study.** The infrastructure is cheap; the main cost is discipline. Expect the most likely outcome (my prior ≈ 85–90%) to be "Jev does not demonstrate useful predictive value," and treat that as a success.
2. **Order of operations that protects you:** build the baseline and verify its *mechanics* first (Phases 1–4). A mildly unprofitable baseline does not stop the Jev test (Model B turning a losing baseline into a winning one would be exactly the interesting result); only broken mechanics or a fundamentally unusable design halt the project. Start forward capture (Phase 13A) as soon as the question set is frozen, because clean forward days cannot be created retroactively.
3. **Treat contamination honestly.** Historical diagnostics are warning signals, not proofs; every historical Jev number is "potentially training-contaminated". Only forward data after the recorded deployment timestamp can raise confidence that a historical result is not a memorization artifact. Ask TypeSafe for the training cutoff.
4. **Pre-register and freeze** (qset hash, thresholds in §16, universe, splits) before generating Jev features; open the 2024–2025 seal exactly once. A pass yields only `HISTORICAL_RESEARCH_PASS` — "strong enough to continue."
5. **$10 live trading is a separate, narrower decision.** `TINY_LIVE_READY` requires the historical pass plus operational safety (production-forward models, ≥ 10 dry-run sessions, kill-switch drill, per-gate safety tests, strategy-capital cap of $10.00 that cannot touch the rest of the account). It needs **no** forward-session count because it is an execution shakedown, and it must never be read as evidence of profitability.
6. **Scaling is blocked** until `SCALE_READY`: ≥ 120 clean forward sessions with Jev's incremental rank-IC/log-loss still positive and consistent with history, no drift, no cost-adjusted degradation, and realistic live execution. Even then the evidence is modest; V1 code cannot exceed $10.00 regardless.
7. **Open items needing your action:** (a) confirm your TypeSafe direct-API access (or choose the OpenRouter route); (b) obtain Tiingo, FRED, TypeSafe and Alpaca keys; (c) ask TypeSafe for the training cutoff; (d) verify current SEC/FINRA sell-side fee rates, Alpaca's current free market-data feed, FRED's VIX publication timing, and Alpaca's account-type fields before Phases 1, 4 and 14 finalize defaults.
8. **Shadow vs paper:** shadow logging (Phase 13A/13B) places no orders and is mandatory because it is the only uncontaminated data source; paper trading is optional and never required.

*Not financial advice. This is a research design; nothing here is a recommendation to trade.*