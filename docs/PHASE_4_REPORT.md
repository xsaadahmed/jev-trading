# Phase 4 Report — Backtest engine, costs, risk, metrics, baseline SANITY

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§10, 12.2, 14, 16.1 G3, 21 Phase 4, 21A  
**Preserved:** Phases 0–3 (scaffold, data, features, labels/WF/baselines)

---

## What was built

### Backtest (`src/jevlab/backtest/`)

| Module | Responsibility |
| ------ | -------------- |
| `costs.py` | `CostModel`: scenarios optimistic/normal/pessimistic/stress (1.0/2.5/6.0/12.0 bps/side), `sell_fee_bps`, per-symbol multipliers; buy/sell fill prices; mandatory costs (no silent zero default) |
| `calendar.py` | Entry open `t+1`, exit open `t+4` session offsets |
| `ledger.py` | Daily event loop: cash/positions/orders/fills; decision after close t; fill at adj open t+1 (+adverse slippage); exit open t+4; K=3 equal weight; rebalance every 3 sessions with phase 0/1/2; fractional shares; long-only; accounting identity each day (rel tol 1e-9) |
| `fast.py` | Fast path matching ledger equity (same flags); cohort vectorized net-return helper for diagnostics |

### Portfolio (`src/jevlab/portfolio/`)

- `state.py` — `PortfolioState` / `Position` for risk and ledger.
- `sizing.py` — equal-weight `equity/K` notionals; unused slots stay cash.

### Risk (`src/jevlab/risk/`)

- `rules.py` — pure `(proposed, portfolio, market, config) → approved/rejections`: max position 34%, max gross 100% of strategy capital, max 6 orders/day, daily-loss halt 5%, drawdown kill 20%, stale data, duplicates, insufficient cash.
- `killswitch.py` — file / env / drawdown halt detection.

### Evaluation (`src/jevlab/evaluation/metrics.py`)

§12.2 economic metrics: total return, CAGR, ann vol, Sharpe, Sortino, max DD, Calmar, hit rate, avg/median trade, profit factor, turnover, exposure, trade count, holding period, CVaR 5%/1%, skewness, beta/alpha vs SPY with HAC SEs, contribution tables, top-5 P&L share. Benchmarks: SPY and equal-weight universe. Power/MDE statement for sanity reports.

### Sanity report (`src/jevlab/reporting/baseline_sanity.py`)

`baseline_sanity_report` covering: accounting invariants, ledger≡fast, trades per phase, exposure, label base rate, realized vs configured costs, drawdown pathology (A @ normal ≤ 40%), cost monotonicity, power/MDE note.

**Profitability is not a gate.** Negative/zero CAGR does not fail the pipeline; only G3 mechanics (or a human-documented design halt) do.

### CLI

```bash
jev run sanity
```

Writes `reports/phase4_sanity/baseline_sanity_report.md`, `.json`, and `baseline_results.csv` (R0/A/D × 4 cost scenarios × phases).

---

## Decisions

1. **Fill costs** are baked into adverse open prices (`± side_cost_dec`); sell fee deducted once from sell proceeds (never double-counted).
2. **Risk on research path** uses strategy capital = backtest initial capital; position/gross caps also respect cash (no leverage).
3. **Fast path** shares the ledger engine under identical flags so ledger≡fast holds at 1e-9; vectorized cohort returns remain available for placebo loops later.
4. **Delayed execution** mode fills buys at `adj_close[t+1]` instead of `adj_open[t+1]`.

No §21A conflicts. Jev not added.

---

## How to run

```bash
# after Phase 1–3 artifacts (or jev run baselines first)
jev run sanity
make check
```

---

## Test results

| Check | Result |
| ----- | ------ |
| ruff / format | pass |
| mypy | pass (69 source files) |
| pytest | all pass (Phase 0–4 suite) |
| `make check` | pass |

Covered: hand-computed toy market (3 assets, 10 days) trades/P&L; cash+holdings=equity daily; ledger≡fast; costs never raise terminal equity; delayed execution; each risk rule rejecting case; metrics/MDE smoke; sanity report mechanics on synthetic panel (3 models × 3 phases × 4 costs).

---

## Known limitations

- Full R0/A/D × cost tables on real history require Phase 1–3 data/features/OOS preds on disk (`jev run sanity`).
- Fast path is parity-first (shared engine); a separate matrix engine can replace it later without changing the public API.
- No Jev, sealed final test, or live trading in this phase.

---

## Stop

Phase 4 is complete. Do not proceed to Phase 5 until explicitly requested.
