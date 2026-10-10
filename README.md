# Jev Trading Research

Controlled experiment measuring whether Jev probabilistic judgments add out-of-sample predictive and economic value to a simple long/cash daily-bar ETF mean-reversion strategy.

**Authoritative specification:** [`docs/MASTER_SPEC.md`](docs/MASTER_SPEC.md) (Section 21A = canonical invariants).

## Quick start

```bash
uv sync --all-extras
cp .env.example .env   # fill keys locally; never commit .env
make check
```

Live trading is hard-disabled by default (`LIVE_TRADING_ENABLED: false`, strategy capital ceiling $10.00 in source).

## Layout

See MASTER_SPEC §17. Package name: `jevlab` under `src/`.

## Phase status

| Phase | Status |
| ----- | ------ |
| 0 Scaffold | complete — see `docs/PHASE_0_REPORT.md` |
| 1 Market data | complete — see `docs/PHASE_1_REPORT.md`, `docs/DATA_NOTES.md` |
| 2 Features | complete — see `docs/PHASE_2_REPORT.md` |
| 3 Labels / WF / baselines | complete — see `docs/PHASE_3_REPORT.md` |
| 4 Backtest / risk / sanity | complete — see `docs/PHASE_4_REPORT.md` |
| 5 Jev mock-first | complete — see `docs/PHASE_5_REPORT.md` |
| 6 Diagnostics / freeze | complete — see `docs/PHASE_6_REPORT.md` |
| 7 Bulk historical Jev | complete (dry-run; no live bulk) — see `docs/PHASE_7_REPORT.md` |
| 13A Forward INPUT capture | complete — see `docs/PHASE_13A_REPORT.md`, `docs/FORWARD_CAPTURE.md` |
| 8 Jev-enhanced lineup | complete — see `docs/PHASE_8_REPORT.md` |
| 9 Statistical evaluation | complete — see `docs/PHASE_9_REPORT.md` |
| 10 Robustness / placebo | complete — see `docs/PHASE_10_REPORT.md` |
| 11 Reporting | complete — see `docs/PHASE_11_REPORT.md` |
| 12 Sealed final test | complete (fixtures; not live) — see `docs/PHASE_12_REPORT.md` |
| 13B+ | not started |
