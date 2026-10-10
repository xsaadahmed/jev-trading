# Live Runbook (Phase 14) — hard-disabled by default

**Mandatory statement (every session):**  
*This live phase is an execution experiment, not evidence of a profitable trading edge.*

V1 strategy capital cannot exceed **$10.00** (`TINY_LIVE_CEILING_USD` in source). Paper trading is optional and **not** required by any gate.

---

## Defaults (safe)

| Setting | Default |
| ------- | ------- |
| `LIVE_TRADING_ENABLED` | `false` |
| `BROKER_ENV` | `disabled` |
| `MAX_STRATEGY_CAPITAL_USD` | `10.00` (≤ ceiling) |
| CLI | dry-run (no `--execute`) |
| Arm token | absent |

Under default config, `build_live_broker` **raises** and no order HTTP is sent.

---

## Prerequisites for any live order

All of L1–L14 must pass (§15.3). Condensed operator checklist:

1. `HISTORICAL_ACCEPTANCE.json` valid → run `jev live evaluate-tiny` (T1–T9) → `TINY_LIVE_ACCEPTANCE.json`
2. Production-forward models A/B hashed before any forward-label join
3. Safety suite green; `reports/safety_test_report.json` recorded
4. ≥ 10 consecutive dry-run sessions, zero discrepancies
5. Kill-switch drill logged (file, `TRADING_HALT`, drawdown latch)
6. Pinned Jev `jev-1.13.0`; resolved id matches every dry-run
7. Cap ≤ $10; isolation tests pass (external holdings untouched)
8. Account verified cash-type, multiplier `1`, shorting disabled
9. No open `critical` rows in `docs/KNOWN_ISSUES.md`
10. Operator arms: `jev live arm` with phrase `I-ACCEPT-REAL-MONEY-RISK-USD-10` (≤ 24 h)
11. Set env/config only when intentionally going live: `LIVE_TRADING_ENABLED=true`, `BROKER_ENV=live`
12. Pass `--execute` (without it: dry-run only; reads allowed, submits blocked)

**Do not** call close-position / close-all. Sells use explicit qty ≤ ledger quantity.

---

## CLI

```bash
jev live status          # flags, arm, ceiling (read-only)
jev live dry-run         # gates + intended path; never submits
jev live arm             # typed phrase → .live_arm
jev live disarm
jev live evaluate-tiny --fixture-metrics PATH
```

This phase does **not** ship an enabled execute command that places real orders by default. Submission is only possible through `maybe_submit_order` after L1–L14 all pass.

---

## Kill switch

- File `KILL_SWITCH` at project root, or
- Env `TRADING_HALT=1`, or
- Latched drawdown breach  

→ all new orders blocked (L7).

---

## Reconciliation

Before trading: ledger vs broker positions; `external_qty = broker_qty − ledger_qty`. Negative ⇒ halt. Open orders with strategy prefix but missing from ledger ⇒ halt.

---

## Secrets

`BROKER_API_KEY` / `BROKER_API_SECRET` never logged (structlog redaction). Prefer OS env / secret store; never commit `.env`.
