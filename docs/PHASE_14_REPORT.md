# Phase 14 Report — Live infrastructure ($10), hard-disabled

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§15, 16.2, 19, 21 Phase 14, 21A  
**Preserved:** Prior phases including 12, 13A, 13B  
**Non-goals:** No real orders; nothing armed or enabled by default

---

## What was built

### Brokers (`src/jevlab/brokers/`)

| Module | Role |
| ------ | ---- |
| `types.py` / `base.py` | `BrokerInterface` protocol + domain types (§15.2) |
| `backtest.py` | `BacktestBroker` — in-memory simulated fills |
| `live.py` | `LiveBroker` (Alpaca REST/httpx; fractional notional market buys, day TIF; fractionable check); `build_paper_broker` optional; **never** DELETE close-position/close-all |
| `strategy_ledger.py` | Append-only ledger; `client_order_id = f"{strategy_id}-{YYYYMMDD}-{symbol}-{side}-{sha1[:8]}"` (≤128); cap / sell-qty / external_qty isolation (§15.1) |
| `live_gate.py` | Independent L1–L14; `maybe_submit_order` choke point (zero submits if any gate fails) |
| `arm.py` | Typed phrase → `.live_arm` ≤ 24 h TTL |
| `audit.py` | `live_orders` audit rows with mandatory execution-experiment sentence |
| `evaluate_tiny.py` | T1–T9 → `TINY_LIVE_ACCEPTANCE.json` (integrity digest; parent = HISTORICAL; **no** forward-session count) |

**Ceiling:** `TINY_LIVE_CEILING_USD = 10.00` in source; config above raises at startup. Caps use strategy equity / ledger — never total account equity.

**Mandatory sentence:** *This live phase is an execution experiment, not evidence of a profitable trading edge.*

### CLI (`jev live`)

```bash
jev live status
jev live dry-run          # reads/gates only; never submits
jev live arm / disarm
jev live evaluate-tiny --fixture-metrics PATH
```

Default config cannot construct an order-capable `LiveBroker`.

### Docs

`docs/LIVE_RUNBOOK.md` — operator checklist; hard-disabled defaults.

---

## Safety tests (`tests/safety/`)

| Invariant | Coverage |
| --------- | -------- |
| L1–L14 each block alone | Parametrized; mock/`BacktestBroker` `submit_order` call count = 0 |
| Default config | `build_live_broker` raises |
| Transport mock | Zero HTTP to order endpoints on read-only path |
| Cap / external holdings / profits | Cap stays at 10; external positive OK; negative halts; config >10 raises |
| Sells ≤ ledger qty | Unit check |
| Idempotent IDs | Deterministic + duplicate ledger refuse |
| Reconciliation mismatch | L12 fails; no submit |
| Tampered / lineage artifacts | L4 fails |
| Secrets redacted | Logging redaction |
| PaperBroker optional | Constructible; unused by gates |
| T1–T9 blocking | Parametrized tiny evaluator |

---

## Checks

```text
ruff check          OK
ruff format --check OK
mypy src/jevlab     OK (113 source files)
pytest -q           OK (full suite incl. tests/safety)
make check          OK
```

**No real order was sent. Nothing was armed or enabled.**

---

## Acceptance mapping

| Spec item | Status |
| --------- | ------ |
| BrokerInterface + Backtest / Paper / Live | Pass |
| Strategy ledger + capital isolation | Pass |
| L1–L14 independent + safety per gate | Pass |
| Dry-run default; reads OK; submit gated | Pass |
| `evaluate-tiny` T1–T9 + integrity digest | Pass |
| LIVE_RUNBOOK | Pass |
| Hard-disabled; no real orders | Pass |

**Stop:** Phase 14 only. Do not proceed to further phases unless requested.
