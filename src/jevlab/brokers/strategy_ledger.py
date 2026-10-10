"""Append-only strategy ledger and capital isolation (MASTER_SPEC §15.1).

The ledger is the source of truth for which shares belong to the strategy.
Broker position quantities are never treated as strategy holdings.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text

from jevlab.constants import STRATEGY_ID, TINY_LIVE_CEILING_USD
from jevlab.utils.registry import init_registry

Side = Literal["buy", "sell"]

CLIENT_ORDER_ID_MAX_LEN: int = 128


class StrategyLedgerError(RuntimeError):
    """Ledger / capital isolation failure (fail closed)."""


def make_client_order_id(
    *,
    session_date: date,
    symbol: str,
    side: Side,
    strategy_id: str = STRATEGY_ID,
) -> str:
    """Deterministic ``client_order_id`` ≤ 128 chars (§15.1)."""
    day = session_date.strftime("%Y%m%d")
    digest = hashlib.sha1(
        f"{strategy_id}|{day}|{symbol}|{side}".encode(), usedforsecurity=False
    ).hexdigest()[:8]
    cid = f"{strategy_id}-{day}-{symbol}-{side}-{digest}"
    if len(cid) > CLIENT_ORDER_ID_MAX_LEN:
        raise StrategyLedgerError(
            f"client_order_id length {len(cid)} exceeds {CLIENT_ORDER_ID_MAX_LEN}"
        )
    return cid


@dataclass(frozen=True, slots=True)
class LedgerRow:
    ledger_id: str
    client_order_id: str
    strategy_id: str
    symbol: str
    side: str
    qty: float
    notional: float
    status: str
    fill_px: float | None
    fill_qty: float | None
    at: str
    mode: str


def append_ledger_row(
    registry_path: Path,
    *,
    client_order_id: str,
    symbol: str,
    side: Side,
    qty: float,
    notional: float,
    status: str,
    mode: str,
    strategy_id: str = STRATEGY_ID,
    fill_px: float | None = None,
    fill_qty: float | None = None,
    at: datetime | None = None,
    ledger_id: str | None = None,
) -> str:
    """Append one strategy_ledger row; refuse duplicate client_order_id."""
    engine = init_registry(registry_path)
    lid = ledger_id or str(uuid.uuid4())
    ts = (at or datetime.now(UTC)).isoformat()
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT ledger_id FROM strategy_ledger WHERE client_order_id = :cid"),
            {"cid": client_order_id},
        ).fetchone()
        if existing is not None:
            raise StrategyLedgerError(
                f"duplicate client_order_id in strategy_ledger: {client_order_id}"
            )
        conn.execute(
            text(
                """
                INSERT INTO strategy_ledger (
                    ledger_id, client_order_id, strategy_id, symbol, side,
                    qty, notional, status, fill_px, fill_qty, at, mode
                ) VALUES (
                    :ledger_id, :client_order_id, :strategy_id, :symbol, :side,
                    :qty, :notional, :status, :fill_px, :fill_qty, :at, :mode
                )
                """
            ),
            {
                "ledger_id": lid,
                "client_order_id": client_order_id,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "notional": float(notional),
                "status": status,
                "fill_px": fill_px,
                "fill_qty": fill_qty,
                "at": ts,
                "mode": mode,
            },
        )
    return lid


def client_order_id_exists(registry_path: Path, client_order_id: str) -> bool:
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM strategy_ledger WHERE client_order_id = :cid"),
            {"cid": client_order_id},
        ).fetchone()
    return row is not None


def list_ledger_rows(registry_path: Path, *, strategy_id: str = STRATEGY_ID) -> list[LedgerRow]:
    engine = init_registry(registry_path)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT ledger_id, client_order_id, strategy_id, symbol, side,
                       qty, notional, status, fill_px, fill_qty, at, mode
                FROM strategy_ledger WHERE strategy_id = :sid ORDER BY at ASC
                """
            ),
            {"sid": strategy_id},
        ).fetchall()
    out: list[LedgerRow] = []
    for r in rows:
        out.append(
            LedgerRow(
                ledger_id=str(r[0]),
                client_order_id=str(r[1]),
                strategy_id=str(r[2]),
                symbol=str(r[3]),
                side=str(r[4]),
                qty=float(r[5] or 0.0),
                notional=float(r[6] or 0.0),
                status=str(r[7]),
                fill_px=float(r[8]) if r[8] is not None else None,
                fill_qty=float(r[9]) if r[9] is not None else None,
                at=str(r[10]),
                mode=str(r[11]),
            )
        )
    return out


def strategy_qty_by_symbol(
    registry_path: Path, *, strategy_id: str = STRATEGY_ID
) -> dict[str, float]:
    """Net filled qty per symbol from the ledger (buys - sells)."""
    qty: dict[str, float] = {}
    skip = {"rejected", "canceled", "cancelled", "expired"}
    for row in list_ledger_rows(registry_path, strategy_id=strategy_id):
        if row.status.lower() in skip:
            continue
        fill = row.fill_qty if row.fill_qty is not None else row.qty
        if row.side.lower() == "buy":
            qty[row.symbol] = qty.get(row.symbol, 0.0) + float(fill)
        elif row.side.lower() == "sell":
            qty[row.symbol] = qty.get(row.symbol, 0.0) - float(fill)
    return {s: q for s, q in qty.items() if abs(q) > 1e-12}


def cumulative_strategy_pnl_usd(registry_path: Path, *, strategy_id: str = STRATEGY_ID) -> float:
    """Rough cumulative realized P&L from filled sell notionals - buy notionals.

    For isolation tests: profits must not raise the deployment cap. Cap uses
    ``min(MAX_STRATEGY_CAPITAL_USD, strategy_equity)`` where
    ``strategy_equity = MAX_STRATEGY_CAPITAL_USD + cumulative_pnl``.
    """
    buys = 0.0
    sells = 0.0
    for row in list_ledger_rows(registry_path, strategy_id=strategy_id):
        if row.status.lower() in {"rejected", "canceled", "cancelled", "expired"}:
            continue
        if row.side.lower() == "buy":
            buys += float(row.notional)
        elif row.side.lower() == "sell":
            sells += float(row.notional)
    return sells - buys


def strategy_equity_usd(
    *,
    max_strategy_capital_usd: float,
    cumulative_pnl: float,
) -> float:
    return float(max_strategy_capital_usd) + float(cumulative_pnl)


def allocatable_usd(
    *,
    max_strategy_capital_usd: float,
    strategy_equity: float,
) -> float:
    """Profits never raise deployment above the cap; losses reduce it."""
    if max_strategy_capital_usd > TINY_LIVE_CEILING_USD + 1e-12:
        raise StrategyLedgerError(
            f"MAX_STRATEGY_CAPITAL_USD={max_strategy_capital_usd} exceeds "
            f"TINY_LIVE_CEILING_USD={TINY_LIVE_CEILING_USD}"
        )
    return min(float(max_strategy_capital_usd), float(strategy_equity))


def check_buy_cap(
    *,
    strategy_market_value: float,
    outstanding_strategy_buy_notional: float,
    proposed_buy_notional: float,
    max_strategy_capital_usd: float,
    strategy_equity: float,
    broker_cash: float,
) -> tuple[bool, str]:
    """§15.1 cap rule; does NOT use total account equity."""
    alloc = allocatable_usd(
        max_strategy_capital_usd=max_strategy_capital_usd,
        strategy_equity=strategy_equity,
    )
    lhs = (
        float(strategy_market_value)
        + float(outstanding_strategy_buy_notional)
        + float(proposed_buy_notional)
    )
    if lhs > alloc + 1e-9:
        return False, f"cap_exceeded lhs={lhs:.4f} allocatable={alloc:.4f}"
    if float(proposed_buy_notional) > float(broker_cash) + 1e-9:
        return False, f"insufficient_broker_cash need={proposed_buy_notional} cash={broker_cash}"
    return True, "ok"


def check_sell_qty(*, proposed_qty: float, ledger_qty: float) -> tuple[bool, str]:
    if proposed_qty <= 0:
        return False, "invalid_sell_qty"
    if proposed_qty > ledger_qty + 1e-12:
        return False, f"sell_exceeds_ledger proposed={proposed_qty} ledger={ledger_qty}"
    return True, "ok"


def external_qty(
    *,
    broker_qty: float,
    ledger_qty: float,
) -> float:
    """``broker_qty - ledger_qty``. Negative ⇒ halt (fail closed)."""
    return float(broker_qty) - float(ledger_qty)


def assert_external_holdings_ok(
    broker_positions: dict[str, float],
    ledger_positions: dict[str, float],
) -> dict[str, float]:
    """Compute external qty; halt if broker holds less than ledger for any symbol."""
    symbols = set(broker_positions) | set(ledger_positions)
    external: dict[str, float] = {}
    for sym in symbols:
        bq = float(broker_positions.get(sym, 0.0))
        lq = float(ledger_positions.get(sym, 0.0))
        ext = external_qty(broker_qty=bq, ledger_qty=lq)
        if ext < -1e-9:
            raise StrategyLedgerError(
                f"external_qty negative for {sym}: broker={bq} ledger={lq}; halt"
            )
        external[sym] = ext
    return external


def reconcile_open_orders(
    *,
    broker_open_client_ids: list[str],
    ledger_client_ids: set[str],
    strategy_prefix: str = STRATEGY_ID,
) -> list[str]:
    """Foreign-prefix open orders that match our prefix but are absent from ledger.

    Returns list of offending client_order_ids (empty = ok).
    """
    bad: list[str] = []
    for cid in broker_open_client_ids:
        if cid.startswith(strategy_prefix) and cid not in ledger_client_ids:
            bad.append(cid)
    return bad


def strategy_market_value_usd(
    ledger_qty: dict[str, float],
    prices: dict[str, float],
) -> float:
    total = 0.0
    for sym, qty in ledger_qty.items():
        px = float(prices.get(sym, 0.0))
        total += abs(qty) * px
    return total


def ledger_summary(registry_path: Path) -> dict[str, Any]:
    rows = list_ledger_rows(registry_path)
    return {
        "n_rows": len(rows),
        "qty_by_symbol": strategy_qty_by_symbol(registry_path),
        "cumulative_pnl": cumulative_strategy_pnl_usd(registry_path),
    }
