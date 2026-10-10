"""BacktestBroker — simulated fills; no network (MASTER_SPEC §15.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from jevlab.brokers.types import (
    AccountState,
    OrderAck,
    OrderRequest,
    OrderStatus,
    Position,
)


class BacktestBroker:
    """In-memory broker for backtests / dry plumbing tests."""

    def __init__(
        self,
        *,
        cash: float = 10.0,
        positions: dict[str, float] | None = None,
        fractionable: set[str] | None = None,
        account_type: str = "cash",
        multiplier: str = "1",
        shorting_enabled: bool = False,
    ) -> None:
        self._cash = float(cash)
        self._positions: dict[str, float] = dict(positions or {})
        self._orders: dict[str, OrderStatus] = {}
        self._fractionable = set(fractionable or set())
        self._account_type = account_type
        self._multiplier = multiplier
        self._shorting_enabled = shorting_enabled
        self._prices: dict[str, float] = {}
        self.submit_calls: list[OrderRequest] = []
        self._market_open = True

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = float(price)

    def set_market_open(self, open_: bool) -> None:
        self._market_open = open_

    def get_account(self) -> AccountState:
        mv = sum(abs(q) * self._prices.get(s, 0.0) for s, q in self._positions.items())
        return AccountState(
            cash=self._cash,
            buying_power=self._cash,
            equity=self._cash + mv,
            account_type=self._account_type,
            multiplier=self._multiplier,
            shorting_enabled=self._shorting_enabled,
        )

    def get_positions(self) -> list[Position]:
        out: list[Position] = []
        for sym, qty in self._positions.items():
            if abs(qty) < 1e-12:
                continue
            px = self._prices.get(sym, 0.0)
            out.append(
                Position(
                    symbol=sym,
                    qty=qty,
                    market_value=abs(qty) * px,
                    avg_entry_price=px,
                )
            )
        return out

    def list_open_orders(self, client_order_prefix: str | None = None) -> list[OrderStatus]:
        open_statuses = {"new", "accepted", "pending_new", "partially_filled"}
        out: list[OrderStatus] = []
        for st in self._orders.values():
            if st.status not in open_statuses:
                continue
            if client_order_prefix and not st.client_order_id.startswith(client_order_prefix):
                continue
            out.append(st)
        return out

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self.submit_calls.append(order)
        # Never call close-position / close-all — only explicit qty/notional.
        if order.side == "buy":
            if order.notional is None or order.notional <= 0:
                raise ValueError("buy requires positive notional")
            px = self._prices.get(order.symbol, 1.0)
            qty = float(order.notional) / max(px, 1e-12)
            self._cash -= float(order.notional)
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + qty
            filled_qty = qty
            notional = float(order.notional)
        else:
            if order.qty is None or order.qty <= 0:
                raise ValueError("sell requires positive qty")
            held = self._positions.get(order.symbol, 0.0)
            if order.qty > held + 1e-12:
                raise ValueError("sell exceeds position")
            px = self._prices.get(order.symbol, 1.0)
            self._positions[order.symbol] = held - float(order.qty)
            self._cash += float(order.qty) * px
            filled_qty = float(order.qty)
            notional = filled_qty * px

        status = OrderStatus(
            client_order_id=order.client_order_id,
            broker_order_id=f"bt-{order.client_order_id}",
            symbol=order.symbol,
            side=order.side,
            status="filled",
            qty=filled_qty if order.side == "sell" else None,
            notional=notional if order.side == "buy" else None,
            filled_qty=filled_qty,
            filled_avg_price=self._prices.get(order.symbol, 1.0),
        )
        self._orders[order.client_order_id] = status
        return OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id=status.broker_order_id,
            status="filled",
            submitted_at=datetime.now(UTC),
        )

    def get_order(self, client_order_id: str) -> OrderStatus:
        if client_order_id not in self._orders:
            raise KeyError(client_order_id)
        return self._orders[client_order_id]

    def cancel_order(self, client_order_id: str) -> None:
        if client_order_id in self._orders:
            prev = self._orders[client_order_id]
            self._orders[client_order_id] = OrderStatus(
                client_order_id=prev.client_order_id,
                broker_order_id=prev.broker_order_id,
                symbol=prev.symbol,
                side=prev.side,
                status="canceled",
                qty=prev.qty,
                notional=prev.notional,
                filled_qty=prev.filled_qty,
                filled_avg_price=prev.filled_avg_price,
            )

    def is_market_open(self) -> bool:
        return self._market_open

    def asset_fractionable(self, symbol: str) -> bool:
        if not self._fractionable:
            return True
        return symbol in self._fractionable

    def debug_state(self) -> dict[str, Any]:
        return {"cash": self._cash, "positions": dict(self._positions)}
