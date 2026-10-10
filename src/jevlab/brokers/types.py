"""Broker domain types (MASTER_SPEC §15.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Side = Literal["buy", "sell"]
OrderType = Literal["market"]
TimeInForce = Literal["day"]
OrderStatusKind = Literal[
    "new",
    "accepted",
    "pending_new",
    "filled",
    "partially_filled",
    "canceled",
    "rejected",
    "expired",
]


@dataclass(frozen=True, slots=True)
class AccountState:
    """Broker account snapshot (cash-type verification fields)."""

    cash: float
    buying_power: float
    equity: float
    account_type: str  # expect "cash" (or equivalent)
    pattern_day_trader: bool = False
    # Margin / shorting — unknown/ambiguous ⇒ fail closed at L14.
    multiplier: str | None = None  # "1" = no margin; None = ambiguous
    shorting_enabled: bool | None = None  # False required; None = ambiguous
    status: str = "ACTIVE"
    raw: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float = 0.0
    side: str = "long"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Fractional buy = notional market; sell = explicit qty ≤ ledger."""

    symbol: str
    side: Side
    client_order_id: str
    notional: float | None = None  # buys
    qty: float | None = None  # sells
    order_type: OrderType = "market"
    time_in_force: TimeInForce = "day"
    session_date: str | None = None  # YYYY-MM-DD for audit


@dataclass(frozen=True, slots=True)
class OrderAck:
    client_order_id: str
    broker_order_id: str
    status: OrderStatusKind
    submitted_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True, slots=True)
class OrderStatus:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: Side
    status: OrderStatusKind
    qty: float | None = None
    notional: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
