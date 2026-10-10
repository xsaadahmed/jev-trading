"""BrokerInterface protocol (MASTER_SPEC §15.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jevlab.brokers.types import AccountState, OrderAck, OrderRequest, OrderStatus, Position


@runtime_checkable
class BrokerInterface(Protocol):
    """Minimal broker surface used by live gates and reconciliation."""

    def get_account(self) -> AccountState: ...

    def get_positions(self) -> list[Position]: ...

    def list_open_orders(self, client_order_prefix: str | None = None) -> list[OrderStatus]: ...

    def submit_order(self, order: OrderRequest) -> OrderAck: ...

    def get_order(self, client_order_id: str) -> OrderStatus: ...

    def cancel_order(self, client_order_id: str) -> None: ...

    def is_market_open(self) -> bool: ...

    def asset_fractionable(self, symbol: str) -> bool: ...
