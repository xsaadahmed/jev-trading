"""Portfolio state for risk checks and ledger marking (MASTER_SPEC §14, §15.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Position:
    """Open long position (no shorts in V1)."""

    symbol: str
    qty: float
    entry_session: date
    exit_session: date
    cost_basis_usd: float
    cohort_id: str


@dataclass
class PortfolioState:
    """Snapshot consumed by ``risk.rules.evaluate_orders``."""

    cash_usd: float
    positions: dict[str, float]  # symbol -> qty (aggregate)
    strategy_equity_usd: float
    strategy_capital_usd: float
    high_water_mark_usd: float
    sod_equity_usd: float
    market_value_usd: float
    outstanding_buy_notional_usd: float = 0.0
    orders_today: int = 0
    kill_switch_active: bool = False
    existing_client_order_ids: set[str] = field(default_factory=set)
    open_position_records: list[Position] = field(default_factory=list)

    @property
    def gross_exposure_usd(self) -> float:
        return float(self.market_value_usd)
