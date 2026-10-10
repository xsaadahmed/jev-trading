"""Pure risk gate: proposed orders → approved / rejections (MASTER_SPEC §14).

Jev and models never import this module; risk cannot be overridden by signals.
Caps are relative to **strategy** capital / equity, not broker account equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from jevlab.config.models import RiskConfig
from jevlab.portfolio.state import PortfolioState
from jevlab.risk.killswitch import is_halted

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class ProposedOrder:
    """Order proposal before risk approval."""

    symbol: str
    side: Side
    qty: float
    notional_usd: float
    client_order_id: str
    session_date: date  # decision or intended submit date


@dataclass(frozen=True, slots=True)
class Rejection:
    client_order_id: str
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class MarketState:
    """Market inputs for stale/missing-data rules (as of decision_time)."""

    session_date: date
    decision_time: datetime
    last_completed_nyse_session: date
    last_bar_session: date | None
    available_at_ok: bool = True
    missing_symbol_count: int = 0
    kill_switch_file: str | None = None
    kill_switch_env_var: str = "TRADING_HALT"


@dataclass
class RiskDecision:
    approved: list[ProposedOrder] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)


def evaluate_orders(
    proposed_orders: list[ProposedOrder],
    portfolio_state: PortfolioState,
    market_state: MarketState,
    config: RiskConfig,
) -> RiskDecision:
    """Apply §14 rules; return approved orders and structured rejections.

    Rules checked (order matters for clarity; each buy is evaluated against
    a running tally of approved buys in this batch):
    kill switch / drawdown, stale data, missing data, max orders/day,
    daily-loss halt (blocks new buys), position cap, gross cap, duplicates,
    no shorts / sell qty, no leverage via cash.
    """
    decision = RiskDecision()
    drawdown_breach = _drawdown_breach(portfolio_state, config)
    halted = portfolio_state.kill_switch_active or is_halted(
        kill_switch_file=market_state.kill_switch_file,
        kill_switch_env_var=market_state.kill_switch_env_var or config.kill_switch_env_var,
        drawdown_breach=drawdown_breach,
    )

    if config.stale_data_halt and not _data_fresh(market_state):
        for order in proposed_orders:
            decision.rejections.append(Rejection(order.client_order_id, order.symbol, "stale_data"))
        return decision

    if market_state.missing_symbol_count > config.max_missing_symbols:
        for order in proposed_orders:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "missing_data_day_skip")
            )
        return decision

    if halted:
        for order in proposed_orders:
            # Exits may still be needed live; backtest flatten is separate.
            # Spec: kill switch → all new orders blocked.
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "kill_switch")
            )
        return decision

    daily_loss_halt = _daily_loss_halt(portfolio_state, config)
    orders_so_far = portfolio_state.orders_today
    running_buy_notional = portfolio_state.outstanding_buy_notional_usd
    running_gross = portfolio_state.market_value_usd
    remaining_cash = portfolio_state.cash_usd
    seen_ids = set(portfolio_state.existing_client_order_ids)
    capital = max(portfolio_state.strategy_capital_usd, 1e-12)
    equity = max(portfolio_state.strategy_equity_usd, 0.0)
    # Caps relative to strategy capital (live) and never above equity (no leverage).
    cap_base = min(capital, equity) if equity > 0 else capital
    max_pos_usd = (config.max_position_pct / 100.0) * cap_base
    max_gross_usd = (config.max_gross_exposure_pct / 100.0) * cap_base

    for order in proposed_orders:
        if order.client_order_id in seen_ids:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "duplicate_order")
            )
            continue
        if order.qty <= 0 or order.notional_usd < 0:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "invalid_qty")
            )
            continue
        if order.side == "sell":
            held = portfolio_state.positions.get(order.symbol, 0.0)
            if order.qty > held + 1e-12:
                decision.rejections.append(
                    Rejection(order.client_order_id, order.symbol, "sell_exceeds_position")
                )
                continue
            decision.approved.append(order)
            seen_ids.add(order.client_order_id)
            orders_so_far += 1
            continue

        # --- buys ---
        if daily_loss_halt:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "daily_loss_halt")
            )
            continue
        if orders_so_far >= config.max_new_orders_per_day:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "max_orders_per_day")
            )
            continue
        if order.notional_usd > max_pos_usd + 1e-9:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "max_position")
            )
            continue
        projected_gross = running_gross + running_buy_notional + order.notional_usd
        if projected_gross > max_gross_usd + 1e-9:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "max_gross_exposure")
            )
            continue
        if order.notional_usd > remaining_cash + 1e-9:
            decision.rejections.append(
                Rejection(order.client_order_id, order.symbol, "insufficient_cash")
            )
            continue

        decision.approved.append(order)
        seen_ids.add(order.client_order_id)
        orders_so_far += 1
        running_buy_notional += order.notional_usd
        remaining_cash -= order.notional_usd

    return decision


def _drawdown_breach(state: PortfolioState, config: RiskConfig) -> bool:
    hwm = state.high_water_mark_usd
    if hwm <= 0:
        return False
    dd = (hwm - state.strategy_equity_usd) / hwm
    return dd >= (config.max_drawdown_pct / 100.0) - 1e-15


def _daily_loss_halt(state: PortfolioState, config: RiskConfig) -> bool:
    sod = state.sod_equity_usd
    if sod <= 0:
        return False
    loss = (sod - state.strategy_equity_usd) / sod
    return loss >= (config.max_daily_loss_pct / 100.0) - 1e-15


def _data_fresh(market: MarketState) -> bool:
    if not market.available_at_ok:
        return False
    if market.last_bar_session is None:
        return False
    return market.last_bar_session == market.last_completed_nyse_session
