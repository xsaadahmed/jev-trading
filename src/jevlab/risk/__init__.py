"""Deterministic risk rules and kill-switch; never imports jev or brokers."""

from jevlab.risk.killswitch import is_halted
from jevlab.risk.rules import MarketState, ProposedOrder, Rejection, evaluate_orders

__all__ = [
    "MarketState",
    "ProposedOrder",
    "Rejection",
    "evaluate_orders",
    "is_halted",
]
