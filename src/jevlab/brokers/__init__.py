"""Broker interfaces, live gates, and strategy ledger (MASTER_SPEC §15 / Phase 14).

Live trading is hard-disabled by default. ``PaperBroker`` is optional and never
required by any acceptance gate. Order submission is gated by L1-L14.
"""

from __future__ import annotations

from jevlab.brokers.base import BrokerInterface
from jevlab.brokers.live_gate import (
    EXECUTION_EXPERIMENT_SENTENCE,
    check_live_gates,
    maybe_submit_order,
)
from jevlab.constants import STRATEGY_ID, TINY_LIVE_CEILING_USD

__all__ = [
    "EXECUTION_EXPERIMENT_SENTENCE",
    "STRATEGY_ID",
    "TINY_LIVE_CEILING_USD",
    "BrokerInterface",
    "check_live_gates",
    "maybe_submit_order",
]
