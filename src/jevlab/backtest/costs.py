"""Fill prices and round-trip costs (MASTER_SPEC §10.2).

Time semantics: entry fill at open of t+1; exit fill at open of t+4.
``cost_bps_per_side`` is the one-way all-in half-spread + slippage baked into
the fill price. ``sell_fee_bps`` is charged once on the sell notional.
"""

from __future__ import annotations

from dataclasses import dataclass

from jevlab.targets.labels import roundtrip_cost_dec
from jevlab.utils.units import bps_to_decimal

# Canonical scenario names → bps/side (must match config/backtest.yaml).
SCENARIO_BPS: dict[str, float] = {
    "optimistic": 1.0,
    "normal": 2.5,
    "pessimistic": 6.0,
    "stress": 12.0,
}


@dataclass(frozen=True, slots=True)
class CostModel:
    """All-in per-side cost plus sell-side fee; names carry units."""

    cost_bps_per_side: float
    sell_fee_bps: float
    per_symbol_cost_multiplier: dict[str, float]
    scenario_name: str = "normal"

    def multiplier(self, symbol: str) -> float:
        return float(self.per_symbol_cost_multiplier.get(symbol, 1.0))

    def side_cost_dec(self, symbol: str) -> float:
        """One-way all-in cost as a decimal return for ``symbol``."""
        return bps_to_decimal(self.cost_bps_per_side * self.multiplier(symbol))

    def roundtrip_cost_dec(self, symbol: str) -> float:
        """Full round-trip decimal drag for ``symbol`` (two sides + fee once)."""
        return roundtrip_cost_dec(
            self.cost_bps_per_side * self.multiplier(symbol),
            self.sell_fee_bps,
        )

    def buy_fill_price_usd(self, adj_open_usd: float, symbol: str) -> float:
        """Buy fill: ``adj_open * (1 + side_cost_dec)`` (adverse slippage)."""
        return float(adj_open_usd) * (1.0 + self.side_cost_dec(symbol))

    def sell_fill_price_usd(self, adj_open_usd: float, symbol: str) -> float:
        """Sell fill: ``adj_open * (1 - side_cost_dec)`` (adverse slippage)."""
        return float(adj_open_usd) * (1.0 - self.side_cost_dec(symbol))

    def sell_fee_usd(self, sell_notional_usd: float) -> float:
        """Regulatory fee on sell notional (once per round trip)."""
        return float(sell_notional_usd) * bps_to_decimal(self.sell_fee_bps)


def require_cost_model(cost: CostModel | None) -> CostModel:
    """Costs are mandatory in the engine — no zero-cost silent default for reports."""
    if cost is None:
        raise ValueError(
            "cost model is required; pass an explicit CostModel "
            "(no zero-cost default for reported backtests)"
        )
    return cost
