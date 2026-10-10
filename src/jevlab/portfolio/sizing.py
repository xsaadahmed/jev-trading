"""Equal-weight position sizing (MASTER_SPEC §10.6).

K targets, each ``equity / K``; unused slots stay cash when fewer than K
eligible names. No leverage: total target ≤ equity. Fractional shares allowed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetNotional:
    symbol: str
    notional_usd: float


def equal_weight_notionals(
    symbols: list[str],
    *,
    equity_usd: float,
    k: int,
) -> list[TargetNotional]:
    """Allocate ``equity_usd / k`` to each selected symbol (cash for empty slots).

    If ``len(symbols) > k``, only the first ``k`` are sized (caller should
    already have truncated). Notionals are non-negative; empty input → [].
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if equity_usd < 0:
        raise ValueError("equity_usd must be non-negative")
    chosen = list(symbols)[:k]
    if not chosen or equity_usd == 0:
        return []
    per = equity_usd / float(k)
    return [TargetNotional(symbol=s, notional_usd=per) for s in chosen]


def shares_for_notional(notional_usd: float, fill_price_usd: float) -> float:
    """Fractional share quantity for a target notional at the fill price."""
    if fill_price_usd <= 0:
        raise ValueError("fill_price_usd must be positive")
    if notional_usd <= 0:
        return 0.0
    return notional_usd / fill_price_usd
