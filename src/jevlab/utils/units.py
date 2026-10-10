"""Unit conversion helpers with unit-typed names (MASTER_SPEC §8, §22)."""

from __future__ import annotations

BPS_PER_UNIT: float = 10_000.0


def bps_to_decimal(value_bps: float) -> float:
    """Convert a value in basis points to a decimal fraction.

    Example: ``bps_to_decimal(2.5) == 0.00025``.
    """
    return float(value_bps) / BPS_PER_UNIT


def decimal_to_bps(value_dec: float) -> float:
    """Convert a decimal fraction to basis points.

    Example: ``decimal_to_bps(0.00025) == 2.5``.
    """
    return float(value_dec) * BPS_PER_UNIT
