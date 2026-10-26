"""Frozen V2 confirmatory statistic, p-value, and Sharpe-advantage tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FROZEN_N: int = 2000
P_THRESHOLD: float = 0.05
MIN_SHARPE_ADVANTAGE: float = 0.10
MAX_DRAWDOWN: float = 0.35  # 35% as a positive bound on |DD| => DD >= -0.35


def empirical_pvalue(*, n_ge: int, n: int = FROZEN_N) -> float:
    """``p = (1 + #{null >= A}) / (1 + N)`` in full float precision."""
    if n_ge < 0 or n < 1:
        raise ValueError("invalid randomization counts")
    if n_ge > n:
        raise ValueError("n_ge cannot exceed N")
    return (1.0 + float(n_ge)) / (1.0 + float(n))


def sharpe_advantage(observed: float, null_stats: np.ndarray | list[float]) -> float:
    arr = np.asarray(null_stats, dtype=float)
    return float(observed) - float(np.median(arr))


def e1_pass(p: float, *, threshold: float = P_THRESHOLD) -> bool:
    if not math.isfinite(p):
        return False
    return p <= threshold


def e2_pass(advantage: float, *, threshold: float = MIN_SHARPE_ADVANTAGE) -> bool:
    if not math.isfinite(advantage):
        return False
    return advantage >= threshold


def e3_pass(cagr_normal: float) -> bool:
    return math.isfinite(cagr_normal) and cagr_normal > 0.0


def e4_pass(cagr_pessimistic: float) -> bool:
    return math.isfinite(cagr_pessimistic) and cagr_pessimistic > 0.0


def e5_pass(max_drawdown_dec: float, *, cap: float = MAX_DRAWDOWN) -> bool:
    """``max_drawdown_dec`` is negative or zero; fail if worse than -35% or non-finite."""
    if not math.isfinite(max_drawdown_dec):
        return False
    return max_drawdown_dec >= -abs(cap)


@dataclass(frozen=True, slots=True)
class NullTestResult:
    observed: float
    null_stats: tuple[float, ...]
    p_value: float
    advantage: float
    n: int
    n_ge: int


def evaluate_randomization(
    observed: float,
    null_stats: np.ndarray | list[float],
    *,
    n: int | None = None,
) -> NullTestResult:
    arr = np.asarray(null_stats, dtype=float)
    n_eff = int(n) if n is not None else len(arr)
    if len(arr) != n_eff:
        raise ValueError("null_stats length must equal N")
    if not math.isfinite(observed) or not np.isfinite(arr).all():
        p = float("nan")
        adv = float("nan")
        n_ge = n_eff
    else:
        n_ge = int(np.sum(arr >= observed))
        p = empirical_pvalue(n_ge=n_ge, n=n_eff)
        adv = sharpe_advantage(observed, arr)
    return NullTestResult(
        observed=float(observed),
        null_stats=tuple(float(x) for x in arr),
        p_value=p,
        advantage=adv,
        n=n_eff,
        n_ge=n_ge,
    )
