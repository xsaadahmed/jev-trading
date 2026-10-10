"""Stationary block bootstrap over dates (MASTER_SPEC §12.3).

Resamples *dates* (not rows) in stationary blocks with mean length 10,
preserving cross-sectional dependence within a day. Default 10,000 draws.
Used for paired (B - A) differences of log-loss, rank-IC, Sharpe, and CAGR.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_N_DRAWS: int = 10_000
DEFAULT_MEAN_BLOCK: int = 10
DEFAULT_SEED: int = 42


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Percentile confidence interval for one scalar statistic."""

    estimate: float
    ci_low: float
    ci_high: float
    n_draws: int
    mean_block: int
    seed: int
    level: float = 0.95

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_draws": self.n_draws,
            "mean_block": self.mean_block,
            "seed": self.seed,
            "level": self.level,
            "excludes_zero": float(
                (self.ci_low > 0.0 and self.ci_high > 0.0)
                or (self.ci_low < 0.0 and self.ci_high < 0.0)
            ),
        }


def stationary_bootstrap_indices(
    n: int,
    *,
    mean_block: int = DEFAULT_MEAN_BLOCK,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix ``(n_draws, n)``.

    Block lengths are i.i.d. Geometric with mean ``mean_block`` (success
    probability ``p = 1/mean_block``). Indices wrap circularly.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    rng = np.random.default_rng(seed)
    p = 1.0 / float(mean_block)
    out = np.empty((n_draws, n), dtype=np.int64)
    for d in range(n_draws):
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            # Geometric(p) with support {1,2,...}; numpy uses number of failures.
            length = int(rng.geometric(p))
            for k in range(length):
                if i >= n:
                    break
                out[d, i] = (start + k) % n
                i += 1
    return out


def _percentile_ci(
    samples: np.ndarray,
    *,
    estimate: float,
    level: float,
    n_draws: int,
    mean_block: int,
    seed: int,
) -> BootstrapCI:
    alpha = 1.0 - level
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return BootstrapCI(
        estimate=float(estimate),
        ci_low=lo,
        ci_high=hi,
        n_draws=n_draws,
        mean_block=mean_block,
        seed=seed,
        level=level,
    )


def bootstrap_mean_difference(
    series_a: Sequence[float] | np.ndarray,
    series_b: Sequence[float] | np.ndarray,
    *,
    mean_block: int = DEFAULT_MEAN_BLOCK,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> BootstrapCI:
    """Paired bootstrap CI for ``mean(B - A)`` on aligned date-level series."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("series_a and series_b must have the same shape")
    if a.ndim != 1:
        raise ValueError("series must be 1-D (one value per date)")
    n = len(a)
    if n == 0:
        return BootstrapCI(
            estimate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_draws=n_draws,
            mean_block=mean_block,
            seed=seed,
            level=level,
        )
    delta = b - a
    estimate = float(np.nanmean(delta))
    idx = stationary_bootstrap_indices(n, mean_block=mean_block, n_draws=n_draws, seed=seed)
    samples = np.nanmean(delta[idx], axis=1)
    return _percentile_ci(
        samples,
        estimate=estimate,
        level=level,
        n_draws=n_draws,
        mean_block=mean_block,
        seed=seed,
    )


def bootstrap_statistic_on_dates(
    dates: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    mean_block: int = DEFAULT_MEAN_BLOCK,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> BootstrapCI:
    """Bootstrap a scalar statistic of a date list (order preserved within blocks)."""
    date_list = list(dates)
    n = len(date_list)
    if n == 0:
        return BootstrapCI(
            estimate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_draws=n_draws,
            mean_block=mean_block,
            seed=seed,
            level=level,
        )
    estimate = float(statistic(date_list))
    idx = stationary_bootstrap_indices(n, mean_block=mean_block, n_draws=n_draws, seed=seed)
    samples = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        resampled = [date_list[i] for i in idx[d]]
        samples[d] = float(statistic(resampled))
    return _percentile_ci(
        samples,
        estimate=estimate,
        level=level,
        n_draws=n_draws,
        mean_block=mean_block,
        seed=seed,
    )


def paired_delta_on_dates(
    dates: Sequence[Any],
    metric_by_date_a: Mapping[Any, float],
    metric_by_date_b: Mapping[Any, float],
) -> float:
    """Mean (B - A) over the given dates (missing dates skipped)."""
    diffs: list[float] = []
    for d in dates:
        if d in metric_by_date_a and d in metric_by_date_b:
            va = metric_by_date_a[d]
            vb = metric_by_date_b[d]
            if np.isfinite(va) and np.isfinite(vb):
                diffs.append(float(vb - va))
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def bootstrap_paired_metric_maps(
    metric_by_date_a: Mapping[Any, float],
    metric_by_date_b: Mapping[Any, float],
    *,
    mean_block: int = DEFAULT_MEAN_BLOCK,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
    return_samples: bool = False,
) -> BootstrapCI | tuple[BootstrapCI, np.ndarray]:
    """Bootstrap mean(B-A) where each model supplies a date->metric map."""
    dates = sorted(set(metric_by_date_a) & set(metric_by_date_b))
    n = len(dates)
    if n == 0:
        empty = BootstrapCI(
            estimate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_draws=n_draws,
            mean_block=mean_block,
            seed=seed,
            level=level,
        )
        if return_samples:
            return empty, np.array([])
        return empty
    estimate = paired_delta_on_dates(dates, metric_by_date_a, metric_by_date_b)
    idx = stationary_bootstrap_indices(n, mean_block=mean_block, n_draws=n_draws, seed=seed)
    samples = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        resampled = [dates[i] for i in idx[d]]
        samples[d] = paired_delta_on_dates(resampled, metric_by_date_a, metric_by_date_b)
    ci = _percentile_ci(
        samples,
        estimate=float(estimate),
        level=level,
        n_draws=n_draws,
        mean_block=mean_block,
        seed=seed,
    )
    if return_samples:
        return ci, samples
    return ci


def one_sided_bootstrap_pvalue(samples: np.ndarray, *, greater: bool) -> float:
    """Empirical one-sided p = (1 + #{unfavorable}) / (1 + N)."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    # greater=True: H1 mean>0, unfavorable = samples<=0; else H1 mean<0.
    count = int(np.sum(s <= 0.0)) if greater else int(np.sum(s >= 0.0))
    return float((1 + count) / (1 + n))


def sharpe_from_daily_returns(rets: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(r) / sd)


def cagr_from_daily_returns(rets: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 1:
        return float("nan")
    wealth = float(np.prod(1.0 + r))
    if wealth <= 0:
        return -1.0
    n = len(r)
    return float(wealth ** (periods_per_year / n) - 1.0)


def bootstrap_paired_return_deltas(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    mean_block: int = DEFAULT_MEAN_BLOCK,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> dict[str, BootstrapCI]:
    """Bootstrap deltaSharpe and deltaCAGR for paired daily return series (B - A)."""
    a = returns_a.astype(float).dropna()
    b = returns_b.astype(float).dropna()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]
    b = b.loc[idx]
    dates = list(idx)
    n = len(dates)
    if n == 0:
        empty = BootstrapCI(
            estimate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_draws=n_draws,
            mean_block=mean_block,
            seed=seed,
            level=level,
        )
        return {"delta_sharpe": empty, "delta_cagr": empty}

    a_vals = a.to_numpy(dtype=float)
    b_vals = b.to_numpy(dtype=float)
    est_sharpe = sharpe_from_daily_returns(b_vals) - sharpe_from_daily_returns(a_vals)
    est_cagr = cagr_from_daily_returns(b_vals) - cagr_from_daily_returns(a_vals)
    boot_idx = stationary_bootstrap_indices(n, mean_block=mean_block, n_draws=n_draws, seed=seed)
    sharpe_s = np.empty(n_draws, dtype=float)
    cagr_s = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        ia = a_vals[boot_idx[d]]
        ib = b_vals[boot_idx[d]]
        sharpe_s[d] = sharpe_from_daily_returns(ib) - sharpe_from_daily_returns(ia)
        cagr_s[d] = cagr_from_daily_returns(ib) - cagr_from_daily_returns(ia)
    return {
        "delta_sharpe": _percentile_ci(
            sharpe_s,
            estimate=float(est_sharpe),
            level=level,
            n_draws=n_draws,
            mean_block=mean_block,
            seed=seed,
        ),
        "delta_cagr": _percentile_ci(
            cagr_s,
            estimate=float(est_cagr),
            level=level,
            n_draws=n_draws,
            mean_block=mean_block,
            seed=seed,
        ),
    }


def bootstrap_ab_deltas(
    *,
    logloss_by_date_a: Mapping[Any, float],
    logloss_by_date_b: Mapping[Any, float],
    ic_by_date_a: Mapping[Any, float],
    ic_by_date_b: Mapping[Any, float],
    returns_a: pd.Series | None = None,
    returns_b: pd.Series | None = None,
    mean_block: int = DEFAULT_MEAN_BLOCK,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap CIs for deltalogLoss, deltaIC, and optionally deltaSharpe/deltaCAGR."""
    ll_pack = bootstrap_paired_metric_maps(
        logloss_by_date_a,
        logloss_by_date_b,
        mean_block=mean_block,
        n_draws=n_draws,
        seed=seed,
        level=level,
        return_samples=True,
    )
    ic_pack = bootstrap_paired_metric_maps(
        ic_by_date_a,
        ic_by_date_b,
        mean_block=mean_block,
        n_draws=n_draws,
        seed=seed + 1,
        level=level,
        return_samples=True,
    )
    assert isinstance(ll_pack, tuple) and isinstance(ic_pack, tuple)
    ll_ci, ll_samples = ll_pack
    ic_ci, ic_samples = ic_pack
    out: dict[str, Any] = {
        "delta_log_loss": {
            **ll_ci.to_dict(),
            "pvalue_one_sided_less": one_sided_bootstrap_pvalue(ll_samples, greater=False),
        },
        "delta_rank_ic": {
            **ic_ci.to_dict(),
            "pvalue_one_sided_greater": one_sided_bootstrap_pvalue(ic_samples, greater=True),
        },
        "mean_block": mean_block,
        "n_draws": n_draws,
        "seed": seed,
    }
    if returns_a is not None and returns_b is not None:
        ret = bootstrap_paired_return_deltas(
            returns_a,
            returns_b,
            mean_block=mean_block,
            n_draws=n_draws,
            seed=seed,
            level=level,
        )
        out["delta_sharpe"] = ret["delta_sharpe"].to_dict()
        out["delta_cagr"] = ret["delta_cagr"].to_dict()
    return out
