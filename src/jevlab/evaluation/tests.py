"""Inferential tests (MASTER_SPEC §12.3, §16.3 S4).

Diebold-Mariano (HAC lag 5), deflated Sharpe ratio (Bailey-López de Prado),
Holm / Benjamini-Hochberg adjustments, MDE for deltaSharpe/deltaIC, and
``forward_consistency_z`` for the scale evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

DEFAULT_NW_LAG: int = 5
EULER_MASCHERONI: float = 0.5772156649015329
FORWARD_CONSISTENCY_Z_THRESHOLD: float = 1.28  # one-sided 10% (§16.3 S4)


def diebold_mariano(
    loss_a: Sequence[float] | np.ndarray,
    loss_b: Sequence[float] | np.ndarray,
    *,
    lag: int = DEFAULT_NW_LAG,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Diebold-Mariano test on per-date mean loss series.

    ``d_t = loss_a[t] - loss_b[t]``. Positive mean(d) => A has higher loss (B better
    for log-loss). HAC Newey-West SE with ``lag`` (default 5).
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss series must align")
    d = a - b
    mask = np.isfinite(d)
    d = d[mask]
    n = len(d)
    if n < max(lag + 2, 3):
        return {
            "mean_diff": float("nan"),
            "se_hac": float("nan"),
            "stat": float("nan"),
            "pvalue": float("nan"),
            "n": float(n),
            "lag": float(lag),
        }
    x = np.ones((n, 1))
    model = OLS(d, x).fit()
    cov = cov_hac(model, nlags=lag)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    mean = float(np.mean(d))
    stat = mean / se if se > 0 else float("nan")
    if not np.isfinite(stat):
        pvalue = float("nan")
    elif alternative == "two-sided":
        pvalue = float(2.0 * stats.norm.sf(abs(stat)))
    elif alternative == "greater":
        pvalue = float(stats.norm.sf(stat))
    elif alternative == "less":
        pvalue = float(stats.norm.cdf(stat))
    else:
        raise ValueError(f"unknown alternative: {alternative}")
    return {
        "mean_diff": mean,
        "se_hac": se,
        "stat": float(stat),
        "pvalue": pvalue,
        "n": float(n),
        "lag": float(lag),
    }


def expected_max_sharpe(
    n_trials: int,
    *,
    variance_sharpe: float = 1.0,
) -> float:
    """Expected maximum Sharpe under ``n_trials`` independent null tests.

    Bailey-López de Prado (2014) approximation (non-annualized SR units)::

        SR_0 = (1-gamma) Phi^{-1}(1-1/N) + gamma Phi^{-1}(1-1/(N e))
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    gamma = EULER_MASCHERONI
    inv1 = float(stats.norm.ppf(1.0 - 1.0 / n))
    inv2 = float(stats.norm.ppf(1.0 - 1.0 / (n * math.e)))
    sr0 = (1.0 - gamma) * inv1 + gamma * inv2
    return float(sr0 * math.sqrt(max(variance_sharpe, 0.0)))


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, float]:
    """Deflated Sharpe ratio (Bailey-López de Prado 2014).

    ``observed_sr`` and the null threshold are in the *same* non-annualized
    units as estimated from ``n_obs`` returns. ``kurtosis`` is the raw fourth
    moment (3 for Gaussian).

    Returns the PSR/DSR probability that the true SR exceeds the multiple-testing
    threshold ``SR_0(n_trials)``.
    """
    n = max(int(n_obs), 2)
    sr_star = float(observed_sr)
    sr0 = expected_max_sharpe(n_trials)
    # Non-normal SE of the Sharpe estimator (Lo 2002 / BLP eq.)
    denom = math.sqrt(
        max(
            1.0 - skew * sr_star + ((kurtosis - 1.0) / 4.0) * sr_star * sr_star,
            1e-12,
        )
    )
    z = ((sr_star - sr0) * math.sqrt(n - 1)) / denom
    dsr = float(stats.norm.cdf(z))
    return {
        "observed_sr": sr_star,
        "sr0": float(sr0),
        "n_obs": float(n),
        "n_trials": float(max(int(n_trials), 1)),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "z": float(z),
        "dsr": dsr,
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (float(pvalues[i]), i))
    adj = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, factor * float(pvalues[idx]))
        running = max(running, val)
        adj[idx] = running
    return adj


def benjamini_hochberg_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (same order as input)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (float(pvalues[i]), i))
    adj = [1.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(1.0, float(pvalues[idx]) * m / (rank + 1))
        prev = min(prev, val)
        adj[idx] = prev
    return adj


def minimum_detectable_effect(
    *,
    n_obs: int,
    corr: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, float]:
    """MDE for paired deltaSharpe and deltaIC given sample length and A-B correlation.

    Uses a normal approximation for the paired difference of approximately
    unit-variance series: ``Var(delta) = 2(1-rho)/n``.
    """
    n = max(int(n_obs), 1)
    rho = float(np.clip(corr, -0.999, 0.999))
    z_a = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_b = float(stats.norm.ppf(power))
    var_delta = max(2.0 * (1.0 - rho), 1e-12)
    se = math.sqrt(var_delta / n)
    mde = (z_a + z_b) * se
    return {
        "n_obs": float(n),
        "corr": rho,
        "alpha": float(alpha),
        "power": float(power),
        "mde_dsharpe": float(mde),
        "mde_dic": float(mde),
        "se_unit": float(se),
    }


def forward_consistency_z(
    ic_hist: float,
    ic_fwd: float,
    se_fwd: float,
) -> dict[str, float]:
    """§16.3 S4 consistency z = (deltaIC_hist - deltaIC_fwd) / SE_fwd.

    Pass when ``z < 1.28`` (one-sided 10%). ``se_fwd`` comes from the
    date-block bootstrap of the forward deltaIC.
    """
    se = float(se_fwd)
    z = float("nan") if se <= 0 or not np.isfinite(se) else (float(ic_hist) - float(ic_fwd)) / se
    return {
        "ic_hist": float(ic_hist),
        "ic_fwd": float(ic_fwd),
        "se_fwd": se,
        "z": float(z),
        "threshold": FORWARD_CONSISTENCY_Z_THRESHOLD,
        "passes_s4": float(bool(np.isfinite(z) and z < FORWARD_CONSISTENCY_Z_THRESHOLD)),
    }


def confirmatory_ab_tests(
    *,
    delta_log_loss_pvalue: float,
    delta_ic_pvalue: float,
) -> dict[str, Any]:
    """Holm-adjust the two confirmatory A->B tests (deltalogLoss, deltaIC)."""
    raw = [float(delta_log_loss_pvalue), float(delta_ic_pvalue)]
    holm = holm_adjust(raw)
    bh = benjamini_hochberg_adjust(raw)
    return {
        "tests": ["delta_log_loss", "delta_rank_ic"],
        "raw_pvalues": raw,
        "holm_pvalues": holm,
        "bh_pvalues": bh,
        "n_confirmatory": 2,
    }
