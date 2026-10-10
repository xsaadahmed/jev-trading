"""Predictive-quality metrics (MASTER_SPEC §12.1).

Log-loss, Brier, ROC-AUC, calibration (10 equal-count bins), reliability data,
and daily cross-sectional Spearman rank-IC with Newey-West t-stat (lag 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

DEFAULT_NW_LAG: int = 5
N_CALIBRATION_BINS: int = 10


def _eligible(frame: pd.DataFrame, eligible_col: str = "eligible") -> pd.DataFrame:
    if eligible_col in frame.columns:
        return frame.loc[frame[eligible_col].fillna(0).astype(int) == 1].copy()
    return frame.copy()


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return float("nan")
    yt = y_true[mask].astype(float)
    yp = np.clip(y_prob[mask].astype(float), 1e-15, 1.0 - 1e-15)
    return float(log_loss(yt, yp, labels=[0.0, 1.0]))


def binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return float("nan")
    return float(brier_score_loss(y_true[mask].astype(float), y_prob[mask].astype(float)))


def binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask].astype(float)
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, y_prob[mask]))


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Equal-count bin reliability diagram data."""

    bin_edges: list[float]
    bin_mean_pred: list[float]
    bin_frac_pos: list[float]
    bin_counts: list[int]
    n_bins: int = N_CALIBRATION_BINS

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_edges": self.bin_edges,
            "bin_mean_pred": self.bin_mean_pred,
            "bin_frac_pos": self.bin_frac_pos,
            "bin_counts": self.bin_counts,
            "n_bins": self.n_bins,
        }


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = N_CALIBRATION_BINS,
) -> CalibrationResult:
    """10 equal-count calibration bins (reliability diagram inputs)."""
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    yt = y_true[mask].astype(float)
    yp = np.clip(y_prob[mask].astype(float), 0.0, 1.0)
    if len(yt) == 0:
        return CalibrationResult([], [], [], [], n_bins=n_bins)
    # Equal-count bins via rank quantiles
    order = np.argsort(yp)
    yt_s = yt[order]
    yp_s = yp[order]
    edges_idx = np.linspace(0, len(yp_s), n_bins + 1, dtype=int)
    mean_pred: list[float] = []
    frac_pos: list[float] = []
    counts: list[int] = []
    edge_vals: list[float] = [float(yp_s[0])]
    for i in range(n_bins):
        lo, hi = int(edges_idx[i]), int(edges_idx[i + 1])
        if hi <= lo:
            continue
        chunk_y = yt_s[lo:hi]
        chunk_p = yp_s[lo:hi]
        mean_pred.append(float(np.mean(chunk_p)))
        frac_pos.append(float(np.mean(chunk_y)))
        counts.append(len(chunk_y))
        edge_vals.append(float(chunk_p[-1]))
    return CalibrationResult(
        bin_edges=edge_vals,
        bin_mean_pred=mean_pred,
        bin_frac_pos=frac_pos,
        bin_counts=counts,
        n_bins=n_bins,
    )


def reliability_data(cal: CalibrationResult) -> pd.DataFrame:
    """Tabular reliability curve: predicted vs empirical frequency."""
    return pd.DataFrame(
        {
            "bin": list(range(len(cal.bin_mean_pred))),
            "mean_predicted": cal.bin_mean_pred,
            "frac_positive": cal.bin_frac_pos,
            "count": cal.bin_counts,
        }
    )


def daily_spearman_ic_series(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "X",
    date_col: str = "date",
    eligible_col: str = "eligible",
) -> pd.Series:
    """Per-date cross-sectional Spearman rank-IC (eligible rows only)."""
    work = _eligible(frame, eligible_col)
    if work.empty:
        return pd.Series(dtype=float)
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    ics: dict[Any, float] = {}
    for d, group in work.groupby(date_col):
        if len(group) < 3:
            continue
        if group[score_col].nunique(dropna=True) < 2:
            continue
        if group[target_col].nunique(dropna=True) < 2:
            continue
        ic = group[score_col].corr(group[target_col], method="spearman")
        if pd.notna(ic):
            ics[d] = float(ic)
    if not ics:
        return pd.Series(dtype=float)
    return pd.Series(ics).sort_index()


def newey_west_tstat(
    series: Sequence[float] | np.ndarray | pd.Series,
    *,
    lag: int = DEFAULT_NW_LAG,
) -> dict[str, float]:
    """t-stat that mean(series)=0 using Newey-West HAC (lag ``lag``)."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < max(lag + 2, 3):
        return {
            "mean": float("nan"),
            "se": float("nan"),
            "t_stat": float("nan"),
            "n": float(n),
            "lag": float(lag),
        }
    # OLS of y on constant; HAC covariance
    x = np.ones((n, 1))
    model = OLS(y, x).fit()
    cov = cov_hac(model, nlags=lag)
    se = float(np.sqrt(max(float(cov[0, 0]), 0.0)))
    mean = float(np.mean(y))
    if se > 0.0:
        t_stat = mean / se
    elif mean == 0.0:
        t_stat = 0.0
    else:
        # Degenerate SE (constant series); report unsigned overflow sentinel.
        t_stat = float(np.sign(mean) * 1e6)
    return {
        "mean": mean,
        "se": se,
        "t_stat": float(t_stat),
        "n": float(n),
        "lag": float(lag),
        "hit_rate_positive": float(np.mean(y > 0)),
    }


def daily_rank_ic_summary(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "X",
    date_col: str = "date",
    eligible_col: str = "eligible",
    nw_lag: int = DEFAULT_NW_LAG,
) -> dict[str, float]:
    """Mean IC, NW t-stat, and hit rate of IC>0."""
    series = daily_spearman_ic_series(
        frame,
        score_col=score_col,
        target_col=target_col,
        date_col=date_col,
        eligible_col=eligible_col,
    )
    nw = newey_west_tstat(series.to_numpy(), lag=nw_lag)
    return {
        "mean_ic": nw["mean"],
        "ic_se_nw": nw["se"],
        "ic_tstat_nw": nw["t_stat"],
        "ic_hit_rate": nw.get("hit_rate_positive", float("nan")),
        "n_dates": nw["n"],
        "nw_lag": float(nw_lag),
    }


def per_date_log_loss(
    frame: pd.DataFrame,
    *,
    prob_col: str = "prob",
    label_col: str = "y",
    date_col: str = "date",
    eligible_col: str = "eligible",
) -> pd.Series:
    """Mean log-loss per date (eligible rows)."""
    work = _eligible(frame, eligible_col)
    if work.empty:
        return pd.Series(dtype=float)
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    out: dict[Any, float] = {}
    for d, group in work.groupby(date_col):
        y = group[label_col].to_numpy(dtype=float)
        p = group[prob_col].to_numpy(dtype=float)
        out[d] = binary_log_loss(y, p)
    return pd.Series(out).sort_index()


def predictive_summary(
    frame: pd.DataFrame,
    *,
    prob_col: str = "prob",
    score_col: str = "score",
    label_col: str = "y",
    target_col: str = "X",
    nw_lag: int = DEFAULT_NW_LAG,
) -> dict[str, Any]:
    """Full §12.1 predictive summary for one model's OOS predictions."""
    work = _eligible(frame)
    if work.empty:
        return {"n_rows": 0.0}
    if prob_col not in work.columns and score_col in work.columns:
        # Rank-map scores to (0,1) for probability metrics only.
        work = work.copy()
        work[prob_col] = work[score_col].rank(pct=True)
    y = work[label_col].to_numpy(dtype=float)
    p = work[prob_col].to_numpy(dtype=float)
    cal = calibration_bins(y, p)
    score_used = score_col if score_col in work.columns else prob_col
    ic = daily_rank_ic_summary(work, score_col=score_used, target_col=target_col, nw_lag=nw_lag)
    return {
        "n_rows": float(len(work)),
        "log_loss": binary_log_loss(y, p),
        "brier": binary_brier(y, p),
        "auc": binary_auc(y, p),
        "calibration": cal.to_dict(),
        "reliability": reliability_data(cal).to_dict(orient="list"),
        "rank_ic": ic,
    }
