"""Simple predictive metrics for Phase 3 OOS evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean log-loss; clips probabilities away from {0,1}."""
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return float("nan")
    yt = y_true[mask].astype(float)
    yp = np.clip(y_prob[mask].astype(float), 1e-15, 1.0 - 1e-15)
    return float(log_loss(yt, yp, labels=[0.0, 1.0]))


def binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask].astype(float)
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, y_prob[mask]))


def binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return float("nan")
    return float(brier_score_loss(y_true[mask].astype(float), y_prob[mask].astype(float)))


def daily_rank_ic(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "X",
    date_col: str = "date",
    eligible_col: str = "eligible",
) -> float:
    """Mean daily cross-sectional Spearman rank-IC of score vs target among eligible."""
    if frame.empty:
        return float("nan")
    work = frame.copy()
    work = work.loc[work[eligible_col].fillna(0).astype(int) == 1]
    ics: list[float] = []
    for _, group in work.groupby(date_col):
        if len(group) < 3:
            continue
        if group[score_col].nunique(dropna=True) < 2:
            continue
        if group[target_col].nunique(dropna=True) < 2:
            continue
        ic = group[score_col].corr(group[target_col], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def daily_rank_ic_series(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "X",
    date_col: str = "date",
    eligible_col: str = "eligible",
) -> list[float]:
    """Per-date rank-IC series among eligible names (for t-stat reporting)."""
    if frame.empty:
        return []
    work = frame.copy()
    if eligible_col in work.columns:
        work = work.loc[work[eligible_col].fillna(0).astype(int) == 1]
    ics: list[float] = []
    for _, group in work.groupby(date_col):
        if len(group) < 3:
            continue
        if group[score_col].nunique(dropna=True) < 2:
            continue
        if group[target_col].nunique(dropna=True) < 2:
            continue
        ic = group[score_col].corr(group[target_col], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    return ics


def summarize_oos(
    predictions: pd.DataFrame,
    *,
    prob_col: str = "prob",
    score_col: str = "score",
    label_col: str = "y",
    target_col: str = "X",
) -> dict[str, float]:
    """Return log-loss, Brier, AUC, mean daily rank-IC, and IC t-stat on OOS rows."""
    if "eligible" in predictions.columns:
        elig = predictions.loc[predictions["eligible"].fillna(0).astype(int) == 1]
    else:
        elig = predictions
    y = elig[label_col].to_numpy(dtype=float)
    # Prefer probability for log-loss/AUC; fall back to score.
    if prob_col in elig.columns:
        p = elig[prob_col].to_numpy(dtype=float)
    else:
        # Map scores to (0,1) via rank percentile within sample for metrics only.
        p = elig[score_col].rank(pct=True).to_numpy(dtype=float)
    score_used = score_col if score_col in elig.columns else prob_col
    ics = daily_rank_ic_series(elig, score_col=score_used, target_col=target_col)
    if len(ics) >= 2:
        mean_ic = float(np.mean(ics))
        se = float(np.std(ics, ddof=1) / np.sqrt(len(ics)))
        ic_t = float(mean_ic / se) if se > 0 else float("nan")
    elif len(ics) == 1:
        mean_ic = float(ics[0])
        ic_t = float("nan")
    else:
        mean_ic = float("nan")
        ic_t = float("nan")
    return {
        "log_loss": binary_log_loss(y, p),
        "brier": binary_brier(y, p),
        "auc": binary_auc(y, p),
        "rank_ic": mean_ic,
        "ic_tstat": ic_t,
        "n_rows": float(len(elig)),
    }
