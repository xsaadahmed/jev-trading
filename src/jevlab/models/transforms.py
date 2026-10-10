"""In-fold feature transforms: clip+logit for Jev probabilities (MASTER_SPEC §9)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jevlab.models.gbm import make_lgbm_pipeline
from jevlab.models.logistic import make_logistic_pipeline

LOGIT_CLIP_LOW: float = 0.01
LOGIT_CLIP_HIGH: float = 0.99


def logit_clip(
    values: np.ndarray,
    *,
    low: float = LOGIT_CLIP_LOW,
    high: float = LOGIT_CLIP_HIGH,
) -> np.ndarray:
    """``logit(clip(p, low, high))`` elementwise."""
    p = np.clip(np.asarray(values, dtype=float), low, high)
    return np.log(p / (1.0 - p))


class SelectiveLogitClip(BaseEstimator, TransformerMixin):
    """Apply clip+logit to selected column indices; pass others through.

    Stateless w.r.t. training data (no fit-time statistics); placed *before*
    ``StandardScaler`` so scaling still fits in-fold on the transformed values.
    """

    def __init__(
        self,
        logit_indices: Sequence[int] | None = None,
        *,
        low: float = LOGIT_CLIP_LOW,
        high: float = LOGIT_CLIP_HIGH,
    ) -> None:
        self.logit_indices = list(logit_indices) if logit_indices is not None else []
        self.low = float(low)
        self.high = float(high)

    def fit(self, X: np.ndarray, y: object | None = None) -> SelectiveLogitClip:
        _ = y
        X = np.asarray(X)
        n_cols = X.shape[1] if X.ndim == 2 else 0
        for i in self.logit_indices:
            if i < 0 or i >= n_cols:
                raise ValueError(f"logit index {i} out of range for n_cols={n_cols}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype=float).copy()
        if out.ndim != 2:
            raise ValueError("SelectiveLogitClip expects 2-D X")
        for i in self.logit_indices:
            out[:, i] = logit_clip(out[:, i], low=self.low, high=self.high)
        return out


def logit_indices_for_columns(
    columns: Sequence[str],
    logit_columns: Sequence[str],
) -> list[int]:
    """Map logit column names to positions within ``columns``."""
    index = {c: i for i, c in enumerate(columns)}
    missing = [c for c in logit_columns if c not in index]
    if missing:
        raise KeyError(f"logit columns not in design matrix: {missing}")
    return [index[c] for c in logit_columns]


def make_lineup_logistic_pipeline(
    *,
    C: float,
    seed: int,
    columns: Sequence[str],
    logit_columns: Sequence[str] = (),
) -> Pipeline:
    """Logit-clip (optional) → StandardScaler → L2 logistic; all fit in-fold."""
    indices = logit_indices_for_columns(columns, logit_columns) if logit_columns else []
    base = make_logistic_pipeline(C=C, seed=seed)
    if not indices:
        return base
    return Pipeline(
        steps=[
            ("logit_clip", SelectiveLogitClip(indices)),
            ("scaler", base.named_steps["scaler"]),
            ("clf", base.named_steps["clf"]),
        ]
    )


def make_lineup_lgbm_pipeline(
    *,
    seed: int,
    columns: Sequence[str],
    logit_columns: Sequence[str] = (),
    params: dict[str, object] | None = None,
) -> Pipeline:
    """Logit-clip (optional) → StandardScaler → LightGBM; all fit in-fold."""
    indices = logit_indices_for_columns(columns, logit_columns) if logit_columns else []
    base = make_lgbm_pipeline(params=params, seed=seed)
    if not indices:
        return base
    return Pipeline(
        steps=[
            ("logit_clip", SelectiveLogitClip(indices)),
            ("scaler", StandardScaler()),
            ("clf", base.named_steps["clf"]),
        ]
    )
