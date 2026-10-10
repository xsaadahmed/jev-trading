"""LightGBM classifier for Model D (fixed small hyperparameters, deterministic)."""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "deterministic": True,
    "n_jobs": 1,
    "verbosity": -1,
}


def make_lgbm_pipeline(
    *,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> Pipeline:
    """StandardScaler + deterministic single-thread LightGBM classifier."""
    cfg = dict(DEFAULT_LGBM_PARAMS)
    if params:
        cfg.update(params)
    cfg["random_state"] = seed
    cfg["deterministic"] = True
    cfg["n_jobs"] = 1
    # Force single-thread OpenMP friendliness for confirmatory runs.
    cfg.setdefault("force_col_wise", True)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(**cfg)),
        ]
    )


def predict_proba_positive(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 in classes or 1.0 in classes:
        idx = classes.index(1) if 1 in classes else classes.index(1.0)
    else:
        idx = int(np.argmax(classes))
    return np.asarray(proba[:, idx], dtype=float)
