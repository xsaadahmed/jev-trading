"""L2 logistic regression pipeline for Model A (MASTER_SPEC §9)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_pipeline(
    *,
    C: float = 1.0,
    seed: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """StandardScaler (fit in-fold) + L2 logistic regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(C),
                    solver="lbfgs",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def predict_proba_positive(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) from a fitted logistic pipeline."""
    proba = pipeline.predict_proba(X)
    # Positive class column.
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 in classes or 1.0 in classes:
        idx = classes.index(1) if 1 in classes else classes.index(1.0)
    else:
        idx = int(np.argmax(classes))
    return np.asarray(proba[:, idx], dtype=float)


def fitted_scaler_mean(pipeline: Pipeline) -> np.ndarray:
    """Expose train-fold scaler mean for leakage tests."""
    scaler: Any = pipeline.named_steps["scaler"]
    return np.asarray(scaler.mean_, dtype=float)
