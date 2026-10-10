"""Walk-forward runners for R0 / A-G (MASTER_SPEC Phases 3 + 8)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from jevlab.cv.sealed_guard import SealedAccessError, SealedDataGuard
from jevlab.cv.walkforward import DEV_MAX_DATE, Fold, make_folds, make_inner_folds
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import ColumnSetSpec, assert_ab_column_diff
from jevlab.models.gbm import make_lgbm_pipeline
from jevlab.models.gbm import predict_proba_positive as gbm_proba
from jevlab.models.logistic import (
    fitted_scaler_mean,
    make_logistic_pipeline,
)
from jevlab.models.logistic import (
    predict_proba_positive as logit_proba,
)
from jevlab.models.metrics import summarize_oos
from jevlab.models.r0 import score_r0
from jevlab.models.transforms import make_lineup_lgbm_pipeline, make_lineup_logistic_pipeline
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

ModelId = Literal["R0", "A", "D"]


@dataclass
class WalkForwardResult:
    """OOS predictions and metrics for one model."""

    model_id: str
    predictions: pd.DataFrame
    metrics: dict[str, float]
    fold_report: list[dict[str, Any]] = field(default_factory=list)
    chosen_C: list[float] = field(default_factory=list)  # per fold for model A


def _xy(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MODEL_A_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Design matrix and labels for eligible rows with finite y and features."""
    work = frame.loc[frame["eligible"].fillna(0).astype(int) == 1].copy()
    work = work.dropna(subset=["y", *columns])
    X = work.loc[:, list(columns)].to_numpy(dtype=float)
    y = work["y"].to_numpy(dtype=int)
    return X, y, work


def _select_C_inner(
    train_frame: pd.DataFrame,
    *,
    sessions: list[date],
    C_grid: list[float],
    seed: int,
    embargo_sessions: int = 5,
    columns: Sequence[str] = MODEL_A_COLUMNS,
    logit_columns: Sequence[str] = (),
) -> float:
    """Pick logistic C by mean inner-fold log-loss (lower is better)."""
    dates = sorted(train_frame["date"].unique().tolist())
    inner = make_inner_folds(
        dates, sessions=sessions, n_splits=3, embargo_sessions=embargo_sessions
    )
    if not inner:
        return float(C_grid[len(C_grid) // 2])

    cols = tuple(columns)
    losses: dict[float, list[float]] = {float(c): [] for c in C_grid}
    for fold in inner:
        tr = train_frame.loc[train_frame["date"].isin(fold.train_dates)]
        te = train_frame.loc[train_frame["date"].isin(fold.test_dates)]
        X_tr, y_tr, _ = _xy(tr, columns=cols)
        X_te, y_te, _ = _xy(te, columns=cols)
        if len(y_tr) < 20 or len(y_te) < 5 or len(np.unique(y_tr)) < 2:
            continue
        for c in C_grid:
            pipe = make_lineup_logistic_pipeline(
                C=c,
                seed=seed,
                columns=cols,
                logit_columns=logit_columns,
            )
            pipe.fit(X_tr, y_tr)
            p = logit_proba(pipe, X_te)
            losses[float(c)].append(
                float(log_loss(y_te, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))
            )
    scored = {c: float(np.mean(v)) for c, v in losses.items() if v}
    if not scored:
        return float(C_grid[len(C_grid) // 2])
    return min(scored.items(), key=lambda kv: kv[1])[0]


def _assert_no_sealed(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    dates = pd.to_datetime(frame["date"]).dt.date
    if (dates > DEV_MAX_DATE).any():
        raise SealedAccessError(
            "dev walk-forward refused rows after 2023-12-29; use SealedDataGuard"
        )


def prepare_model_frame(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Join features and labels on (symbol, date); drop sealed dates."""
    feat = features.copy()
    lab = labels.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.date
    lab["date"] = pd.to_datetime(lab["date"]).dt.date
    cols = ["symbol", "date", *MODEL_A_COLUMNS, *extra_columns]
    # Keep Jev / placebo / one-hot columns when present.
    for c in (*JEV_FEATURE_COLUMNS, *extra_columns):
        if c in feat.columns and c not in cols:
            cols.append(c)
    for c in feat.columns:
        if str(c).startswith(("goh_", "placebo_")) and c not in cols:
            cols.append(c)
    if "eligible" in feat.columns:
        cols.append("eligible")
    if "ret_5d" not in cols and "ret_5d" in feat.columns:
        cols.append("ret_5d")
    feat = feat.loc[:, [c for c in cols if c in feat.columns]]
    lab_cols = ["symbol", "date", "y", "X", "R", "Rbar", "y_cost", "eligible"]
    lab = lab.loc[:, [c for c in lab_cols if c in lab.columns]]
    # Prefer label eligibility if both present.
    if "eligible" in feat.columns and "eligible" in lab.columns:
        feat = feat.drop(columns=["eligible"])
    merged = feat.merge(lab, on=["symbol", "date"], how="inner")
    merged = merged.loc[merged["date"] <= DEV_MAX_DATE].reset_index(drop=True)
    _assert_no_sealed(merged)
    return merged


def run_r0_walkforward(frame: pd.DataFrame, folds: list[Fold]) -> WalkForwardResult:
    """Score R0 on each test fold (no fitting)."""
    parts: list[pd.DataFrame] = []
    report: list[dict[str, Any]] = []
    for fold in folds:
        te = frame.loc[frame["date"].isin(fold.test_dates)].copy()
        scored = score_r0(te)
        scored["fold_id"] = fold.fold_id
        parts.append(scored)
        report.append({"fold_id": fold.fold_id, "n_test": len(scored)})
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics = summarize_oos(preds) if not preds.empty else {}
    return WalkForwardResult("R0", preds, metrics, report)


def run_logistic_walkforward(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> WalkForwardResult:
    """Model A: in-fold scaler+logistic with inner-CV C selection."""
    parts: list[pd.DataFrame] = []
    report: list[dict[str, Any]] = []
    chosen: list[float] = []
    for fold in folds:
        tr = frame.loc[frame["date"].isin(fold.train_dates)]
        te = frame.loc[frame["date"].isin(fold.test_dates)].copy()
        C_star = _select_C_inner(
            tr,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        chosen.append(C_star)
        X_tr, y_tr, _ = _xy(tr)
        X_te, y_te, te_rows = _xy(te)
        if len(y_tr) < 20 or len(np.unique(y_tr)) < 2:
            report.append({"fold_id": fold.fold_id, "status": "skip_insufficient_train"})
            continue
        pipe = make_logistic_pipeline(C=C_star, seed=seed)
        pipe.fit(X_tr, y_tr)
        if len(te_rows) == 0:
            report.append({"fold_id": fold.fold_id, "status": "skip_empty_test"})
            continue
        p = logit_proba(pipe, X_te)
        out = te_rows.copy()
        out["prob"] = p
        out["score"] = p
        out["model_id"] = "A"
        out["fold_id"] = fold.fold_id
        out["C"] = C_star
        parts.append(out)
        report.append(
            {
                "fold_id": fold.fold_id,
                "C": C_star,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "scaler_mean_sum": float(np.sum(fitted_scaler_mean(pipe))),
            }
        )
        log.info("experiment.finish", model="A", fold=fold.fold_id, C=C_star)
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics = summarize_oos(preds) if not preds.empty else {}
    return WalkForwardResult("A", preds, metrics, report, chosen)


def run_lgbm_walkforward(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    lgbm_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> WalkForwardResult:
    """Model D: fixed-hyperparameter LightGBM, fit in-fold only."""
    parts: list[pd.DataFrame] = []
    report: list[dict[str, Any]] = []
    for fold in folds:
        tr = frame.loc[frame["date"].isin(fold.train_dates)]
        te = frame.loc[frame["date"].isin(fold.test_dates)].copy()
        X_tr, y_tr, _ = _xy(tr)
        X_te, y_te, te_rows = _xy(te)
        if len(y_tr) < 20 or len(np.unique(y_tr)) < 2:
            report.append({"fold_id": fold.fold_id, "status": "skip_insufficient_train"})
            continue
        pipe = make_lgbm_pipeline(params=lgbm_params, seed=seed)
        pipe.fit(X_tr, y_tr)
        if len(te_rows) == 0:
            report.append({"fold_id": fold.fold_id, "status": "skip_empty_test"})
            continue
        p = gbm_proba(pipe, X_te)
        out = te_rows.copy()
        out["prob"] = p
        out["score"] = p
        out["model_id"] = "D"
        out["fold_id"] = fold.fold_id
        parts.append(out)
        report.append({"fold_id": fold.fold_id, "n_train": len(y_tr), "n_test": len(y_te)})
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics = summarize_oos(preds) if not preds.empty else {}
    return WalkForwardResult("D", preds, metrics, report)


def run_baselines(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    C_grid: list[float] | None = None,
    lgbm_params: dict[str, Any] | None = None,
    seed: int = 42,
    embargo_sessions: int = 5,
    sealed_guard: SealedDataGuard | None = None,
) -> dict[str, WalkForwardResult]:
    """Run R0, A, D walk-forward on joined feature/label frame (dev dates only)."""
    frame = prepare_model_frame(features, labels)
    if sealed_guard is not None:
        sealed_guard.assert_dev_dates(frame["date"])
    sessions = sorted(frame["date"].unique().tolist())
    folds = make_folds(sessions, embargo_sessions=embargo_sessions)
    if not folds:
        raise RuntimeError("no walk-forward folds produced; check date coverage")
    C_grid = C_grid or [0.01, 0.1, 1.0]
    return {
        "R0": run_r0_walkforward(frame, folds),
        "A": run_logistic_walkforward(
            frame,
            folds,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        ),
        "D": run_lgbm_walkforward(frame, folds, lgbm_params=lgbm_params, seed=seed),
    }


def assert_identical_folds(folds_a: list[Fold], folds_b: list[Fold]) -> None:
    """A and B must share identical fold ids and train/test date sets."""
    if len(folds_a) != len(folds_b):
        raise AssertionError(f"fold count mismatch: {len(folds_a)} vs {len(folds_b)}")
    for fa, fb in zip(folds_a, folds_b, strict=True):
        if fa.fold_id != fb.fold_id:
            raise AssertionError(f"fold_id mismatch: {fa.fold_id} vs {fb.fold_id}")
        if fa.train_dates != fb.train_dates or fa.test_dates != fb.test_dates:
            raise AssertionError(f"fold dates differ for {fa.fold_id}")


def run_column_set_logistic(
    frame: pd.DataFrame,
    folds: list[Fold],
    spec: ColumnSetSpec,
    *,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> WalkForwardResult:
    """Logistic walk-forward for one column-set (A/B/C/F/G)."""
    if spec.learner != "logistic":
        raise ValueError(f"{spec.model_id} is not logistic")
    if spec.interface_only:
        raise RuntimeError(f"{spec.model_id} is interface-only; values not available yet")

    cols = tuple(spec.columns)
    parts: list[pd.DataFrame] = []
    report: list[dict[str, Any]] = []
    chosen: list[float] = []
    for fold in folds:
        tr = frame.loc[frame["date"].isin(fold.train_dates)]
        te = frame.loc[frame["date"].isin(fold.test_dates)].copy()
        C_star = _select_C_inner(
            tr,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
            columns=cols,
            logit_columns=spec.logit_columns,
        )
        chosen.append(C_star)
        X_tr, y_tr, _ = _xy(tr, columns=cols)
        X_te, y_te, te_rows = _xy(te, columns=cols)
        if len(y_tr) < 20 or len(np.unique(y_tr)) < 2:
            report.append({"fold_id": fold.fold_id, "status": "skip_insufficient_train"})
            continue
        pipe = make_lineup_logistic_pipeline(
            C=C_star,
            seed=seed,
            columns=cols,
            logit_columns=spec.logit_columns,
        )
        pipe.fit(X_tr, y_tr)
        if len(te_rows) == 0:
            report.append({"fold_id": fold.fold_id, "status": "skip_empty_test"})
            continue
        p = logit_proba(pipe, X_te)
        out = te_rows.copy()
        out["prob"] = p
        out["score"] = p
        out["model_id"] = spec.model_id
        out["fold_id"] = fold.fold_id
        out["C"] = C_star
        parts.append(out)
        report.append(
            {
                "fold_id": fold.fold_id,
                "C": C_star,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "n_columns": len(cols),
                "scaler_mean_sum": float(np.sum(fitted_scaler_mean(pipe))),
            }
        )
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics = summarize_oos(preds) if not preds.empty else {}
    return WalkForwardResult(spec.model_id, preds, metrics, report, chosen)


def run_column_set_lgbm(
    frame: pd.DataFrame,
    folds: list[Fold],
    spec: ColumnSetSpec,
    *,
    lgbm_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> WalkForwardResult:
    """LightGBM walk-forward for one column-set (D/E)."""
    if spec.learner != "lightgbm":
        raise ValueError(f"{spec.model_id} is not lightgbm")
    cols = tuple(spec.columns)
    parts: list[pd.DataFrame] = []
    report: list[dict[str, Any]] = []
    for fold in folds:
        tr = frame.loc[frame["date"].isin(fold.train_dates)]
        te = frame.loc[frame["date"].isin(fold.test_dates)].copy()
        X_tr, y_tr, _ = _xy(tr, columns=cols)
        X_te, y_te, te_rows = _xy(te, columns=cols)
        if len(y_tr) < 20 or len(np.unique(y_tr)) < 2:
            report.append({"fold_id": fold.fold_id, "status": "skip_insufficient_train"})
            continue
        pipe = make_lineup_lgbm_pipeline(
            seed=seed,
            columns=cols,
            logit_columns=spec.logit_columns,
            params=lgbm_params,
        )
        pipe.fit(X_tr, y_tr)
        if len(te_rows) == 0:
            report.append({"fold_id": fold.fold_id, "status": "skip_empty_test"})
            continue
        p = gbm_proba(pipe, X_te)
        out = te_rows.copy()
        out["prob"] = p
        out["score"] = p
        out["model_id"] = spec.model_id
        out["fold_id"] = fold.fold_id
        parts.append(out)
        report.append(
            {
                "fold_id": fold.fold_id,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "n_columns": len(cols),
            }
        )
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics = summarize_oos(preds) if not preds.empty else {}
    return WalkForwardResult(spec.model_id, preds, metrics, report)


def run_lineup_walkforward(
    frame: pd.DataFrame,
    folds: list[Fold],
    registry: dict[str, ColumnSetSpec],
    *,
    sessions: list[date],
    C_grid: list[float],
    lgbm_params: dict[str, Any] | None = None,
    seed: int = 42,
    embargo_sessions: int = 5,
    model_ids: Sequence[str] | None = None,
) -> dict[str, WalkForwardResult]:
    """Run requested lineup models on a shared fold schedule.

    A and B share the same ``folds``, ``C_grid``, ``seed``, and row universe;
    only columns differ (asserted).
    """
    assert_ab_column_diff()
    if "A" in registry and "B" in registry:
        assert_identical_folds(folds, folds)  # shared object identity not required
        a_cols = set(registry["A"].columns)
        b_cols = set(registry["B"].columns)
        if b_cols - a_cols != set(JEV_FEATURE_COLUMNS):
            raise AssertionError("B columns must equal A + 8 Jev features")
        if registry["A"].learner != registry["B"].learner:
            raise AssertionError("A and B must share learner")

    wanted = (
        list(model_ids)
        if model_ids is not None
        else [m for m, spec in registry.items() if not spec.interface_only]
    )
    results: dict[str, WalkForwardResult] = {}
    for mid in wanted:
        spec = registry[mid]  # type: ignore[index]
        if spec.interface_only:
            continue
        if spec.learner == "logistic":
            results[mid] = run_column_set_logistic(
                frame,
                folds,
                spec,
                sessions=sessions,
                C_grid=C_grid,
                seed=seed,
                embargo_sessions=embargo_sessions,
            )
        else:
            results[mid] = run_column_set_lgbm(
                frame,
                folds,
                spec,
                lgbm_params=lgbm_params,
                seed=seed,
            )
    return results
