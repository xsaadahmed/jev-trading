"""Locked V2 sealed training cutoff and fit-once Model A."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from jevlab.config.v2_research import EXPECTED_T0, EXPECTED_TRAINING_CUTOFF
from jevlab.cv.purge import embargo_cutoff_date
from jevlab.cv.walkforward import make_inner_folds
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.models.logistic import predict_proba_positive
from jevlab.models.pipelines import _select_C_inner, _xy
from jevlab.models.transforms import make_lineup_logistic_pipeline
from jevlab.utils.calendar import nyse_session_dates
from jevlab.v2.candidate import V2_C_GRID, V2_EMBARGO, V2_SEED
from jevlab.v2.dev_io import assert_frame_pre_sealed

SEALED_T0: date = date.fromisoformat(EXPECTED_T0)
EXPECTED_CUTOFF: date = date.fromisoformat(EXPECTED_TRAINING_CUTOFF)


class TrainingLeakError(AssertionError):
    """Training set violated the embargo / sealed-year rule."""


def assert_sealed_training_cutoff(
    *,
    t0: date = SEALED_T0,
    embargo_sessions: int = V2_EMBARGO,
    calendar_start: date = date(2023, 11, 1),
    calendar_end: date = date(2024, 1, 15),
) -> date:
    """Derive and assert the frozen cutoff ``2023-12-22`` from the NYSE calendar.

    The calendar window includes T0 so ``embargo_cutoff_date`` can index it.
    This does **not** load market data or sealed feature rows.
    """
    sessions = nyse_session_dates(calendar_start, calendar_end)
    cutoff = embargo_cutoff_date(sessions, t0, embargo_sessions=embargo_sessions)
    if cutoff is None:
        raise TrainingLeakError("embargo cutoff could not be derived")
    if t0 == SEALED_T0 and cutoff != EXPECTED_CUTOFF:
        raise TrainingLeakError(f"expected training cutoff {EXPECTED_CUTOFF}, derived {cutoff}")
    # Labels for a decision at cutoff use open(t+4); with embargo=5 that exit
    # is strictly before T0.
    i0 = sessions.index(t0) if t0 in sessions else None
    ic = sessions.index(cutoff)
    if i0 is not None and ic > i0 - embargo_sessions:
        raise TrainingLeakError("cutoff does not satisfy s(t) <= s(T0) - 5")
    return cutoff


def filter_training_rows(
    frame: pd.DataFrame,
    *,
    cutoff: date,
    t0: date,
) -> pd.DataFrame:
    """Keep decision rows ``date <= cutoff``; refuse any date >= T0."""
    assert_frame_pre_sealed(frame)
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    if (work["date"] >= t0).any():
        raise TrainingLeakError("training frame contains T0 or later rows")
    out = work.loc[work["date"] <= cutoff].reset_index(drop=True)
    if out.empty:
        raise TrainingLeakError("no training rows at embargo cutoff")
    return out


@dataclass(frozen=True, slots=True)
class FittedModelA:
    pipeline: Pipeline
    C: float
    cutoff: date
    n_train_rows: int
    fit_count: int
    seed: int
    columns: tuple[str, ...]


def fit_once_model_a(
    train_frame: pd.DataFrame,
    *,
    sessions: list[date],
    t0: date,
    cutoff: date,
    C_grid: list[float] | None = None,
    seed: int = V2_SEED,
    embargo_sessions: int = V2_EMBARGO,
) -> FittedModelA:
    """Inner-CV C on pre-cutoff rows; fit one Model A; never refit."""
    train = filter_training_rows(train_frame, cutoff=cutoff, t0=t0)
    if train["date"].gt(cutoff).any():
        raise TrainingLeakError("post-cutoff rows reached inner CV")
    inner = make_inner_folds(
        sorted(train["date"].unique().tolist()),
        sessions=sessions,
        n_splits=3,
        embargo_sessions=embargo_sessions,
    )
    _ = inner
    grid = list(C_grid) if C_grid is not None else list(V2_C_GRID)
    C_star = _select_C_inner(
        train,
        sessions=sessions,
        C_grid=grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
        columns=MODEL_A_COLUMNS,
        logit_columns=(),
    )
    X, y, _ = _xy(train, columns=MODEL_A_COLUMNS)
    if len(y) < 20 or len(np.unique(y)) < 2:
        raise TrainingLeakError("insufficient training rows for Model A")
    pipe = make_lineup_logistic_pipeline(
        C=C_star,
        seed=seed,
        columns=MODEL_A_COLUMNS,
        logit_columns=(),
    )
    pipe.fit(X, y)
    return FittedModelA(
        pipeline=pipe,
        C=float(C_star),
        cutoff=cutoff,
        n_train_rows=len(y),
        fit_count=1,
        seed=seed,
        columns=MODEL_A_COLUMNS,
    )


def score_model_a(fitted: FittedModelA, score_frame: pd.DataFrame) -> pd.DataFrame:
    """Score eligible rows with the frozen artifact (no refit).

    ``y`` may be missing at the sealed boundary; ranking does not require it.
    Ineligible and incomplete-feature rows are retained with NaN scores.
    """
    work = score_frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    missing = [c for c in fitted.columns if c not in work.columns]
    if missing:
        raise TrainingLeakError(f"score frame missing Model A columns: {missing}")
    out = work.copy()
    out["prob"] = np.nan
    out["score"] = np.nan
    out["model_id"] = "A"
    out["C"] = fitted.C
    elig = (
        out["eligible"].fillna(0).astype(int) == 1
        if "eligible" in out.columns
        else pd.Series(True, index=out.index)
    )
    feat = out.loc[:, list(fitted.columns)].apply(pd.to_numeric, errors="coerce")
    finite = feat.notna().all(axis=1)
    rows = out.loc[elig & finite]
    if rows.empty:
        return out
    X = feat.loc[rows.index].to_numpy(dtype=float)
    p = predict_proba_positive(fitted.pipeline, X)
    out.loc[rows.index, "prob"] = p
    out.loc[rows.index, "score"] = p
    return out
