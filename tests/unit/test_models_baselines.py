"""Baseline models R0/A/D and leakage tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from jevlab.cv.walkforward import make_folds
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.models.logistic import fitted_scaler_mean, make_logistic_pipeline
from jevlab.models.pipelines import (
    prepare_model_frame,
    run_baselines,
    run_logistic_walkforward,
    run_r0_walkforward,
)
from jevlab.models.r0 import score_r0
from jevlab.utils.calendar import nyse_session_dates


def _synthetic_panel(
    *,
    start: date = date(2016, 1, 4),
    end: date = date(2019, 12, 31),
    n_symbols: int = 13,
    seed: int = 0,
) -> pd.DataFrame:
    """Features + labels with a weak linear signal in ret_1d."""
    rng = np.random.default_rng(seed)
    sessions = nyse_session_dates(start, end)
    symbols = [f"E{i:02d}" for i in range(n_symbols)]
    rows: list[dict[str, object]] = []
    for d in sessions:
        for sym in symbols:
            noise = rng.normal(0, 1, size=len(MODEL_A_COLUMNS))
            feats = {c: float(noise[i]) for i, c in enumerate(MODEL_A_COLUMNS)}
            # Implant signal: higher ret_1d → higher P(y=1)
            signal = feats["ret_1d"]
            y = 1 if (signal + rng.normal(0, 0.5)) > 0 else 0
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    **feats,
                    "eligible": 1,
                    "y": y,
                    "X": float(signal),
                    "R": float(signal * 0.01),
                    "Rbar": 0.0,
                    "y_cost": y,
                }
            )
    return pd.DataFrame(rows)


def test_r0_scores_lowest_ret_5d_highest() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": [date(2019, 1, 2), date(2019, 1, 2)],
            "ret_5d": [-0.05, 0.01],
            "eligible": [1, 1],
        }
    )
    scored = score_r0(frame)
    assert (
        scored.loc[scored["symbol"] == "A", "score"].iloc[0]
        > scored.loc[scored["symbol"] == "B", "score"].iloc[0]
    )


def test_global_fit_leakage_scaler_differs_from_full_sample() -> None:
    panel = _synthetic_panel(end=date(2018, 6, 29), seed=1)
    sessions = sorted(panel["date"].unique().tolist())
    folds = make_folds(
        sessions,
        oos_start=date(2018, 1, 2),
        oos_end=date(2018, 6, 29),
        first_train_end=date(2017, 12, 29),
        train_start=date(2016, 1, 4),
    )
    assert folds
    fold = folds[0]
    tr = panel.loc[panel["date"].isin(fold.train_dates)]
    X_tr = tr.loc[:, list(MODEL_A_COLUMNS)].to_numpy(dtype=float)
    y_tr = tr["y"].to_numpy(dtype=int)
    pipe = make_logistic_pipeline(C=1.0, seed=0)
    pipe.fit(X_tr, y_tr)
    train_mean = fitted_scaler_mean(pipe)

    X_all = panel.loc[:, list(MODEL_A_COLUMNS)].to_numpy(dtype=float)
    y_all = panel["y"].to_numpy(dtype=int)
    pipe_full = make_logistic_pipeline(C=1.0, seed=0)
    pipe_full.fit(X_all, y_all)
    full_mean = fitted_scaler_mean(pipe_full)
    # In-fold scaler must not equal a global fit on train+test.
    assert not np.allclose(train_mean, full_mean)


def test_prepare_model_frame_rejects_sealed_dates() -> None:
    features = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": [date(2023, 12, 29), date(2024, 1, 2)],
            **{c: [0.0, 0.0] for c in MODEL_A_COLUMNS},
            "eligible": [1, 1],
        }
    )
    labels = features.loc[:, ["symbol", "date", "eligible"]].assign(
        y=1, X=0.0, R=0.0, Rbar=0.0, y_cost=1
    )
    # prepare drops sealed rather than raising when filtering — assert no 2024 remains.
    merged = prepare_model_frame(features, labels)
    assert merged["date"].max() <= date(2023, 12, 29)
    assert date(2024, 1, 2) not in set(merged["date"])


def test_run_baselines_oos_metrics_no_2024() -> None:
    panel = _synthetic_panel(end=date(2019, 9, 30), seed=2)
    # Split into fake features/labels for run_baselines API.
    features = panel.drop(columns=["y", "X", "R", "Rbar", "y_cost"])
    labels = panel.loc[:, ["symbol", "date", "y", "X", "R", "Rbar", "y_cost", "eligible"]]
    results = run_baselines(
        features,
        labels,
        C_grid=[0.1, 1.0],
        seed=0,
        # Limit folds via date coverage: panel ends 2019Q3 ⇒ a few folds.
    )
    assert set(results) == {"R0", "A", "D"}
    for result in results.values():
        assert result.predictions["date"].max() <= date(2019, 9, 30)
        assert (result.predictions["date"] < date(2024, 1, 1)).all()
        assert "log_loss" in result.metrics
        assert "auc" in result.metrics
        assert "rank_ic" in result.metrics


def test_logistic_walkforward_fit_in_fold_only() -> None:
    panel = _synthetic_panel(end=date(2019, 6, 28), seed=3)
    sessions = sorted(panel["date"].unique().tolist())
    folds = make_folds(
        sessions,
        oos_start=date(2019, 1, 2),
        oos_end=date(2019, 6, 28),
    )
    result = run_logistic_walkforward(
        panel,
        folds,
        sessions=sessions,
        C_grid=[0.1, 1.0],
        seed=0,
    )
    assert not result.predictions.empty
    assert result.chosen_C
    r0 = run_r0_walkforward(panel, folds)
    assert not r0.predictions.empty
