"""Placebo suite P1-P6 (MASTER_SPEC §13).

Jev columns are replaced; market features, folds, and pipeline stay fixed.
Empirical p = (1 + #{placebo >= observed}) / (1 + N). P5 and P6 are single runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jevlab.backtest.costs import CostModel
from jevlab.backtest.fast import run_fast
from jevlab.config.models import RiskConfig
from jevlab.cv.walkforward import Fold
from jevlab.evaluation.predictive import binary_log_loss
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import (
    JEV_LOGIT_COLUMNS,
    MODEL_B_COLUMNS,
    ColumnSetSpec,
    column_set_registry,
)
from jevlab.models.pipelines import run_column_set_logistic
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

PlaceboId = Literal["P1", "P2", "P3", "P4", "P5", "P6"]

PLACEBO_N: dict[PlaceboId, int] = {
    "P1": 1000,
    "P2": 500,
    "P3": 500,
    "P4": 500,
    "P5": 1,
    "P6": 1,
}

MIN_CIRCULAR_LAG: int = 60
CONTAMINATION_LABEL: str = "potentially_training_contaminated"
P5_RESIDUAL_GAIN_FRACTION_NOTE: str = (
    "G4 requires residual-Jev to retain >= 50% of the observed gain; this module reports data only."
)


def empirical_pvalue(observed: float, placebos: Sequence[float]) -> float:
    """One-sided empirical p = (1 + #{placebo >= observed}) / (1 + N)."""
    arr = np.asarray(placebos, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0 or not np.isfinite(observed):
        return float("nan")
    count = int(np.sum(arr >= float(observed)))
    return float((1 + count) / (1 + n))


def _require_jev_columns(frame: pd.DataFrame, columns: Sequence[str] = JEV_FEATURE_COLUMNS) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing Jev columns: {missing}")


def within_date_shuffle(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = JEV_FEATURE_COLUMNS,
    seed: int = 0,
    date_col: str = "date",
) -> pd.DataFrame:
    """P1: permute Jev vectors across ETFs within each date."""
    _require_jev_columns(frame, columns)
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for _, idx in out.groupby(date_col, sort=False).groups.items():
        ix = list(idx)
        if len(ix) < 2:
            continue
        perm = rng.permutation(len(ix))
        src = [ix[i] for i in perm]
        out.loc[ix, list(columns)] = out.loc[src, list(columns)].to_numpy()
    return out


def global_shuffle(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = JEV_FEATURE_COLUMNS,
    seed: int = 0,
) -> pd.DataFrame:
    """P2: permute Jev vectors across all rows."""
    _require_jev_columns(frame, columns)
    out = frame.copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out))
    out.loc[:, list(columns)] = out.iloc[perm][list(columns)].to_numpy()
    return out


def circular_shift_within_asset(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = JEV_FEATURE_COLUMNS,
    seed: int = 0,
    min_lag: int = MIN_CIRCULAR_LAG,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.DataFrame:
    """P3: circular-shift each asset's Jev series by a random lag >= min_lag."""
    _require_jev_columns(frame, columns)
    out = frame.copy().sort_values([symbol_col, date_col]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    for _, idx in out.groupby(symbol_col, sort=False).groups.items():
        ix = list(idx)
        n = len(ix)
        if n < 2:
            continue
        # Lag in [min_lag, n-1] when possible; otherwise any nonzero lag.
        if n > min_lag:
            lag = int(rng.integers(min_lag, n))
        else:
            lag = int(rng.integers(1, n)) if n > 1 else 0
        if lag == 0:
            continue
        block = out.loc[ix, list(columns)].to_numpy()
        out.loc[ix, list(columns)] = np.roll(block, shift=lag, axis=0)
    return out


def fit_ar1(series: np.ndarray) -> tuple[float, float]:
    """Return (phi, residual_std) for a simple AR(1) fit; phi clipped to (-0.99, 0.99)."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return 0.0, float(np.nanstd(y)) if len(y) else 1.0
    x = y[:-1]
    z = y[1:]
    var_x = float(np.var(x))
    if var_x <= 0:
        return 0.0, float(np.std(z, ddof=1)) if len(z) > 1 else 1.0
    phi = float(np.cov(x, z, ddof=1)[0, 1] / var_x)
    phi = float(np.clip(phi, -0.99, 0.99))
    resid = z - phi * x
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return phi, sigma


def ar1_matched_noise(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = JEV_FEATURE_COLUMNS,
    seed: int = 0,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.DataFrame:
    """P4: Gaussian AR(1) noise matching each column's per-asset autocorrelation."""
    _require_jev_columns(frame, columns)
    out = frame.copy().sort_values([symbol_col, date_col]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    for col in columns:
        for _, idx in out.groupby(symbol_col, sort=False).groups.items():
            ix = list(idx)
            vals = out.loc[ix, col].to_numpy(dtype=float)
            phi, sigma = fit_ar1(vals)
            mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
            noise = np.empty(len(ix), dtype=float)
            noise[0] = mean + rng.normal(0.0, sigma)
            for t in range(1, len(ix)):
                noise[t] = mean + phi * (noise[t - 1] - mean) + rng.normal(0.0, sigma)
            out.loc[ix, col] = noise
    return out


TransformFn = Callable[[pd.DataFrame, int], pd.DataFrame]


def _transform_dispatch(kind: PlaceboId) -> TransformFn:
    if kind == "P1":

        def _p1(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
            return within_date_shuffle(frame, seed=seed)

        return _p1
    if kind == "P2":

        def _p2(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
            return global_shuffle(frame, seed=seed)

        return _p2
    if kind == "P3":

        def _p3(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
            return circular_shift_within_asset(frame, seed=seed)

        return _p3
    if kind == "P4":

        def _p4(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
            return ar1_matched_noise(frame, seed=seed)

        return _p4
    raise ValueError(f"{kind} is not a transform-loop placebo")


def delta_log_loss_gain(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> float:
    """Gain = logloss(A) - logloss(B); positive => B better."""
    ya = preds_a["y"].to_numpy(dtype=float)
    yb = preds_b["y"].to_numpy(dtype=float)
    pa = preds_a["prob"].to_numpy(dtype=float)
    pb = preds_b["prob"].to_numpy(dtype=float)
    return float(binary_log_loss(ya, pa) - binary_log_loss(yb, pb))


def _spec_b() -> ColumnSetSpec:
    return column_set_registry()["B"]


def train_b_oos(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> pd.DataFrame:
    """Walk-forward OOS predictions for model B on ``frame``."""
    result = run_column_set_logistic(
        frame,
        folds,
        _spec_b(),
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    return result.predictions


def sharpe_from_equity(equity: pd.DataFrame, *, periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe from an equity curve (``equity_usd`` column)."""
    if equity is None or equity.empty or "equity_usd" not in equity.columns:
        return float("nan")
    eq = equity.sort_values("date")["equity_usd"].to_numpy(dtype=float)
    if len(eq) < 3:
        return float("nan")
    rets = np.diff(eq) / np.clip(eq[:-1], 1e-12, None)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return float("nan")
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(rets) / sd)


@dataclass
class PlaceboLoopResult:
    placebo_id: PlaceboId
    n: int
    seed: int
    observed_delta_log_loss_gain: float
    placebo_delta_log_loss_gains: list[float]
    pvalue_log_loss: float
    observed_delta_sharpe: float | None = None
    placebo_delta_sharpes: list[float] = field(default_factory=list)
    pvalue_sharpe: float | None = None
    contamination_label: str = CONTAMINATION_LABEL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_placebo_loop(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    kind: PlaceboId,
    preds_a: pd.DataFrame,
    sessions: list[date],
    C_grid: list[float],
    n: int | None = None,
    seed: int = 42,
    embargo_sessions: int = 5,
    bars: pd.DataFrame | None = None,
    cost: CostModel | None = None,
    risk_config: RiskConfig | None = None,
    top_k: int = 3,
    phase: int = 0,
    compute_sharpe: bool = False,
) -> PlaceboLoopResult:
    """Run P1-P4 placebo loop; optional ``run_fast`` Sharpe deltas when bars provided."""
    if kind not in {"P1", "P2", "P3", "P4"}:
        raise ValueError(f"run_placebo_loop supports P1-P4 only; got {kind}")
    n_draws = int(PLACEBO_N[kind] if n is None else n)
    if n_draws < 1:
        raise ValueError("n must be >= 1")
    transform = _transform_dispatch(kind)
    preds_b_obs = train_b_oos(
        frame,
        folds,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    observed_ll = delta_log_loss_gain(preds_a, preds_b_obs)
    observed_sh: float | None = None
    if compute_sharpe and bars is not None and cost is not None and risk_config is not None:
        res_a = run_fast(
            bars,
            preds_a,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=phase,
            apply_risk=False,
        )
        res_b = run_fast(
            bars,
            preds_b_obs,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=phase,
            apply_risk=False,
        )
        observed_sh = sharpe_from_equity(res_b.equity_curve) - sharpe_from_equity(
            res_a.equity_curve
        )

    gains: list[float] = []
    sharpes: list[float] = []
    for i in range(n_draws):
        draw_seed = int(seed + 17 * i + hash(kind) % 10_000)
        placebo_frame = transform(frame, draw_seed)
        preds_b = train_b_oos(
            placebo_frame,
            folds,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        gains.append(delta_log_loss_gain(preds_a, preds_b))
        if (
            compute_sharpe
            and bars is not None
            and cost is not None
            and risk_config is not None
            and observed_sh is not None
        ):
            res_pb = run_fast(
                bars,
                preds_b,
                cost=cost,
                risk_config=risk_config,
                top_k=top_k,
                phase=phase,
                apply_risk=False,
            )
            res_a = run_fast(
                bars,
                preds_a,
                cost=cost,
                risk_config=risk_config,
                top_k=top_k,
                phase=phase,
                apply_risk=False,
            )
            sharpes.append(
                sharpe_from_equity(res_pb.equity_curve) - sharpe_from_equity(res_a.equity_curve)
            )
        if (i + 1) % 50 == 0 or i == 0:
            log.info("placebo.progress", kind=kind, draw=i + 1, n=n_draws)

    p_ll = empirical_pvalue(observed_ll, gains)
    p_sh = empirical_pvalue(observed_sh, sharpes) if sharpes and observed_sh is not None else None
    return PlaceboLoopResult(
        placebo_id=kind,
        n=n_draws,
        seed=seed,
        observed_delta_log_loss_gain=observed_ll,
        placebo_delta_log_loss_gains=gains,
        pvalue_log_loss=p_ll,
        observed_delta_sharpe=observed_sh,
        placebo_delta_sharpes=sharpes,
        pvalue_sharpe=p_sh,
    )


def _lgbm_regressor(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "reg",
                LGBMRegressor(
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    deterministic=True,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def project_jev_from_deterministic(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    seed: int = 42,
    jev_columns: Sequence[str] = JEV_FEATURE_COLUMNS,
    det_columns: Sequence[str] = MODEL_A_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """In-fold LightGBM: MODEL_A_COLUMNS -> each Jev feature.

    Returns (predicted_frame, residual_frame) with Jev columns replaced.
    """
    _require_jev_columns(frame, jev_columns)
    pred = frame.copy()
    resid = frame.copy()
    for col in jev_columns:
        pred[col] = np.nan
        resid[col] = np.nan
    for fold in folds:
        tr = frame.loc[frame["date"].isin(fold.train_dates)]
        te = frame.loc[frame["date"].isin(fold.test_dates)]
        if tr.empty or te.empty:
            continue
        X_tr = tr.loc[:, list(det_columns)].to_numpy(dtype=float)
        X_te = te.loc[:, list(det_columns)].to_numpy(dtype=float)
        for j, col in enumerate(jev_columns):
            y_tr = tr[col].to_numpy(dtype=float)
            mask = np.isfinite(y_tr)
            if int(mask.sum()) < 10:
                continue
            pipe = _lgbm_regressor(seed + j)
            pipe.fit(X_tr[mask], y_tr[mask])
            yhat_te = pipe.predict(X_te)
            yhat_tr = pipe.predict(X_tr)
            pred.loc[te.index, col] = yhat_te
            resid.loc[te.index, col] = te[col].to_numpy(dtype=float) - yhat_te
            # Also fill train with in-sample predictions for completeness (unused in OOS).
            pred.loc[tr.index, col] = yhat_tr
            resid.loc[tr.index, col] = tr[col].to_numpy(dtype=float) - yhat_tr
    return pred, resid


@dataclass
class P5Result:
    observed_gain: float
    predicted_only_gain: float
    residual_only_gain: float
    residual_fraction_of_gain: float
    contamination_label: str = CONTAMINATION_LABEL
    note: str = P5_RESIDUAL_GAIN_FRACTION_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_p5_projection(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    preds_a: pd.DataFrame,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> P5Result:
    """P5 single run: B with predicted-Jev only and residual-Jev only."""
    preds_b = train_b_oos(
        frame,
        folds,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    observed = delta_log_loss_gain(preds_a, preds_b)
    pred_frame, resid_frame = project_jev_from_deterministic(frame, folds, seed=seed)
    preds_pred = train_b_oos(
        pred_frame,
        folds,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    preds_resid = train_b_oos(
        resid_frame,
        folds,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    g_pred = delta_log_loss_gain(preds_a, preds_pred)
    g_resid = delta_log_loss_gain(preds_a, preds_resid)
    frac = float(g_resid / observed) if observed != 0 and np.isfinite(observed) else float("nan")
    return P5Result(
        observed_gain=observed,
        predicted_only_gain=g_pred,
        residual_only_gain=g_resid,
        residual_fraction_of_gain=frac,
    )


def write_placebo_results(payload: Mapping[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "placebo_results.json"
    clean = json.loads(json.dumps(payload, default=str))
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    digest = sha256_of(canonical_json(clean))
    (out_dir / "placebo_results.digest").write_text(digest + "\n", encoding="utf-8")
    return path


__all__ = [
    "JEV_LOGIT_COLUMNS",
    "MIN_CIRCULAR_LAG",
    "MODEL_B_COLUMNS",
    "PLACEBO_N",
    "P5Result",
    "PlaceboLoopResult",
    "ar1_matched_noise",
    "circular_shift_within_asset",
    "delta_log_loss_gain",
    "empirical_pvalue",
    "fit_ar1",
    "global_shuffle",
    "project_jev_from_deterministic",
    "run_p5_projection",
    "run_placebo_loop",
    "train_b_oos",
    "within_date_shuffle",
    "write_placebo_results",
]
