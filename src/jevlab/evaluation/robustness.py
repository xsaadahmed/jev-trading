"""Robustness suite (MASTER_SPEC §13) — data only, no narrative conclusions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jevlab.backtest.costs import SCENARIO_BPS, CostModel
from jevlab.backtest.fast import run_fast
from jevlab.config.models import RiskConfig
from jevlab.cv.walkforward import Fold
from jevlab.evaluation.placebo import (
    CONTAMINATION_LABEL,
    delta_log_loss_gain,
    sharpe_from_equity,
    train_b_oos,
)
from jevlab.evaluation.predictive import binary_log_loss
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import JEV_LOGIT_COLUMNS, ColumnSetSpec, column_set_registry
from jevlab.models.pipelines import run_column_set_logistic
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0)
K_GRID: tuple[int, ...] = (2, 3, 4)
PHASES: tuple[int, ...] = (0, 1, 2)
REGIME_KEYS: tuple[str, ...] = (
    "bull_spy_above_sma200",
    "bear_spy_below_sma200",
    "vix_tercile_low",
    "vix_tercile_mid",
    "vix_tercile_high",
    "year_2020",
    "year_2022",
)


@dataclass
class RobustnessBundle:
    """Container for all robustness data tables (no conclusions)."""

    meta: dict[str, Any] = field(default_factory=dict)
    feature_ablation: list[dict[str, Any]] = field(default_factory=list)
    jev_question_ablation: list[dict[str, Any]] = field(default_factory=list)
    regimes: list[dict[str, Any]] = field(default_factory=list)
    leave_one_etf_out: list[dict[str, Any]] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    k_sweep: list[dict[str, Any]] = field(default_factory=list)
    regularization: list[dict[str, Any]] = field(default_factory=list)
    cost_sweep: list[dict[str, Any]] = field(default_factory=list)
    delayed_execution: list[dict[str, Any]] = field(default_factory=list)
    choice_order: list[dict[str, Any]] = field(default_factory=list)
    paraphrase: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["meta"] = {
            **d.get("meta", {}),
            "contamination_label": CONTAMINATION_LABEL,
            "narrative_conclusions": None,
        }
        return d


def _logloss(preds: pd.DataFrame) -> float:
    return binary_log_loss(preds["y"].to_numpy(dtype=float), preds["prob"].to_numpy(dtype=float))


def _spec_b_with_columns(columns: Sequence[str], logit_columns: Sequence[str]) -> ColumnSetSpec:
    return ColumnSetSpec(
        model_id="B",
        columns=tuple(columns),
        learner="logistic",
        purpose="confirmatory",
        n_columns=len(columns),
        logit_columns=tuple(logit_columns),
        notes="robustness ablation / grid variant",
    )


def train_logistic_oos(
    frame: pd.DataFrame,
    folds: list[Fold],
    columns: Sequence[str],
    *,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
    logit_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    logit = (
        tuple(c for c in logit_columns if c in columns)
        if logit_columns is not None
        else tuple(c for c in JEV_LOGIT_COLUMNS if c in columns)
    )
    spec = _spec_b_with_columns(columns, logit)
    return run_column_set_logistic(
        frame,
        folds,
        spec,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    ).predictions


def feature_ablation(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    preds_a: pd.DataFrame,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> list[dict[str, Any]]:
    """Drop each deterministic MODEL_A feature one at a time from B."""
    base_cols = list(MODEL_A_COLUMNS) + list(JEV_FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for drop in MODEL_A_COLUMNS:
        cols = [c for c in base_cols if c != drop]
        preds_b = train_logistic_oos(
            frame,
            folds,
            cols,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        rows.append(
            {
                "dropped_feature": drop,
                "n_columns": len(cols),
                "delta_log_loss_gain": delta_log_loss_gain(preds_a, preds_b),
                "log_loss_b": _logloss(preds_b),
            }
        )
    return rows


def jev_question_ablation(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    preds_a: pd.DataFrame,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> list[dict[str, Any]]:
    """Drop each Jev feature (question-derived column) one at a time from B."""
    rows: list[dict[str, Any]] = []
    for drop in JEV_FEATURE_COLUMNS:
        cols = list(MODEL_A_COLUMNS) + [c for c in JEV_FEATURE_COLUMNS if c != drop]
        preds_b = train_logistic_oos(
            frame,
            folds,
            cols,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        rows.append(
            {
                "dropped_jev_feature": drop,
                "n_columns": len(cols),
                "delta_log_loss_gain": delta_log_loss_gain(preds_a, preds_b),
                "log_loss_b": _logloss(preds_b),
            }
        )
    return rows


def _date_mask(dates: pd.Series, start: date, end: date) -> pd.Series:
    d = pd.to_datetime(dates).dt.date
    return (d >= start) & (d <= end)


def regime_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for regime splits (lagged VIX terciles; SPY vs SMA200)."""
    dates = pd.to_datetime(frame["date"]).dt.date
    masks: dict[str, pd.Series] = {
        "year_2020": _date_mask(frame["date"], date(2020, 1, 1), date(2020, 12, 31)),
        "year_2022": _date_mask(frame["date"], date(2022, 1, 1), date(2022, 12, 31)),
    }
    if "spy_px_vs_sma200" in frame.columns:
        spy = frame["spy_px_vs_sma200"].astype(float)
        masks["bull_spy_above_sma200"] = spy > 0
        masks["bear_spy_below_sma200"] = spy <= 0
    if "vix_lag1_pct_252" in frame.columns:
        v = frame["vix_lag1_pct_252"].astype(float)
        # Terciles on unique dates to avoid cross-sectional overweight.
        by_date = frame.groupby(dates, sort=True)["vix_lag1_pct_252"].first()
        qs = by_date.quantile([1 / 3, 2 / 3])
        lo, hi = float(qs.iloc[0]), float(qs.iloc[1])
        masks["vix_tercile_low"] = v <= lo
        masks["vix_tercile_mid"] = (v > lo) & (v <= hi)
        masks["vix_tercile_high"] = v > hi
    return masks


def regime_splits(
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Evaluate observed A/B OOS metrics inside each regime (no retrain)."""
    rows: list[dict[str, Any]] = []
    # Align predictions to frame keys
    key = ["symbol", "date"]
    fa = preds_a.merge(
        frame[key + [c for c in ("spy_px_vs_sma200", "vix_lag1_pct_252") if c in frame.columns]],
        on=key,
        how="left",
    )
    fb = preds_b.merge(
        frame[key + [c for c in ("spy_px_vs_sma200", "vix_lag1_pct_252") if c in frame.columns]],
        on=key,
        how="left",
    )
    masks = regime_masks(fa)
    for name, mask in masks.items():
        if mask.sum() < 20:
            rows.append({"regime": name, "n_rows": int(mask.sum()), "status": "insufficient"})
            continue
        sub_a = fa.loc[mask]
        sub_b = fb.loc[mask]
        # Re-merge B on same index length via keys
        sub_b = fb.merge(sub_a[key], on=key, how="inner")
        sub_a = fa.merge(sub_b[key], on=key, how="inner")
        rows.append(
            {
                "regime": name,
                "n_rows": len(sub_a),
                "delta_log_loss_gain": delta_log_loss_gain(sub_a, sub_b),
                "log_loss_a": _logloss(sub_a),
                "log_loss_b": _logloss(sub_b),
            }
        )
    return rows


def leave_one_etf_out(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    preds_a: pd.DataFrame,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> list[dict[str, Any]]:
    """Retrain A/B with each ETF removed from the panel."""
    symbols = sorted(frame["symbol"].astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []
    a_cols = list(column_set_registry()["A"].columns)
    for sym in symbols:
        sub = frame.loc[frame["symbol"].astype(str) != sym].copy()
        if sub.empty:
            continue
        preds_a_loo = train_logistic_oos(
            sub,
            folds,
            a_cols,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
            logit_columns=(),
        )
        preds_b_loo = train_b_oos(
            sub,
            folds,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        rows.append(
            {
                "left_out_symbol": sym,
                "n_rows": len(sub),
                "delta_log_loss_gain": delta_log_loss_gain(preds_a_loo, preds_b_loo),
            }
        )
    # Mark best contributor as the leave-out that hurts gain the most vs full-panel baseline.
    if rows:
        full_gain = delta_log_loss_gain(
            preds_a,
            train_b_oos(
                frame,
                folds,
                sessions=sessions,
                C_grid=C_grid,
                seed=seed,
                embargo_sessions=embargo_sessions,
            ),
        )
        for r in rows:
            r["gain_drop_vs_full"] = float(full_gain - float(r["delta_log_loss_gain"]))
        best = max(rows, key=lambda r: float(r["gain_drop_vs_full"]))
        for r in rows:
            r["is_best_contributor_left_out"] = r["left_out_symbol"] == best["left_out_symbol"]
    return rows


def phase_sweep(
    bars: pd.DataFrame,
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    *,
    cost: CostModel,
    risk_config: RiskConfig,
    top_k: int = 3,
    phases: Sequence[int] = PHASES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in phases:
        ra = run_fast(
            bars,
            preds_a,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=int(phase),
            apply_risk=False,
        )
        rb = run_fast(
            bars,
            preds_b,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=int(phase),
            apply_risk=False,
        )
        rows.append(
            {
                "phase": int(phase),
                "sharpe_a": sharpe_from_equity(ra.equity_curve),
                "sharpe_b": sharpe_from_equity(rb.equity_curve),
                "delta_sharpe": sharpe_from_equity(rb.equity_curve)
                - sharpe_from_equity(ra.equity_curve),
                "n_trades_a": len(ra.trades) if ra.trades is not None else 0,
                "n_trades_b": len(rb.trades) if rb.trades is not None else 0,
            }
        )
    if rows:
        rows.append(
            {
                "phase": "mean",
                "delta_sharpe": float(np.nanmean([r["delta_sharpe"] for r in rows])),
                "sharpe_a": float(np.nanmean([r["sharpe_a"] for r in rows])),
                "sharpe_b": float(np.nanmean([r["sharpe_b"] for r in rows])),
            }
        )
    return rows


def k_sweep(
    bars: pd.DataFrame,
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    *,
    cost: CostModel,
    risk_config: RiskConfig,
    ks: Sequence[int] = K_GRID,
    phase: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in ks:
        ra = run_fast(
            bars,
            preds_a,
            cost=cost,
            risk_config=risk_config,
            top_k=int(k),
            phase=phase,
            apply_risk=False,
        )
        rb = run_fast(
            bars,
            preds_b,
            cost=cost,
            risk_config=risk_config,
            top_k=int(k),
            phase=phase,
            apply_risk=False,
        )
        rows.append(
            {
                "top_k": int(k),
                "delta_sharpe": sharpe_from_equity(rb.equity_curve)
                - sharpe_from_equity(ra.equity_curve),
                "sharpe_a": sharpe_from_equity(ra.equity_curve),
                "sharpe_b": sharpe_from_equity(rb.equity_curve),
            }
        )
    return rows


def regularization_grid_step(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    preds_a: pd.DataFrame,
    sessions: list[date],
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    seed: int = 42,
    embargo_sessions: int = 5,
) -> list[dict[str, Any]]:
    """Threshold-free: fix C one step below / at / above the mid grid point for B."""
    grid = sorted(float(c) for c in C_grid)
    if len(grid) < 2:
        grid = list(DEFAULT_C_GRID)
    mid = len(grid) // 2
    variants = {
        "C_minus_1": grid[max(0, mid - 1)],
        "C_mid": grid[mid],
        "C_plus_1": grid[min(len(grid) - 1, mid + 1)],
    }
    rows: list[dict[str, Any]] = []
    for name, c_val in variants.items():
        preds_b = train_b_oos(
            frame,
            folds,
            sessions=sessions,
            C_grid=[c_val],
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        rows.append(
            {
                "variant": name,
                "C": c_val,
                "delta_log_loss_gain": delta_log_loss_gain(preds_a, preds_b),
            }
        )
    return rows


def cost_sweep(
    bars: pd.DataFrame,
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    *,
    risk_config: RiskConfig,
    sell_fee_bps: float = 0.3,
    per_symbol_cost_multiplier: dict[str, float] | None = None,
    top_k: int = 3,
    phase: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mult = per_symbol_cost_multiplier or {}
    for name, bps in SCENARIO_BPS.items():
        cost = CostModel(
            cost_bps_per_side=float(bps),
            sell_fee_bps=sell_fee_bps,
            per_symbol_cost_multiplier=mult,
            scenario_name=name,
        )
        ra = run_fast(
            bars,
            preds_a,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=phase,
            apply_risk=False,
        )
        rb = run_fast(
            bars,
            preds_b,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=phase,
            apply_risk=False,
        )
        rows.append(
            {
                "scenario": name,
                "cost_bps_per_side": float(bps),
                "delta_sharpe": sharpe_from_equity(rb.equity_curve)
                - sharpe_from_equity(ra.equity_curve),
                "sharpe_a": sharpe_from_equity(ra.equity_curve),
                "sharpe_b": sharpe_from_equity(rb.equity_curve),
            }
        )
    return rows


def delayed_execution_sweep(
    bars: pd.DataFrame,
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    *,
    cost: CostModel,
    risk_config: RiskConfig,
    top_k: int = 3,
    phase: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("open_t1", "delayed"):
        ra = run_fast(
            bars,
            preds_a,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=phase,
            fill_mode=mode,  # type: ignore[arg-type]
            apply_risk=False,
        )
        rb = run_fast(
            bars,
            preds_b,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=phase,
            fill_mode=mode,  # type: ignore[arg-type]
            apply_risk=False,
        )
        rows.append(
            {
                "fill_mode": mode,
                "delta_sharpe": sharpe_from_equity(rb.equity_curve)
                - sharpe_from_equity(ra.equity_curve),
                "sharpe_a": sharpe_from_equity(ra.equity_curve),
                "sharpe_b": sharpe_from_equity(rb.equity_curve),
            }
        )
    return rows


def choice_order_sensitivity(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    preds_a: pd.DataFrame,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
) -> list[dict[str, Any]]:
    """Permute Jev column *order* (same values) to probe Choice-order fragility.

    Values are unchanged; only column permutation of the 8 Jev slots — a cheap
    proxy for choice-order effects when re-calling Jev is unavailable.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(3):
        order = list(JEV_FEATURE_COLUMNS)
        rng.shuffle(order)
        renamed = frame.copy()
        # Rotate values among Jev columns according to a permutation.
        perm = rng.permutation(len(JEV_FEATURE_COLUMNS))
        src = [JEV_FEATURE_COLUMNS[j] for j in perm]
        renamed.loc[:, list(JEV_FEATURE_COLUMNS)] = frame.loc[:, src].to_numpy()
        preds_b = train_b_oos(
            renamed,
            folds,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        rows.append(
            {
                "draw": i,
                "source_order": src,
                "delta_log_loss_gain": delta_log_loss_gain(preds_a, preds_b),
            }
        )
    return rows


def paraphrase_sensitivity_plan(
    *,
    n_states: int = 3000,
    n_paraphrases: int = 2,
    n_questions: int = 5,
) -> dict[str, Any]:
    """Plan for 2 paraphrases x n_states (logged as trials when executed live)."""
    return {
        "n_states": n_states,
        "n_paraphrases_per_question": n_paraphrases,
        "n_questions": n_questions,
        "planned_calls": n_states * n_paraphrases * n_questions,
        "status": "planned",
        "requires_live": True,
        "contamination_label": CONTAMINATION_LABEL,
    }


def run_robustness_suite(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    sessions: list[date],
    C_grid: list[float],
    seed: int = 42,
    embargo_sessions: int = 5,
    bars: pd.DataFrame | None = None,
    cost: CostModel | None = None,
    risk_config: RiskConfig | None = None,
    top_k: int = 3,
    include_ablations: bool = True,
    include_loo: bool = True,
) -> RobustnessBundle:
    """Assemble robustness data tables. Omits narrative conclusions."""
    bundle = RobustnessBundle(
        meta={
            "seed": seed,
            "C_grid": list(C_grid),
            "n_folds": len(folds),
            "n_rows": len(frame),
            "contamination_label": CONTAMINATION_LABEL,
        }
    )
    if include_ablations:
        log.info("robustness.feature_ablation")
        bundle.feature_ablation = feature_ablation(
            frame,
            folds,
            preds_a=preds_a,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
        log.info("robustness.jev_ablation")
        bundle.jev_question_ablation = jev_question_ablation(
            frame,
            folds,
            preds_a=preds_a,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
    bundle.regimes = regime_splits(preds_a, preds_b, frame)
    bundle.regularization = regularization_grid_step(
        frame,
        folds,
        preds_a=preds_a,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    bundle.choice_order = choice_order_sensitivity(
        frame,
        folds,
        preds_a=preds_a,
        sessions=sessions,
        C_grid=C_grid,
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    bundle.paraphrase = [paraphrase_sensitivity_plan()]
    if include_loo:
        log.info("robustness.loo")
        bundle.leave_one_etf_out = leave_one_etf_out(
            frame,
            folds,
            preds_a=preds_a,
            sessions=sessions,
            C_grid=C_grid,
            seed=seed,
            embargo_sessions=embargo_sessions,
        )
    if bars is not None and cost is not None and risk_config is not None:
        bundle.phases = phase_sweep(
            bars, preds_a, preds_b, cost=cost, risk_config=risk_config, top_k=top_k
        )
        bundle.k_sweep = k_sweep(bars, preds_a, preds_b, cost=cost, risk_config=risk_config)
        bundle.cost_sweep = cost_sweep(bars, preds_a, preds_b, risk_config=risk_config, top_k=top_k)
        bundle.delayed_execution = delayed_execution_sweep(
            bars, preds_a, preds_b, cost=cost, risk_config=risk_config, top_k=top_k
        )
    return bundle


def write_robustness_report(bundle: RobustnessBundle, out_dir: Path) -> Path:
    """Write robustness JSON data only (no narrative)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = bundle.to_dict()
    path = out_dir / "robustness_data.json"
    clean = json.loads(json.dumps(payload, default=str))
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    digest = sha256_of(canonical_json(clean))
    (out_dir / "robustness_data.digest").write_text(digest + "\n", encoding="utf-8")
    # Per-section files for Phase 11 consumption
    for key in (
        "feature_ablation",
        "jev_question_ablation",
        "regimes",
        "leave_one_etf_out",
        "phases",
        "k_sweep",
        "regularization",
        "cost_sweep",
        "delayed_execution",
        "choice_order",
        "paraphrase",
    ):
        section_path = out_dir / f"{key}.json"
        section_path.write_text(
            json.dumps(clean.get(key, []), indent=2, default=str), encoding="utf-8"
        )
    readme = out_dir / "README.md"
    readme.write_text(
        "# Robustness report (data only)\n\n"
        "This directory contains numerical robustness / placebo outputs.\n"
        "It intentionally contains **no narrative conclusions**.\n"
        f"contamination_label: `{CONTAMINATION_LABEL}`\n",
        encoding="utf-8",
    )
    return path
