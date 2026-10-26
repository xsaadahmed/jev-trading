"""V2 candidate adapter: exact V1 Model A - no silent copy of semantics.

All ranking/trading rules are delegated to committed V1 modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from jevlab.backtest.costs import SCENARIO_BPS, CostModel
from jevlab.backtest.fast import equity_curves_equal, run_fast
from jevlab.backtest.ledger import EQUITY_TOL_REL, FillMode, LedgerResult, run_ledger
from jevlab.config.loader import load_all_configs
from jevlab.config.models import RiskConfig
from jevlab.cv.walkforward import OOS_END, OOS_START, Fold, make_folds
from jevlab.evaluation.metrics import compute_economic_metrics
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.models.column_sets import column_set_registry
from jevlab.models.pipelines import (
    WalkForwardResult,
    prepare_model_frame,
    run_column_set_logistic,
)
from jevlab.portfolio.sizing import equal_weight_notionals
from jevlab.signals.selection import rebalance_dates, select_top_k
from jevlab.v2.dev_io import assert_frame_pre_sealed
from jevlab.v2.hashes import assert_frozen_research_hashes

V2_UNIVERSE: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
)
V2_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0)
V2_TOP_K: int = 3
V2_PHASES: tuple[int, ...] = (0, 1, 2)
V2_EMBARGO: int = 5
V2_SEED: int = 42
V2_ELIGIBILITY_RULE: str = "adj_close > sma200"
# MASTER_SPEC §20 same-environment logistic probability tolerance.
PRED_ATOL: float = 1e-9
PRED_RTOL: float = 1e-9
METRIC_RTOL: float = 1e-8


def assert_v2_candidate_semantics(*, root: Any | None = None) -> None:
    """Hash + structural assertions that V2 still is exact V1 Model A."""
    assert_frozen_research_hashes(root=root)
    configs = load_all_configs()
    assert tuple(configs.universe.symbols) == V2_UNIVERSE
    assert list(configs.features.model_a_columns) == list(MODEL_A_COLUMNS)
    assert len(MODEL_A_COLUMNS) == 17
    assert configs.strategy.eligibility_rule == V2_ELIGIBILITY_RULE
    assert list(configs.strategy.logistic_C_grid) == list(V2_C_GRID)
    assert configs.strategy.top_k == V2_TOP_K
    assert list(configs.strategy.phases) == list(V2_PHASES)
    assert configs.strategy.seed == V2_SEED
    assert configs.strategy.entry_lag_sessions == 1
    assert configs.strategy.horizon_sessions == 3
    spec = column_set_registry()["A"]
    assert spec.columns == MODEL_A_COLUMNS
    assert spec.n_columns == 17
    assert spec.logit_columns == ()
    assert spec.learner == "logistic"


def research_cost_model(scenario: str, configs: Any | None = None) -> CostModel:
    configs = configs or load_all_configs()
    bps = SCENARIO_BPS[scenario]
    match = next(s for s in configs.backtest.cost_scenarios if s.name == scenario)
    if match.cost_bps_per_side != bps:
        raise AssertionError(f"cost scenario {scenario} drifted from SCENARIO_BPS")
    return CostModel(
        cost_bps_per_side=float(match.cost_bps_per_side),
        sell_fee_bps=float(configs.backtest.sell_fee_bps),
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
        scenario_name=scenario,
    )


def research_risk_config(configs: Any | None = None) -> RiskConfig:
    configs = configs or load_all_configs()
    return configs.risk


@dataclass
class ModelAWalkForward:
    """Walk-forward Model A result using V1 lineup logistic path."""

    result: WalkForwardResult
    folds: list[Fold]
    sessions: list[date]
    frame: pd.DataFrame


def run_v2_model_a_walkforward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    embargo_sessions: int = V2_EMBARGO,
    seed: int = V2_SEED,
    C_grid: list[float] | None = None,
    oos_start: date = OOS_START,
    oos_end: date = OOS_END,
) -> ModelAWalkForward:
    """Exact V1 confirmatory Model A walk-forward (Phase-8 column-set path)."""
    assert_v2_candidate_semantics()
    assert_frame_pre_sealed(features)
    assert_frame_pre_sealed(labels)
    frame = prepare_model_frame(features, labels)
    assert_frame_pre_sealed(frame)
    sessions = sorted(frame["date"].unique().tolist())
    folds = make_folds(
        sessions,
        embargo_sessions=embargo_sessions,
        oos_start=oos_start,
        oos_end=oos_end,
    )
    spec = column_set_registry()["A"]
    result = run_column_set_logistic(
        frame,
        folds,
        spec,
        sessions=sessions,
        C_grid=list(C_grid) if C_grid is not None else list(V2_C_GRID),
        seed=seed,
        embargo_sessions=embargo_sessions,
    )
    return ModelAWalkForward(result=result, folds=folds, sessions=sessions, frame=frame)


def select_model_a_top_k(
    scores: pd.DataFrame,
    *,
    sessions: list[date],
    phase: int,
    k: int = V2_TOP_K,
    every: int = 3,
) -> pd.DataFrame:
    """V1 ``select_top_k`` on phase rebalance dates: min(K, n_eligible)."""
    reb = rebalance_dates(sessions, every=every, phase=phase)
    return select_top_k(scores, k=k, rebalance_dates=reb)


def shortage_cash_weights(n_selected: int, *, k: int = V2_TOP_K) -> tuple[float, float]:
    """Return (weight_per_selected, cash_weight) matching ``equal_weight_notionals``."""
    dummy = equal_weight_notionals(["X"] * n_selected, equity_usd=1.0, k=k)
    invested = sum(t.notional_usd for t in dummy)
    return (1.0 / k if n_selected else 0.0), (1.0 - invested)


def run_model_a_backtest(
    bars: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    phase: int,
    cost: CostModel,
    risk_config: RiskConfig | None = None,
    initial_capital_usd: float = 100_000.0,
    top_k: int = V2_TOP_K,
    apply_risk: bool = True,
    enable_drawdown_kill: bool = False,
    fill_mode: FillMode = "open_t1",
    metric_start: date | None = OOS_START,
    metric_end: date | None = OOS_END,
) -> tuple[LedgerResult, LedgerResult, dict[str, float]]:
    """Ledger + fast Model A backtest; research kill-switch off."""
    assert_frame_pre_sealed(bars)
    assert_frame_pre_sealed(scores)
    risk = risk_config or research_risk_config()
    kwargs: dict[str, Any] = dict(
        cost=cost,
        risk_config=risk,
        initial_capital_usd=initial_capital_usd,
        top_k=top_k,
        phase=phase,
        fill_mode=fill_mode,
        apply_risk=apply_risk,
        enable_drawdown_kill=enable_drawdown_kill,
        score_col="score",
    )
    ledger = run_ledger(bars, scores, **kwargs)
    fast = run_fast(bars, scores, **kwargs)
    if not equity_curves_equal(ledger.equity_curve, fast.equity_curve, tol_rel=EQUITY_TOL_REL):
        raise AssertionError("V2 Model A ledger vs fast equity mismatch")
    metrics = compute_economic_metrics(
        ledger.equity_curve,
        ledger.trades,
        initial_capital_usd=initial_capital_usd,
        metric_start=metric_start,
        metric_end=metric_end,
    ).to_dict()
    return ledger, fast, metrics


def mean_phase_stat(
    phase_metrics: dict[int, dict[str, float]],
    key: str,
) -> float:
    vals = [float(phase_metrics[p][key]) for p in V2_PHASES if p in phase_metrics]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))
