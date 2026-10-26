"""Mechanics-matched random ranking null (frozen V2 seed policy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from jevlab.backtest.costs import CostModel
from jevlab.backtest.fast import equity_curves_equal, run_fast
from jevlab.backtest.ledger import EQUITY_TOL_REL, LedgerResult, run_ledger
from jevlab.config.models import RiskConfig
from jevlab.config.v2_research import EXPECTED_BASE_SEED, SEED_CANONICAL_STRING
from jevlab.evaluation.metrics import compute_economic_metrics
from jevlab.signals.selection import rebalance_dates, select_top_k
from jevlab.v2.candidate import V2_PHASES, V2_TOP_K, research_risk_config
from jevlab.v2.dev_io import assert_frame_pre_sealed
from jevlab.v2.statistics import FROZEN_N, NullTestResult, evaluate_randomization

MOD_64: int = 2**64
BASE_SEED: int = EXPECTED_BASE_SEED
assert BASE_SEED == 15224036858585574915


def child_seed(draw_id: int, phase: int, *, base_seed: int = BASE_SEED) -> int:
    """``(base_seed + draw_id + phase) mod 2**64``; draw_id is 1-based."""
    if draw_id < 1:
        raise ValueError("draw_id is 1-based")
    if phase not in V2_PHASES:
        raise ValueError(f"phase must be in {V2_PHASES}")
    return int((base_seed + int(draw_id) + int(phase)) % MOD_64)


def rng_for(draw_id: int, phase: int, *, base_seed: int = BASE_SEED) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(child_seed(draw_id, phase, base_seed=base_seed)))


def randomize_ranking(
    scores: pd.DataFrame,
    *,
    draw_id: int,
    phase: int,
    sessions: list[date],
    k: int = V2_TOP_K,
    every: int = 3,
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Replace scores among eligible names only; keep dates/eligibility/K."""
    assert_frame_pre_sealed(scores)
    frame = scores.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    reb = set(rebalance_dates(sessions, every=every, phase=phase))
    rng = rng_for(draw_id, phase, base_seed=base_seed)
    parts: list[pd.DataFrame] = []
    for dt, group in frame.groupby("date", sort=True):
        g = group.copy()
        if dt not in reb:
            g["score"] = np.nan
            parts.append(g)
            continue
        elig_mask = g["eligible"].fillna(0).astype(int) == 1
        elig_syms = sorted(g.loc[elig_mask, "symbol"].astype(str).unique().tolist())
        if elig_syms:
            perm = rng.permutation(elig_syms)
            rank_score = {sym: float(len(perm) - i) for i, sym in enumerate(perm)}
            g.loc[elig_mask, "score"] = g.loc[elig_mask, "symbol"].map(rank_score)
        g.loc[~elig_mask, "score"] = -1e18
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return select_top_k(out, k=k, rebalance_dates=sorted(reb))


@dataclass
class NullDrawResult:
    draw_id: int
    phase_sharpes: dict[int, float] = field(default_factory=dict)
    mean_sharpe: float = float("nan")
    selections: dict[int, pd.DataFrame] = field(default_factory=dict)


def _backtest_sharpe(
    bars: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    phase: int,
    cost: CostModel,
    risk_config: RiskConfig,
    initial_capital_usd: float,
    top_k: int,
    metric_start: date | None,
    metric_end: date | None,
    check_fast: bool,
) -> tuple[float, LedgerResult]:
    kwargs: dict[str, Any] = dict(
        cost=cost,
        risk_config=risk_config,
        initial_capital_usd=initial_capital_usd,
        top_k=top_k,
        phase=phase,
        apply_risk=True,
        enable_drawdown_kill=False,
        score_col="score",
    )
    ledger = run_ledger(bars, selected, **kwargs)
    if check_fast:
        fast = run_fast(bars, selected, **kwargs)
        if not equity_curves_equal(ledger.equity_curve, fast.equity_curve, tol_rel=EQUITY_TOL_REL):
            raise AssertionError("random-null ledger vs fast mismatch")
    metrics = compute_economic_metrics(
        ledger.equity_curve,
        ledger.trades,
        initial_capital_usd=initial_capital_usd,
        metric_start=metric_start,
        metric_end=metric_end,
    ).to_dict()
    return float(metrics["sharpe"]), ledger


def run_null_draw(
    bars: pd.DataFrame,
    model_a_scores: pd.DataFrame,
    *,
    draw_id: int,
    sessions: list[date],
    cost: CostModel,
    risk_config: RiskConfig | None = None,
    initial_capital_usd: float = 100_000.0,
    top_k: int = V2_TOP_K,
    phases: tuple[int, ...] = V2_PHASES,
    metric_start: date | None,
    metric_end: date | None,
    check_fast: bool = False,
    base_seed: int = BASE_SEED,
) -> NullDrawResult:
    """One randomization draw: ranking only; same mechanics as Model A."""
    assert_frame_pre_sealed(bars)
    assert_frame_pre_sealed(model_a_scores)
    risk = risk_config or research_risk_config()
    out = NullDrawResult(draw_id=draw_id)
    sharpes: list[float] = []
    for phase in phases:
        selected = randomize_ranking(
            model_a_scores,
            draw_id=draw_id,
            phase=phase,
            sessions=sessions,
            k=top_k,
            base_seed=base_seed,
        )
        out.selections[phase] = selected
        sh, _ledger = _backtest_sharpe(
            bars,
            selected,
            phase=phase,
            cost=cost,
            risk_config=risk,
            initial_capital_usd=initial_capital_usd,
            top_k=top_k,
            metric_start=metric_start,
            metric_end=metric_end,
            check_fast=check_fast,
        )
        out.phase_sharpes[phase] = sh
        sharpes.append(sh)
    out.mean_sharpe = float(sum(sharpes) / len(sharpes)) if sharpes else float("nan")
    return out


def run_null_distribution(
    bars: pd.DataFrame,
    model_a_scores: pd.DataFrame,
    *,
    sessions: list[date],
    cost: CostModel,
    n: int = FROZEN_N,
    **kwargs: Any,
) -> np.ndarray:
    stats = [
        run_null_draw(
            bars,
            model_a_scores,
            draw_id=r,
            sessions=sessions,
            cost=cost,
            **kwargs,
        ).mean_sharpe
        for r in range(1, n + 1)
    ]
    return np.asarray(stats, dtype=float)


def compare_to_null(
    observed_mean_sharpe: float, null_stats: np.ndarray, *, n: int | None = None
) -> NullTestResult:
    return evaluate_randomization(observed_mean_sharpe, null_stats, n=n)


def seed_policy_dict() -> dict[str, Any]:
    return {
        "canonical_string": SEED_CANONICAL_STRING,
        "base_seed": BASE_SEED,
        "child_seed_formula": "(base_seed + draw_id + phase) mod 2**64",
        "generator": "numpy.random.PCG64",
        "n": FROZEN_N,
        "draw_id_indexing": "one_based",
    }
