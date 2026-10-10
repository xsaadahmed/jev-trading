"""Vectorized backtest path used for placebo/bootstrap loops (§10.5).

Must match ``run_ledger`` equity within relative tolerance 1e-9 on the same
inputs (regression-tested). Omits per-order risk rejections by default so
inner loops stay cheap; pass ``apply_risk=True`` to mirror the ledger gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jevlab.backtest.costs import CostModel, require_cost_model
from jevlab.backtest.ledger import EQUITY_TOL_REL, FillMode, LedgerResult, run_ledger
from jevlab.config.models import RiskConfig


def run_fast(
    bars: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    cost: CostModel,
    risk_config: RiskConfig,
    initial_capital_usd: float = 100_000.0,
    top_k: int = 3,
    horizon_sessions: int = 3,
    entry_lag_sessions: int = 1,
    rebalance_every_sessions: int = 3,
    phase: int = 0,
    strategy_id: str = "jev-pullback-v1",
    fill_mode: FillMode = "open_t1",
    score_col: str = "score",
    apply_risk: bool = False,
) -> LedgerResult:
    """Fast path: currently delegates to the ledger engine for bit-stable parity.

    A pure matrix implementation lives in ``equity_from_holdings`` for cohort
    diagnostics; the event loop remains the source of truth so ledger≡fast
    holds under the same risk settings. For bootstrap/placebo (no risk),
    ``apply_risk=False`` skips the risk gate (ledger default is True — callers
    that need equivalence must pass the same flag to both).
    """
    cost = require_cost_model(cost)
    return run_ledger(
        bars,
        scores,
        cost=cost,
        risk_config=risk_config,
        initial_capital_usd=initial_capital_usd,
        top_k=top_k,
        horizon_sessions=horizon_sessions,
        entry_lag_sessions=entry_lag_sessions,
        rebalance_every_sessions=rebalance_every_sessions,
        phase=phase,
        strategy_id=strategy_id,
        fill_mode=fill_mode,
        score_col=score_col,
        apply_risk=apply_risk,
    )


def equity_curves_equal(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    tol_rel: float = EQUITY_TOL_REL,
) -> bool:
    """True if equity columns match within relative tolerance."""
    if list(a.columns) != list(b.columns) and not (
        {"date", "equity_usd"} <= set(a.columns) and {"date", "equity_usd"} <= set(b.columns)
    ):
        return False
    left = a.sort_values("date").reset_index(drop=True)
    right = b.sort_values("date").reset_index(drop=True)
    if len(left) != len(right):
        return False
    if not (left["date"].to_numpy() == right["date"].to_numpy()).all():
        return False
    ea = left["equity_usd"].to_numpy(dtype=float)
    eb = right["equity_usd"].to_numpy(dtype=float)
    denom = np.maximum(np.maximum(np.abs(ea), np.abs(eb)), 1e-12)
    return bool(np.all(np.abs(ea - eb) / denom <= tol_rel))


def cohort_net_returns(
    bars: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    cost: CostModel,
    sessions: list[Any],
    entry_lag: int = 1,
    horizon: int = 3,
) -> pd.DataFrame:
    """Vectorized per-cohort net returns (equal-weight legs) for diagnostics.

    ``picks`` columns: decision_date, symbol. Uses open entry/exit with costs.
    """
    from jevlab.backtest.calendar import entry_session, exit_session

    cost = require_cost_model(cost)
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    open_map = {
        (r.date, r.symbol): float(r.adj_open)
        for r in frame.itertuples(index=False)
        if hasattr(r, "adj_open")
    }
    rows: list[dict[str, Any]] = []
    sess = [pd.Timestamp(s).date() for s in sessions]
    for decision, group in picks.groupby("decision_date"):
        d = pd.Timestamp(decision).date()
        entry = entry_session(d, sess, entry_lag=entry_lag)
        exit_d = exit_session(d, sess, entry_lag=entry_lag, horizon_sessions=horizon)
        if entry is None or exit_d is None:
            continue
        rets: list[float] = []
        for sym in group["symbol"].astype(str):
            o0 = open_map.get((entry, sym))
            o1 = open_map.get((exit_d, sym))
            if o0 is None or o1 is None or o0 <= 0:
                continue
            buy = cost.buy_fill_price_usd(o0, sym)
            sell = cost.sell_fill_price_usd(o1, sym)
            fee_frac = cost.sell_fee_bps / 10_000.0
            net = (sell * (1.0 - fee_frac) / buy) - 1.0
            rets.append(net)
        if not rets:
            continue
        rows.append(
            {
                "decision_date": d,
                "entry_session": entry,
                "exit_session": exit_d,
                "n_legs": len(rets),
                "net_return_dec": float(np.mean(rets)),
            }
        )
    return pd.DataFrame(rows)
