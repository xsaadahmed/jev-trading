"""Economic metrics (MASTER_SPEC §12.2) and a minimal power/MDE helper.

Benchmarks: equal-weight universe buy-and-hold and SPY. Beta/alpha use daily
OLS with HAC (Newey-West) standard errors via statsmodels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

TRADING_DAYS_PER_YEAR: float = 252.0


def _slice_equity_curve(
    equity_curve: pd.DataFrame,
    *,
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    """Restrict the equity curve to ``[start, end]`` (inclusive) when given."""
    if equity_curve.empty or (start is None and end is None):
        return equity_curve
    frame = equity_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if start is not None:
        frame = frame.loc[frame["date"] >= start]
    if end is not None:
        frame = frame.loc[frame["date"] <= end]
    return frame.reset_index(drop=True)


def _slice_trades(
    trades: pd.DataFrame,
    *,
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    """Keep round-trip legs whose decision_date falls in ``[start, end]``."""
    if trades is None or trades.empty or (start is None and end is None):
        return trades if trades is not None else pd.DataFrame()
    frame = trades.copy()
    if "decision_date" in frame.columns:
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.date
        key = "decision_date"
    else:
        frame["session"] = pd.to_datetime(frame["session"]).dt.date
        key = "session"
    if start is not None:
        frame = frame.loc[frame[key] >= start]
    if end is not None:
        frame = frame.loc[frame[key] <= end]
    return frame.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class EconomicMetrics:
    """Container for §12.2 statistics (all decimal returns unless noted)."""

    values: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


def daily_returns(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float)
    return eq.pct_change().dropna()


def max_drawdown_dec(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def cagr_dec(equity: pd.Series, *, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    eq = equity.astype(float).dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    n = len(eq) - 1
    total = float(eq.iloc[-1] / eq.iloc[0])
    if total <= 0:
        return -1.0
    return float(total ** (periods_per_year / n) - 1.0)


def sharpe_ratio(rets: pd.Series, *, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    r = rets.astype(float).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1))


def sortino_ratio(rets: pd.Series, *, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    r = rets.astype(float).dropna()
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / dstd)


def cvar_dec(rets: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall (mean of losses beyond alpha quantile). Negative = loss."""
    r = rets.astype(float).dropna()
    if r.empty:
        return 0.0
    q = float(r.quantile(alpha))
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else 0.0


def beta_alpha_hac(
    strategy_rets: pd.Series,
    bench_rets: pd.Series,
    *,
    hac_lags: int = 5,
) -> dict[str, float]:
    """Daily OLS of strategy on benchmark with Newey-West HAC SEs."""
    aligned = pd.concat(
        [strategy_rets.astype(float), bench_rets.astype(float)],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 10:
        return {
            "beta": float("nan"),
            "alpha_daily_dec": float("nan"),
            "alpha_ann_dec": float("nan"),
            "beta_se_hac": float("nan"),
            "alpha_se_hac": float("nan"),
        }
    y = aligned.iloc[:, 0].to_numpy()
    x = sm.add_constant(aligned.iloc[:, 1].to_numpy())
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    alpha_d = float(model.params[0])
    beta = float(model.params[1])
    return {
        "beta": beta,
        "alpha_daily_dec": alpha_d,
        "alpha_ann_dec": float((1.0 + alpha_d) ** TRADING_DAYS_PER_YEAR - 1.0),
        "beta_se_hac": float(model.bse[1]),
        "alpha_se_hac": float(model.bse[0]),
    }


def compute_economic_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    benchmark_rets: pd.Series | None = None,
    initial_capital_usd: float | None = None,
    metric_start: date | None = None,
    metric_end: date | None = None,
) -> EconomicMetrics:
    """Compute §12.2 metrics from a ledger/fast result.

    When ``metric_start`` / ``metric_end`` are set (walk-forward OOS window),
    equity and trades are sliced to that window so idle warm-up years do not
    dilute exposure/CAGR/Sharpe.
    """
    curve = _slice_equity_curve(equity_curve, start=metric_start, end=metric_end)
    trade_frame = _slice_trades(trades, start=metric_start, end=metric_end)
    if curve.empty:
        return EconomicMetrics(
            values={
                "total_return_dec": 0.0,
                "cagr_dec": 0.0,
                "ann_vol_dec": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown_dec": 0.0,
                "calmar": 0.0,
                "hit_rate": 0.0,
                "avg_trade_dec": 0.0,
                "median_trade_dec": 0.0,
                "profit_factor": 0.0,
                "turnover": 0.0,
                "exposure": 0.0,
                "trade_count": 0.0,
                "avg_holding_sessions": 0.0,
                "cvar_5pct_dec": 0.0,
                "cvar_1pct_dec": 0.0,
                "skewness": 0.0,
                "top5_trade_share_pnl": 0.0,
                "beta": float("nan"),
                "alpha_daily_dec": float("nan"),
                "alpha_ann_dec": float("nan"),
                "beta_se_hac": float("nan"),
                "alpha_se_hac": float("nan"),
            }
        )
    eq = curve.sort_values("date")["equity_usd"].astype(float)
    rets = daily_returns(eq)
    init = float(initial_capital_usd) if initial_capital_usd is not None else float(eq.iloc[0])
    total_ret = float(eq.iloc[-1] / init - 1.0) if init > 0 and len(eq) else 0.0
    dd = max_drawdown_dec(eq)
    cagr = cagr_dec(eq)
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(rets) > 1 else 0.0
    sh = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    calmar = float(cagr / abs(dd)) if dd < 0 else 0.0

    trade_stats = _trade_stats(trade_frame)
    exposure = _exposure(curve)
    turnover = _turnover(trade_frame, curve)

    out: dict[str, float] = {
        "total_return_dec": total_ret,
        "cagr_dec": cagr,
        "ann_vol_dec": vol,
        "sharpe": sh,
        "sortino": so,
        "max_drawdown_dec": dd,
        "calmar": calmar,
        "hit_rate": trade_stats["hit_rate"],
        "avg_trade_dec": trade_stats["avg_trade_dec"],
        "median_trade_dec": trade_stats["median_trade_dec"],
        "profit_factor": trade_stats["profit_factor"],
        "turnover": turnover,
        "exposure": exposure,
        "trade_count": trade_stats["trade_count"],
        "avg_holding_sessions": trade_stats["avg_holding_sessions"],
        "cvar_5pct_dec": cvar_dec(rets, 0.05),
        "cvar_1pct_dec": cvar_dec(rets, 0.01),
        "skewness": float(stats.skew(rets)) if len(rets) > 2 else 0.0,
        "top5_trade_share_pnl": trade_stats["top5_trade_share_pnl"],
    }
    if benchmark_rets is not None:
        ba = beta_alpha_hac(rets, benchmark_rets)
        out.update(ba)
    else:
        out.update(
            {
                "beta": float("nan"),
                "alpha_daily_dec": float("nan"),
                "alpha_ann_dec": float("nan"),
                "beta_se_hac": float("nan"),
                "alpha_se_hac": float("nan"),
            }
        )
    return EconomicMetrics(values=out)


def contribution_tables(
    trades: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """P&L contribution by symbol and by year (from round-trip cohorts)."""
    pnl = _round_trip_pnl(trades)
    if pnl.empty:
        return {"by_symbol": pd.DataFrame(), "by_year": pd.DataFrame()}
    by_sym = pd.DataFrame(pnl.groupby("symbol", as_index=False)["pnl_usd"].sum())
    pnl = pnl.copy()
    pnl["year"] = pd.to_datetime(pnl["exit_session"]).dt.year
    by_year = pd.DataFrame(pnl.groupby("year", as_index=False)["pnl_usd"].sum())
    return {"by_symbol": by_sym, "by_year": by_year}


def _round_trip_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    buys = trades.loc[trades["side"] == "buy"].set_index("cohort_id")
    sells = trades.loc[trades["side"] == "sell"].set_index("cohort_id")
    rows: list[dict[str, Any]] = []
    for cid in buys.index.intersection(sells.index):
        b = buys.loc[cid]
        s = sells.loc[cid]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        buy_cost = float(b["notional_usd"])
        sell_net = float(s["notional_usd"]) - float(s.get("fee_usd", 0.0))
        rows.append(
            {
                "cohort_id": cid,
                "symbol": str(b["symbol"]),
                "entry_session": b["session"],
                "exit_session": s["session"],
                "pnl_usd": sell_net - buy_cost,
                "ret_dec": (sell_net / buy_cost - 1.0) if buy_cost > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    pnl = _round_trip_pnl(trades)
    if pnl.empty:
        return {
            "hit_rate": 0.0,
            "avg_trade_dec": 0.0,
            "median_trade_dec": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0.0,
            "avg_holding_sessions": 0.0,
            "top5_trade_share_pnl": 0.0,
        }
    rets = pnl["ret_dec"].astype(float)
    wins = pnl.loc[pnl["pnl_usd"] > 0, "pnl_usd"].sum()
    losses = -pnl.loc[pnl["pnl_usd"] < 0, "pnl_usd"].sum()
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    hold = pd.to_datetime(pnl["exit_session"]).dt.date.apply(
        lambda x: x.toordinal()
    ) - pd.to_datetime(pnl["entry_session"]).dt.date.apply(lambda x: x.toordinal())
    # Session distance approximate via calendar days; callers use session index when needed.
    top5 = pnl["pnl_usd"].astype(float).abs().nlargest(min(5, len(pnl))).sum()
    total_abs = float(pnl["pnl_usd"].astype(float).abs().sum())
    return {
        "hit_rate": float((rets > 0).mean()),
        "avg_trade_dec": float(rets.mean()),
        "median_trade_dec": float(rets.median()),
        "profit_factor": pf if np.isfinite(pf) else 0.0,
        "trade_count": float(len(pnl)),
        "avg_holding_sessions": float(hold.mean()) if len(hold) else 0.0,
        "top5_trade_share_pnl": float(top5 / total_abs) if total_abs > 0 else 0.0,
    }


def _exposure(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    eq = equity_curve["equity_usd"].astype(float).replace(0, np.nan)
    mv = equity_curve["market_value_usd"].astype(float)
    frac = (mv / eq).dropna()
    return float(frac.mean()) if len(frac) else 0.0


def _turnover(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> float:
    if trades is None or trades.empty or equity_curve.empty:
        return 0.0
    notional = float(trades["notional_usd"].astype(float).sum())
    avg_eq = float(equity_curve["equity_usd"].astype(float).mean())
    n_days = max(len(equity_curve), 1)
    # Annualized one-way turnover approximation.
    return float((notional / avg_eq) * (TRADING_DAYS_PER_YEAR / n_days)) if avg_eq > 0 else 0.0


def benchmark_returns(
    bars: pd.DataFrame,
    *,
    sessions: list[Any] | None = None,
    universe_symbols: list[str] | None = None,
) -> dict[str, pd.Series]:
    """SPY and equal-weight universe daily close-to-close returns."""
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if sessions is not None:
        sess = {pd.Timestamp(s).date() for s in sessions}
        frame = frame.loc[frame["date"].isin(sess)]
    close = frame.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    close = close.sort_index()
    out: dict[str, pd.Series] = {}
    if "SPY" in close.columns:
        out["SPY"] = close["SPY"].pct_change().dropna()
    if universe_symbols:
        cols = [c for c in universe_symbols if c in close.columns]
        if cols:
            ew = close[cols].pct_change().mean(axis=1).dropna()
            out["equal_weight_universe"] = ew
    return out


def minimum_detectable_effect(
    *,
    n_blocks: int,
    corr: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, float]:
    """Rough MDE for paired deltaSharpe / deltaIC given independent date-blocks.

    Walk-forward OOS ≈ 1,258 sessions ≈ 419 non-overlapping 3-session blocks (§12.3).
    Uses two-sample normal approximation with paired correlation.
    """
    # z for two-sided alpha and one-sided power
    z_a = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_b = float(stats.norm.ppf(power))
    # Effective variance inflation for paired difference of unit-variance series.
    var_factor = max(2.0 * (1.0 - corr), 1e-6)
    se_unit = np.sqrt(var_factor / max(n_blocks, 1))
    mde = (z_a + z_b) * se_unit
    return {
        "n_blocks": float(n_blocks),
        "corr": float(corr),
        "mde_delta_sharpe_unit": float(mde),
        "mde_delta_ic_unit": float(mde),
        "alpha": alpha,
        "power": power,
    }


def power_statement(
    *,
    n_sessions: int = 1258,
    horizon_sessions: int = 3,
    corr: float = 0.9,
) -> str:
    """Human-readable power/MDE blurb for sanity reports."""
    n_blocks = max(n_sessions // horizon_sessions, 1)
    mde = minimum_detectable_effect(n_blocks=n_blocks, corr=corr)
    return (
        f"Effective independent date-blocks ~ {n_blocks} "
        f"(from {n_sessions} sessions / {horizon_sessions}-session horizon), "
        f"not n_assets x n_sessions. With corr(A,B)~{corr:.2f}, "
        f"approximate MDE for paired dSharpe/dIC (alpha=0.05, power=0.8) is "
        f"{mde['mde_delta_sharpe_unit']:.3f} in unit-variance terms. "
        f"If underpowered relative to the observed gap, report 'inconclusive', "
        f"not 'no effect'."
    )
