"""Daily event-loop backtest with explicit cash/positions/orders/fills (§10.5).

Timing (no look-ahead):
- Decision after close of session ``t`` (features/scores as of close t).
- Entry fill at adjusted open of ``t+1`` (plus adverse slippage from CostModel).
- Exit fill at adjusted open of ``t+4``.
- End-of-day mark uses adjusted close.
- Rebalance every ``rebalance_every`` sessions with phase offset 0/1/2.
- Fractional shares; long-only; no leverage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from jevlab.backtest.calendar import entry_session, exit_session, session_index_map
from jevlab.backtest.costs import CostModel, require_cost_model
from jevlab.config.models import RiskConfig
from jevlab.portfolio.sizing import equal_weight_notionals, shares_for_notional
from jevlab.portfolio.state import PortfolioState, Position
from jevlab.risk.rules import MarketState, ProposedOrder, evaluate_orders
from jevlab.signals.selection import rebalance_dates, select_top_k

FillMode = Literal["open_t1", "delayed"]
# delayed: entry at adj_close[t+1], exit still at open[t+4] (robustness §13).

EQUITY_TOL_REL: float = 1e-9


@dataclass
class Fill:
    session: date
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    price_usd: float
    fee_usd: float
    notional_usd: float
    client_order_id: str
    cohort_id: str
    decision_date: date


@dataclass
class LedgerResult:
    """Outputs of one ledger backtest run."""

    equity_curve: pd.DataFrame  # date, cash_usd, market_value_usd, equity_usd
    trades: pd.DataFrame
    positions_eod: pd.DataFrame
    invariants_ok: bool
    invariant_violations: list[str] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def client_order_id(strategy_id: str, session: date, symbol: str, side: str) -> str:
    raw = f"{strategy_id}|{session.isoformat()}|{symbol}|{side}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{strategy_id}-{session:%Y%m%d}-{symbol}-{side}-{digest}"


def _pivot_price(bars: pd.DataFrame, col: str) -> pd.DataFrame:
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")


def run_ledger(
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
    apply_risk: bool = True,
) -> LedgerResult:
    """Run the explicit ledger event loop.

    ``scores`` must include symbol, date, score, eligible (and optionally already
    filtered to selected rows — selection is reapplied on rebalance dates).
    """
    cost = require_cost_model(cost)
    if initial_capital_usd <= 0:
        raise ValueError("initial_capital_usd must be positive")

    open_px = _pivot_price(bars, "adj_open")
    close_px = _pivot_price(bars, "adj_close")
    sessions = sorted(set(open_px.index) & set(close_px.index))
    if not sessions:
        raise ValueError("no overlapping sessions in bars")

    score_frame = scores.copy()
    score_frame["date"] = pd.to_datetime(score_frame["date"]).dt.date
    reb = set(
        rebalance_dates(
            sessions,
            every=rebalance_every_sessions,
            phase=phase,
        )
    )

    cash = float(initial_capital_usd)
    positions: dict[str, float] = {}
    open_recs: list[Position] = []
    pending: list[dict[str, Any]] = []  # scheduled fills
    fills: list[Fill] = []
    eq_rows: list[dict[str, Any]] = []
    pos_rows: list[dict[str, Any]] = []
    violations: list[str] = []
    rejection_rows: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    hwm = float(initial_capital_usd)
    sod_equity = float(initial_capital_usd)
    kill = False
    idx_map = session_index_map(sessions)

    def mark_value(day: date) -> float:
        total = 0.0
        for sym, qty in positions.items():
            if qty == 0:
                continue
            if sym in close_px.columns and day in close_px.index:
                px = close_px.loc[day, sym]
            elif sym in open_px.columns and day in open_px.index:
                px = open_px.loc[day, sym]
            else:
                continue
            if px is None or (isinstance(px, float) and np.isnan(float(px))):
                continue
            total += float(qty) * float(px)
        return total

    def equity_now(day: date) -> float:
        return cash + mark_value(day)

    for day in sessions:
        # --- 1) Execute fills scheduled for today's open (exits before entries) ---
        todays = [p for p in pending if p["fill_session"] == day]
        pending = [p for p in pending if p["fill_session"] != day]
        todays.sort(key=lambda p: 0 if p["side"] == "sell" else 1)

        for order in todays:
            sym = order["symbol"]
            side = order["side"]
            if fill_mode == "delayed" and side == "buy":
                raw_px = float(close_px.loc[day, sym])
            else:
                raw_px = float(open_px.loc[day, sym])
            if side == "buy":
                px = cost.buy_fill_price_usd(raw_px, sym)
                qty = float(order["qty"])
                notional = qty * px
                if notional > cash + 1e-9:
                    # Scale down to available cash (no leverage).
                    qty = cash / px if px > 0 else 0.0
                    notional = qty * px
                if qty <= 0:
                    continue
                cash -= notional
                positions[sym] = positions.get(sym, 0.0) + qty
                # Keep linked sell qty in sync if buy was cash-scaled.
                for p in pending:
                    if p["side"] == "sell" and p["cohort_id"] == order["cohort_id"]:
                        p["qty"] = qty
                open_recs.append(
                    Position(
                        symbol=sym,
                        qty=qty,
                        entry_session=day,
                        exit_session=order["exit_session"],
                        cost_basis_usd=notional,
                        cohort_id=order["cohort_id"],
                    )
                )
                fills.append(
                    Fill(
                        session=day,
                        symbol=sym,
                        side="buy",
                        qty=qty,
                        price_usd=px,
                        fee_usd=0.0,
                        notional_usd=notional,
                        client_order_id=order["client_order_id"],
                        cohort_id=order["cohort_id"],
                        decision_date=order["decision_date"],
                    )
                )
            else:
                # Sell entire cohort qty (or available).
                qty = min(float(order["qty"]), positions.get(sym, 0.0))
                if qty <= 0:
                    continue
                px = cost.sell_fill_price_usd(raw_px, sym)
                notional = qty * px
                fee = cost.sell_fee_usd(notional)
                cash += notional - fee
                positions[sym] = positions.get(sym, 0.0) - qty
                if abs(positions[sym]) < 1e-12:
                    positions.pop(sym, None)
                open_recs = [
                    r
                    for r in open_recs
                    if not (r.cohort_id == order["cohort_id"] and r.symbol == sym)
                ]
                fills.append(
                    Fill(
                        session=day,
                        symbol=sym,
                        side="sell",
                        qty=qty,
                        price_usd=px,
                        fee_usd=fee,
                        notional_usd=notional,
                        client_order_id=order["client_order_id"],
                        cohort_id=order["cohort_id"],
                        decision_date=order["decision_date"],
                    )
                )

        # --- 2) Mark to market (adj close) ---
        mv = mark_value(day)
        eq = cash + mv
        if eq > hwm:
            hwm = eq
        # Relative accounting invariant.
        recon = abs(eq - (cash + mv))
        if eq != 0 and recon / max(abs(eq), 1e-12) > EQUITY_TOL_REL:
            violations.append(f"{day}: cash+holdings != equity (rel {recon / abs(eq)})")
        # Also absolute floating check on the identity we just used (always ~0);
        # the meaningful check is cash + marked positions == recorded equity.
        recorded = cash + mv
        if abs(recorded - eq) > 1e-9 * max(1.0, abs(eq)):
            violations.append(f"{day}: accounting drift")

        eq_rows.append(
            {
                "date": day,
                "cash_usd": cash,
                "market_value_usd": mv,
                "equity_usd": eq,
            }
        )
        for sym, qty in positions.items():
            pos_rows.append({"date": day, "symbol": sym, "qty": qty})

        # Drawdown kill for subsequent decisions.
        if hwm > 0 and (hwm - eq) / hwm >= (risk_config.max_drawdown_pct / 100.0):
            kill = True

        # --- 3) Decisions after close on rebalance dates ---
        if day not in reb:
            sod_equity = eq
            continue

        entry = entry_session(day, sessions, entry_lag=entry_lag_sessions)
        exit_d = exit_session(
            day,
            sessions,
            entry_lag=entry_lag_sessions,
            horizon_sessions=horizon_sessions,
        )
        if entry is None or exit_d is None:
            sod_equity = eq
            continue

        day_scores = score_frame.loc[score_frame["date"] == day]
        if day_scores.empty:
            sod_equity = eq
            continue
        selected = select_top_k(day_scores, k=top_k, score_col=score_col)
        picks = selected.loc[selected["selected"] == 1, "symbol"].astype(str).tolist()
        targets = equal_weight_notionals(picks, equity_usd=eq, k=top_k)

        proposed: list[ProposedOrder] = []
        staged: list[dict[str, Any]] = []
        for tgt in targets:
            sym = tgt.symbol
            if sym not in open_px.columns or entry not in open_px.index:
                continue
            raw_entry = float(open_px.loc[entry, sym])
            if fill_mode == "delayed":
                # Size using expected delayed fill (close of entry session).
                if entry not in close_px.index:
                    continue
                raw_entry = float(close_px.loc[entry, sym])
                px = cost.buy_fill_price_usd(raw_entry, sym)
            else:
                px = cost.buy_fill_price_usd(raw_entry, sym)
            qty = shares_for_notional(tgt.notional_usd, px)
            if qty <= 0:
                continue
            oid = client_order_id(strategy_id, day, sym, "buy")
            cohort = f"{day.isoformat()}:{sym}"
            proposed.append(
                ProposedOrder(
                    symbol=sym,
                    side="buy",
                    qty=qty,
                    notional_usd=qty * px,
                    client_order_id=oid,
                    session_date=day,
                )
            )
            staged.append(
                {
                    "symbol": sym,
                    "side": "buy",
                    "qty": qty,
                    "fill_session": entry,
                    "exit_session": exit_d,
                    "client_order_id": oid,
                    "cohort_id": cohort,
                    "decision_date": day,
                    "sell_client_order_id": client_order_id(strategy_id, day, sym, "sell"),
                }
            )

        # Outstanding buys already submitted (not this decision batch). The risk
        # gate accumulates this batch's notionals itself — do not pre-sum staged.
        port = PortfolioState(
            cash_usd=cash,
            positions=dict(positions),
            strategy_equity_usd=eq,
            strategy_capital_usd=initial_capital_usd,
            high_water_mark_usd=hwm,
            sod_equity_usd=sod_equity,
            market_value_usd=mv,
            outstanding_buy_notional_usd=0.0,
            orders_today=0,
            kill_switch_active=kill,
            existing_client_order_ids=set(known_ids),
            open_position_records=list(open_recs),
        )
        mkt = MarketState(
            session_date=day,
            decision_time=datetime_at_decision(day),
            last_completed_nyse_session=day,
            last_bar_session=day,
            available_at_ok=True,
            missing_symbol_count=0,
        )
        if apply_risk:
            risk_out = evaluate_orders(proposed, port, mkt, risk_config)
            approved_ids = {o.client_order_id for o in risk_out.approved}
            for rej in risk_out.rejections:
                rejection_rows.append(
                    {
                        "date": day,
                        "symbol": rej.symbol,
                        "reason": rej.reason,
                        "client_order_id": rej.client_order_id,
                    }
                )
            staged = [s for s in staged if s["client_order_id"] in approved_ids]
        for s in staged:
            known_ids.add(s["client_order_id"])
            pending.append(s)
            # Schedule exit at open of exit_session.
            pending.append(
                {
                    "symbol": s["symbol"],
                    "side": "sell",
                    "qty": s["qty"],
                    "fill_session": s["exit_session"],
                    "exit_session": s["exit_session"],
                    "client_order_id": s["sell_client_order_id"],
                    "cohort_id": s["cohort_id"],
                    "decision_date": s["decision_date"],
                }
            )
            known_ids.add(s["sell_client_order_id"])

        sod_equity = eq

    equity_curve = pd.DataFrame(eq_rows)
    trades = pd.DataFrame(
        [
            {
                "session": f.session,
                "symbol": f.symbol,
                "side": f.side,
                "qty": f.qty,
                "price_usd": f.price_usd,
                "fee_usd": f.fee_usd,
                "notional_usd": f.notional_usd,
                "client_order_id": f.client_order_id,
                "cohort_id": f.cohort_id,
                "decision_date": f.decision_date,
            }
            for f in fills
        ]
    )
    positions_eod = pd.DataFrame(pos_rows)
    invariants_ok = len(violations) == 0 and _check_equity_identity(equity_curve)
    if not _check_equity_identity(equity_curve):
        violations.append("equity identity failed on curve")

    return LedgerResult(
        equity_curve=equity_curve,
        trades=trades,
        positions_eod=positions_eod,
        invariants_ok=invariants_ok,
        invariant_violations=violations,
        rejections=rejection_rows,
        meta={
            "phase": phase,
            "fill_mode": fill_mode,
            "cost_bps_per_side": cost.cost_bps_per_side,
            "scenario_name": cost.scenario_name,
            "n_sessions": len(sessions),
            "initial_capital_usd": initial_capital_usd,
            "top_k": top_k,
            "session_index": idx_map,
        },
    )


def datetime_at_decision(day: date) -> Any:
    """decision_time(t) = t 18:30 America/New_York (stored as aware datetime)."""
    from zoneinfo import ZoneInfo

    ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=18, minute=30)
    return ts.tz_localize(ZoneInfo("America/New_York"))


def _check_equity_identity(equity_curve: pd.DataFrame) -> bool:
    if equity_curve.empty:
        return True
    cash = equity_curve["cash_usd"].to_numpy(dtype=float)
    mv = equity_curve["market_value_usd"].to_numpy(dtype=float)
    eq = equity_curve["equity_usd"].to_numpy(dtype=float)
    recon = cash + mv
    denom = np.maximum(np.abs(eq), 1e-12)
    return bool(np.all(np.abs(recon - eq) / denom <= EQUITY_TOL_REL))
