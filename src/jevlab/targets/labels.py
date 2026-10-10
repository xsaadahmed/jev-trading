"""Label construction — sole module for target logic (MASTER_SPEC §8).

Time semantics: decision after close of session t; label uses adjusted opens at
entry open(t+1) and exit open(t+4). The label for date t is unknown until
session t+4 begins. Features at t use only data <= close t.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from jevlab.utils.units import bps_to_decimal

LABEL_VERSION: str = "v1"
HORIZON_SESSIONS: int = 3  # exit is entry + horizon; entry is t+1 ⇒ exit session index t+4
ENTRY_LAG_SESSIONS: int = 1


def roundtrip_cost_dec(cost_bps_per_side: float, sell_fee_bps: float) -> float:
    """Round-trip cost as a decimal return: ``2 * cost_bps / 1e4 + fee_bps / 1e4``.

    Counts the one-way all-in cost once on each leg (2x) and the sell-side
    regulatory fee once. Never double-counts the round trip.
    Example: 2.5 bps/side + 0.3 bp fee -> 0.00053.
    """
    return 2.0 * bps_to_decimal(cost_bps_per_side) + bps_to_decimal(sell_fee_bps)


def _session_index(sessions: list[date]) -> dict[date, int]:
    return {d: i for i, d in enumerate(sessions)}


def build_labels(
    bars: pd.DataFrame,
    *,
    eligible: pd.DataFrame | None = None,
    universe_symbols: list[str],
    cost_bps_per_side: float = 2.5,
    sell_fee_bps: float = 0.3,
    per_symbol_cost_multiplier: dict[str, float] | None = None,
    label_version: str = LABEL_VERSION,
    horizon_sessions: int = HORIZON_SESSIONS,
    entry_lag_sessions: int = ENTRY_LAG_SESSIONS,
) -> pd.DataFrame:
    """Build primary and secondary labels for every (symbol, decision date).

    Parameters
    ----------
    bars:
        Must include ``symbol``, ``date``, ``adj_open``. Session dates are the
        sorted unique bar dates (NYSE sessions).
    eligible:
        Optional frame with ``symbol``, ``date``, ``eligible`` (1/0). If omitted
        and ``eligible`` is a column on ``bars``, that is used; otherwise all
        rows are treated as eligible for computing R but y is still gated when
        an eligible frame is provided.
    cost_bps_per_side:
        One-way all-in cost in bps before per-symbol multiplier.
    per_symbol_cost_multiplier:
        Scales ``cost_bps_per_side`` only (not ``sell_fee_bps``).

    Returns
    -------
    DataFrame with columns:
    ``symbol, date, horizon, label_version, R, Rbar, X, y, roundtrip_cost_dec,
    y_cost, eligible, entry_date, exit_date, feature_time_end, label_window_start``.

    ``feature_time_end`` is the decision session date t (features as-of close t).
    ``label_window_start`` is the entry session date (open t+1), strictly after t.
    """
    if bars.empty:
        return pd.DataFrame()

    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if "adj_open" not in frame.columns:
        raise ValueError("bars must include adj_open for label returns")

    multipliers = per_symbol_cost_multiplier or {}
    sessions = sorted(frame["date"].unique().tolist())
    idx = _session_index(sessions)
    # Exit session offset from decision date: entry_lag + horizon_sessions
    # entry at i+entry_lag, exit at i+entry_lag+horizon ⇒ offset = entry_lag + horizon
    exit_offset = entry_lag_sessions + horizon_sessions

    opens = (
        frame.loc[:, ["symbol", "date", "adj_open"]]
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .set_index(["symbol", "date"])["adj_open"]
    )

    if eligible is None and "eligible" in frame.columns:
        elig = frame.loc[:, ["symbol", "date", "eligible"]].drop_duplicates(
            subset=["symbol", "date"], keep="last"
        )
    elif eligible is not None:
        elig = eligible.loc[:, ["symbol", "date", "eligible"]].copy()
        elig["date"] = pd.to_datetime(elig["date"]).dt.date
    else:
        elig = frame.loc[:, ["symbol", "date"]].drop_duplicates().assign(eligible=1)

    elig_map = elig.set_index(["symbol", "date"])["eligible"]

    # Decision dates that have room for exit open.
    max_decision_i = len(sessions) - 1 - exit_offset
    if max_decision_i < 0:
        return pd.DataFrame()
    decision_dates = sessions[: max_decision_i + 1]

    rows: list[dict[str, Any]] = []
    for t in decision_dates:
        i = idx[t]
        entry_date = sessions[i + entry_lag_sessions]
        exit_date = sessions[i + exit_offset]
        # Gross returns per symbol for this decision date.
        r_by_symbol: dict[str, float] = {}
        for sym in universe_symbols:
            try:
                a_entry = float(opens.loc[(sym, entry_date)])
                a_exit = float(opens.loc[(sym, exit_date)])
            except KeyError:
                continue
            if a_entry <= 0 or not np.isfinite(a_entry) or not np.isfinite(a_exit):
                continue
            r_by_symbol[sym] = a_exit / a_entry - 1.0

        # Require full universe for Rbar (equal-weight over all 13).
        if len(r_by_symbol) < len(universe_symbols):
            continue

        rbar = float(np.mean(list(r_by_symbol.values())))
        for sym, r_dec in r_by_symbol.items():
            x_dec = r_dec - rbar
            try:
                is_eligible = int(elig_map.loc[(sym, t)])
            except KeyError:
                is_eligible = 0
            mult = float(multipliers.get(sym, 1.0))
            cost_bps_eff = cost_bps_per_side * mult
            rtc_dec = roundtrip_cost_dec(cost_bps_eff, sell_fee_bps)
            if is_eligible == 1:
                y_val: float = 1.0 if x_dec > 0.0 else 0.0
                y_cost_val: float = 1.0 if r_dec > rtc_dec else 0.0
            else:
                y_val = float("nan")
                y_cost_val = float("nan")
            rows.append(
                {
                    "symbol": sym,
                    "date": t,
                    "horizon": horizon_sessions,
                    "label_version": label_version,
                    "R": r_dec,
                    "Rbar": rbar,
                    "X": x_dec,
                    "y": y_val,
                    "roundtrip_cost_dec": rtc_dec,
                    "y_cost": y_cost_val,
                    "eligible": is_eligible,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "feature_time_end": t,
                    "label_window_start": entry_date,
                    "cost_bps_per_side_effective": cost_bps_eff,
                    "sell_fee_bps": sell_fee_bps,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def assert_label_after_feature(labels: pd.DataFrame) -> None:
    """Invariant: every label window starts strictly after the feature timestamp."""
    if labels.empty:
        return
    bad = labels["label_window_start"] <= labels["feature_time_end"]
    if bad.any():
        sample = labels.loc[bad, ["symbol", "date", "feature_time_end", "label_window_start"]]
        raise AssertionError(
            f"label window must start after feature time; examples:\n{sample.head()}"
        )
