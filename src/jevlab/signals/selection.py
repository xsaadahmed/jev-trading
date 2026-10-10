"""Top-K eligible selection; ties broken by symbol order (MASTER_SPEC §4.2)."""

from __future__ import annotations

from datetime import date

import pandas as pd


def select_top_k(
    scores: pd.DataFrame,
    *,
    k: int = 3,
    score_col: str = "score",
    eligible_col: str = "eligible",
    date_col: str = "date",
    symbol_col: str = "symbol",
    rebalance_dates: list[date] | None = None,
) -> pd.DataFrame:
    """Select top-K eligible symbols by score on each date.

    Ties: higher score wins; equal scores broken by ascending symbol order
    (deterministic). Adds ``rank`` (1=best) and ``selected`` (1/0).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    required = {score_col, eligible_col, date_col, symbol_col}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing columns: {sorted(missing)}")

    frame = scores.copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.date
    if rebalance_dates is not None:
        allowed = {pd.Timestamp(d).date() for d in rebalance_dates}
        frame = frame.loc[frame[date_col].isin(allowed)]

    parts: list[pd.DataFrame] = []
    for _dt, group in frame.groupby(date_col, sort=True):
        elig = group.loc[group[eligible_col].fillna(0).astype(int) == 1].copy()
        # Sort by score descending, symbol ascending for ties.
        elig = elig.sort_values(
            [score_col, symbol_col],
            ascending=[False, True],
            kind="mergesort",
        )
        elig["rank"] = range(1, len(elig) + 1)
        elig["selected"] = (elig["rank"] <= k).astype(int)
        # Non-eligible rows: not selected.
        non = group.loc[group[eligible_col].fillna(0).astype(int) != 1].copy()
        non["rank"] = pd.NA
        non["selected"] = 0
        parts.append(pd.concat([elig, non], ignore_index=True))

    if not parts:
        out = frame.copy()
        out["rank"] = pd.NA
        out["selected"] = 0
        return out
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values([date_col, symbol_col])
        .reset_index(drop=True)
    )


def rebalance_dates(
    sessions: list[date],
    *,
    every: int = 3,
    phase: int = 0,
) -> list[date]:
    """Fixed-calendar rebalance dates: sessions where index % every == phase."""
    ordered = sorted({pd.Timestamp(d).date() for d in sessions})
    return [d for i, d in enumerate(ordered) if i % every == phase]
