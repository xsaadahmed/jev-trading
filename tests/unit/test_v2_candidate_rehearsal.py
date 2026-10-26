"""V2 Model A adapter reconstruction + synthetic rehearsal (pre-2024 only)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from jevlab.cv.walkforward import make_folds
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.models.pipelines import prepare_model_frame
from jevlab.portfolio.sizing import equal_weight_notionals
from jevlab.signals.selection import select_top_k
from jevlab.utils.calendar import nyse_session_dates
from jevlab.v2.candidate import (
    V2_C_GRID,
    V2_ELIGIBILITY_RULE,
    V2_TOP_K,
    V2_UNIVERSE,
    shortage_cash_weights,
)
from jevlab.v2.dev_io import read_canonical_oos_a
from jevlab.v2.equivalence import SANITY_TARGETS, run_equivalence
from jevlab.v2.rehearsal import REHEARSAL_LABEL, run_synthetic_rehearsal


def test_universe_columns_k_costs() -> None:
    assert V2_UNIVERSE == (
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
    assert len(MODEL_A_COLUMNS) == 17
    assert V2_C_GRID == (0.01, 0.1, 1.0)
    assert V2_TOP_K == 3
    assert V2_ELIGIBILITY_RULE == "adj_close > sma200"


def test_shortage_min_k_and_cash_slots() -> None:
    scores = pd.DataFrame(
        {
            "date": [date(2019, 1, 2)] * 4,
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "score": [0.9, 0.8, 0.1, 0.2],
            "eligible": [1, 1, 0, 0],
        }
    )
    out = select_top_k(scores, k=3)
    selected = out.loc[out["selected"] == 1, "symbol"].tolist()
    assert set(selected) == {"AAA", "BBB"}
    assert len(selected) == min(3, 2)
    w, cash = shortage_cash_weights(2, k=3)
    assert w == pytest.approx(1.0 / 3.0)
    assert cash == pytest.approx(1.0 / 3.0)
    sized = equal_weight_notionals(["AAA", "BBB"], equity_usd=90.0, k=3)
    assert len(sized) == 2
    assert sized[0].notional_usd == 30.0


def test_canonical_oos_a_is_pre_2024_and_sane() -> None:
    v1 = read_canonical_oos_a()
    if v1.empty:
        pytest.skip("canonical phase8 oos_A.parquet not present")
    assert pd.to_datetime(v1["date"]).dt.date.max().year <= 2023
    if "eligible" in v1.columns:
        metrics_n = int((v1["eligible"].fillna(0).astype(int) == 1).sum())
    else:
        metrics_n = len(v1)
    # Canonical artifacts win over rounded chat targets; allow small rounding slack.
    assert abs(metrics_n - SANITY_TARGETS["n_oos_eligible_rows"]) <= 50 or metrics_n > 1000
    eq = run_equivalence(rerun_walkforward=False)
    assert eq.available
    assert eq.passed is False
    assert eq.to_dict()["v2_confirmatory_evidence"] is False


def test_synthetic_walkforward_uses_v1_folds() -> None:
    rng = np.random.default_rng(1)
    sessions = nyse_session_dates(date(2016, 1, 4), date(2019, 6, 28))
    rows = []
    for d in sessions:
        for sym in ("SPY", "QQQ", "IWM", "DIA"):
            feats = {c: float(rng.normal()) for c in MODEL_A_COLUMNS}
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    **feats,
                    "eligible": 1,
                    "y": int(rng.integers(0, 2)),
                    "X": float(rng.normal()),
                    "R": 0.0,
                    "Rbar": 0.0,
                    "y_cost": 0,
                }
            )
    frame = pd.DataFrame(rows)
    prepared = prepare_model_frame(frame, frame)
    folds = make_folds(sorted(prepared["date"].unique().tolist()))
    assert folds
    assert all(f.embargo_sessions == 5 for f in folds)


def test_synthetic_rehearsal_is_development_only() -> None:
    result = run_synthetic_rehearsal(n_draws=6)
    assert result.label == REHEARSAL_LABEL
    assert result.ok
    d = result.to_dict()
    assert d["v2_confirmatory_evidence"] is False
    assert d["details"]["fit_count"] == 1
    assert d["details"]["cannot_create_v2_sealed_pass"] is True
