"""Label construction and cost-formula tests (MASTER_SPEC §8)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from jevlab.data.availability import assign_bars_available_at
from jevlab.targets.labels import (
    assert_label_after_feature,
    build_labels,
    roundtrip_cost_dec,
)
from tests.unit.feature_fixtures import make_sessions


def test_roundtrip_cost_dec_exact_normal_scenario() -> None:
    # 2 * 2.5 bps + 0.3 bp = 5.3 bp = 0.00053; two-sided counted once.
    assert roundtrip_cost_dec(2.5, 0.3) == pytest.approx(0.00053)
    assert roundtrip_cost_dec(2.5, 0.3) == 2 * 2.5 / 10_000 + 0.3 / 10_000


def test_roundtrip_cost_dec_never_double_counts_legs() -> None:
    # One-way 2.5 → round trip contribution is 5.0 bps, not 10.
    assert roundtrip_cost_dec(2.5, 0.0) == pytest.approx(0.0005)
    assert roundtrip_cost_dec(2.5, 0.0) != pytest.approx(0.0010)


def test_label_window_starts_after_feature_time() -> None:
    sessions = make_sessions(40, start=date(2019, 1, 2))
    symbols = [f"S{i:02d}" for i in range(13)]
    # Constant opens so R=0; still produces label rows.
    rows = []
    for sym in symbols:
        for i, d in enumerate(sessions):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "adj_open": 100.0 + 0.01 * i,
                    "adj_close": 100.0 + 0.01 * i,
                    "eligible": 1,
                }
            )
    bars = assign_bars_available_at(pd.DataFrame(rows))
    labels = build_labels(
        bars,
        universe_symbols=symbols,
        cost_bps_per_side=2.5,
        sell_fee_bps=0.3,
    )
    assert not labels.empty
    assert_label_after_feature(labels)
    assert (labels["label_window_start"] > labels["feature_time_end"]).all()
    # Entry is next session after decision.
    for _, row in labels.head(20).iterrows():
        sess = sessions
        i = sess.index(row["date"])
        assert row["entry_date"] == sess[i + 1]
        assert row["exit_date"] == sess[i + 4]


def test_per_symbol_multiplier_scales_cost_bps_only() -> None:
    sessions = make_sessions(30, start=date(2019, 6, 3))
    symbols = [f"S{i:02d}" for i in range(13)]
    rows = []
    for sym in symbols:
        for d in sessions:
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "adj_open": 100.0,
                    "adj_close": 100.0,
                    "eligible": 1,
                }
            )
    bars = pd.DataFrame(rows)
    labels = build_labels(
        bars,
        universe_symbols=symbols,
        cost_bps_per_side=2.5,
        sell_fee_bps=0.3,
        per_symbol_cost_multiplier={"S00": 2.0, **{s: 1.0 for s in symbols[1:]}},
    )
    s0 = labels.loc[labels["symbol"] == "S00"].iloc[0]
    s1 = labels.loc[labels["symbol"] == "S01"].iloc[0]
    assert s0["roundtrip_cost_dec"] == pytest.approx(roundtrip_cost_dec(5.0, 0.3))
    assert s1["roundtrip_cost_dec"] == pytest.approx(roundtrip_cost_dec(2.5, 0.3))


def test_y_only_on_eligible_rows() -> None:
    sessions = make_sessions(25, start=date(2019, 3, 1))
    symbols = [f"S{i:02d}" for i in range(13)]
    rows = []
    for sym in symbols:
        for d in sessions:
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "adj_open": 100.0,
                    "adj_close": 100.0,
                    "eligible": 1 if sym == "S00" else 0,
                }
            )
    labels = build_labels(pd.DataFrame(rows), universe_symbols=symbols)
    assert labels.loc[labels["symbol"] == "S00", "y"].notna().all()
    assert labels.loc[labels["symbol"] != "S00", "y"].isna().all()
