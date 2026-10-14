"""Unit tests for live Phase 6 audit helpers (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jevlab.jev.audit import (
    degeneracy_check,
    determinism_stats,
    plan_audit_calls,
    summarize_numeric_series,
)
from jevlab.jev.audit_live import sample_d2_states, update_disposition_factual


def test_plan_includes_d5_fine_and_stays_under_default_budget() -> None:
    plan = plan_audit_calls()
    assert plan["calls_by_section"]["d5_fine"] == 3000
    assert plan["calls_by_section"]["d5_coarse"] == 3000
    assert plan["total_calls"] == 300 + 600 + 600 + 2000 + 3000 + 3000
    assert plan["estimated_cost_usd"] < 25.0


def test_determinism_stats_reports_median_and_threshold() -> None:
    n = 20
    a = pd.DataFrame({"j_dislocation": np.full(n, 0.5), "j_regime": np.linspace(0.1, 0.9, n)})
    b = pd.DataFrame(
        {"j_dislocation": np.full(n, 0.5), "j_regime": np.linspace(0.1, 0.9, n) + 0.001}
    )
    c = pd.DataFrame(
        {"j_dislocation": np.full(n, 0.5), "j_regime": np.linspace(0.1, 0.9, n) - 0.001}
    )
    stats = determinism_stats(
        [a, b, c],
        ["j_dislocation", "j_regime"],
        noul_cols=["j_dislocation", "j_regime"],
    )
    assert "median_within_state_sd" in stats["features"]["j_dislocation"]
    assert stats["mitigation_triggered"] is False
    # Large within-state SD triggers mitigation.
    noisy = [
        pd.DataFrame({"j_dislocation": np.zeros(n), "j_regime": np.zeros(n)}),
        pd.DataFrame({"j_dislocation": np.ones(n), "j_regime": np.zeros(n)}),
        pd.DataFrame({"j_dislocation": np.full(n, 0.5), "j_regime": np.zeros(n)}),
    ]
    hot = determinism_stats(noisy, ["j_dislocation", "j_regime"], noul_cols=["j_dislocation"])
    assert hot["mitigation_triggered"] is True


def test_sample_d2_disjoint_from_dev() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-01-04", "2018-12-31").date
    rows = []
    for sym in ("AAA", "BBB"):
        for d in dates:
            rows.append({"symbol": sym, "date": d, "ret_1d": float(rng.normal())})
    feats = pd.DataFrame(rows)
    dev = feats.iloc[:50].copy()
    d2 = sample_d2_states(feats, n=100, seed=1, exclude=dev)
    keys_dev = set(zip(dev["symbol"], dev["date"], strict=True))
    keys_d2 = set(zip(d2["symbol"], d2["date"], strict=True))
    assert keys_dev.isdisjoint(keys_d2)
    assert len(d2) == 100


def test_disposition_factual_never_checks_decision(tmp_path) -> None:
    path = tmp_path / "contamination_disposition.md"
    path.write_text(
        "# Contamination disposition (HUMAN COMPLETE — do not auto-fill)\n\n"
        "- [ ] `proceed`\n- [ ] `proceed_with_caveats`\n- [ ] `stop`\n",
        encoding="utf-8",
    )
    update_disposition_factual(
        path,
        report_hash="abc",
        audit_report_path="reports/audit/AUDIT_REPORT.md",
        audit_qset_version="audit_qset_v1",
        audit_qset_hash="h1",
        research_qset_hash="h2",
        live_calls=True,
        warnings=["D1_WARNING_SIGNAL"],
        diagnostics=["D1", "D2"],
    )
    text = path.read_text(encoding="utf-8")
    assert "- [ ] `proceed`" in text
    assert "- [x]" not in text
    assert "abc" in text
    assert "D1_WARNING_SIGNAL" in text


def test_summarize_and_degeneracy_smoke() -> None:
    s = summarize_numeric_series([0.1, 0.2, 0.9, 0.4])
    assert s["n"] == 4
    assert s["min"] == 0.1
    deg = degeneracy_check({"dislocation": [0.2, 0.5, 0.8]}, {"setup": [{"a": 0.5, "b": 0.5}] * 5})
    assert deg["flags"] == []
