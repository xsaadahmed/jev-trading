"""Phase 6 audit diagnostics tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.audit import (
    ERA_BOUNDS,
    BudgetExceededError,
    BudgetGuard,
    SealedPartitionError,
    assign_era,
    degeneracy_check,
    exact_binomial_pvalue_ge,
    logistic_outcome_benchmark,
    min_k_for_pvalue,
    multinomial_era_benchmark,
    plan_audit_calls,
    refuse_sealed_path,
    run_audit_dry,
    write_audit_report,
)


def test_exact_binomial_orientation_values() -> None:
    # Orientation from §11.4: ~227/600 for p<0.01 under p=1/3
    k_star = min_k_for_pvalue(600, 1.0 / 3.0, 0.01)
    assert 220 <= k_star <= 235
    assert exact_binomial_pvalue_ge(k_star, 600, 1.0 / 3.0) < 0.01
    assert exact_binomial_pvalue_ge(k_star - 1, 600, 1.0 / 3.0) >= 0.01
    # Known small case: Binom(10, 0.5), P(X>=8)
    assert exact_binomial_pvalue_ge(8, 10, 0.5) == pytest.approx(0.0546875)


def test_era_boundary_assignment() -> None:
    assert assign_era(date(2016, 1, 4)) == "E1"
    assert assign_era(date(2018, 6, 29)) == "E1"
    assert assign_era(date(2018, 7, 2)) == "E2"
    assert assign_era(date(2021, 3, 31)) == "E2"
    assert assign_era(date(2021, 4, 1)) == "E3"
    assert assign_era(date(2023, 12, 29)) == "E3"
    # Gap between E1 and E2
    assert assign_era(date(2018, 7, 1)) is None
    assert assign_era(date(2024, 1, 2)) is None
    assert ERA_BOUNDS["E1"][0] == date(2016, 1, 4)


def test_benchmark_chance_vs_strong_signal() -> None:
    rng = np.random.default_rng(0)
    n = 300
    # Chance: features independent of label
    X_chance = rng.normal(size=(n, len(MODEL_A_COLUMNS)))
    y_chance = rng.integers(0, 3, size=n)
    chance = multinomial_era_benchmark(X_chance, y_chance, n_splits=3, seed=0)
    # Strong signal: feature 0 encodes class
    X_strong = rng.normal(size=(n, len(MODEL_A_COLUMNS)))
    y_strong = rng.integers(0, 3, size=n)
    X_strong[:, 0] = y_strong.astype(float) * 5.0
    strong = multinomial_era_benchmark(X_strong, y_strong, n_splits=3, seed=0)
    assert strong["accuracy"] > chance["accuracy"]
    assert strong["accuracy"] > 0.6

    # Binary logistic chance vs signal
    yb = rng.integers(0, 2, size=n)
    Xb = rng.normal(size=(n, len(MODEL_A_COLUMNS)))
    weak = logistic_outcome_benchmark(Xb, yb, seed=0)
    Xs = Xb.copy()
    Xs[:, 0] = yb.astype(float) * 4.0
    good = logistic_outcome_benchmark(Xs, yb, seed=0)
    assert good["auc"] > weak["auc"] or (good["auc"] > 0.7)


def test_sealed_path_and_future_dates_refused(tmp_path: Path) -> None:
    sealed = tmp_path / "data" / "sealed"
    sealed.mkdir(parents=True)
    with pytest.raises(SealedPartitionError):
        refuse_sealed_path(sealed / "part.parquet")
    from jevlab.jev.audit import assert_audit_dates_allowed, refuse_sealed_read

    with pytest.raises(SealedPartitionError):
        refuse_sealed_read(sealed / "x.parquet", sealed)
    # Non-sealed path is fine
    refuse_sealed_read(tmp_path / "features" / "x.parquet", sealed)
    with pytest.raises(SealedPartitionError):
        assert_audit_dates_allowed([date(2024, 1, 2)])


def test_dry_run_makes_zero_calls(tmp_path: Path) -> None:
    calls = {"n": 0}

    class Boom:
        def infer(self, *_a: object, **_k: object) -> None:
            calls["n"] += 1
            raise AssertionError("dry-run must not call Jev")

    _ = Boom  # document intent; run_audit_dry never constructs a client
    report = run_audit_dry(None, sealed_root=None)
    assert report.meta["live"] is False
    assert report.meta["api_calls"] == 0
    assert calls["n"] == 0
    md = write_audit_report(report, tmp_path / "audit")
    text = md.read_text(encoding="utf-8")
    assert "warning signal, not proof" in text
    assert "potentially training-contaminated" in text


def test_budget_cap() -> None:
    guard = BudgetGuard(budget_usd=0.01, tokens_per_call=800)
    plan = plan_audit_calls()
    with pytest.raises(BudgetExceededError):
        guard.plan(int(plan["total_calls"]))
    tiny = BudgetGuard(budget_usd=100.0, tokens_per_call=10)
    cost = tiny.plan(10)
    assert cost > 0
    with pytest.raises(BudgetExceededError):
        BudgetGuard(budget_usd=0.0000001, tokens_per_call=1_000_000).record_usage(1_000_000)


def test_degeneracy_flags() -> None:
    ok = degeneracy_check(
        {"dislocation": [0.2, 0.5, 0.8, 0.4, 0.6]},
        {"setup": [{"a": 0.4, "b": 0.6}] * 10},
    )
    assert ok["flags"] == []
    bad = degeneracy_check(
        {"dislocation": [0.5] * 20},
        {"setup": [{"a": 0.99, "b": 0.01}] * 20},
    )
    assert any("noul_degenerate" in f for f in bad["flags"])
    assert any("choice_degenerate" in f for f in bad["flags"])


def test_disposition_template_has_no_autofilled_decision() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "contamination_disposition.md").read_text(encoding="utf-8")
    assert "HUMAN COMPLETE" in text
    assert "do not auto-fill" in text.lower() or "never auto-fill" in text.lower()
    # Decision checkboxes present but unchecked
    assert "- [ ] `proceed`" in text
    assert "- [ ] `stop`" in text
