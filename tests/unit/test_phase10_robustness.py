"""Phase 10: placebo transforms, empirical p, robustness data writers."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from jevlab.cv.walkforward import make_folds
from jevlab.evaluation.placebo import (
    PLACEBO_N,
    ar1_matched_noise,
    circular_shift_within_asset,
    empirical_pvalue,
    fit_ar1,
    global_shuffle,
    run_p5_projection,
    run_placebo_loop,
    train_b_oos,
    within_date_shuffle,
    write_placebo_results,
)
from jevlab.evaluation.robustness import (
    feature_ablation,
    paraphrase_sensitivity_plan,
    run_robustness_suite,
    write_robustness_report,
)
from jevlab.experiments.manifest import insert_trial_ledger
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.audit import build_p6_shuffled_states, plan_p6_state_shuffle, run_p6_state_shuffle
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import column_set_registry
from jevlab.models.pipelines import run_column_set_logistic
from jevlab.utils.calendar import nyse_session_dates
from jevlab.utils.registry import init_registry


def _synthetic_panel(
    *,
    start: date = date(2016, 1, 4),
    end: date = date(2019, 6, 28),
    n_symbols: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = nyse_session_dates(start, end)
    symbols = [f"E{i:02d}" for i in range(n_symbols)]
    rows: list[dict[str, object]] = []
    for d in sessions:
        for i, sym in enumerate(symbols):
            noise = rng.normal(0, 1, size=len(MODEL_A_COLUMNS))
            feats = {c: float(noise[j]) for j, c in enumerate(MODEL_A_COLUMNS)}
            # Mild persistence in Jev for autocorrelation tests
            base = 0.5 + 0.05 * np.sin(i + d.toordinal() / 40.0)
            jev = {
                "j_dislocation": float(np.clip(base + rng.normal(0, 0.05), 0.05, 0.95)),
                "j_downside_E": float(1.0 + 0.1 * rng.normal()),
                "j_downside_p3": float(np.clip(base + rng.normal(0, 0.05), 0.05, 0.95)),
                "j_regime": float(np.clip(0.4 + 0.1 * rng.normal(), 0.05, 0.95)),
                "j_pullback_p": float(np.clip(base + rng.normal(0, 0.05), 0.05, 0.95)),
                "j_breakdown_p": float(np.clip(0.3 + 0.1 * rng.normal(), 0.05, 0.95)),
                "j_sideways_p": float(np.clip(0.35 + 0.1 * rng.normal(), 0.05, 0.95)),
                "j_setup_conf": float(np.clip(base + rng.normal(0, 0.05), 0.05, 0.95)),
            }
            feats["spy_px_vs_sma200"] = float(rng.normal(0, 0.02))
            feats["vix_lag1_pct_252"] = float(rng.uniform(0, 1))
            signal = feats["ret_1d"] + 0.4 * jev["j_dislocation"]
            y = 1 if (signal + rng.normal(0, 0.5)) > 0 else 0
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    **feats,
                    **jev,
                    "eligible": 1,
                    "y": y,
                    "X": float(signal),
                    "R": float(signal * 0.01),
                    "Rbar": 0.0,
                    "y_cost": y,
                }
            )
    return pd.DataFrame(rows)


def test_placebo_n_defaults() -> None:
    assert PLACEBO_N["P1"] == 1000
    assert PLACEBO_N["P2"] == 500
    assert PLACEBO_N["P3"] == 500
    assert PLACEBO_N["P4"] == 500
    assert PLACEBO_N["P5"] == 1
    assert PLACEBO_N["P6"] == 1


def test_empirical_pvalue_formula() -> None:
    # observed = 2; placebos = [0,1,2,3] => #{>=2}=2 => (1+2)/(1+4)=0.6
    assert empirical_pvalue(2.0, [0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.6)
    assert empirical_pvalue(10.0, [0.0, 1.0, 2.0]) == pytest.approx(1 / 4)


def test_within_date_and_global_preserve_marginals() -> None:
    frame = _synthetic_panel(seed=1)
    col = "j_dislocation"
    orig = frame[col].to_numpy()
    p1 = within_date_shuffle(frame, seed=7)
    p2 = global_shuffle(frame, seed=7)
    assert np.sort(p1[col].to_numpy()) == pytest.approx(np.sort(orig))
    assert np.sort(p2[col].to_numpy()) == pytest.approx(np.sort(orig))
    # Market columns untouched
    assert np.allclose(p1["ret_1d"].to_numpy(), frame["ret_1d"].to_numpy())
    assert np.allclose(p2["ret_1d"].to_numpy(), frame["ret_1d"].to_numpy())


def test_circular_shift_preserves_autocorrelation() -> None:
    frame = _synthetic_panel(seed=2, end=date(2018, 12, 31))
    col = "j_pullback_p"
    # Focus on one asset with enough length
    sym = frame["symbol"].iloc[0]
    series = frame.loc[frame["symbol"] == sym, col].to_numpy(dtype=float)
    phi0, _ = fit_ar1(series)
    shifted = circular_shift_within_asset(frame, seed=3, min_lag=5)
    series_s = shifted.loc[shifted["symbol"] == sym, col].to_numpy(dtype=float)
    phi1, _ = fit_ar1(series_s)
    # Circular shift preserves lag-1 autocorrelation exactly for long series
    assert phi1 == pytest.approx(phi0, abs=0.15)
    assert np.sort(series_s) == pytest.approx(np.sort(series))


def test_ar1_matched_noise_seed_determinism() -> None:
    frame = _synthetic_panel(seed=3, end=date(2017, 12, 29))
    a = ar1_matched_noise(frame, seed=11)
    b = ar1_matched_noise(frame, seed=11)
    c = ar1_matched_noise(frame, seed=12)
    assert np.allclose(
        a[list(JEV_FEATURE_COLUMNS)].to_numpy(), b[list(JEV_FEATURE_COLUMNS)].to_numpy()
    )
    assert not np.allclose(
        a[list(JEV_FEATURE_COLUMNS)].to_numpy(), c[list(JEV_FEATURE_COLUMNS)].to_numpy()
    )


def test_shuffle_seed_determinism() -> None:
    frame = _synthetic_panel(seed=4, end=date(2017, 6, 30))
    a = within_date_shuffle(frame, seed=0)
    b = within_date_shuffle(frame, seed=0)
    c = within_date_shuffle(frame, seed=1)
    assert a.equals(b)
    assert not a.equals(c)


def test_placebo_loop_small_n_and_runtime_budget() -> None:
    panel = _synthetic_panel(seed=5)
    sessions = sorted(panel["date"].unique().tolist())
    folds = make_folds(
        sessions,
        first_train_end=date(2018, 6, 29),
        oos_start=date(2018, 7, 2),
        oos_end=date(2019, 6, 28),
        embargo_sessions=5,
    )
    assert folds
    C_grid = [0.1, 1.0]
    registry = column_set_registry()
    preds_a = run_column_set_logistic(
        panel,
        folds,
        registry["A"],
        sessions=sessions,
        C_grid=C_grid,
        seed=0,
    ).predictions
    t0 = time.perf_counter()
    result = run_placebo_loop(
        panel,
        folds,
        kind="P2",
        preds_a=preds_a,
        sessions=sessions,
        C_grid=C_grid,
        n=3,
        seed=0,
        compute_sharpe=False,
    )
    elapsed = time.perf_counter() - t0
    assert result.n == 3
    assert len(result.placebo_delta_log_loss_gains) == 3
    assert 0.0 <= result.pvalue_log_loss <= 1.0
    # Small fixture must stay well under a loose laptop budget.
    assert elapsed < 120.0


def test_p5_projection_runs() -> None:
    panel = _synthetic_panel(seed=6)
    sessions = sorted(panel["date"].unique().tolist())
    folds = make_folds(
        sessions,
        first_train_end=date(2018, 6, 29),
        oos_start=date(2018, 7, 2),
        oos_end=date(2019, 6, 28),
    )
    registry = column_set_registry()
    preds_a = run_column_set_logistic(
        panel, folds, registry["A"], sessions=sessions, C_grid=[1.0], seed=0
    ).predictions
    p5 = run_p5_projection(panel, folds, preds_a=preds_a, sessions=sessions, C_grid=[1.0], seed=0)
    assert np.isfinite(p5.observed_gain)
    assert np.isfinite(p5.predicted_only_gain)
    assert np.isfinite(p5.residual_only_gain)


def test_p6_dry_and_budget_plan() -> None:
    panel = _synthetic_panel(seed=7, end=date(2018, 12, 31))
    plan = plan_p6_state_shuffle(n_states=100, budget_usd=1.0)
    assert plan.n_states == 100
    assert plan.estimated_usd > 0
    shuffled = build_p6_shuffled_states(panel, n_states=50, seed=0)
    assert len(shuffled) == 50
    assert "donor_row_index" in shuffled.columns
    dry = run_p6_state_shuffle(panel, live=False, n_states=40, seed=1)
    assert dry["status"] == "dry_plan"
    assert dry["live"] is False


def test_robustness_writer_and_ablation(tmp_path: Path) -> None:
    panel = _synthetic_panel(seed=8)
    sessions = sorted(panel["date"].unique().tolist())
    folds = make_folds(
        sessions,
        first_train_end=date(2018, 6, 29),
        oos_start=date(2018, 7, 2),
        oos_end=date(2019, 6, 28),
    )
    registry = column_set_registry()
    preds_a = run_column_set_logistic(
        panel, folds, registry["A"], sessions=sessions, C_grid=[1.0], seed=0
    ).predictions
    preds_b = train_b_oos(panel, folds, sessions=sessions, C_grid=[1.0], seed=0)
    ab = feature_ablation(panel, folds, preds_a=preds_a, sessions=sessions, C_grid=[1.0], seed=0)
    assert len(ab) == len(MODEL_A_COLUMNS)
    bundle = run_robustness_suite(
        panel,
        folds,
        preds_a=preds_a,
        preds_b=preds_b,
        sessions=sessions,
        C_grid=[1.0],
        seed=0,
        include_ablations=False,
        include_loo=False,
    )
    assert bundle.paraphrase[0]["n_paraphrases_per_question"] == 2
    assert paraphrase_sensitivity_plan()["requires_live"] is True
    path = write_robustness_report(bundle, tmp_path / "robustness_report")
    assert path.is_file()
    text_body = path.read_text(encoding="utf-8")
    assert "narrative_conclusions" in text_body
    assert "null" in text_body.lower() or "None" in text_body
    # No verdict language required absent — just data.
    assert "Jev does not" not in text_body

    placebo_path = write_placebo_results(
        {"P2": {"n": 3, "pvalue_log_loss": 0.5}, "narrative_conclusions": None},
        tmp_path / "robustness_report",
    )
    assert placebo_path.is_file()

    reg = tmp_path / "experiments.sqlite"
    init_registry(reg)
    tid = insert_trial_ledger(
        reg,
        purpose="placebo",
        feature_set="placebo_P2",
        model_class="logistic_B",
        grid={"n": 3},
        experiment_id="t10",
        result_summary="{}",
    )
    engine = init_registry(reg)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT purpose FROM trial_ledger WHERE trial_id = :t"),
            {"t": tid},
        ).fetchone()
    assert row is not None
    assert row[0] == "placebo"
