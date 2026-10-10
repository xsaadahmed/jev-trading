"""Phase 8: column sets, A/B identity, ledger, manifest, sealed refusal."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from jevlab.cv.sealed_guard import SealedAccessError
from jevlab.cv.walkforward import make_folds
from jevlab.experiments.manifest import (
    assert_manifest_complete,
    insert_trial_ledger,
    manifest_completeness_keys,
    write_manifest,
)
from jevlab.experiments.runner import (
    build_comparison_table,
    comparison_table_markdown,
    run_model_set,
)
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.models.column_sets import (
    CONTAMINATION_LABEL,
    MODEL_B_COLUMNS,
    MODEL_C_COLUMNS,
    MODEL_E_COLUMNS,
    MODEL_F_COLUMNS,
    PLACEBO_JEV_COLUMNS,
    assert_ab_column_diff,
    column_set_registry,
)
from jevlab.models.controls import (
    expected_g_onehot_columns,
    model_g_columns,
)
from jevlab.models.pipelines import (
    assert_identical_folds,
    prepare_model_frame,
    run_lineup_walkforward,
)
from jevlab.models.transforms import logit_clip, make_lineup_logistic_pipeline
from jevlab.utils.calendar import nyse_session_dates
from jevlab.utils.registry import init_registry


def _synthetic_lineup_panel(
    *,
    start: date = date(2016, 1, 4),
    end: date = date(2019, 6, 28),
    n_symbols: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = nyse_session_dates(start, end)
    symbols = [f"E{i:02d}" for i in range(n_symbols)]
    rows: list[dict[str, object]] = []
    for d in sessions:
        for sym in symbols:
            noise = rng.normal(0, 1, size=len(MODEL_A_COLUMNS))
            feats = {c: float(noise[i]) for i, c in enumerate(MODEL_A_COLUMNS)}
            jev = {
                "j_dislocation": float(rng.uniform(0.05, 0.95)),
                "j_downside_E": float(rng.uniform(0.5, 3.5)),
                "j_downside_p3": float(rng.uniform(0.05, 0.95)),
                "j_regime": float(rng.uniform(0.05, 0.95)),
                "j_pullback_p": float(rng.uniform(0.05, 0.95)),
                "j_breakdown_p": float(rng.uniform(0.05, 0.95)),
                "j_sideways_p": float(rng.uniform(0.05, 0.95)),
                "j_setup_conf": float(rng.uniform(0.05, 0.95)),
            }
            signal = feats["ret_1d"] + 0.3 * jev["j_dislocation"]
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


def test_asserted_column_counts() -> None:
    assert_ab_column_diff()
    assert len(MODEL_A_COLUMNS) == 17
    assert len(MODEL_B_COLUMNS) == 25
    assert len(MODEL_C_COLUMNS) == 8
    assert len(MODEL_E_COLUMNS) == 25
    assert len(MODEL_F_COLUMNS) == 25
    assert len(JEV_FEATURE_COLUMNS) == 8
    assert len(PLACEBO_JEV_COLUMNS) == 8
    assert MODEL_B_COLUMNS[:17] == MODEL_A_COLUMNS
    assert set(MODEL_B_COLUMNS) - set(MODEL_A_COLUMNS) == set(JEV_FEATURE_COLUMNS)
    g = model_g_columns()
    assert g[:17] == MODEL_A_COLUMNS
    n_oh = len(expected_g_onehot_columns())
    assert len(g) == 17 + n_oh
    assert n_oh > 0
    reg = column_set_registry(model_g_columns=g)
    assert reg["A"].n_columns == 17
    assert reg["B"].n_columns == 25
    assert reg["F"].interface_only is True
    assert reg["A"].purpose == "confirmatory"
    assert reg["B"].purpose == "confirmatory"
    assert reg["C"].purpose == "exploratory"


def test_ab_column_diff_assertion() -> None:
    assert_ab_column_diff()
    with pytest.raises(AssertionError):
        # Mutate a copy path indirectly by checking inequality
        assert set(MODEL_B_COLUMNS) - set(MODEL_A_COLUMNS) == {"not_a_feature"}


def test_logit_clip_bounds() -> None:
    out = logit_clip(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(out).all()
    assert out[1] == pytest.approx(0.0)
    pipe = make_lineup_logistic_pipeline(
        C=1.0,
        seed=0,
        columns=("a", "j_dislocation"),
        logit_columns=("j_dislocation",),
    )
    X = np.array([[0.1, 0.2], [0.3, 0.8], [0.0, 0.5], [-0.1, 0.9]] * 10)
    y = np.array([0, 1, 0, 1] * 10)
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert len(p) == len(y)


def test_identical_fold_indices() -> None:
    panel = _synthetic_lineup_panel()
    sessions = sorted(panel["date"].unique().tolist())
    folds_a = make_folds(
        sessions,
        first_train_end=date(2018, 6, 29),
        oos_start=date(2018, 7, 2),
        oos_end=date(2019, 6, 28),
        embargo_sessions=5,
    )
    folds_b = make_folds(
        sessions,
        first_train_end=date(2018, 6, 29),
        oos_start=date(2018, 7, 2),
        oos_end=date(2019, 6, 28),
        embargo_sessions=5,
    )
    assert_identical_folds(folds_a, folds_b)
    assert len(folds_a) >= 1


def test_lineup_ab_share_folds_and_row_set(tmp_path: Path) -> None:
    panel = _synthetic_lineup_panel(seed=1)
    # Attach tiny goh columns so G can run without full one-hot build
    panel["goh_dummy_b1"] = 0.0
    frame = prepare_model_frame(
        panel, panel, extra_columns=[*list(JEV_FEATURE_COLUMNS), "goh_dummy_b1"]
    )
    sessions = sorted(frame["date"].unique().tolist())
    folds = make_folds(
        sessions,
        first_train_end=date(2018, 6, 29),
        oos_start=date(2018, 7, 2),
        oos_end=date(2019, 6, 28),
    )
    g_cols = (*MODEL_A_COLUMNS, "goh_dummy_b1")
    registry = column_set_registry(model_g_columns=g_cols)
    results = run_lineup_walkforward(
        frame,
        folds,
        registry,  # type: ignore[arg-type]
        sessions=sessions,
        C_grid=[0.1, 1.0],
        seed=42,
        model_ids=["A", "B"],
    )
    assert set(results) == {"A", "B"}
    # Same fold_ids present in both reports
    fa = {r["fold_id"] for r in results["A"].fold_report if "fold_id" in r}
    fb = {r["fold_id"] for r in results["B"].fold_report if "fold_id" in r}
    assert fa == fb
    # Predictions share the same (symbol, date) keys when both succeed
    if not results["A"].predictions.empty and not results["B"].predictions.empty:
        keys_a = set(
            zip(
                results["A"].predictions["symbol"],
                results["A"].predictions["date"],
                strict=True,
            )
        )
        keys_b = set(
            zip(
                results["B"].predictions["symbol"],
                results["B"].predictions["date"],
                strict=True,
            )
        )
        assert keys_a == keys_b


def test_manifest_completeness_and_ledger(tmp_path: Path) -> None:
    reg = tmp_path / "experiments.sqlite"
    init_registry(reg)
    payload = {
        "experiment_id": "exp-test",
        "models": ["A", "B"],
        "folds": [{"fold_id": "2019Q1", "n_train": 10, "n_test": 5}],
        "C_grid": [0.01, 0.1, 1.0],
        "seed": 42,
        "contamination_label": CONTAMINATION_LABEL,
        "potentially_training_contaminated": True,
        "column_counts": {"A": 17, "B": 25},
        "purpose_by_model": {"A": "confirmatory", "B": "confirmatory"},
        "metrics_by_model": {"A": {"log_loss": 0.7}, "B": {"log_loss": 0.69}},
    }
    assert_manifest_complete(payload)
    path = write_manifest(payload, tmp_path / "out")
    assert path.is_file()
    text_body = path.read_text(encoding="utf-8")
    assert CONTAMINATION_LABEL in text_body

    tid = insert_trial_ledger(
        reg,
        purpose="confirmatory",
        feature_set="A_vs_B",
        model_class="logistic",
        grid={"C_grid": [0.1]},
        experiment_id="exp-test",
        result_summary="{}",
    )
    engine = init_registry(reg)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT purpose, feature_set FROM trial_ledger WHERE trial_id = :t"),
            {"t": tid},
        ).fetchone()
    assert row is not None
    assert row[0] == "confirmatory"
    assert row[1] == "A_vs_B"
    assert "experiment_id" in manifest_completeness_keys()


def test_no_sealed_access_in_prepare() -> None:
    panel = _synthetic_lineup_panel(end=date(2018, 12, 31))
    sealed_row = panel.iloc[[0]].copy()
    sealed_row["date"] = date(2024, 6, 3)
    from jevlab.models.pipelines import _assert_no_sealed

    with pytest.raises(SealedAccessError):
        _assert_no_sealed(sealed_row)


def test_run_model_set_end_to_end(tmp_path: Path) -> None:
    panel = _synthetic_lineup_panel(seed=2)
    # Pre-attach a couple of goh columns; skip expensive bucket build
    panel["goh_long_term_trend_b1"] = (panel["ret_1d"] > 0).astype(float)
    panel["goh_long_term_trend_b2"] = (panel["ret_1d"] <= 0).astype(float)
    reg = tmp_path / "experiments.sqlite"
    init_registry(reg)
    out = tmp_path / "lineup"
    result = run_model_set(
        panel,
        panel,
        out_dir=out,
        registry_path=reg,
        C_grid=[0.1, 1.0],
        seed=7,
        model_ids=["A", "B", "C", "D", "G"],
        build_g_onehot=False,
        run_backtests=False,
        oos_start=date(2018, 7, 2),
        oos_end=date(2019, 6, 28),
        first_train_end=date(2018, 6, 29),
    )
    assert result.manifest_path is not None and result.manifest_path.is_file()
    assert (out / "comparison_table.md").is_file()
    md = (out / "comparison_table.md").read_text(encoding="utf-8")
    assert "no conclusion language" in md.lower()
    assert CONTAMINATION_LABEL in md
    assert "A" in result.results and "B" in result.results
    assert any(t for t in result.trial_ids)
    # Confirmatory A vs B + exploratory others
    engine = init_registry(reg)
    with engine.connect() as conn:
        purposes = [r[0] for r in conn.execute(text("SELECT purpose FROM trial_ledger")).fetchall()]
    assert "confirmatory" in purposes
    assert "exploratory" in purposes
    table = build_comparison_table(result.results, registry=result.registry)
    assert "conclusion" not in comparison_table_markdown(table).lower() or (
        "no conclusion" in comparison_table_markdown(table).lower()
    )


def test_comparison_table_has_no_verdict_words() -> None:
    from jevlab.models.pipelines import WalkForwardResult

    fake = {
        "A": WalkForwardResult(
            "A",
            pd.DataFrame(),
            {"log_loss": 0.7, "auc": 0.5, "rank_ic": 0.01, "n_rows": 10},
        ),
        "B": WalkForwardResult(
            "B",
            pd.DataFrame(),
            {"log_loss": 0.68, "auc": 0.52, "rank_ic": 0.02, "n_rows": 10},
        ),
    }
    g = model_g_columns()
    reg = column_set_registry(model_g_columns=g)
    table = build_comparison_table(fake, registry=reg)
    md = comparison_table_markdown(table)
    banned = ("outperforms", "proves", "significant edge", "accept", "reject h1")
    low = md.lower()
    for word in banned:
        assert word not in low
    assert CONTAMINATION_LABEL in md
