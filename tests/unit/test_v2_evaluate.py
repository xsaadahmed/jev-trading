"""Real V2-7 evaluator: fail-closed metrics, concentration, oracle, no caller gates."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pandas as pd
import pytest

from jevlab.experiments.artifacts import ArtifactError
from jevlab.v2.acceptance import (
    V2GateMetrics,
    concentration_from_trades,
    evaluate_e5,
    evaluate_s2,
    evaluate_s3,
    evaluate_v2_gates,
    passing_fixture_metrics,
)
from jevlab.v2.artifacts import write_v2_sealed_pass
from jevlab.v2.evaluate import evaluate_v2_study, oracle_from_artifacts
from jevlab.v2.rehearsal import synthetic_pre2024_payload
from jevlab.v2.sealed_run import run_v2_final_test
from jevlab.v2.statistics import e5_pass


def test_production_run_v2_final_test_rejects_caller_metrics() -> None:
    sig = inspect.signature(run_v2_final_test)
    assert "metrics" not in sig.parameters


def test_default_gate_metrics_cannot_write_pass(tmp_path: Path) -> None:
    report = evaluate_v2_gates(V2GateMetrics())
    assert report.all_passed is False
    with pytest.raises(ArtifactError):
        write_v2_sealed_pass(
            report,
            out_dir=tmp_path,
            git_commit="t",
            sealed_result_digest="x",
            training_artifact_digest="y",
            random_null_artifact_digest="z",
        )
    assert not (tmp_path / "V2_SEALED_PASS.json").exists()


def test_e5_sign_and_magnitude() -> None:
    assert e5_pass(-0.35) is True
    assert evaluate_e5(V2GateMetrics(max_drawdown_dec=-0.35)).passed is True
    assert e5_pass(-0.3500001) is False
    assert evaluate_e5(V2GateMetrics(max_drawdown_dec=-0.3500001)).passed is False
    assert e5_pass(-0.80) is False
    assert evaluate_e5(V2GateMetrics(max_drawdown_dec=-0.80)).passed is False


def _trades(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
    """rows: (cohort, symbol, notional, side) with buy then sell per cohort."""
    out = []
    for cohort, symbol, notional, side in rows:
        out.append(
            {
                "cohort_id": cohort,
                "symbol": symbol,
                "notional_usd": notional,
                "side": side,
                "fee_usd": 0.0,
                "session": "2020-01-02",
            }
        )
    return pd.DataFrame(out)


def test_concentration_uses_positive_completed_trade_pnl() -> None:
    trades = _trades(
        [
            ("a", "AAA", 100.0, "buy"),
            ("a", "AAA", 150.0, "sell"),  # +50
            ("b", "BBB", 100.0, "buy"),
            ("b", "BBB", 130.0, "sell"),  # +30
            ("c", "CCC", 100.0, "buy"),
            ("c", "CCC", 110.0, "sell"),  # +10
            ("d", "DDD", 100.0, "buy"),
            ("d", "DDD", 90.0, "sell"),  # -10 ignored
            ("e", "EEE", 100.0, "buy"),
            ("e", "EEE", 105.0, "sell"),  # +5
            ("f", "FFF", 100.0, "buy"),
            ("f", "FFF", 104.0, "sell"),  # +4
            ("g", "GGG", 100.0, "buy"),
            ("g", "GGG", 103.0, "sell"),  # +3
        ]
    )
    conc = concentration_from_trades(trades)
    assert conc["positive_pnl"] == pytest.approx(50 + 30 + 10 + 5 + 4 + 3)
    assert conc["top5_profitable_share"] == pytest.approx((50 + 30 + 10 + 5 + 4) / 102.0)
    assert conc["max_etf_positive_share"] == pytest.approx(50.0 / 102.0)
    even = _trades(
        [(str(i), f"S{i}", 100.0, "buy") for i in range(21)]
        + [(str(i), f"S{i}", 110.0, "sell") for i in range(21)]
    )
    even_c = concentration_from_trades(even)
    assert even_c["positive_pnl"] == pytest.approx(210.0)
    assert even_c["top5_profitable_share"] == pytest.approx(50.0 / 210.0)
    assert even_c["max_etf_positive_share"] == pytest.approx(10.0 / 210.0)
    metrics = passing_fixture_metrics().with_overrides(**even_c)
    assert evaluate_s2(metrics).passed is True
    assert evaluate_s3(metrics).passed is True


def test_concentration_zero_positive_pnl_fails_closed() -> None:
    trades = _trades(
        [
            ("a", "AAA", 100.0, "buy"),
            ("a", "AAA", 90.0, "sell"),
        ]
    )
    conc = concentration_from_trades(trades)
    assert conc["positive_pnl"] == 0.0
    assert math.isnan(conc["top5_profitable_share"])
    metrics = passing_fixture_metrics().with_overrides(**conc)
    assert evaluate_s2(metrics).passed is False
    assert evaluate_s3(metrics).passed is False


def test_evaluate_v2_study_oracle_and_no_2024_train() -> None:
    payload = synthetic_pre2024_payload(n_sessions=48, n_symbols=5, seed=9)
    result = evaluate_v2_study(
        features=payload["features"],
        labels=payload["labels"],
        bars=payload["bars"],
        t0=payload["t0"],
        n=6,
        allow_reduced_n=True,
        allow_sealed_dates=False,
        label="development_rehearsal_only",
        run_robustness=False,
        check_fast_first_null=False,
    )
    assert result.fitted.fit_count == 1
    assert result.details["n_2024_train"] == 0
    assert result.details["n_2025_train"] == 0
    assert result.details["dropped_draws"] == 0
    assert len(result.null_stats) == 6
    oracle = oracle_from_artifacts(result.details, result.null_stats, result.observed_mean_sharpe)
    assert oracle["p_value"] == pytest.approx(result.test.p_value)
    assert oracle["sharpe_advantage"] == pytest.approx(result.test.advantage)
    assert oracle["null_median"] == pytest.approx(result.details["null_median"])
    assert oracle["phase_wins"] == result.metrics.phases_beat_null_median
    assert result.label == "development_rehearsal_only"
    assert result.to_public_dict()["v2_confirmatory_evidence"] is False


def test_production_path_rehearsal_small_n(tmp_path: Path) -> None:
    from jevlab.v2.rehearsal import REHEARSAL_LABEL, run_production_path_rehearsal

    out = run_production_path_rehearsal(
        n=6,
        repeat=True,
        n_sessions=290,
        n_symbols=4,
        n_score=16,
        seed=11,
        run_robustness=False,
        sandbox_root=tmp_path / "sandbox",
    )
    assert out.ok
    assert out.label == REHEARSAL_LABEL
    assert out.details["n"] == 6
    assert out.details["n_executed"] == 6
    assert out.details["dropped_draws"] == 0 if "dropped_draws" in out.details else True
    assert out.details["deterministic_repeat"] is True
    assert out.details["oracle_gate_comparison"] is True
    assert out.details["acceptance_written"] is False
    assert out.details["n_2024_train"] == 0
    assert out.details["n_2025_train"] == 0
    assert out.details["fit_count"] == 1
    assert out.details["v2_confirmatory_evidence"] is False


def test_cli_source_does_not_import_caller_gate_metrics() -> None:
    from jevlab.cli import v2 as v2_cli

    source = Path(v2_cli.__file__).read_text(encoding="utf-8")
    assert "from jevlab.v2.acceptance import V2GateMetrics" not in source
    assert "metrics=V2GateMetrics" not in source
    assert "metrics=" not in source
