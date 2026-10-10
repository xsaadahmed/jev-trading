"""Phase 9: bootstrap, DM, DSR, Holm/BH, MDE, predictive metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sqlalchemy import text

from jevlab.evaluation.bootstrap import (
    bootstrap_mean_difference,
    bootstrap_paired_return_deltas,
    one_sided_bootstrap_pvalue,
    stationary_bootstrap_indices,
)
from jevlab.evaluation.predictive import (
    binary_brier,
    binary_log_loss,
    calibration_bins,
    daily_rank_ic_summary,
    predictive_summary,
)
from jevlab.evaluation.stats_summary import (
    build_ab_stats_summary,
    log_confirmatory_trial,
    write_stats_summary,
)
from jevlab.evaluation.tests import (
    benjamini_hochberg_adjust,
    confirmatory_ab_tests,
    deflated_sharpe_ratio,
    diebold_mariano,
    expected_max_sharpe,
    forward_consistency_z,
    holm_adjust,
    minimum_detectable_effect,
)
from jevlab.models.column_sets import CONTAMINATION_LABEL
from jevlab.utils.registry import init_registry


def test_stationary_bootstrap_shape_and_seed() -> None:
    a = stationary_bootstrap_indices(50, mean_block=10, n_draws=20, seed=0)
    b = stationary_bootstrap_indices(50, mean_block=10, n_draws=20, seed=0)
    c = stationary_bootstrap_indices(50, mean_block=10, n_draws=20, seed=1)
    assert a.shape == (20, 50)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0 and a.max() < 50


def test_planted_effect_bootstrap_ci_coverage() -> None:
    """CI covers the true mean difference in ~95% of replications."""
    rng = np.random.default_rng(0)
    truth = 0.05
    n = 80
    n_sims = 80
    n_draws = 400
    covered = 0
    for i in range(n_sims):
        noise = rng.normal(0, 0.1, size=n)
        a = rng.normal(0, 0.1, size=n)
        b = a + truth + noise * 0.0  # exact shift + independent noise on a already
        b = a + truth + rng.normal(0, 0.02, size=n)
        ci = bootstrap_mean_difference(a, b, mean_block=5, n_draws=n_draws, seed=10 + i, level=0.95)
        if ci.ci_low <= truth <= ci.ci_high:
            covered += 1
    rate = covered / n_sims
    # Allow sampling slack around 95%
    assert 0.85 <= rate <= 1.0


def test_null_type_i_dm_and_uniform_pvalues() -> None:
    """Under the null, DM Type-I ≈ 5% over 200 sims; p-values ~uniform."""
    rng = np.random.default_rng(1)
    n = 100
    n_sims = 200
    pvals: list[float] = []
    rejects = 0
    for _ in range(n_sims):
        # Identical predictive ability: same losses + tiny iid noise
        base = rng.normal(0.7, 0.05, size=n)
        la = base + rng.normal(0, 0.01, size=n)
        lb = base + rng.normal(0, 0.01, size=n)
        dm = diebold_mariano(la, lb, lag=5)
        p = float(dm["pvalue"])
        pvals.append(p)
        if p < 0.05:
            rejects += 1
    type_i = rejects / n_sims
    assert 0.01 <= type_i <= 0.12
    # KS uniformity of p-values (under null ≈ U(0,1))
    ks = stats.kstest(pvals, "uniform")
    assert ks.pvalue > 0.001


def test_dsr_published_identity_and_example() -> None:
    """DSR matches Bailey-López de Prado identity; published-style numerical check."""
    # Identity: N=1, skew=0, kurtosis=3 => DSR = Phi(SR * sqrt(n-1))
    sr = 0.5
    n_obs = 1250
    got = deflated_sharpe_ratio(sr, n_obs=n_obs, n_trials=1, skew=0.0, kurtosis=3.0)
    expected = float(stats.norm.cdf(sr * np.sqrt(n_obs - 1)))
    assert got["dsr"] == pytest.approx(expected, rel=1e-9)

    # SR0 for N=100 from BLP formula (Euler-Mascheroni mixture of normal quantiles)
    sr0 = expected_max_sharpe(100)
    inv1 = float(stats.norm.ppf(1.0 - 1.0 / 100))
    inv2 = float(stats.norm.ppf(1.0 - 1.0 / (100 * np.e)))
    gamma = 0.5772156649015329
    assert sr0 == pytest.approx((1 - gamma) * inv1 + gamma * inv2, rel=1e-12)

    # Higher trial count deflates DSR (same observed SR)
    dsr_1 = deflated_sharpe_ratio(1.0, n_obs=1250, n_trials=1, skew=0.0, kurtosis=3.0)["dsr"]
    dsr_100 = deflated_sharpe_ratio(1.0, n_obs=1250, n_trials=100, skew=0.0, kurtosis=3.0)["dsr"]
    assert dsr_100 < dsr_1


def test_mde_monotonicity() -> None:
    small_n = minimum_detectable_effect(n_obs=50, corr=0.5)
    large_n = minimum_detectable_effect(n_obs=500, corr=0.5)
    assert large_n["mde_dsharpe"] < small_n["mde_dsharpe"]
    assert large_n["mde_dic"] < small_n["mde_dic"]
    low_corr = minimum_detectable_effect(n_obs=200, corr=0.1)
    high_corr = minimum_detectable_effect(n_obs=200, corr=0.9)
    assert high_corr["mde_dsharpe"] < low_corr["mde_dsharpe"]


def test_holm_bh_and_forward_z() -> None:
    raw = [0.01, 0.04, 0.03]
    holm = holm_adjust(raw)
    assert holm[0] <= holm[1] or holm[0] <= 0.03
    assert all(0 <= p <= 1 for p in holm)
    bh = benjamini_hochberg_adjust(raw)
    assert all(0 <= p <= 1 for p in bh)
    # Holm is at least as large as raw for the smallest
    assert holm[0] >= raw[0]

    z_ok = forward_consistency_z(0.05, 0.04, 0.02)
    assert z_ok["z"] == pytest.approx(0.5)
    assert z_ok["passes_s4"] == 1.0
    z_bad = forward_consistency_z(0.10, 0.01, 0.02)
    assert z_bad["z"] == pytest.approx(4.5)
    assert z_bad["passes_s4"] == 0.0


def test_predictive_metrics_and_calibration() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, size=n).astype(float)
    p = np.clip(y * 0.7 + rng.uniform(0, 0.3, size=n), 0, 1)
    assert binary_log_loss(y, p) < binary_log_loss(y, np.full(n, 0.5))
    assert 0 <= binary_brier(y, p) <= 1
    cal = calibration_bins(y, p, n_bins=10)
    assert len(cal.bin_mean_pred) == 10
    assert sum(cal.bin_counts) == n

    # Build a mini panel for IC
    rows = []
    for d in range(30):
        for s in range(8):
            score = rng.normal()
            rows.append(
                {
                    "date": pd.Timestamp("2019-01-02") + pd.Timedelta(days=d),
                    "symbol": f"E{s}",
                    "score": score,
                    "prob": 1 / (1 + np.exp(-score)),
                    "y": int(score + rng.normal() > 0),
                    "X": score + rng.normal(0, 0.5),
                    "eligible": 1,
                }
            )
    frame = pd.DataFrame(rows)
    ic = daily_rank_ic_summary(frame)
    assert np.isfinite(ic["mean_ic"])
    assert np.isfinite(ic["ic_tstat_nw"])
    summary = predictive_summary(frame)
    assert "log_loss" in summary and "calibration" in summary


def test_one_sided_bootstrap_pvalue_null() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=2000)
    p = one_sided_bootstrap_pvalue(samples, greater=True)
    assert 0.4 < p < 0.6


def test_return_delta_bootstrap() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=120, freq="B")
    a = pd.Series(rng.normal(0.0002, 0.01, size=len(idx)), index=idx)
    b = a + 0.0005
    out = bootstrap_paired_return_deltas(a, b, mean_block=5, n_draws=200, seed=0)
    assert out["delta_sharpe"].estimate > 0
    assert out["delta_cagr"].estimate > 0


def test_stats_summary_and_confirmatory_ledger(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    rows_a = []
    rows_b = []
    for d in range(40):
        day = pd.Timestamp("2019-01-02") + pd.Timedelta(days=d)
        for s in range(6):
            x = rng.normal()
            # B slightly better calibrated
            pa = 1 / (1 + np.exp(-(0.3 * x)))
            pb = 1 / (1 + np.exp(-(0.8 * x)))
            y = int(x + rng.normal(0, 0.8) > 0)
            base = {
                "date": day,
                "symbol": f"E{s}",
                "X": x,
                "y": y,
                "eligible": 1,
            }
            rows_a.append({**base, "prob": pa, "score": pa})
            rows_b.append({**base, "prob": pb, "score": pb})
    preds_a = pd.DataFrame(rows_a)
    preds_b = pd.DataFrame(rows_b)
    summary = build_ab_stats_summary(
        preds_a,
        preds_b,
        n_draws=300,
        mean_block=5,
        seed=0,
        n_trials_ledger=3,
        observed_sr_b=0.4,
        n_obs_returns=252,
        ic_hist=0.05,
        ic_fwd=0.04,
        se_fwd=0.02,
    )
    assert summary["purpose"] == "confirmatory"
    assert summary["contamination_label"] == CONTAMINATION_LABEL
    assert "bootstrap" in summary and "diebold_mariano" in summary
    assert summary["multiple_testing"]["n_confirmatory"] == 2
    assert "mde" in summary
    path = write_stats_summary(summary, tmp_path / "stats")
    assert path.is_file()
    text_body = path.read_text(encoding="utf-8")
    assert "delta_log_loss" in text_body

    reg = tmp_path / "experiments.sqlite"
    init_registry(reg)
    tid = log_confirmatory_trial(summary, registry_path=reg, experiment_id="exp-phase9")
    engine = init_registry(reg)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT purpose, feature_set FROM trial_ledger WHERE trial_id = :t"),
            {"t": tid},
        ).fetchone()
    assert row is not None
    assert row[0] == "confirmatory"
    assert row[1] == "A_vs_B"

    multi = confirmatory_ab_tests(delta_log_loss_pvalue=0.01, delta_ic_pvalue=0.04)
    assert len(multi["holm_pvalues"]) == 2
