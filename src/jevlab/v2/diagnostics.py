"""Secondary predictive and robustness diagnostics - never gates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from jevlab.evaluation.bootstrap import (
    DEFAULT_MEAN_BLOCK,
    DEFAULT_SEED,
    bootstrap_statistic_on_dates,
)
from jevlab.evaluation.predictive import (
    binary_auc,
    binary_brier,
    binary_log_loss,
    calibration_bins,
)
from jevlab.models.metrics import daily_rank_ic, daily_rank_ic_series
from jevlab.v2.acceptance import V2GateMetrics, evaluate_v2_gates
from jevlab.v2.candidate import V2_C_GRID, V2_TOP_K

ROBUSTNESS_VARIANTS: tuple[str, ...] = (
    "k_2",
    "k_4",
    "adjacent_C",
    "leave_one_etf_out",
    "phase_results",
    "cost_scenarios",
    "delayed_execution",
    "horizon_sensitivity",
)


def predictive_diagnostics(preds: pd.DataFrame) -> dict[str, Any]:
    work = preds.copy()
    if "eligible" in work.columns:
        work = work.loc[work["eligible"].fillna(0).astype(int) == 1]
    y = work["y"].to_numpy(dtype=float) if "y" in work.columns else None
    p = work["prob"].to_numpy(dtype=float) if "prob" in work.columns else None
    out: dict[str, Any] = {
        "role": "secondary_diagnostics_only",
        "participates_in_v2_sealed_pass": False,
    }
    if y is not None and p is not None and len(y):
        out["log_loss"] = binary_log_loss(y, p)
        out["brier"] = binary_brier(y, p)
        out["auc"] = binary_auc(y, p)
        out["mean_rank_ic"] = daily_rank_ic(work)
        ics = daily_rank_ic_series(work)
        if ics:
            ci = bootstrap_statistic_on_dates(
                list(range(len(ics))),
                lambda idx: float(sum(ics[i] for i in idx) / len(idx)),
                mean_block=DEFAULT_MEAN_BLOCK,
                n_draws=200,
                seed=DEFAULT_SEED,
            )
            out["rank_ic_bootstrap"] = ci.to_dict()
        out["calibration"] = calibration_bins(y, p).to_dict()
    return out


def robustness_spec() -> dict[str, Any]:
    return {
        "role": "diagnostic_only",
        "participates_in_v2_sealed_pass": False,
        "cannot_rescue_failed_canonical": True,
        "cannot_fail_passing_canonical": True,
        "k_grid": [2, V2_TOP_K, 4],
        "adjacent_C_grid": list(V2_C_GRID),
        "variants": list(ROBUSTNESS_VARIANTS),
        "delayed_execution_fill_mode": "delayed",
        "horizon_sensitivity": "exploratory_relabel_only",
    }


def gates_ignore_diagnostics(metrics: V2GateMetrics) -> None:
    """Assert predictive/robustness fields cannot change hard-gate outcomes."""
    base = evaluate_v2_gates(metrics)
    bad = metrics.with_overrides(
        predictive_log_loss=10.0,
        predictive_auc=0.0,
        robustness_notes={"k_2_sharpe": -9.0, "all_variants_fail": True},
    )
    good = metrics.with_overrides(
        predictive_log_loss=0.01,
        predictive_auc=0.99,
        robustness_notes={"k_2_sharpe": 5.0, "all_variants_pass": True},
    )
    assert evaluate_v2_gates(bad).all_passed == base.all_passed
    assert evaluate_v2_gates(good).all_passed == base.all_passed
    for a, b in zip(base.results, evaluate_v2_gates(bad).results, strict=True):
        assert a.passed == b.passed
    for a, b in zip(base.results, evaluate_v2_gates(good).results, strict=True):
        assert a.passed == b.passed
