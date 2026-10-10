"""Assemble ``stats_summary.json`` for the A vs B confirmatory hypothesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jevlab.evaluation.bootstrap import (
    DEFAULT_MEAN_BLOCK,
    DEFAULT_N_DRAWS,
    bootstrap_ab_deltas,
)
from jevlab.evaluation.predictive import (
    daily_spearman_ic_series,
    per_date_log_loss,
    predictive_summary,
)
from jevlab.evaluation.tests import (
    confirmatory_ab_tests,
    deflated_sharpe_ratio,
    diebold_mariano,
    forward_consistency_z,
    minimum_detectable_effect,
)
from jevlab.experiments.manifest import insert_trial_ledger
from jevlab.models.column_sets import CONTAMINATION_LABEL
from jevlab.utils.hashing import canonical_json, sha256_of


def _finite_for_digest(obj: Any) -> Any:
    """Replace non-finite floats with None so content digests stay canonical."""
    if isinstance(obj, dict):
        return {k: _finite_for_digest(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite_for_digest(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _one_sided_pvalue_from_ci(estimate: float, ci_low: float, ci_high: float) -> float:
    """Crude one-sided p proxy from percentile CI (for Holm packaging).

    Prefer DM / NW p-values when available; this is a fallback for IC when only
    a bootstrap CI is on hand.
    """
    if not np.isfinite(estimate):
        return float("nan")
    # If CI wholly on the favorable side of 0 for deltalogLoss (negative) or deltaIC (positive)
    if estimate < 0 and ci_high < 0:
        return 0.01
    if estimate > 0 and ci_low > 0:
        return 0.01
    if (ci_low <= 0 <= ci_high) or not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return 0.5
    return 0.05


def build_ab_stats_summary(
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    *,
    returns_a: pd.Series | None = None,
    returns_b: pd.Series | None = None,
    n_trials_ledger: int = 1,
    observed_sr_b: float | None = None,
    n_obs_returns: int | None = None,
    mean_block: int = DEFAULT_MEAN_BLOCK,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = 42,
    ic_hist: float | None = None,
    ic_fwd: float | None = None,
    se_fwd: float | None = None,
) -> dict[str, Any]:
    """Primary A vs B statistical summary (confirmatory deltalogLoss + deltaIC)."""
    sum_a = predictive_summary(preds_a)
    sum_b = predictive_summary(preds_b)

    ll_a = per_date_log_loss(preds_a)
    ll_b = per_date_log_loss(preds_b)
    ic_a = daily_spearman_ic_series(preds_a)
    ic_b = daily_spearman_ic_series(preds_b)

    # Align dates
    dates = sorted(set(ll_a.index) & set(ll_b.index))
    ll_a_m = {d: float(ll_a.loc[d]) for d in dates}
    ll_b_m = {d: float(ll_b.loc[d]) for d in dates}
    ic_dates = sorted(set(ic_a.index) & set(ic_b.index))
    ic_a_m = {d: float(ic_a.loc[d]) for d in ic_dates}
    ic_b_m = {d: float(ic_b.loc[d]) for d in ic_dates}

    boot = bootstrap_ab_deltas(
        logloss_by_date_a=ll_a_m,
        logloss_by_date_b=ll_b_m,
        ic_by_date_a=ic_a_m,
        ic_by_date_b=ic_b_m,
        returns_a=returns_a,
        returns_b=returns_b,
        mean_block=mean_block,
        n_draws=n_draws,
        seed=seed,
    )

    dm = diebold_mariano(
        [ll_a_m[d] for d in dates],
        [ll_b_m[d] for d in dates],
        lag=5,
    )

    # One-sided p for deltaIC>0 from bootstrap percentile (fraction of draws ≤ 0)
    # Approximate via CI fallback when draws not retained.
    d_ll = boot["delta_log_loss"]
    d_ic = boot["delta_rank_ic"]
    # H1 for log-loss: B better => mean(loss_a - loss_b) > 0
    dm_less = diebold_mariano(
        [ll_a_m[d] for d in dates],
        [ll_b_m[d] for d in dates],
        lag=5,
        alternative="greater",
    )
    p_ll_onesided = float(dm_less["pvalue"])
    p_ic = float(d_ic.get("pvalue_one_sided_greater", float("nan")))
    if not np.isfinite(p_ic):
        p_ic = _one_sided_pvalue_from_ci(
            float(d_ic["estimate"]), float(d_ic["ci_low"]), float(d_ic["ci_high"])
        )

    multi = confirmatory_ab_tests(
        delta_log_loss_pvalue=p_ll_onesided,
        delta_ic_pvalue=p_ic,
    )

    # Correlation of per-date metrics for MDE
    aligned = sorted(set(ic_a_m) & set(ic_b_m))
    if len(aligned) >= 3:
        va = np.array([ic_a_m[d] for d in aligned], dtype=float)
        vb = np.array([ic_b_m[d] for d in aligned], dtype=float)
        corr = float(np.corrcoef(va, vb)[0, 1]) if np.std(va) > 0 and np.std(vb) > 0 else 0.0
    else:
        corr = 0.0
    mde = minimum_detectable_effect(n_obs=max(len(aligned), len(dates)), corr=corr)

    n_obs = int(n_obs_returns) if n_obs_returns is not None else max(len(dates), 2)
    sr = float(observed_sr_b) if observed_sr_b is not None else 0.0
    dsr = deflated_sharpe_ratio(sr, n_obs=n_obs, n_trials=max(n_trials_ledger, 1))

    fwd = None
    if ic_hist is not None and ic_fwd is not None and se_fwd is not None:
        fwd = forward_consistency_z(ic_hist, ic_fwd, se_fwd)

    summary: dict[str, Any] = {
        "hypothesis": "H1: OOS log-loss(B) < log-loss(A) and mean daily rank-IC(B) > rank-IC(A)",
        "purpose": "confirmatory",
        "contamination_label": CONTAMINATION_LABEL,
        "potentially_training_contaminated": True,
        "model_a": sum_a,
        "model_b": sum_b,
        "bootstrap": boot,
        "diebold_mariano": {
            "two_sided": dm,
            "one_sided_b_better": dm_less,
        },
        "multiple_testing": multi,
        "mde": mde,
        "deflated_sharpe": dsr,
        "forward_consistency": fwd,
        "n_dates_logloss": len(dates),
        "n_dates_ic": len(aligned),
        "corr_ic_a_b": corr,
        "bootstrap_config": {
            "mean_block": mean_block,
            "n_draws": n_draws,
            "seed": seed,
        },
    }
    # Point deltas
    summary["point_estimates"] = {
        "delta_log_loss_B_minus_A": float(d_ll["estimate"]),
        "delta_rank_ic_B_minus_A": float(d_ic["estimate"]),
        "log_loss_A": sum_a.get("log_loss"),
        "log_loss_B": sum_b.get("log_loss"),
        "rank_ic_A": (sum_a.get("rank_ic") or {}).get("mean_ic"),
        "rank_ic_B": (sum_b.get("rank_ic") or {}).get("mean_ic"),
    }
    summary["content_digest"] = sha256_of(canonical_json(_finite_for_digest(summary)))
    return summary


def write_stats_summary(summary: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stats_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def log_confirmatory_trial(
    summary: dict[str, Any],
    *,
    registry_path: Path,
    experiment_id: str,
    qset_version: str = "qset_v1",
    state_schema_version: str = "state_v1",
) -> str:
    """Append a confirmatory trial_ledger row for the A vs B stats summary."""
    return insert_trial_ledger(
        registry_path,
        purpose="confirmatory",
        feature_set="A_vs_B",
        model_class="statistical_evaluation",
        grid={
            "bootstrap_mean_block": summary.get("bootstrap_config", {}).get("mean_block"),
            "bootstrap_n_draws": summary.get("bootstrap_config", {}).get("n_draws"),
            "seed": summary.get("bootstrap_config", {}).get("seed"),
        },
        experiment_id=experiment_id,
        result_summary=json.dumps(
            {
                "point_estimates": summary.get("point_estimates"),
                "multiple_testing": summary.get("multiple_testing"),
                "content_digest": summary.get("content_digest"),
                "contamination_label": CONTAMINATION_LABEL,
            },
            default=str,
        ),
        qset_version=qset_version,
        state_schema_version=state_schema_version,
    )
