"""V1 Model A 2019-2023 reconstruction comparison (NOT V2 confirmation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from jevlab.backtest.fast import equity_curves_equal
from jevlab.backtest.ledger import EQUITY_TOL_REL
from jevlab.cv.walkforward import OOS_END, OOS_START
from jevlab.models.metrics import summarize_oos
from jevlab.signals.selection import rebalance_dates, select_top_k
from jevlab.v2.candidate import (
    METRIC_RTOL,
    PRED_ATOL,
    PRED_RTOL,
    V2_PHASES,
    V2_TOP_K,
    V2_UNIVERSE,
    research_cost_model,
    research_risk_config,
    run_model_a_backtest,
    run_v2_model_a_walkforward,
)
from jevlab.v2.dev_io import (
    assert_frame_pre_sealed,
    read_canonical_oos_a,
    read_dev_bars,
    read_dev_feature_store,
    v1_canonical_artifact_paths,
)

SANITY_TARGETS: dict[str, Any] = {
    "n_oos_eligible_rows": 11470,
    "log_loss": 0.695210,
    "brier": 0.25102,
    "auc": 0.49473,
    "mean_rank_ic": 0.01613,
    "cagr_normal": 0.0778,
    "sharpe_normal": 0.5827,
    "max_dd": -0.218,
    "mean_exposure": 0.766,
    "phase_round_trips": (1187, 1182, 1188),
}


def _eligible_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    if "eligible" in work.columns:
        work = work.loc[work["eligible"].fillna(0).astype(int) == 1]
    return {(str(r.symbol), str(r.date)) for r in work.itertuples(index=False)}


def _rel_close(a: float, b: float, *, rtol: float = METRIC_RTOL) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return bool(np.isnan(a) and np.isnan(b))
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom <= rtol or abs(a - b) <= 1e-12


def compare_predictions(v2: pd.DataFrame, v1: pd.DataFrame) -> dict[str, Any]:
    keys_v2 = _eligible_keys(v2)
    keys_v1 = _eligible_keys(v1)
    left = v2.copy()
    right = v1.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.date
    right["date"] = pd.to_datetime(right["date"]).dt.date
    merged = left.merge(right, on=["symbol", "date"], suffixes=("_v2", "_v1"), how="inner")
    prob_ok = False
    max_abs = float("nan")
    if "prob_v2" in merged.columns and "prob_v1" in merged.columns:
        p2 = merged["prob_v2"].to_numpy(dtype=float)
        p1 = merged["prob_v1"].to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(p2 - p1))) if len(merged) else 0.0
        prob_ok = bool(np.allclose(p2, p1, rtol=PRED_RTOL, atol=PRED_ATOL, equal_nan=True))
    elif "score_v2" in merged.columns and "score_v1" in merged.columns:
        p2 = merged["score_v2"].to_numpy(dtype=float)
        p1 = merged["score_v1"].to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(p2 - p1))) if len(merged) else 0.0
        prob_ok = bool(np.allclose(p2, p1, rtol=PRED_RTOL, atol=PRED_ATOL, equal_nan=True))
    c_ok = True
    if "C_v2" in merged.columns and "C_v1" in merged.columns:
        c_ok = bool((merged["C_v2"] == merged["C_v1"]).all())
    fold_ok = True
    if "fold_id_v2" in merged.columns and "fold_id_v1" in merged.columns:
        fold_ok = bool((merged["fold_id_v2"].astype(str) == merged["fold_id_v1"].astype(str)).all())
    return {
        "eligible_keys_equal": keys_v2 == keys_v1,
        "n_eligible_v2": len(keys_v2),
        "n_eligible_v1": len(keys_v1),
        "n_inner_join": len(merged),
        "prediction_allclose": prob_ok,
        "prediction_max_abs_diff": max_abs,
        "chosen_C_equal": c_ok,
        "fold_id_equal": fold_ok,
        "tolerance": {"rtol": PRED_RTOL, "atol": PRED_ATOL},
    }


def _selection_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    work = work.loc[work["selected"].fillna(0).astype(int) == 1]
    return {(str(r.symbol), str(r.date)) for r in work.itertuples(index=False)}


def _trade_identities(trades: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    if trades is None or trades.empty:
        return set()
    work = trades.copy()
    work["decision_date"] = pd.to_datetime(work["decision_date"]).dt.date
    sess_col = "session" if "session" in work.columns else "date"
    work[sess_col] = pd.to_datetime(work[sess_col]).dt.date
    return {
        (str(r.symbol), str(r.decision_date), str(r.side), str(getattr(r, sess_col)))
        for r in work.itertuples(index=False)
    }


@dataclass
class EquivalenceReport:
    available: bool
    checks: dict[str, Any] = field(default_factory=dict)
    metrics_v2: dict[str, float] = field(default_factory=dict)
    metrics_v1: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": "implementation_equivalence_only",
            "v2_confirmatory_evidence": False,
            "available": self.available,
            "passed": self.passed,
            "checks": self.checks,
            "metrics_v2": self.metrics_v2,
            "metrics_v1": self.metrics_v1,
            "sanity_targets_rounded": SANITY_TARGETS,
            "canonical_artifacts_win_over_rounded_targets": True,
            "notes": self.notes,
        }


def run_equivalence(
    *,
    features: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    bars: pd.DataFrame | None = None,
    rerun_walkforward: bool = True,
) -> EquivalenceReport:
    """Compare V2 Model A reconstruction to canonical V1 Phase-8 artifacts."""
    paths = v1_canonical_artifact_paths()
    v1 = read_canonical_oos_a(paths["oos_A"])
    notes = [
        "2019-2023 comparison is prior/development evidence only.",
        "Not V2 confirmatory evidence.",
    ]
    if v1.empty:
        return EquivalenceReport(available=False, notes=[*notes, "canonical oos_A.parquet missing"])
    assert_frame_pre_sealed(v1)
    metrics_v1 = summarize_oos(v1)
    report = EquivalenceReport(available=True, metrics_v1=metrics_v1, notes=notes)
    report.checks["canonical_n_rows"] = int(metrics_v1.get("n_rows", len(v1)))
    report.checks["canonical_fold_ids"] = sorted(v1["fold_id"].astype(str).unique().tolist())
    report.checks["canonical_C_by_fold"] = (
        v1.groupby(v1["fold_id"].astype(str))["C"].first().astype(float).to_dict()
        if "C" in v1.columns
        else {}
    )

    if not rerun_walkforward or features is None or labels is None:
        report.checks["walkforward_rerun"] = False
        report.passed = False
        report.notes.append("Walk-forward not re-run in this call; canonical artifact loaded only.")
        return report

    assert_frame_pre_sealed(features)
    assert_frame_pre_sealed(labels)
    wf = run_v2_model_a_walkforward(features, labels)
    v2_preds = wf.result.predictions
    assert_frame_pre_sealed(v2_preds)
    pred_cmp = compare_predictions(v2_preds, v1)
    report.metrics_v2 = dict(wf.result.metrics)
    report.checks["predictions"] = pred_cmp
    report.checks["n_folds_v2"] = len(wf.folds)
    report.checks["fold_ids_v2"] = [f.fold_id for f in wf.folds]
    report.checks["fold_ids_equal"] = [f.fold_id for f in wf.folds] == report.checks[
        "canonical_fold_ids"
    ]
    chosen = {
        str(item["fold_id"]): float(item["C"]) for item in wf.result.fold_report if "C" in item
    }
    report.checks["chosen_C_v2"] = chosen
    report.checks["chosen_C_equal"] = chosen == report.checks["canonical_C_by_fold"]
    report.checks["walkforward_rerun"] = True

    pred_metrics_ok = True
    for key in ("log_loss", "brier", "auc", "rank_ic"):
        if key in report.metrics_v1 and key in report.metrics_v2:
            ok = _rel_close(float(report.metrics_v2[key]), float(report.metrics_v1[key]))
            report.checks[f"metric_{key}_close"] = ok
            pred_metrics_ok = pred_metrics_ok and ok

    selection_ok = True
    if "score" not in v2_preds.columns:
        v2_preds = v2_preds.copy()
        v2_preds["score"] = v2_preds["prob"]
    v1_scores = v1.copy()
    if "score" not in v1_scores.columns:
        v1_scores["score"] = v1_scores["prob"]
    sessions = wf.sessions
    for phase in V2_PHASES:
        reb = rebalance_dates(sessions, every=3, phase=phase)
        sel_v2 = select_top_k(v2_preds, k=V2_TOP_K, rebalance_dates=reb)
        sel_v1 = select_top_k(v1_scores, k=V2_TOP_K, rebalance_dates=reb)
        eq = _selection_keys(sel_v2) == _selection_keys(sel_v1)
        report.checks[f"selection_phase_{phase}_equal"] = eq
        selection_ok = selection_ok and eq
    report.checks["selection_equal"] = selection_ok

    ledger_fast_ok = True
    trade_ok = True
    metric_ok = True
    trade_counts_v2: list[int] = []
    trade_counts_v1: list[int] = []
    if bars is not None and not v2_preds.empty:
        assert_frame_pre_sealed(bars)
        cost = research_cost_model("normal")
        risk = research_risk_config()
        phase_metrics_v2: dict[int, dict[str, float]] = {}
        phase_metrics_v1: dict[int, dict[str, float]] = {}
        for phase in V2_PHASES:
            ledger_v2, fast_v2, m_v2 = run_model_a_backtest(
                bars,
                v2_preds,
                phase=phase,
                cost=cost,
                risk_config=risk,
                metric_start=OOS_START,
                metric_end=OOS_END,
            )
            ledger_v1, fast_v1, m_v1 = run_model_a_backtest(
                bars,
                v1_scores,
                phase=phase,
                cost=cost,
                risk_config=risk,
                metric_start=OOS_START,
                metric_end=OOS_END,
            )
            lf = equity_curves_equal(
                ledger_v2.equity_curve, fast_v2.equity_curve, tol_rel=EQUITY_TOL_REL
            ) and equity_curves_equal(
                ledger_v1.equity_curve, fast_v1.equity_curve, tol_rel=EQUITY_TOL_REL
            )
            ledger_fast_ok = ledger_fast_ok and lf
            ids_v2 = _trade_identities(ledger_v2.trades)
            ids_v1 = _trade_identities(ledger_v1.trades)
            t_eq = ids_v2 == ids_v1
            trade_ok = trade_ok and t_eq
            report.checks[f"trade_ids_phase_{phase}_equal"] = t_eq
            report.checks[f"trade_count_phase_{phase}_v2"] = len(ids_v2)
            report.checks[f"trade_count_phase_{phase}_v1"] = len(ids_v1)
            trade_counts_v2.append(int(m_v2.get("trade_count", 0)))
            trade_counts_v1.append(int(m_v1.get("trade_count", 0)))
            for key in ("cagr_dec", "sharpe", "max_drawdown_dec", "exposure"):
                if not _rel_close(float(m_v2[key]), float(m_v1[key])):
                    metric_ok = False
            # Also vs V2 vs V1 reconstructed backtest of canonical scores: curves.
            if not equity_curves_equal(
                ledger_v2.equity_curve, ledger_v1.equity_curve, tol_rel=EQUITY_TOL_REL
            ):
                metric_ok = False
            phase_metrics_v2[phase] = m_v2
            phase_metrics_v1[phase] = m_v1
        report.checks["ledger_fast_equal"] = ledger_fast_ok
        report.checks["phase_trade_counts_v2"] = trade_counts_v2
        report.checks["phase_trade_counts_v1"] = trade_counts_v1
        report.checks["trade_identities_equal"] = trade_ok
        report.checks["economic_metrics_close"] = metric_ok
        report.checks["mean_sharpe_v2"] = float(
            sum(phase_metrics_v2[p]["sharpe"] for p in V2_PHASES) / 3.0
        )
        report.checks["mean_sharpe_v1"] = float(
            sum(phase_metrics_v1[p]["sharpe"] for p in V2_PHASES) / 3.0
        )
        report.checks["mean_cagr_v2"] = float(
            sum(phase_metrics_v2[p]["cagr_dec"] for p in V2_PHASES) / 3.0
        )
        report.checks["mean_cagr_v1"] = float(
            sum(phase_metrics_v1[p]["cagr_dec"] for p in V2_PHASES) / 3.0
        )

    report.passed = bool(
        pred_cmp.get("eligible_keys_equal")
        and pred_cmp.get("prediction_allclose")
        and pred_cmp.get("fold_id_equal")
        and pred_cmp.get("chosen_C_equal")
        and report.checks.get("fold_ids_equal")
        and report.checks.get("chosen_C_equal")
        and selection_ok
        and pred_metrics_ok
        and (bars is None or (ledger_fast_ok and trade_ok and metric_ok))
    )
    return report


def load_dev_inputs_or_none() -> tuple[
    pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None
]:
    """Best-effort local load of pre-2024 features/labels/bars."""
    try:
        features = read_dev_feature_store()
    except Exception:
        return None, None, None
    if features.empty:
        return None, None, None
    features = features.loc[features["symbol"].isin(V2_UNIVERSE)].reset_index(drop=True)
    try:
        from jevlab.config.loader import load_all_configs
        from jevlab.targets.labels import build_labels

        configs = load_all_configs()
        bars = read_dev_bars(
            vendor=configs.data.primary_vendor,
            symbols=list(V2_UNIVERSE),
        )
        if bars.empty:
            return features, None, None
        elig = features.loc[:, ["symbol", "date", "eligible"]]
        labels = build_labels(
            bars,
            eligible=elig,
            universe_symbols=list(V2_UNIVERSE),
            cost_bps_per_side=2.5,
            sell_fee_bps=configs.backtest.sell_fee_bps,
            per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
        )
        assert_frame_pre_sealed(labels)
        return features, labels, bars
    except Exception:
        return features, None, None
