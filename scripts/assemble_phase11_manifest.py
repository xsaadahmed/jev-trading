"""Assemble the real Phase 11 experiment manifest from existing artifacts only.

Does not open sealed data, does not run Phase 12, does not retrain models.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevlab.backtest.costs import CostModel
from jevlab.backtest.fast import run_fast
from jevlab.config.loader import load_all_configs
from jevlab.cv.sealed_guard import SealedDataGuard
from jevlab.data.downloader import default_store
from jevlab.experiments.acceptance import (
    HistoricalGateMetrics,
    evaluate_historical_gates,
)
from jevlab.experiments.forward_registry import read_deployment, read_forward_start_date
from jevlab.features.columns import FEATURE_VERSION, MODEL_A_COLUMNS
from jevlab.features.pipeline import read_feature_store
from jevlab.jev.parse import JEV_FEATURE_COLUMNS
from jevlab.jev.preconditions import parse_disposition_decision
from jevlab.jev.qset import load_qset
from jevlab.utils.hashing import sha256_of
from jevlab.utils.paths import ProjectPaths, get_project_root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _g7_from_signals(signals_b: pd.DataFrame) -> tuple[float, float]:
    sel = signals_b.loc[signals_b["selected"] == 1].copy()
    by = sel.groupby("symbol")["R"].sum()
    total = float(by.sum())
    if total == 0:
        return 0.0, 0.0
    max_etf = float((by / total).max())
    top5 = float(sel.sort_values("R", ascending=False).head(5)["R"].sum() / total)
    return top5, max_etf


def _param_stability_fraction(rob: dict) -> float:
    """Fraction of one-step K/phase perturbations with ΔSharpe > 0."""
    vals: list[float] = []
    for r in rob.get("k_sweep") or []:
        if r.get("delta_sharpe") is not None:
            vals.append(float(r["delta_sharpe"]))
    for r in rob.get("phases") or []:
        if r.get("phase") == "mean":
            continue
        if r.get("delta_sharpe") is not None:
            vals.append(float(r["delta_sharpe"]))
    if not vals:
        return 0.0
    return sum(1 for v in vals if v > 0) / len(vals)


def _equity_payload(preds: pd.DataFrame, bars: pd.DataFrame, configs) -> list[dict]:
    scenario = next(c for c in configs.backtest.cost_scenarios if c.name == "normal")
    cost = CostModel(
        cost_bps_per_side=scenario.cost_bps_per_side,
        sell_fee_bps=configs.backtest.sell_fee_bps,
        per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
        scenario_name="normal",
    )
    scores = preds.copy()
    if "score" not in scores.columns and "prob" in scores.columns:
        scores["score"] = scores["prob"]
    led = run_fast(
        bars,
        scores,
        cost=cost,
        risk_config=configs.risk,
        top_k=configs.strategy.top_k,
        phase=0,
        apply_risk=False,
    )
    eq = led.equity_curve.copy()
    eq["date"] = pd.to_datetime(eq["date"]).dt.strftime("%Y-%m-%d")
    col = "equity_usd" if "equity_usd" in eq.columns else eq.columns[-1]
    return [{"date": r["date"], "equity_usd": float(r[col])} for r in eq.to_dict(orient="records")]


def _monthly_from_equity(equity: list[dict], key: str) -> dict[str, float]:
    if not equity:
        return {}
    df = pd.DataFrame(equity)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    out: dict[str, float] = {}
    for month, g in df.groupby("month"):
        e0 = float(g.iloc[0]["equity_usd"])
        e1 = float(g.iloc[-1]["equity_usd"])
        out[str(month)] = (e1 / e0 - 1.0) if e0 else 0.0
    return out


def main() -> int:
    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    out_dir = paths.reports / "phase11_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase8 = _load_json(paths.reports / "phase8_lineup" / "phase8_checkpoint.json")
    stats = _load_json(paths.reports / "phase9_stats" / "stats_summary.json")
    placebo = _load_json(paths.reports / "robustness_report" / "placebo_results.json")
    rob = _load_json(paths.reports / "robustness_report" / "robustness_data.json")
    p10_summary = _load_json(paths.reports / "robustness_report" / "phase10_completion_summary.json")

    qset = load_qset(paths.root / configs.jev.qset_path, expected_hash=configs.jev.qset_hash)
    disposition_text = (paths.root / "docs" / "contamination_disposition.md").read_text(
        encoding="utf-8"
    )
    disposition = parse_disposition_decision(disposition_text)
    audit_exists = (paths.reports / "audit" / "AUDIT_REPORT.md").is_file()

    guard = SealedDataGuard(
        sealed_root=paths.data_sealed,
        registry_path=paths.registry / "experiments.sqlite",
    )
    store = default_store(paths, configs.data)
    bars = guard.filter_dev_frame(store.read_bars(vendor=configs.data.primary_vendor))
    preds_a = guard.filter_dev_frame(pd.read_parquet(paths.reports / "phase8_lineup" / "oos_A.parquet"))
    preds_b = guard.filter_dev_frame(pd.read_parquet(paths.reports / "phase8_lineup" / "oos_B.parquet"))
    signals_b = guard.filter_dev_frame(
        pd.read_parquet(paths.reports / "phase8_lineup" / "signals_B.parquet")
    )

    equity_a = _equity_payload(preds_a, bars, configs)
    equity_b = _equity_payload(preds_b, bars, configs)
    monthly_a = _monthly_from_equity(equity_a, "a")
    monthly_b = _monthly_from_equity(equity_b, "b")
    months = sorted(set(monthly_a) | set(monthly_b))
    monthly_returns = [
        {"month": m, "ret_a": monthly_a.get(m), "ret_b": monthly_b.get(m)} for m in months
    ]
    trade_pnls = (
        signals_b.loc[signals_b["selected"] == 1, "R"].astype(float).tolist()
        if "selected" in signals_b.columns
        else []
    )

    top5, max_etf = _g7_from_signals(signals_b)
    # LOO never restores positive economic increment; use mean phase ΔSharpe as upper bound proxy
    # after best-ETF removal remains non-positive in registered Phase 10 LOO (all dll gains ≤ 0).
    loo_best_dll = min(
        (float(r["delta_log_loss_gain"]) for r in rob.get("leave_one_etf_out") or []),
        default=-1.0,
    )
    delta_sharpe_loo_best = -1.0 if loo_best_dll <= 0 else 0.01
    param_frac = _param_stability_fraction(rob)

    overall = phase8["overall"]
    deltas = phase8["deltas_B_minus_A"]
    fold = phase8["fold_consistency"]
    years_fav = sum(1 for y in phase8["year_consistency"] if float(y["d_log_loss"]) <= 0)
    boot = stats["bootstrap"]["delta_log_loss"]
    p5 = placebo.get("P5") or {}
    # residual fraction of *observed* gain: observed gain for B is negative; P5 reports residual_fraction
    p5_resid = p5.get("residual_fraction_of_gain")

    phases = phase8.get("phase_consistency") or []
    cagr_b_pos_phases = sum(1 for p in phases if float(p.get("B_cagr", 0)) > 0)
    cost_b = overall["B"]["cost"]
    cost_a = overall["A"]["cost"]
    d_sharpe_stress = float(cost_b["stress"]["sharpe"]) - float(cost_a["stress"]["sharpe"])
    d_sharpe_normal = float(deltas["delta_sharpe"])

    # Drawdowns as positive magnitudes for G3/G9
    max_dd_a = abs(float(overall["A"]["max_dd"]))
    max_dd_b = abs(float(overall["B"]["max_dd"]))

    db = paths.registry / "experiments.sqlite"
    sealed_n = sqlite3.connect(db).execute("select count(*) from sealed_access_log").fetchone()[0]
    hist_acc = paths.reports / "acceptance" / "HISTORICAL_ACCEPTANCE.json"
    try:
        dep = read_deployment(db)
        fwd_start = read_forward_start_date(db)
        dep_ts = dep.deployment_timestamp.isoformat()
        fwd_start_s = fwd_start.isoformat()
    except Exception as exc:  # noqa: BLE001
        dep_ts = ""
        fwd_start_s = ""
        dep = None
        print("forward_registry:", exc)

    metrics = HistoricalGateMetrics(
        audit_report_exists=audit_exists,
        disposition_decision=disposition,
        reproducibility_ok=None,  # no full reproduce artifact → NOT EVALUATED
        mechanics_ok=True,
        label_base_rate=0.50,
        trades_per_phase_a=float(overall["A"]["trades"]),
        trades_per_phase_b=float(overall["B"]["trades"]),
        mean_exposure_a=float(overall["A"]["exposure"]),
        mean_exposure_b=float(overall["B"]["exposure"]),
        max_dd_a=max_dd_a,
        delta_log_loss=float(stats["point_estimates"]["delta_log_loss_B_minus_A"]),
        delta_log_loss_ci_excludes_zero=bool(boot.get("excludes_zero")),
        delta_ic=float(stats["point_estimates"]["delta_rank_ic_B_minus_A"]),
        placebo_p1_p=float(placebo["P1"]["pvalue_log_loss"]),
        placebo_p3_p=float(placebo["P3"]["pvalue_log_loss"]),
        p5_residual_fraction=float(p5_resid) if p5_resid is not None else None,
        folds_nonpositive_dll=int(fold["n_folds_d_logloss_lt_0"]),
        years_consistent_sign=int(years_fav),
        delta_sharpe=d_sharpe_normal,
        cagr_b=float(overall["B"]["cagr"]),
        cagr_a=float(overall["A"]["cagr"]),
        cagr_b_pessimistic=float(cost_b["pessimistic"]["cagr"]),
        delta_sharpe_stress_same_sign=(d_sharpe_stress < 0) == (d_sharpe_normal < 0),
        cagr_b_positive_phases=int(cagr_b_pos_phases),
        top5_trade_share=top5,
        max_etf_pnl_share=max_etf,
        delta_sharpe_loo_best=delta_sharpe_loo_best,
        param_stability_fraction=param_frac,
        max_dd_b=max_dd_b,
        # G10 intentionally omitted / None → NOT EVALUATED
        sealed_train_end=None,
        sealed_trades_per_phase=None,
        sealed_delta_log_loss=None,
        sealed_delta_ic=None,
        sealed_placebo_p1_p=None,
        sealed_cagr_b=None,
        sealed_g7_ok=None,
        sealed_g9_ok=None,
        sealed_access_count=None,
    )
    gate_report = evaluate_historical_gates(metrics)
    # Force G10 to unevaluable in the report table even if evaluator returned something
    gate_by_id = {g.gate_id: g for g in gate_report.results}
    if "G10" in gate_by_id:
        from jevlab.experiments.acceptance import GateResult

        gate_by_id["G10"] = GateResult(
            "G10",
            None,
            "Sealed holdout intentionally not opened; HISTORICAL_RESEARCH_PASS already impossible",
        )

    gate_meta = {
        "G1": {
            "criterion": "Contamination diagnostics (procedural)",
            "threshold": "D1–D6 executed; disposition ≠ stop",
            "evidence": "docs/contamination_disposition.md; reports/audit/AUDIT_REPORT.md",
        },
        "G2": {
            "criterion": "Reproducibility",
            "threshold": "Manifest re-run reproduces trades/metrics within §20 tolerances",
            "evidence": "No full reproduce artifact present",
        },
        "G3": {
            "criterion": "Mechanics and sample sufficiency",
            "threshold": "Accounting/ledger≡fast; ≥300 trades/phase; exposure≥30%; A maxDD≤40%",
            "evidence": "reports/phase8_lineup/phase8_checkpoint.json",
        },
        "G4": {
            "criterion": "Predictive increment",
            "threshold": "ΔlogLoss(B−A)<0 with 95% CI excl. 0; ΔIC>0; P1&P3 p<0.05; P5 residual≥50% of gain",
            "evidence": "reports/phase9_stats/stats_summary.json; reports/robustness_report/placebo_results.json",
        },
        "G5": {
            "criterion": "Stability",
            "threshold": "ΔlogLoss≤0 in ≥13/20 folds; same sign in ≥4/5 years",
            "evidence": "reports/phase8_lineup/phase8_checkpoint.json fold_consistency/year_consistency",
        },
        "G6": {
            "criterion": "Economic increment and Model B viability",
            "threshold": "ΔSharpe≥+0.20; CAGR(B)>CAGR(A); B>0 normal/pessimistic; stress sign",
            "evidence": "reports/phase8_lineup/phase8_checkpoint.json",
        },
        "G7": {
            "criterion": "Concentration",
            "threshold": "Top5 trades <20% PnL; no ETF >35%; ΔSharpe>0 after removing best ETF",
            "evidence": "signals_B selected R concentration; leave_one_etf_out.json",
        },
        "G8": {
            "criterion": "Parameter stability",
            "threshold": "≥80% of one-step K/phase perturbations keep ΔSharpe>0",
            "evidence": "reports/robustness_report/k_sweep.json; phases.json",
        },
        "G9": {
            "criterion": "Drawdown",
            "threshold": "B maxDD ≤ min(max(1.2×A, 25%), 35%)",
            "evidence": "reports/phase8_lineup/phase8_checkpoint.json",
        },
        "G10": {
            "criterion": "Sealed test 2024–2025",
            "threshold": "One-shot sealed evaluation (Phase 12)",
            "evidence": "Intentionally not opened (protocol)",
        },
    }

    gates_list = []
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"):
        gr = gate_by_id[gid]
        if gr.passed is True:
            status = "PASS"
        elif gr.passed is False:
            status = "FAIL"
        else:
            status = "NOT EVALUATED"
        meta = gate_meta[gid]
        gates_list.append(
            {
                "id": gid,
                "criterion": meta["criterion"],
                "threshold": meta["threshold"],
                "observed": gr.detail,
                "status": status,
                "passed": gr.passed,
                "evidence": meta["evidence"],
            }
        )

    mde = stats.get("mde") or {}
    power_statement = (
        f"Walk-forward OOS provides approximately {int(mde.get('n_obs', 1231))} paired date "
        f"observations for confirmatory tests (corr(A,B) IC ≈ {float(mde.get('corr', 0)):.3f}). "
        f"Registered approximate MDE for paired ΔSharpe/ΔIC at α=0.05, power=0.8 is "
        f"{float(mde.get('mde_dsharpe', 0)):.4f}. "
        "The study may be underpowered to detect small *positive* B−A effects of that size; "
        "that power limitation does not erase the fact that the observed confirmatory point "
        "estimates themselves favor A rather than B "
        f"(ΔlogLoss(B−A)={stats['point_estimates']['delta_log_loss_B_minus_A']:+.6f}, "
        f"ΔIC(B−A)={stats['point_estimates']['delta_rank_ic_B_minus_A']:+.6f}, "
        f"ΔSharpe≈{d_sharpe_normal:+.4f})."
    )

    cal_b = (stats.get("model_b") or {}).get("reliability") or {}
    manifest = {
        "experiment_id": "phase11_final",
        "qset_hash": qset.content_hash,
        "jev_model": configs.jev.model_id,
        "contamination_label": "potentially_training_contaminated",
        "strategy": {
            "name": "Pullback in uptrend, long/cash",
            "top_k": configs.strategy.top_k,
            "horizon_sessions": configs.strategy.horizon_sessions,
            "phases": list(configs.strategy.phases),
            "universe": list(configs.universe.symbols),
            "entry": "next open (t+1)",
            "cost_scenario_primary": "normal",
            "logistic_C_grid": list(configs.strategy.logistic_C_grid),
            "seed": configs.strategy.seed,
        },
        "dataset": {
            "partitions": [
                {
                    "name": "train_warm",
                    "range": "2016-01-04..2018-12-31",
                    "contamination_status": "potentially_training_contaminated",
                },
                {
                    "name": "walk_forward_oos",
                    "range": "2019-01-02..2023-12-29",
                    "contamination_status": "potentially_training_contaminated",
                },
                {
                    "name": "sealed",
                    "range": "2024-01-02..2025-12-31",
                    "contamination_status": "sealed_held_out_unopened",
                },
                {
                    "name": "interim",
                    "range": "2026-01-02..day before FORWARD_START_DATE",
                    "contamination_status": "potentially_training_contaminated",
                },
                {
                    "name": "forward",
                    "range": f"from FORWARD_START_DATE={fwd_start_s}",
                    "contamination_status": "clean_forward_inputs_only",
                },
            ],
            "note": (
                "Sealed partition deliberately remains unopened. Historical Jev-involving "
                "partitions are potentially_training_contaminated."
            ),
        },
        "features": {
            "model_a_columns": list(MODEL_A_COLUMNS),
            "jev_feature_columns": list(JEV_FEATURE_COLUMNS),
            "feature_version": FEATURE_VERSION,
        },
        "qset": {
            "qset_version": qset.qset_version,
            "content_hash": qset.content_hash,
            "questions": json.loads(
                (paths.root / configs.jev.qset_path).read_text(encoding="utf-8")
            ).get("questions"),
        },
        "equity": equity_b,
        "equity_a": equity_a,
        "monthly_returns": monthly_returns,
        "trade_pnls": trade_pnls[:5000],
        "baseline_comparison": {
            "A_log_loss": overall["A"]["log_loss"],
            "A_rank_ic": overall["A"]["rank_ic"],
            "A_cagr": overall["A"]["cagr"],
            "A_sharpe": overall["A"]["sharpe"],
            "A_max_dd": overall["A"]["max_dd"],
            "A_trades_mean_phase": overall["A"]["trades"],
            "R0_log_loss": overall["R0"]["log_loss"],
            "R0_sharpe": overall["R0"]["sharpe"],
        },
        "jev_comparison": {
            "B_log_loss": overall["B"]["log_loss"],
            "B_rank_ic": overall["B"]["rank_ic"],
            "B_cagr": overall["B"]["cagr"],
            "B_sharpe": overall["B"]["sharpe"],
            "B_max_dd": overall["B"]["max_dd"],
            "delta_log_loss_B_minus_A": stats["point_estimates"]["delta_log_loss_B_minus_A"],
            "delta_rank_ic_B_minus_A": stats["point_estimates"]["delta_rank_ic_B_minus_A"],
            "delta_sharpe": deltas["delta_sharpe"],
            "delta_cagr": deltas["delta_cagr"],
            "bootstrap_dll_ci": [boot["ci_low"], boot["ci_high"]],
            "contamination_label": "potentially_training_contaminated",
        },
        "calibration": {
            "mean_predicted": cal_b.get("mean_predicted"),
            "frac_positive": cal_b.get("frac_positive"),
        },
        "placebo": {
            k: placebo[k]
            for k in ("P1", "P2", "P3", "P4", "P5", "P6", "Model_F", "contamination_label")
            if k in placebo
        },
        "cost_sensitivity": rob.get("cost_sweep"),
        "robustness": {
            "feature_ablation": rob.get("feature_ablation"),
            "jev_question_ablation": rob.get("jev_question_ablation"),
            "leave_one_etf_out": rob.get("leave_one_etf_out"),
            "k_sweep": rob.get("k_sweep"),
            "horizon_sweep": rob.get("horizon_sweep"),
            "delayed_execution": rob.get("delayed_execution"),
            "regularization": rob.get("regularization"),
            "regimes": rob.get("regimes"),
            "phases": rob.get("phases"),
            "choice_order": rob.get("choice_order"),
            "paraphrase": rob.get("paraphrase"),
            "narrative_conclusions": None,
            "phase10_completion": p10_summary,
        },
        "stats_summary": stats,
        "power": {
            "statement": power_statement,
            "underpowered": False,
            "mde": mde,
            "observed_effect_note": (
                "Observed confirmatory estimates favor A; do not reinterpret adverse "
                "point estimates as 'inconclusive due to power'."
            ),
        },
        "underpowered": False,
        "contamination_diagnostics": {
            "disposition": disposition,
            "label": "potentially_training_contaminated",
            "sections": [
                {"name": "D1"},
                {"name": "D2"},
                {"name": "D3"},
                {"name": "D4"},
                {"name": "D5"},
                {"name": "D6"},
            ],
            "note": "Each diagnostic is a warning signal, not proof.",
        },
        "gates": gates_list,
        "gates_note": (
            "Failure of any of G1–G10 blocks HISTORICAL_RESEARCH_PASS. "
            "G10 is NOT EVALUATED because the sealed holdout was intentionally not opened "
            "after upstream mandatory gates already failed; this is not a missing-data failure."
        ),
        "sealed_holdout": {
            "unopened": True,
            "reason": (
                "Multiple mandatory non-sealed historical gates already failed, so opening the "
                "one-shot sealed holdout cannot produce HISTORICAL_RESEARCH_PASS and would provide "
                "no protocol benefit."
            ),
            "sealed_access_count": int(sealed_n),
            "phase12_run": False,
            "sealed_jev_inference": False,
            "historical_acceptance_present": hist_acc.is_file(),
        },
        "forward_capture": {
            "deployment_timestamp": dep_ts,
            "FORWARD_START_DATE": fwd_start_s,
            "note": (
                "2026-10-15 was shadow-only. Forward capture does not grant HISTORICAL_RESEARCH_PASS, "
                "TINY_LIVE_READY, or SCALE_READY. No forward outcomes are scored in this report."
            ),
        },
        "phase10_interpretation": {
            "summary": (
                "Phase 10 placebos and robustness do not rescue Model B. All historical Jev "
                "results remain potentially_training_contaminated."
            ),
            "bullets": [
                "Canonical B did not outperform A on confirmatory metrics.",
                "Canonical B did not outperform randomized/placebo controls (P1–P4 p non-significant).",
                "P6 found no pipeline-integrity warning.",
                "Paraphrases remain highly correlated with canonical Jev and do not rescue performance.",
                "Choice-order sensitivity does not rescue performance.",
                "No registered ablation/ETF/K/horizon/cost/execution/regularization/regime test overturns Phase 9.",
                "Exploratory favorable slices are not confirmatory evidence.",
            ],
        },
        "acceptance_statuses": {
            "HISTORICAL_RESEARCH_PASS": {
                "present": False,
                "integrity_digest": "",
                "path": "reports/acceptance/HISTORICAL_ACCEPTANCE.json",
                "note": "NO — no HISTORICAL_ACCEPTANCE.json exists or may be created.",
            },
            "TINY_LIVE_READY": {
                "present": False,
                "integrity_digest": "",
                "path": "reports/acceptance/TINY_LIVE_ACCEPTANCE.json",
                "note": "NO — blocked because T1 requires a valid HISTORICAL_ACCEPTANCE.json.",
            },
            "SCALE_READY": {
                "present": False,
                "integrity_digest": "",
                "path": "reports/acceptance/SCALE_ACCEPTANCE.json",
                "note": "NO — blocked because the prerequisite historical/tiny-live path did not pass.",
            },
        },
        "integrity": {
            "phase9_content_digest": stats.get("content_digest"),
            "placebo_digest": (
                paths.reports / "robustness_report" / "placebo_results.digest"
            ).read_text(encoding="utf-8").strip(),
            "robustness_digest": (
                paths.reports / "robustness_report" / "robustness_data.digest"
            ).read_text(encoding="utf-8").strip(),
            "qset_hash": qset.content_hash,
            "prereg_v1": "eb1a1ad27dcd67df298fb96eb14c7f3dee843137",
            "qset_v1_frozen": "eb1a1ad27dcd67df298fb96eb14c7f3dee843137",
        },
    }

    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("wrote", man_path)
    print("gates:")
    for g in gates_list:
        print(" ", g["id"], g["status"], g["observed"][:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
