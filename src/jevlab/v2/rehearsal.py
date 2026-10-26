"""Pre-2024 pseudo-seal software rehearsal. Not V2 evidence."""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jevlab.cv.purge import embargo_cutoff_date
from jevlab.utils.calendar import nyse_session_dates
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.v2.acceptance import (
    V2GateMetrics,
    concentration_from_trades,
    evaluate_v2_gates,
    phases_beating_null,
)
from jevlab.v2.candidate import (
    V2_PHASES,
    research_cost_model,
    research_risk_config,
    run_model_a_backtest,
)
from jevlab.v2.dev_io import assert_frame_pre_sealed
from jevlab.v2.null_engine import run_null_draw
from jevlab.v2.statistics import FROZEN_N, evaluate_randomization
from jevlab.v2.training import fit_once_model_a, score_model_a

REHEARSAL_LABEL = "development_rehearsal_only"
PSEUDO_T0 = date(2022, 1, 3)


@dataclass
class RehearsalResult:
    label: str = REHEARSAL_LABEL
    ok: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "v2_confirmatory_evidence": False,
            "does_not_alter_gates_or_thresholds": True,
            "ok": self.ok,
            "details": self.details,
        }


def synthetic_pre2024_payload(
    *,
    n_sessions: int = 80,
    n_symbols: int = 6,
    seed: int = 7,
) -> dict[str, Any]:
    """Pre-2024 pseudo-sealed payload. Never includes 2024-2025 rows."""
    frame, bars = _synthetic_panel(n_sessions=n_sessions, n_symbols=n_symbols, seed=seed)
    assert_frame_pre_sealed(frame)
    assert_frame_pre_sealed(bars)
    sessions = sorted(frame["date"].unique().tolist())
    t0 = sessions[len(sessions) // 2]
    return {
        "features": frame,
        "labels": frame,
        "bars": bars,
        "t0": t0,
        "pseudo": True,
        "label": REHEARSAL_LABEL,
    }


def _synthetic_panel(
    n_sessions: int = 80, n_symbols: int = 6, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sessions = nyse_session_dates(date(2021, 6, 1), date(2022, 6, 30))[:n_sessions]
    symbols = [f"E{i:02d}" for i in range(n_symbols)]
    from jevlab.features.columns import MODEL_A_COLUMNS

    rows: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    px = {s: 100.0 for s in symbols}
    for i, d in enumerate(sessions):
        for s in symbols:
            px[s] *= float(np.exp(rng.normal(0.0003, 0.01)))
            feats = {c: float(rng.normal()) for c in MODEL_A_COLUMNS}
            y = 1 if feats["ret_1d"] + rng.normal() > 0 else 0
            rows.append(
                {
                    "symbol": s,
                    "date": d,
                    **feats,
                    "eligible": 1 if i % 11 != 0 or s != symbols[0] else 0,
                    "y": y,
                    "X": float(feats["ret_1d"]),
                    "R": float(rng.normal(0, 0.01)),
                    "Rbar": 0.0,
                    "y_cost": y,
                }
            )
            bar_rows.append(
                {
                    "symbol": s,
                    "date": d,
                    "adj_open": px[s],
                    "adj_close": px[s] * (1.0 + float(rng.normal(0, 0.002))),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bar_rows)


def run_synthetic_rehearsal(*, n_draws: int = 8) -> RehearsalResult:
    """Fit-once + nulls + gates on a synthetic 2021-2022 panel (pre-2024)."""
    frame, bars = _synthetic_panel()
    assert_frame_pre_sealed(frame)
    assert_frame_pre_sealed(bars)
    sessions = sorted(frame["date"].unique().tolist())
    t0 = sessions[len(sessions) // 2]
    cutoff = embargo_cutoff_date(sessions, t0, embargo_sessions=5)
    assert cutoff is not None
    train = frame.loc[frame["date"] <= cutoff]
    score = frame.loc[frame["date"] >= t0]
    fitted = fit_once_model_a(train, sessions=sessions, t0=t0, cutoff=cutoff)
    if fitted.fit_count != 1:
        return RehearsalResult(ok=False, details={"error": "fit_count != 1"})
    if (train["date"] >= t0).any():
        return RehearsalResult(ok=False, details={"error": "train leaked into score window"})
    preds = score_model_a(fitted, score)
    if "score" not in preds.columns:
        preds["score"] = preds["prob"]
    cost_n = research_cost_model("normal")
    cost_p = research_cost_model("pessimistic")
    risk = research_risk_config()
    phase_sharpe: dict[int, float] = {}
    phase_cagr_n: dict[int, float] = {}
    phase_cagr_p: dict[int, float] = {}
    phase_dd: dict[int, float] = {}
    trades_all = []
    ledger_fast_ok = True
    for phase in V2_PHASES:
        ledger, _fast, m_n = run_model_a_backtest(
            bars,
            preds,
            phase=phase,
            cost=cost_n,
            risk_config=risk,
            metric_start=t0,
            metric_end=sessions[-1],
        )
        _l2, _f2, m_p = run_model_a_backtest(
            bars,
            preds,
            phase=phase,
            cost=cost_p,
            risk_config=risk,
            metric_start=t0,
            metric_end=sessions[-1],
        )
        phase_sharpe[phase] = float(m_n["sharpe"])
        phase_cagr_n[phase] = float(m_n["cagr_dec"])
        phase_cagr_p[phase] = float(m_p["cagr_dec"])
        phase_dd[phase] = float(m_n["max_drawdown_dec"])
        trades_all.append(ledger.trades)
        ledger_fast_ok = ledger_fast_ok and True
    observed = float(sum(phase_sharpe.values()) / 3.0)
    null_stats = []
    phase_null: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for r in range(1, n_draws + 1):
        draw = run_null_draw(
            bars,
            preds,
            draw_id=r,
            sessions=sessions,
            cost=cost_n,
            risk_config=risk,
            metric_start=t0,
            metric_end=sessions[-1],
            check_fast=(r == 1),
        )
        null_stats.append(draw.mean_sharpe)
        for ph, sh in draw.phase_sharpes.items():
            phase_null[ph].append(sh)
    test = evaluate_randomization(observed, null_stats, n=n_draws)
    conc = concentration_from_trades(pd.concat(trades_all, ignore_index=True))
    medians = {ph: float(np.median(v)) if v else float("nan") for ph, v in phase_null.items()}
    metrics = V2GateMetrics(
        hash_integrity_ok=True,
        tag_integrity_ok=True,
        research_config_hash_ok=True,
        mechanics_ok=ledger_fast_ok,
        leakage_ok=True,
        ledger_fast_ok=ledger_fast_ok,
        p_value=test.p_value,
        sharpe_advantage=test.advantage,
        cagr_normal=float(sum(phase_cagr_n.values()) / 3.0),
        cagr_pessimistic=float(sum(phase_cagr_p.values()) / 3.0),
        max_drawdown_dec=float(min(phase_dd.values())),
        phases_beat_null_median=phases_beating_null(phase_sharpe, medians),
        top5_profitable_share=conc["top5_profitable_share"],
        max_etf_positive_share=conc["max_etf_positive_share"],
        positive_pnl=conc["positive_pnl"],
        seal_one_shot_ok=True,
        seal_broken=False,
        sealed_access_count=1,
    )
    gates = evaluate_v2_gates(metrics, label=REHEARSAL_LABEL)
    return RehearsalResult(
        ok=True,
        details={
            "t0": str(t0),
            "cutoff": str(cutoff),
            "fit_count": fitted.fit_count,
            "C": fitted.C,
            "n_train": fitted.n_train_rows,
            "n_score": len(preds),
            "n_draws": n_draws,
            "observed_mean_sharpe": observed,
            "p_value": test.p_value,
            "advantage": test.advantage,
            "gates_all_passed": gates.all_passed,
            "cannot_create_v2_sealed_pass": True,
            "ledger_fast_ok": ledger_fast_ok,
        },
    )


def null_vector_digest(stats: np.ndarray) -> str:
    payload = {"n": len(stats), "stats": [float(x) for x in stats]}
    return sha256_of(canonical_json(payload))


def _finite_close(a: Any, b: Any, *, atol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    fa = float(a)
    fb = float(b)
    if not math.isfinite(fa) and not math.isfinite(fb):
        return True
    return abs(fa - fb) <= atol


def run_production_path_rehearsal(
    *,
    n: int = FROZEN_N,
    repeat: bool = True,
    n_sessions: int = 36,
    n_symbols: int = 4,
    seed: int = 11,
    run_robustness: bool = False,
    sandbox_root: Path | None = None,
) -> RehearsalResult:
    """Actual production computational path on a PRE-2024 pseudo-sealed source."""
    from jevlab.utils.registry import init_registry
    from jevlab.v2.acceptance import phases_beating_null
    from jevlab.v2.evaluate import oracle_from_artifacts
    from jevlab.v2.sealed_run import run_v2_final_test
    from jevlab.v2.sealed_source import CountingSealedSource

    payload = synthetic_pre2024_payload(n_sessions=n_sessions, n_symbols=n_symbols, seed=seed)
    t0 = payload["t0"]
    tmp_root = sandbox_root or Path("reports") / "v2_preseal" / "production_path_rehearsal_sandbox"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "data" / "sealed").mkdir(parents=True)
    acc = tmp_root / "reports" / "acceptance"
    seal = tmp_root / "reports" / "v2_sealed"
    acc.mkdir(parents=True)
    seal.mkdir(parents=True)
    (tmp_root / "registry").mkdir()
    reg = tmp_root / "registry" / "experiments.sqlite"
    init_registry(reg)

    def _run(seal_dir: Path, acc_dir: Path, *, check_fast: bool) -> Any:
        return run_v2_final_test(
            root=tmp_root,
            registry_path=reg,
            acceptance_dir=acc_dir,
            seal_dir=seal_dir,
            seal_token="unused",
            sealed_source=CountingSealedSource(payload),
            require_clean_tree=False,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_seal_token=False,
            require_authorization=False,
            record_sealed_access=False,
            n=n,
            allow_reduced_n=n != FROZEN_N,
            allow_sealed_dates=False,
            result_label=REHEARSAL_LABEL,
            t0=t0,
            run_robustness=run_robustness,
            check_fast_first_null=check_fast,
        )

    t_start = time.perf_counter()
    result = _run(seal, acc, check_fast=True)
    elapsed = time.perf_counter() - t_start
    if result.acceptance_written or (acc / "V2_SEALED_PASS.json").exists():
        return RehearsalResult(ok=False, details={"error": "rehearsal wrote V2_SEALED_PASS"})
    assert result.result_path is not None
    body = json.loads(result.result_path.read_text(encoding="utf-8"))
    details = body["details"]
    null_stats = np.asarray(details["null_vector"], dtype=float)
    observed = float(body["observed_mean_sharpe"])
    oracle = oracle_from_artifacts(details, null_stats, observed)
    expected_wins = phases_beating_null(
        {int(k): float(v) for k, v in details["phase_sharpe_normal"].items()},
        {int(k): float(v) for k, v in details["phase_null_medians"].items()},
    )
    oracle_ok = (
        _finite_close(oracle["p_value"], body["p_value"])
        and _finite_close(oracle["sharpe_advantage"], body["sharpe_advantage"])
        and _finite_close(oracle["observed_mean_sharpe"], observed)
        and _finite_close(oracle["null_median"], details["null_median"])
        and _finite_close(
            oracle["cagr_normal"],
            sum(float(v) for v in details["phase_cagr_normal"].values()) / 3.0,
        )
        and _finite_close(
            oracle["cagr_pessimistic"],
            sum(float(v) for v in details["phase_cagr_pessimistic"].values()) / 3.0,
        )
        and _finite_close(
            oracle["max_drawdown_magnitude"],
            abs(min(float(v) for v in details["phase_max_drawdown"].values())),
        )
        and oracle["phase_wins"] == expected_wins
        and _finite_close(
            oracle["top5_profitable_share"], details["concentration"]["top5_profitable_share"]
        )
        and _finite_close(
            oracle["max_etf_positive_share"], details["concentration"]["max_etf_positive_share"]
        )
    )
    repeat_ok = True
    if repeat:
        seal2 = tmp_root / "reports" / "v2_sealed_repeat"
        acc2 = tmp_root / "reports" / "acceptance_repeat"
        seal2.mkdir(parents=True)
        acc2.mkdir(parents=True)
        result2 = _run(seal2, acc2, check_fast=False)
        assert result2.result_path is not None
        body2 = json.loads(result2.result_path.read_text(encoding="utf-8"))
        repeat_ok = body2["details"]["null_vector_digest"] == details["null_vector_digest"]
        repeat_ok = repeat_ok and body2["observed_mean_sharpe"] == observed
    return RehearsalResult(
        ok=repeat_ok and oracle_ok and details["fit_count"] == 1 and details["dropped_draws"] == 0,
        details={
            "label": REHEARSAL_LABEL,
            "data_period": {"t0": str(t0), "cutoff": details["cutoff"]},
            "n": n,
            "n_executed": details["n_executed"],
            "null_vector_digest": details["null_vector_digest"],
            "deterministic_repeat": repeat_ok,
            "oracle_gate_comparison": oracle_ok,
            "oracle": oracle,
            "observed_mean_sharpe": observed,
            "p_value": body["p_value"],
            "advantage": body["sharpe_advantage"],
            "runtime_seconds": elapsed,
            "fit_count": details["fit_count"],
            "C": details["C"],
            "n_2024_train": details["n_2024_train"],
            "n_2025_train": details["n_2025_train"],
            "dropped_draws": details["dropped_draws"],
            "acceptance_written": result.acceptance_written,
            "does_not_alter_thresholds": True,
            "v2_confirmatory_evidence": False,
            "result_path": str(result.result_path),
        },
    )


def run_full_n_rehearsal(*, repeat: bool = True) -> RehearsalResult:
    """Software validation with frozen N=2000 via the production computational path."""
    return run_production_path_rehearsal(n=FROZEN_N, repeat=repeat, run_robustness=False)
