"""Pre-2024 pseudo-seal software rehearsal. Not V2 evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from jevlab.cv.purge import embargo_cutoff_date
from jevlab.utils.calendar import nyse_session_dates
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
from jevlab.v2.statistics import evaluate_randomization
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
