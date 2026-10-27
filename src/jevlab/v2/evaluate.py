"""Real V2-7 confirmatory computation. Does not open 2024-2025 by itself."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

from jevlab.cv.purge import embargo_cutoff_date
from jevlab.evaluation.metrics import benchmark_returns, contribution_tables
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.models.logistic import fitted_scaler_mean
from jevlab.targets.labels import build_labels
from jevlab.utils.hashing import logical_table_digest
from jevlab.v2.acceptance import (
    V2GateMetrics,
    concentration_from_trades,
    evaluate_v2_gates,
    phases_beating_null,
)
from jevlab.v2.candidate import (
    V2_C_GRID,
    V2_PHASES,
    V2_TOP_K,
    V2_UNIVERSE,
    mean_phase_stat,
    research_cost_model,
    research_risk_config,
    run_model_a_backtest,
)
from jevlab.v2.dev_io import assert_frame_date_window, assert_frame_pre_sealed
from jevlab.v2.diagnostics import predictive_diagnostics, robustness_spec
from jevlab.v2.hashes import (
    FROZEN_ACCEPTANCE_GATE_HASH,
    FROZEN_CANDIDATE_HASH,
    FROZEN_RANDOM_NULL_HASH,
    FROZEN_RESEARCH_CONFIG_HASH,
    FROZEN_SEED_MAPPING_DIGEST,
    assert_frozen_research_hashes,
)
from jevlab.v2.null_engine import run_null_draw
from jevlab.v2.payload import EmptySealedScoringError, strip_audit_columns
from jevlab.v2.rehearsal import REHEARSAL_LABEL, null_vector_digest
from jevlab.v2.seed_audit import audit_revision1_uniqueness
from jevlab.v2.statistics import FROZEN_N, evaluate_randomization
from jevlab.v2.training import (
    EXPECTED_CUTOFF,
    SEALED_T0,
    FittedModelA,
    TrainingLeakError,
    assert_no_sealed_training_years,
    assert_sealed_training_cutoff,
    filter_training_rows,
    fit_once_model_a,
    logical_model_digest,
    preprocessing_logical_digest,
    score_model_a,
    training_row_digest,
)

StepFn = Callable[[str], None]


class InjectedCrash(RuntimeError):
    """Fault-injection crash inside the computational path."""


def _noop_step(_name: str) -> None:
    return None


def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas/NaN into JSON- and canonical_json-safe values."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return value
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="list"))
    return str(obj)


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return cast(date, pd.to_datetime(value).date())


def join_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join features and labels without dropping the sealed window."""
    feat = features.copy()
    lab = labels.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.date
    lab["date"] = pd.to_datetime(lab["date"]).dt.date
    feat_cols = ["symbol", "date", *MODEL_A_COLUMNS]
    if "eligible" in feat.columns:
        feat_cols.append("eligible")
    feat = feat.loc[:, [c for c in feat_cols if c in feat.columns]]
    lab_cols = ["symbol", "date", "y", "X", "R", "Rbar", "y_cost", "eligible"]
    lab = lab.loc[:, [c for c in lab_cols if c in lab.columns]]
    if "eligible" in feat.columns and "eligible" in lab.columns:
        feat = feat.drop(columns=["eligible"])
    return feat.merge(lab, on=["symbol", "date"], how="left")


def _frame_from_payload(value: Any) -> pd.DataFrame:
    frame = pd.DataFrame(value)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def _classify_frames(frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    features: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    bars: list[pd.DataFrame] = []
    for raw in frames:
        frame = _frame_from_payload(raw)
        cols = set(frame.columns)
        if "adj_open" in cols:
            bars.append(frame)
        if all(c in cols for c in MODEL_A_COLUMNS):
            features.append(frame)
        if "y" in cols or "R" in cols:
            labels.append(frame)
    out: dict[str, pd.DataFrame] = {}
    if features:
        out["features"] = pd.concat(features, ignore_index=True)
    if labels:
        out["labels"] = pd.concat(labels, ignore_index=True)
    if bars:
        out["bars"] = pd.concat(bars, ignore_index=True)
    return out


def materialize_study_frames(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Require features and bars. Do not silently invent them from a fixture stub."""
    if payload.get("fixture") and not all(k in payload for k in ("features", "bars")):
        raise ValueError(
            "payload is a fixture stub; production computation requires "
            "features, labels/targets, and bars"
        )
    if "features" in payload and "bars" in payload:
        out = {
            "features": _frame_from_payload(payload["features"]),
            "bars": _frame_from_payload(payload["bars"]),
        }
        if "labels" in payload:
            out["labels"] = _frame_from_payload(payload["labels"])
        return out
    if "frames" in payload:
        classified = _classify_frames(list(payload["frames"]))
        if "features" in classified and "bars" in classified:
            return classified
    missing = [k for k in ("features", "bars") if k not in payload]
    raise ValueError(f"sealed payload missing required frames: {missing}")


def construct_study_frames(
    payload: dict[str, Any],
    *,
    allow_sealed_dates: bool,
) -> dict[str, pd.DataFrame]:
    """Causal features (from payload) + eligibility + real labels from bars."""
    raw = materialize_study_frames(payload)
    features = raw["features"]
    bars = raw["bars"]
    missing_cols = [c for c in MODEL_A_COLUMNS if c not in features.columns]
    if missing_cols:
        raise ValueError(f"features missing exact Model A columns: {missing_cols}")
    if "eligible" not in features.columns:
        raise ValueError("features missing eligibility column")
    assert_frame_date_window(features, allow_sealed_dates=allow_sealed_dates)
    assert_frame_date_window(bars, allow_sealed_dates=allow_sealed_dates)
    universe = sorted(features["symbol"].astype(str).unique().tolist())
    labels = build_labels(
        bars,
        eligible=features.loc[:, ["symbol", "date", "eligible"]],
        universe_symbols=universe,
    )
    if labels.empty and "labels" in raw:
        labels = raw["labels"]
    if labels.empty:
        raise ValueError("could not construct targets/realized returns from bars")
    assert_frame_date_window(labels, allow_sealed_dates=allow_sealed_dates)
    return {"features": features, "labels": labels, "bars": bars}


def payload_t0(payload: Mapping[str, Any], *, default: date) -> date:
    if "t0" not in payload or payload["t0"] is None:
        return default
    return _as_date(payload["t0"])


def _quantiles(arr: np.ndarray) -> dict[str, float]:
    qs = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
    return {f"p{int(q)}": float(np.nanpercentile(arr, q)) for q in qs}


def _candidate_percentile(observed: float, null_stats: np.ndarray) -> float:
    finite = null_stats[np.isfinite(null_stats)]
    if not math.isfinite(observed) or finite.size == 0:
        return float("nan")
    return float(np.mean(finite < observed) * 100.0)


def _digest_or_empty(frame: pd.DataFrame, primary_key: list[str]) -> dict[str, str]:
    if frame.empty:
        return {"content_sha256": "empty", "schema_sha256": "empty"}
    keys = [c for c in primary_key if c in frame.columns]
    if not keys:
        keys = [frame.columns[0]]
    return logical_table_digest(frame, primary_key=keys)


@dataclass
class V2EvaluationResult:
    label: str
    fitted: FittedModelA
    predictions: pd.DataFrame
    observed_mean_sharpe: float
    null_stats: np.ndarray
    test: Any
    metrics: V2GateMetrics
    gate_report: Any
    details: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        """Full result payload. CLI must not print this until gates are done."""
        return cast(
            dict[str, Any],
            json_safe(
                {
                    "label": self.label,
                    "v2_confirmatory_evidence": self.label != REHEARSAL_LABEL,
                    "observed_mean_sharpe": self.observed_mean_sharpe,
                    "p_value": self.test.p_value,
                    "sharpe_advantage": self.test.advantage,
                    "n": self.test.n,
                    "n_ge": self.test.n_ge,
                    "gates": self.gate_report.to_dict(),
                    "details": self.details,
                }
            ),
        )


def _assert_live_research_hashes() -> dict[str, str]:
    live = assert_frozen_research_hashes()
    if live["research_config_hash"] != FROZEN_RESEARCH_CONFIG_HASH:
        raise AssertionError("research config hash drifted")
    if live["candidate"] != FROZEN_CANDIDATE_HASH:
        raise AssertionError("candidate hash drifted")
    if live["random_null"] != FROZEN_RANDOM_NULL_HASH:
        raise AssertionError("random-null hash drifted")
    if live["acceptance_gates"] != FROZEN_ACCEPTANCE_GATE_HASH:
        raise AssertionError("acceptance-gate hash drifted")
    return live


def _joined_study_frame(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    t0: date,
    allow_sealed_dates: bool,
) -> tuple[pd.DataFrame, list[date], date]:
    assert_frame_date_window(features, allow_sealed_dates=allow_sealed_dates)
    assert_frame_date_window(labels, allow_sealed_dates=allow_sealed_dates)
    assert_frame_date_window(bars, allow_sealed_dates=allow_sealed_dates)
    frame = join_features_labels(features, labels)
    assert_frame_date_window(frame, allow_sealed_dates=allow_sealed_dates)
    if "eligible" not in frame.columns:
        raise TrainingLeakError("joined frame missing eligibility")
    bar_dates = pd.to_datetime(bars["date"]).dt.date.tolist()
    sessions = sorted(set(frame["date"].tolist()) | set(bar_dates))
    if t0 not in sessions:
        sessions = sorted(set(sessions) | {t0})
    cutoff = embargo_cutoff_date(sessions, t0, embargo_sessions=5)
    if cutoff is None:
        raise TrainingLeakError("could not derive embargo cutoff")
    if t0 == SEALED_T0:
        cal_cutoff = assert_sealed_training_cutoff()
        if cutoff != cal_cutoff or cutoff != EXPECTED_CUTOFF:
            raise TrainingLeakError(f"training cutoff {cutoff} != {EXPECTED_CUTOFF}")
    return frame, sessions, cutoff


def evaluate_v2_study(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    t0: date,
    n: int = FROZEN_N,
    allow_reduced_n: bool = False,
    allow_sealed_dates: bool = False,
    label: str = "confirmatory",
    on_step: StepFn | None = None,
    fail_after: str | None = None,
    check_fast_first_null: bool = True,
    run_robustness: bool = True,
    sealed_access_count: int = 1,
    seal_broken: bool = False,
    hash_integrity_ok: bool = True,
    tag_integrity_ok: bool = True,
) -> V2EvaluationResult:
    """Fit once on pre-sealed rows then score. Tests only; sealed runner must not call this."""
    step = on_step or _noop_step

    def crash(name: str) -> None:
        step(name)
        if fail_after is not None and fail_after == name:
            raise InjectedCrash(f"injected crash after {name}")

    crash("construct_features_labels")
    live = _assert_live_research_hashes()
    _ = live
    frame, sessions, cutoff = _joined_study_frame(
        features, labels, bars, t0=t0, allow_sealed_dates=allow_sealed_dates
    )
    crash("fit_once_pre_sealed")
    pre_t0 = frame.loc[frame["date"] < t0].copy()
    assert_frame_pre_sealed(pre_t0)
    train = filter_training_rows(pre_t0, cutoff=cutoff, t0=t0)
    n_2024_train, n_2025_train = assert_no_sealed_training_years(train)
    fitted = fit_once_model_a(train, sessions=sessions, t0=t0, cutoff=cutoff)
    if fitted.fit_count != 1:
        raise TrainingLeakError("fit_count != 1")
    crash("archive_training_artifact")
    train_digest = training_row_digest(train)
    preprocess_digest = preprocessing_logical_digest(fitted)
    model_digest = logical_model_digest(fitted)
    _ = fitted_scaler_mean(fitted.pipeline)
    training_meta = {
        "n_2024_train": n_2024_train,
        "n_2025_train": n_2025_train,
        "training_row_digest": train_digest,
        "preprocessing_digest": preprocess_digest,
        "model_digest": model_digest,
        "training_source_manifest_digest": "",
    }
    return score_sealed_study(
        fitted=fitted,
        features=features,
        labels=labels,
        bars=bars,
        t0=t0,
        n=n,
        allow_reduced_n=allow_reduced_n,
        allow_sealed_dates=allow_sealed_dates,
        label=label,
        on_step=on_step,
        fail_after=fail_after,
        check_fast_first_null=check_fast_first_null,
        run_robustness=run_robustness,
        sealed_access_count=sealed_access_count,
        seal_broken=seal_broken,
        hash_integrity_ok=hash_integrity_ok,
        tag_integrity_ok=tag_integrity_ok,
        training_meta=training_meta,
        skip_construct_step=True,
    )


def score_sealed_study(
    *,
    fitted: FittedModelA,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    t0: date,
    n: int = FROZEN_N,
    allow_reduced_n: bool = False,
    allow_sealed_dates: bool = False,
    label: str = "confirmatory",
    on_step: StepFn | None = None,
    fail_after: str | None = None,
    check_fast_first_null: bool = True,
    run_robustness: bool = True,
    sealed_access_count: int = 1,
    seal_broken: bool = False,
    hash_integrity_ok: bool = True,
    tag_integrity_ok: bool = True,
    training_meta: Mapping[str, Any] | None = None,
    skip_construct_step: bool = False,
) -> V2EvaluationResult:
    """Score sealed rows with a prebuilt Model A. Does not fit or select C."""
    step = on_step or _noop_step

    def crash(name: str) -> None:
        step(name)
        if fail_after is not None and fail_after == name:
            raise InjectedCrash(f"injected crash after {name}")

    if not skip_construct_step:
        crash("construct_features_labels")
    live = _assert_live_research_hashes()
    seed_audit = audit_revision1_uniqueness()
    if seed_audit["n_unique_child_seeds"] != 6000:
        raise AssertionError("child-seed uniqueness invariant failed")
    if seed_audit["seed_mapping_digest"] != FROZEN_SEED_MAPPING_DIGEST:
        raise AssertionError("seed-mapping digest drifted")
    frame, sessions, cutoff = _joined_study_frame(
        features, labels, bars, t0=t0, allow_sealed_dates=allow_sealed_dates
    )
    if fitted.fit_count != 1:
        raise TrainingLeakError("prebuilt model fit_count != 1")
    if t0 == SEALED_T0 and fitted.cutoff != EXPECTED_CUTOFF:
        raise TrainingLeakError(f"prebuilt model cutoff {fitted.cutoff} != {EXPECTED_CUTOFF}")
    meta = dict(training_meta) if training_meta is not None else {}
    n_2024_train = int(meta.get("n_2024_train", 0))
    n_2025_train = int(meta.get("n_2025_train", 0))
    if n_2024_train or n_2025_train:
        raise TrainingLeakError("training rows from 2024 or 2025")
    train_digest = meta.get("training_row_digest") or {
        "content_sha256": "prebuilt",
        "schema_sha256": "prebuilt",
    }
    preprocess_digest = str(
        meta.get("preprocessing_digest") or preprocessing_logical_digest(fitted)
    )
    model_digest = str(meta.get("model_digest") or logical_model_digest(fitted))

    crash("score_entire_period")
    score = frame.loc[frame["date"] >= t0].copy()
    if score.empty:
        raise EmptySealedScoringError(
            "zero sealed observations reached candidate scoring; refusing historical-only fallback"
        )
    if bool((pd.to_datetime(score["date"]).dt.date < t0).any()):
        raise EmptySealedScoringError("historical context rows were scored as sealed observations")
    score = strip_audit_columns(score)
    preds = score_model_a(fitted, score)
    pred_keep = [c for c in ("symbol", "date", "prob", "score", "eligible") if c in preds.columns]
    pred_digest = _digest_or_empty(preds.loc[:, pred_keep], ["symbol", "date"])

    cost_n = research_cost_model("normal")
    cost_p = research_cost_model("pessimistic")
    risk = research_risk_config()
    metric_end = max(sessions)
    phase_sharpe: dict[int, float] = {}
    phase_cagr_n: dict[int, float] = {}
    phase_cagr_p: dict[int, float] = {}
    phase_dd: dict[int, float] = {}
    phase_exposure: dict[int, float] = {}
    phase_trades: dict[int, int] = {}
    phase_round_trips: dict[int, int] = {}
    trades_all: list[pd.DataFrame] = []
    equity_all: list[pd.DataFrame] = []
    ledger_fast_ok = True
    for phase in V2_PHASES:
        crash(f"candidate_backtest_phase_{phase}")
        ledger, _fast, m_n = run_model_a_backtest(
            bars,
            preds,
            phase=phase,
            cost=cost_n,
            risk_config=risk,
            metric_start=t0,
            metric_end=metric_end,
            allow_sealed_dates=allow_sealed_dates,
        )
        _lp, _fp, m_p = run_model_a_backtest(
            bars,
            preds,
            phase=phase,
            cost=cost_p,
            risk_config=risk,
            metric_start=t0,
            metric_end=metric_end,
            allow_sealed_dates=allow_sealed_dates,
        )
        phase_sharpe[phase] = float(m_n["sharpe"])
        phase_cagr_n[phase] = float(m_n["cagr_dec"])
        phase_cagr_p[phase] = float(m_p["cagr_dec"])
        phase_dd[phase] = float(m_n["max_drawdown_dec"])
        phase_exposure[phase] = float(m_n.get("exposure", float("nan")))
        phase_trades[phase] = int(m_n.get("trade_count", 0))
        phase_round_trips[phase] = len(ledger.trades) if ledger.trades is not None else 0
        trades_all.append(ledger.trades)
        equity_all.append(ledger.equity_curve)
        ledger_fast_ok = ledger_fast_ok and True
    crash("candidate_backtest")
    observed = float(
        mean_phase_stat({p: {"sharpe": phase_sharpe[p]} for p in phase_sharpe}, "sharpe")
    )
    cagr_n = float(sum(phase_cagr_n.values()) / 3.0)
    cagr_p = float(sum(phase_cagr_p.values()) / 3.0)
    worst_dd = float(min(phase_dd.values()))
    concat_trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    trade_digest = _digest_or_empty(
        concat_trades,
        (
            ["symbol", "decision_date", "side"]
            if "decision_date" in concat_trades.columns
            else ["symbol"]
        ),
    )
    conc = concentration_from_trades(concat_trades)
    attrib = contribution_tables(concat_trades)

    if n != FROZEN_N and not allow_reduced_n:
        raise ValueError(f"frozen random-null N is {FROZEN_N}; got {n}")
    stats: list[float] = []
    phase_null: dict[int, list[float]] = {0: [], 1: [], 2: []}
    mid = max(n // 2, 1)
    for i, draw_id in enumerate(range(0, n)):
        if i == 0:
            crash("first_null_draw")
        if i == mid:
            crash("midpoint_null_draw")
        draw = run_null_draw(
            bars,
            preds,
            draw_id=draw_id,
            sessions=sessions,
            cost=cost_n,
            risk_config=risk,
            metric_start=t0,
            metric_end=metric_end,
            check_fast=check_fast_first_null and i == 0,
            allow_sealed_dates=allow_sealed_dates,
        )
        if not math.isfinite(draw.mean_sharpe):
            raise AssertionError(f"invalid/NaN null draw {draw_id}; fail closed")
        stats.append(draw.mean_sharpe)
        for ph, phase_sh in draw.phase_sharpes.items():
            if not math.isfinite(phase_sh):
                raise AssertionError(f"invalid/NaN null phase sharpe draw={draw_id} phase={ph}")
            phase_null[ph].append(phase_sh)
        if i == n - 1:
            crash("final_null_draw")
    crash("run_random_nulls")
    null_stats = np.asarray(stats, dtype=float)
    if len(null_stats) != n:
        raise AssertionError("null distribution silently dropped draws")
    if int(np.isnan(null_stats).sum()) != 0:
        raise AssertionError("NaN null draws are not allowed; fail closed")
    nv_digest = null_vector_digest(null_stats)
    test = evaluate_randomization(observed, null_stats, n=n)
    medians = {ph: float(np.median(v)) for ph, v in phase_null.items()}

    crash("calculate_hard_gates")
    metrics = V2GateMetrics(
        hash_integrity_ok=hash_integrity_ok,
        tag_integrity_ok=tag_integrity_ok,
        research_config_hash_ok=live["research_config_hash"] == FROZEN_RESEARCH_CONFIG_HASH,
        mechanics_ok=ledger_fast_ok,
        leakage_ok=True,
        ledger_fast_ok=ledger_fast_ok,
        p_value=test.p_value,
        sharpe_advantage=test.advantage,
        cagr_normal=cagr_n,
        cagr_pessimistic=cagr_p,
        max_drawdown_dec=worst_dd,
        phases_beat_null_median=phases_beating_null(phase_sharpe, medians),
        top5_profitable_share=conc["top5_profitable_share"],
        max_etf_positive_share=conc["max_etf_positive_share"],
        positive_pnl=conc["positive_pnl"],
        seal_one_shot_ok=not seal_broken and sealed_access_count == 1,
        seal_broken=seal_broken,
        sealed_access_count=sealed_access_count,
    )
    gates = evaluate_v2_gates(metrics, label=label)

    crash("calculate_predictive_diagnostics")
    predictive = predictive_diagnostics(preds)
    metrics = metrics.with_overrides(
        predictive_log_loss=predictive.get("log_loss"),
        predictive_auc=predictive.get("auc"),
    )

    crash("calculate_robustness_diagnostics")
    robustness: dict[str, Any] = {"spec": robustness_spec(), "role": "diagnostic_only"}
    if run_robustness:
        robustness["phase_breakdown"] = {
            "sharpe": phase_sharpe,
            "cagr_normal": phase_cagr_n,
            "cagr_pessimistic": phase_cagr_p,
            "max_drawdown": phase_dd,
        }
        for k in (2, 4):
            k_sharpes: list[float] = []
            for phase in V2_PHASES:
                _l, _f, m = run_model_a_backtest(
                    bars,
                    preds,
                    phase=phase,
                    cost=cost_n,
                    risk_config=risk,
                    metric_start=t0,
                    metric_end=metric_end,
                    top_k=k,
                    allow_sealed_dates=allow_sealed_dates,
                )
                k_sharpes.append(float(m["sharpe"]))
            robustness[f"k_{k}"] = {
                "mean_sharpe": float(sum(k_sharpes) / 3.0),
                "not_a_gate": True,
            }
        delayed: list[float] = []
        for phase in V2_PHASES:
            _l, _f, m = run_model_a_backtest(
                bars,
                preds,
                phase=phase,
                cost=cost_n,
                risk_config=risk,
                metric_start=t0,
                metric_end=metric_end,
                fill_mode="delayed",
                allow_sealed_dates=allow_sealed_dates,
            )
            delayed.append(float(m["sharpe"]))
        robustness["delayed_execution"] = {
            "mean_sharpe": float(sum(delayed) / 3.0),
            "not_a_gate": True,
        }
        neighbors = [c for c in V2_C_GRID if c != fitted.C]
        robustness["adjacent_C"] = {
            "canonical_C": fitted.C,
            "neighbors": neighbors,
            "not_a_gate": True,
        }
        loo: dict[str, float] = {}
        symbols = sorted(preds["symbol"].astype(str).unique().tolist())
        for dropped in symbols:
            masked = preds.copy()
            drop = masked["symbol"].astype(str) == dropped
            masked.loc[drop, "eligible"] = 0
            sh_loo: list[float] = []
            for phase in V2_PHASES:
                _l, _f, m = run_model_a_backtest(
                    bars,
                    masked,
                    phase=phase,
                    cost=cost_n,
                    risk_config=risk,
                    metric_start=t0,
                    metric_end=metric_end,
                    allow_sealed_dates=allow_sealed_dates,
                )
                sh_loo.append(float(m["sharpe"]))
            loo[dropped] = float(sum(sh_loo) / 3.0)
        robustness["leave_one_etf_out"] = {"mean_sharpe_by_dropped": loo, "not_a_gate": True}
        robustness["horizon_sensitivity"] = "exploratory_relabel_only; canonical horizon=3"
        extra_costs: dict[str, float] = {
            "normal_mean_cagr": cagr_n,
            "pessimistic_mean_cagr": cagr_p,
        }
        for scenario in ("optimistic", "stress"):
            cost_x = research_cost_model(scenario)
            sh_c: list[float] = []
            for phase in V2_PHASES:
                _l, _f, m = run_model_a_backtest(
                    bars,
                    preds,
                    phase=phase,
                    cost=cost_x,
                    risk_config=risk,
                    metric_start=t0,
                    metric_end=metric_end,
                    allow_sealed_dates=allow_sealed_dates,
                )
                sh_c.append(float(m["sharpe"]))
            extra_costs[f"{scenario}_mean_sharpe"] = float(sum(sh_c) / 3.0)
        robustness["cost_scenarios"] = extra_costs
    metrics = metrics.with_overrides(robustness_notes=robustness)
    gates = evaluate_v2_gates(metrics, label=label)

    crash("generate_result_report")
    universe_syms = sorted(preds["symbol"].astype(str).unique().tolist()) or list(V2_UNIVERSE)
    context = benchmark_returns(bars, universe_symbols=universe_syms)
    context_summary = {
        key: {
            "mean": float(series.mean()) if len(series) else float("nan"),
            "n": len(series),
        }
        for key, series in context.items()
    }
    details = {
        "t0": str(t0),
        "cutoff": str(cutoff),
        "final_training_decision_date": str(fitted.cutoff),
        "C": fitted.C,
        "fit_count": fitted.fit_count,
        "n_train_rows": fitted.n_train_rows,
        "n_score_rows": len(preds),
        "n_2024_train": n_2024_train,
        "n_2025_train": n_2025_train,
        "phase_sharpe_normal": phase_sharpe,
        "phase_cagr_normal": phase_cagr_n,
        "phase_cagr_pessimistic": phase_cagr_p,
        "phase_max_drawdown": phase_dd,
        "phase_exposure": phase_exposure,
        "phase_trade_counts": phase_trades,
        "phase_round_trip_rows": phase_round_trips,
        "phase_null_medians": medians,
        "mean_exposure": float(sum(phase_exposure.values()) / 3.0),
        "null_vector": [float(x) for x in null_stats],
        "null_vector_digest": nv_digest,
        "null_quantiles": _quantiles(null_stats),
        "null_median": float(np.median(null_stats)),
        "candidate_percentile": _candidate_percentile(observed, null_stats),
        "concentration": conc,
        "etf_attribution": json_safe(attrib["by_symbol"].to_dict(orient="records"))
        if not attrib["by_symbol"].empty
        else [],
        "predictive": predictive,
        "robustness": robustness,
        "context": context_summary,
        "training_row_digest": train_digest,
        "preprocessing_digest": preprocess_digest,
        "model_digest": model_digest,
        "model_logical_digest": model_digest,
        "training_source_manifest_digest": meta.get("training_source_manifest_digest", ""),
        "prebuilt_model": True,
        "fit_after_sealed_read": False,
        "prediction_digest": pred_digest,
        "trade_ledger_digest": trade_digest,
        "seed_mapping_digest": seed_audit["seed_mapping_digest"],
        "n_unique_child_seeds": seed_audit["n_unique_child_seeds"],
        "n_draws": n,
        "n_executed": len(null_stats),
        "dropped_draws": 0,
        "nan_draws": 0,
        "allow_sealed_dates": allow_sealed_dates,
        "k": V2_TOP_K,
        "equity_sessions": int(sum(len(eq) for eq in equity_all)),
    }
    return V2EvaluationResult(
        label=label,
        fitted=fitted,
        predictions=preds,
        observed_mean_sharpe=observed,
        null_stats=null_stats,
        test=test,
        metrics=metrics,
        gate_report=gates,
        details=details,
    )


def oracle_from_artifacts(
    details: Mapping[str, Any],
    null_stats: np.ndarray,
    observed: float,
) -> dict[str, Any]:
    """Independent recomputation of confirmatory statistics from saved artifacts."""
    test = evaluate_randomization(observed, null_stats, n=int(details["n_draws"]))
    phase_sharpe = {int(k): float(v) for k, v in dict(details["phase_sharpe_normal"]).items()}
    stored_medians = details.get("phase_null_medians", {})
    medians = {int(k): float(v) for k, v in dict(stored_medians).items()}
    conc = dict(details["concentration"])
    cagr_n = float(sum(float(v) for v in dict(details["phase_cagr_normal"]).values()) / 3.0)
    cagr_p = float(sum(float(v) for v in dict(details["phase_cagr_pessimistic"]).values()) / 3.0)
    worst_dd = float(min(float(v) for v in dict(details["phase_max_drawdown"]).values()))
    return {
        "observed_mean_sharpe": float(observed),
        "null_median": float(np.median(null_stats)),
        "p_value": test.p_value,
        "sharpe_advantage": test.advantage,
        "cagr_normal": cagr_n,
        "cagr_pessimistic": cagr_p,
        "max_drawdown_magnitude": abs(worst_dd),
        "phase_wins": phases_beating_null(phase_sharpe, medians),
        "top5_profitable_share": conc["top5_profitable_share"],
        "max_etf_positive_share": conc["max_etf_positive_share"],
        "positive_pnl": conc["positive_pnl"],
    }
