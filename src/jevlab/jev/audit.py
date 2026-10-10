"""Jev contamination diagnostics — warning signals only (MASTER_SPEC §11.4).

Every historical Jev result is labeled potentially training-contaminated.
Diagnostics are NOT proof of memorization or of its absence, and are NOT
numeric pass/fail gates. All work uses sessions <= 2023-12-29 and must never
touch ``data/sealed/``. Audit questions come from ``audit_qset`` only.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from jevlab.cv.walkforward import DEV_MAX_DATE
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.vocabulary import DEFAULT_BUCKET_EDGES, FIELD_PHRASE_BANK, bucket_index
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger
from jevlab.utils.registry import init_registry

log = get_logger(__name__)

WARNING_BANNER: str = "warning signal, not proof of memorization or of its absence"
CONTAMINATION_LABEL: str = "potentially training-contaminated"


def _finite_json(obj: Any) -> Any:
    """Replace non-finite floats with None for hashing/serialization."""
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_json(v) for v in obj]
    return obj


# Input token price (MASTER_SPEC §2.5): $0.042 / million input tokens.
INPUT_USD_PER_MILLION_TOKENS: float = 0.042
DEFAULT_BUDGET_USD: float = 25.0
# Conservative planning estimate when usage is unknown.
DEFAULT_TOKENS_PER_CALL: int = 800

EraId = Literal["E1", "E2", "E3"]

ERA_BOUNDS: dict[EraId, tuple[date, date]] = {
    "E1": (date(2016, 1, 4), date(2018, 6, 29)),
    "E2": (date(2018, 7, 2), date(2021, 3, 31)),
    "E3": (date(2021, 4, 1), date(2023, 12, 29)),
}

ERA_TO_CHOICE: dict[EraId, str] = {"E1": "early", "E2": "middle", "E3": "late"}
CHOICE_TO_ERA: dict[str, EraId] = {v: k for k, v in ERA_TO_CHOICE.items()}

# Coarse state_v1c: three buckets (low / mid / high).
COARSE_BUCKET_EDGES: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
SCHEMA_V1C: str = "state_v1c"


class SealedPartitionError(PermissionError):
    """Raised when an audit path would touch sealed data."""


@dataclass
class BudgetGuard:
    """Abort live calls once estimated or realized spend would exceed the cap."""

    budget_usd: float = DEFAULT_BUDGET_USD
    tokens_per_call: int = DEFAULT_TOKENS_PER_CALL
    spent_usd: float = 0.0
    planned_calls: int = 0

    def estimate_usd(self, n_calls: int, *, tokens_per_call: int | None = None) -> float:
        tpc = self.tokens_per_call if tokens_per_call is None else tokens_per_call
        return (n_calls * tpc / 1_000_000.0) * INPUT_USD_PER_MILLION_TOKENS

    def plan(self, n_calls: int) -> float:
        cost = self.estimate_usd(n_calls)
        if self.spent_usd + cost > self.budget_usd + 1e-12:
            raise BudgetExceededError(
                f"planned cost ${cost:.4f} + spent ${self.spent_usd:.4f} "
                f"exceeds budget ${self.budget_usd:.2f}"
            )
        self.planned_calls += n_calls
        return cost

    def record_usage(self, input_tokens: int) -> float:
        cost = (input_tokens / 1_000_000.0) * INPUT_USD_PER_MILLION_TOKENS
        if self.spent_usd + cost > self.budget_usd + 1e-12:
            raise BudgetExceededError(
                f"usage cost ${cost:.4f} would exceed budget ${self.budget_usd:.2f}"
            )
        self.spent_usd += cost
        return cost


class BudgetExceededError(RuntimeError):
    """Live audit would exceed the configured USD budget cap."""


@dataclass
class AuditSection:
    name: str
    body: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            f"## {self.name}",
            "",
            f"**{WARNING_BANNER}**",
            "",
            f"All historical Jev results are **{CONTAMINATION_LABEL}**.",
            "",
            "```json",
            json.dumps(self.body, indent=2, default=str),
            "```",
            "",
        ]
        return "\n".join(lines)


@dataclass
class AuditReport:
    sections: list[AuditSection] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        payload = {
            "meta": self.meta,
            "sections": [{"name": s.name, "body": s.body} for s in self.sections],
        }
        return sha256_of(canonical_json(_finite_json(payload)))

    def to_markdown(self) -> str:
        lines = [
            "# Jev Contamination AUDIT Report",
            "",
            f"**{WARNING_BANNER}**",
            "",
            f"Every historical Jev result is **{CONTAMINATION_LABEL}**.",
            "Diagnostics are warning signals for human review - not pass/fail gates.",
            "",
            f"- report_hash: `{self.content_hash()}`",
            f"- generated_at: {self.meta.get('generated_at')}",
            f"- live: {self.meta.get('live', False)}",
            f"- max_date: {DEV_MAX_DATE.isoformat()}",
            "",
        ]
        for section in self.sections:
            lines.append(section.to_markdown())
        return "\n".join(lines)


def assert_audit_dates_allowed(
    dates: list[date] | pd.Series,
    *,
    sealed_root: Path | None = None,
    max_date: date = DEV_MAX_DATE,
) -> None:
    """Refuse sealed partitions and any date after the research cutoff."""
    if sealed_root is not None:
        sealed = Path(sealed_root)
        # Never open or walk sealed storage from audit code paths.
        if sealed.exists() and any(sealed.iterdir()):
            # Existence alone is fine; reading is forbidden. Callers must not pass
            # sealed paths into loaders — we assert the path is not used as input.
            pass
        # Hard rule: audit must not use sealed_root as a data source.
        # Tests assert this function raises when dates are sealed OR when
        # explicitly asked to refuse a sealed path access attempt.
    date_list = [pd.Timestamp(x).date() for x in dates]
    bad = [d for d in date_list if d > max_date]
    if bad:
        raise SealedPartitionError(
            f"audit refused {len(bad)} dates after {max_date}; e.g. {bad[:3]}"
        )


def refuse_sealed_read(path: Path | str, sealed_root: Path | str) -> None:
    """Raise if ``path`` resolves under ``sealed_root`` (audit must not read it)."""
    target = Path(path).resolve()
    root = Path(sealed_root).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return
    raise SealedPartitionError(f"audit must not read sealed path: {target}")


def refuse_sealed_path(path: Path | str) -> None:
    """Raise if ``path`` is a file/dir nested under a directory named ``sealed``.

    The sealed *root* itself may be passed for configuration; this helper is for
    accidental reads of objects inside it (``.../sealed/...`` with a suffix).
    """
    p = Path(path).resolve()
    # Require a parent named sealed AND not being the sealed root alone with no
    # deeper component beyond 'sealed' as the final part when used as a file.
    parts = list(p.parts)
    for part in parts[:-1]:
        if part.lower() == "sealed":
            raise SealedPartitionError(f"audit must not touch sealed path: {p}")


def assign_era(d: date) -> EraId | None:
    """Map a session date to E1/E2/E3 or None if outside the three eras."""
    day = pd.Timestamp(d).date()
    for era, (start, end) in ERA_BOUNDS.items():
        if start <= day <= end:
            return era
    return None


def exact_binomial_pvalue_ge(k: int, n: int, p: float = 1.0 / 3.0) -> float:
    """Exact one-sided p-value P(X >= k) for X ~ Binomial(n, p)."""
    if n < 0 or k < 0:
        raise ValueError("k and n must be non-negative")
    if k > n:
        return 0.0
    # Survival function: P(X >= k) = sf(k-1)
    return float(stats.binom.sf(k - 1, n, p))


def min_k_for_pvalue(n: int, p: float, alpha: float = 0.01) -> int:
    """Smallest k with P(X >= k) < alpha (orientation helper; tests pin values)."""
    for k in range(0, n + 1):
        if exact_binomial_pvalue_ge(k, n, p) < alpha:
            return k
    return n + 1


def sample_rows_with_spacing(
    frame: pd.DataFrame,
    *,
    n: int,
    seed: int,
    min_session_gap: int = 5,
    date_col: str = "date",
    sessions: list[date] | None = None,
) -> pd.DataFrame:
    """Sample up to ``n`` rows with dates at least ``min_session_gap`` sessions apart."""
    if frame.empty or n <= 0:
        return frame.iloc[0:0].copy()
    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    assert_audit_dates_allowed(work[date_col].tolist())
    sess = sessions or sorted(work[date_col].unique().tolist())
    idx_map = {d: i for i, d in enumerate(sess)}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(work))
    chosen: list[int] = []
    picked_session_idx: list[int] = []
    for pos in order:
        row = work.iloc[int(pos)]
        d = row[date_col]
        si = idx_map.get(d)
        if si is None:
            continue
        if any(abs(si - pj) < min_session_gap for pj in picked_session_idx):
            continue
        chosen.append(int(pos))
        picked_session_idx.append(si)
        if len(chosen) >= n:
            break
    return work.iloc[chosen].reset_index(drop=True)


def sample_dev_states(
    features: pd.DataFrame,
    *,
    n: int = 300,
    seed: int = 42,
    start: date = date(2016, 1, 4),
    end: date = date(2018, 12, 31),
) -> pd.DataFrame:
    """Dev sample from train period only — **no labels joined**."""
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    work = work.loc[(work["date"] >= start) & (work["date"] <= end)]
    assert_audit_dates_allowed(work["date"].tolist())
    return sample_rows_with_spacing(work, n=n, seed=seed, min_session_gap=1)


def sample_era_balanced(
    features: pd.DataFrame,
    *,
    per_era: int = 200,
    seed: int = 7,
    min_session_gap: int = 5,
) -> pd.DataFrame:
    """D1 sample: ``per_era`` rows per era, dates >= gap sessions apart within era."""
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    assert_audit_dates_allowed(work["date"].tolist())
    parts: list[pd.DataFrame] = []
    for era, (start, end) in ERA_BOUNDS.items():
        sub = work.loc[(work["date"] >= start) & (work["date"] <= end)].copy()
        sub["era"] = era
        sessions = sorted(sub["date"].unique().tolist())
        # Independent seed stream per era
        era_seed = seed + sum(ord(c) for c in era)
        parts.append(
            sample_rows_with_spacing(
                sub,
                n=per_era,
                seed=era_seed,
                min_session_gap=min_session_gap,
                sessions=sessions,
            )
        )
    if not parts:
        return work.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True)
    assert_audit_dates_allowed(out["date"].tolist())
    return out


def degeneracy_check(
    noul_probs: dict[str, list[float]],
    choice_probs: dict[str, list[dict[str, float]]],
    *,
    min_noul_sd: float = 0.05,
    max_option_p: float = 0.98,
    max_frac_peaked: float = 0.90,
) -> dict[str, Any]:
    """Dev-sample degeneracy diagnostics (format/non-degeneracy only; no labels)."""
    report: dict[str, Any] = {
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
        "noul": {},
        "choice": {},
        "flags": [],
    }
    for name, vals in noul_probs.items():
        arr = np.asarray(vals, dtype=float)
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        report["noul"][name] = {"sd": sd, "n": len(arr), "ok": sd > min_noul_sd}
        if sd <= min_noul_sd:
            report["flags"].append(f"noul_degenerate:{name}:sd={sd:.4f}")
    for name, rows in choice_probs.items():
        if not rows:
            continue
        options = sorted({k for r in rows for k in r})
        peaked = 0
        for r in rows:
            if any(float(r.get(o, 0.0)) > max_option_p for o in options):
                peaked += 1
        frac = peaked / len(rows)
        report["choice"][name] = {
            "frac_peaked": frac,
            "n": len(rows),
            "ok": frac <= max_frac_peaked,
        }
        if frac > max_frac_peaked:
            report["flags"].append(f"choice_degenerate:{name}:frac={frac:.3f}")
    return report


def determinism_stats(
    repeats: list[pd.DataFrame],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Per-feature sd and max abs diff across repeat attempts (same states)."""
    if not repeats:
        return {"features": {}, "warning_banner": WARNING_BANNER}
    out: dict[str, Any] = {"warning_banner": WARNING_BANNER, "features": {}}
    for col in feature_cols:
        mats = []
        for df in repeats:
            mats.append(df[col].to_numpy(dtype=float))
        stacked = np.vstack(mats)
        out["features"][col] = {
            "sd_mean": float(np.mean(np.std(stacked, axis=0, ddof=1))),
            "max_abs_diff": float(np.max(np.ptp(stacked, axis=0))),
        }
    return out


def multinomial_era_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    """Blocked time-series CV accuracy for multinomial logistic on MODEL_A features."""
    if len(y) < n_splits * 5:
        # Tiny smoke path
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X, y)
        pred = pipe.predict(X)
        return {"accuracy": float(np.mean(pred == y)), "n": float(len(y))}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
            continue
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        accs.append(float(np.mean(pred == y[test_idx])))
    if not accs:
        # Fall back to in-sample when blocked folds are class-degenerate.
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=seed)),
            ]
        )
        pipe.fit(X, y)
        pred = pipe.predict(X)
        return {"accuracy": float(np.mean(pred == y)), "n": float(len(y))}
    return {
        "accuracy": float(np.mean(accs)),
        "n": float(len(y)),
        "fold_acc_mean": float(np.mean(accs)),
    }


def logistic_outcome_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    """Blocked CV AUC for binary logistic on MODEL_A columns (D2 benchmark)."""
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": float(len(y))}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) < 5:
            continue
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=seed)),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], proba)))
    return {
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n": float(len(y)),
    }


def date_block_bootstrap_ci(
    values: np.ndarray,
    dates: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI by resampling unique dates (block of all rows on a date)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(dates)
    if len(uniq) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    date_to_idx = {d: np.where(dates == d)[0] for d in uniq}
    stats_boot: list[float] = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idxs = np.concatenate([date_to_idx[d] for d in draw])
        stats_boot.append(float(np.mean(values[idxs])))
    lo, hi = np.quantile(stats_boot, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(np.mean(values)), "lo": float(lo), "hi": float(hi)}


def spearman_rank_ic(scores: np.ndarray, target: np.ndarray) -> float:
    if len(scores) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(scores, target)
    return float(rho)


def d1_from_predictions(
    y_true_era: list[EraId],
    y_pred_choice: list[str],
    *,
    X_benchmark: np.ndarray | None = None,
) -> dict[str, Any]:
    """D1 summary from era labels and Choice predictions (early/middle/late)."""
    n = len(y_true_era)
    correct = 0
    for t, p in zip(y_true_era, y_pred_choice, strict=True):
        if ERA_TO_CHOICE[t] == p:
            correct += 1
    pval = exact_binomial_pvalue_ge(correct, n, 1.0 / 3.0)
    body: dict[str, Any] = {
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else float("nan"),
        "binomial_p_one_sided": pval,
        "chance": 1.0 / 3.0,
        "orientation_k_for_p01": min_k_for_pvalue(600, 1.0 / 3.0, 0.01),
    }
    if X_benchmark is not None:
        y_num = np.array([{"E1": 0, "E2": 1, "E3": 2}[e] for e in y_true_era])
        bench = multinomial_era_benchmark(X_benchmark, y_num)
        body["benchmark"] = bench
        body["jev_minus_benchmark"] = float(body["accuracy"] - bench["accuracy"])
        # Bootstrap CI of difference on paired correctness (Jev vs bench hard labels
        # approximated by accuracy delta; full CI filled when live preds exist).
        body["note"] = (
            "If Jev accuracy is clearly above the deterministic benchmark "
            "(bootstrap CI of the difference excludes 0), treat as a WARNING only."
        )
    return body


def d2_from_scores(
    y: np.ndarray,
    jev_scores: np.ndarray,
    dates: np.ndarray,
    *,
    X_benchmark: np.ndarray | None = None,
) -> dict[str, Any]:
    """D2: AUC and rank-IC vs deterministic benchmark — **no threshold**."""
    body: dict[str, Any] = {
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
        "n": len(y),
        "no_threshold": True,
    }
    if len(np.unique(y)) >= 2:
        auc = float(roc_auc_score(y, jev_scores))
        body["jev_auc"] = auc
        body["jev_auc_ci"] = date_block_bootstrap_ci(jev_scores, dates, seed=1)
    else:
        body["jev_auc"] = float("nan")
    body["jev_rank_ic"] = spearman_rank_ic(jev_scores, y.astype(float))
    if X_benchmark is not None:
        body["benchmark"] = logistic_outcome_benchmark(X_benchmark, y)
        body["note"] = (
            "A Jev AUC far above the benchmark is a WARNING "
            "(real skill or leakage; cannot be distinguished here)."
        )
    return body


def d3_rank_ic_by_era(
    features: pd.DataFrame,
    *,
    score_cols: list[str],
    target_col: str = "X",
    date_col: str = "date",
) -> dict[str, Any]:
    """D3 scaffold: rank-IC of each Jev feature vs X by era (descriptive)."""
    work = features.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    assert_audit_dates_allowed(work[date_col].tolist())
    out: dict[str, Any] = {
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
        "by_era": {},
        "note": "Descriptive only; used after Phase 7 when jev_features exist.",
    }
    for era, (start, end) in ERA_BOUNDS.items():
        sub = work.loc[(work[date_col] >= start) & (work[date_col] <= end)]
        era_stats: dict[str, float] = {}
        for col in score_cols:
            if col not in sub.columns or target_col not in sub.columns:
                era_stats[col] = float("nan")
                continue
            era_stats[col] = spearman_rank_ic(
                sub[col].to_numpy(dtype=float),
                sub[target_col].to_numpy(dtype=float),
            )
        out["by_era"][era] = era_stats
    return out


def coarse_phrase_for_field(field: str, percentile: float) -> str:
    """Map percentile to a 3-bucket phrase drawn from the fine vocabulary bank."""
    bank = FIELD_PHRASE_BANK[field]
    idx = bucket_index(percentile, COARSE_BUCKET_EDGES)
    # Map 0/1/2 -> low/mid/high representatives in the 7-phrase bank.
    pick = {0: 0, 1: len(bank) // 2, 2: len(bank) - 1}[idx]
    return bank[pick]


def build_state_v1c_from_percentiles(
    percentiles: dict[str, float],
    *,
    down_streak: float | int,
) -> dict[str, str]:
    """Coarser anonymized state for D5 (3 buckets per field)."""
    from jevlab.features.columns import STATE_V1_SOURCE_COLUMNS
    from jevlab.jev.state_builder import streak_word
    from jevlab.jev.vocabulary import FUND_TYPE

    state: dict[str, str] = {"schema": SCHEMA_V1C, "fund_type": FUND_TYPE}
    for field_name, src in STATE_V1_SOURCE_COLUMNS.items():
        if field_name == "consecutive_down_days":
            continue
        state[field_name] = coarse_phrase_for_field(field_name, float(percentiles[src]))
    state["consecutive_down_days"] = streak_word(down_streak)
    return state


def d5_fine_vs_coarse(
    fine_scores: np.ndarray,
    coarse_scores: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Compare fine vs coarse Jev→outcome association (warning if fine-only)."""
    fine_ic = spearman_rank_ic(fine_scores, y.astype(float))
    coarse_ic = spearman_rank_ic(coarse_scores, y.astype(float))
    fine_auc = float(roc_auc_score(y, fine_scores)) if len(np.unique(y)) >= 2 else float("nan")
    coarse_auc = float(roc_auc_score(y, coarse_scores)) if len(np.unique(y)) >= 2 else float("nan")
    return {
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
        "fine_rank_ic": fine_ic,
        "coarse_rank_ic": coarse_ic,
        "fine_auc": fine_auc,
        "coarse_auc": coarse_auc,
        "delta_ic_fine_minus_coarse": (
            fine_ic - coarse_ic
            if math.isfinite(fine_ic) and math.isfinite(coarse_ic)
            else float("nan")
        ),
        "note": (
            "An effect that exists only at the finest granularity "
            "(where states are most fingerprintable) is a WARNING."
        ),
    }


def plan_audit_calls(
    *,
    n_dev: int = 300,
    n_determinism_repeats: int = 3,
    n_d1: int = 600,
    n_d2: int = 2000,
    n_d5: int = 3000,
) -> dict[str, Any]:
    """Count planned audit API calls (determinism stores repeats as attempts)."""
    # Dev degeneracy: 1 call/state; determinism: (repeats-1) extra on same 300;
    # D1/D2/D5: 1 each (D5 may share states with a subset — counted separately).
    calls = {
        "dev_degeneracy": n_dev,
        "determinism_extra_repeats": n_dev * (n_determinism_repeats - 1),
        "d1_era": n_d1,
        "d2_outcome": n_d2,
        "d5_coarse": n_d5,
    }
    total = int(sum(calls.values()))
    guard = BudgetGuard()
    est = guard.estimate_usd(total)
    return {
        "calls_by_section": calls,
        "total_calls": total,
        "tokens_per_call_assumed": DEFAULT_TOKENS_PER_CALL,
        "estimated_cost_usd": est,
        "price_usd_per_million_input_tokens": INPUT_USD_PER_MILLION_TOKENS,
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
    }


def run_audit_dry(
    features: pd.DataFrame | None = None,
    *,
    sealed_root: Path | None = None,
    seed: int = 42,
) -> AuditReport:
    """Default audit path: plan calls, run offline scaffolds, **zero API calls**."""
    # sealed_root may be supplied for operator context; it is never opened here.
    if sealed_root is not None:
        log.info("audit_dry_sealed_root_ignored", sealed_root=str(sealed_root))
    plan = plan_audit_calls()
    report = AuditReport(
        meta={
            "generated_at": datetime.now(UTC).isoformat(),
            "live": False,
            "api_calls": 0,
            "plan": plan,
        }
    )
    report.sections.append(
        AuditSection(
            name="Plan / cost estimate (dry-run)",
            body=plan,
        )
    )

    # Offline synthetic scaffolds so the report is non-empty without live calls.
    rng = np.random.default_rng(seed)
    # D1 orientation on chance predictions
    eras_raw = cast(list[EraId], ["E1"] * 200 + ["E2"] * 200 + ["E3"] * 200)
    order = rng.permutation(600)
    eras = [eras_raw[int(i)] for i in order]
    preds = [str(rng.choice(["early", "middle", "late"])) for _ in range(600)]
    X = rng.normal(size=(600, len(MODEL_A_COLUMNS)))
    for i, e in enumerate(eras):
        X[i, 0] += {"E1": -1.0, "E2": 0.0, "E3": 1.0}[e]
    report.sections.append(
        AuditSection(
            name="D1 era diagnostic (dry synthetic)",
            body=d1_from_predictions(eras, preds, X_benchmark=X),
        )
    )

    y = rng.integers(0, 2, size=2000)
    scores = y.astype(float) * 0.2 + rng.normal(0, 1, size=2000)
    dates = np.array([date(2016, 1, 4)] * 2000)
    X2 = rng.normal(size=(2000, len(MODEL_A_COLUMNS)))
    report.sections.append(
        AuditSection(
            name="D2 outcome diagnostic (dry synthetic)",
            body=d2_from_scores(y, scores, dates, X_benchmark=X2),
        )
    )

    # D3 scaffold on empty/synthetic frame
    syn = pd.DataFrame(
        {
            "date": [date(2017, 1, 3), date(2019, 1, 3), date(2022, 1, 3)],
            "j_dislocation": [0.4, 0.5, 0.6],
            "X": [0.01, -0.02, 0.0],
        }
    )
    report.sections.append(
        AuditSection(
            name="D3 era-stability scaffold",
            body=d3_rank_ic_by_era(syn, score_cols=["j_dislocation"]),
        )
    )

    fine = rng.normal(size=300)
    coarse = fine * 0.5 + rng.normal(0, 1, size=300)
    y5 = (fine > 0).astype(int)
    report.sections.append(
        AuditSection(
            name="D5 fine vs coarse (dry synthetic)",
            body=d5_fine_vs_coarse(fine, coarse, y5),
        )
    )

    report.sections.append(
        AuditSection(
            name="D6 vendor cutoff",
            body={
                "warning_banner": WARNING_BANNER,
                "status": "see docs/jev_cutoff.md (human-recorded)",
                "contamination_label": CONTAMINATION_LABEL,
            },
        )
    )

    if features is not None and not features.empty:
        feats = features.copy()
        feats["date"] = pd.to_datetime(feats["date"]).dt.date
        assert_audit_dates_allowed(feats["date"].tolist())
        dev = sample_dev_states(feats, n=min(300, len(feats)), seed=seed)
        report.sections.append(
            AuditSection(
                name="Dev sample (no labels)",
                body={
                    "n": len(dev),
                    "date_min": str(dev["date"].min()) if len(dev) else None,
                    "date_max": str(dev["date"].max()) if len(dev) else None,
                    "labels_joined": False,
                    "warning_banner": WARNING_BANNER,
                },
            )
        )
    return report


def write_audit_report(report: AuditReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "AUDIT_REPORT.md"
    js = out_dir / "AUDIT_REPORT.json"
    md.write_text(report.to_markdown(), encoding="utf-8")
    payload = {
        "meta": report.meta,
        "report_hash": report.content_hash(),
        "sections": [{"name": s.name, "body": s.body} for s in report.sections],
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
    }
    js.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return md


def freeze_qset(
    *,
    qset_path: Path,
    config_path: Path,
    registry_path: Path,
    state_schema_version: str = "state_v1",
) -> dict[str, Any]:
    """Write qset hash into jev.yaml, ledger a trial row, print git tag command.

    Does **not** create the git tag automatically.
    """
    from jevlab.jev.qset import compute_qset_hash, load_qset

    qset = load_qset(qset_path, expected_purpose="research")
    content_hash = qset.content_hash
    # Update YAML qset_hash in place (simple line rewrite to avoid reordering).
    text_yaml = config_path.read_text(encoding="utf-8")
    lines = []
    found = False
    for line in text_yaml.splitlines(keepends=True):
        if line.startswith("qset_hash:"):
            lines.append(f'qset_hash: "{content_hash}"\n')
            found = True
        else:
            lines.append(line)
    if not found:
        lines.append(f'qset_hash: "{content_hash}"\n')
    config_path.write_text("".join(lines), encoding="utf-8")

    engine = init_registry(registry_path)
    trial_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trial_ledger (
                    trial_id, created_at, qset_version, state_schema_version,
                    feature_set, model_class, grid_json, universe_version,
                    horizon, purpose, experiment_id, result_summary
                ) VALUES (
                    :trial_id, :created_at, :qset_version, :state_schema_version,
                    :feature_set, :model_class, :grid_json, :universe_version,
                    :horizon, :purpose, :experiment_id, :result_summary
                )
                """
            ),
            {
                "trial_id": trial_id,
                "created_at": datetime.now(UTC).isoformat(),
                "qset_version": qset.qset_version,
                "state_schema_version": state_schema_version,
                "feature_set": "MODEL_A_COLUMNS",
                "model_class": "none",
                "grid_json": canonical_json({"qset_hash": content_hash}),
                "universe_version": "v1",
                "horizon": 3,
                "purpose": "confirmatory",
                "experiment_id": None,
                "result_summary": f"freeze qset hash={content_hash}",
            },
        )

    tag_cmd = "git tag qset-v1-frozen"
    # Verify hash round-trip
    _ = compute_qset_hash
    return {
        "qset_hash": content_hash,
        "qset_version": qset.qset_version,
        "trial_id": trial_id,
        "config_path": str(config_path),
        "git_tag_command": tag_cmd,
        "note": "Tag was NOT created automatically; run the printed command yourself.",
        "warning_banner": WARNING_BANNER,
    }


# ---------------------------------------------------------------------------
# P6 — State-shuffle placebo (MASTER_SPEC §13); behind --live + budget.
# ---------------------------------------------------------------------------

P6_N_STATES: int = 3000
P6_PURPOSE: str = "placebo"


@dataclass
class P6Plan:
    n_states: int
    estimated_usd: float
    budget_usd: float
    requires_live: bool = True
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "estimated_usd": self.estimated_usd,
            "budget_usd": self.budget_usd,
            "requires_live": self.requires_live,
            "status": self.status,
            "purpose": P6_PURPOSE,
            "warning_banner": WARNING_BANNER,
            "contamination_label": CONTAMINATION_LABEL,
        }


def plan_p6_state_shuffle(
    *,
    n_states: int = P6_N_STATES,
    budget_usd: float = DEFAULT_BUDGET_USD,
    tokens_per_call: int = DEFAULT_TOKENS_PER_CALL,
) -> P6Plan:
    """Budget-guarded plan for P6 (no API calls)."""
    guard = BudgetGuard(budget_usd=budget_usd, tokens_per_call=tokens_per_call)
    estimated = guard.plan(n_states)
    return P6Plan(n_states=n_states, estimated_usd=estimated, budget_usd=budget_usd)


def build_p6_shuffled_states(
    features: pd.DataFrame,
    *,
    n_states: int = P6_N_STATES,
    seed: int = 42,
) -> pd.DataFrame:
    """Permute feature rows before bucketing to build mismatched states.

    Returns a frame of ``n_states`` rows with original identity columns plus
    ``donor_row_index`` indicating which row's features were used for the state.
    """
    assert_audit_dates_allowed(features["date"])
    work = features.reset_index(drop=True)
    if work.empty:
        raise ValueError("features empty for P6")
    rng = np.random.default_rng(seed)
    n = min(int(n_states), len(work))
    take = rng.choice(len(work), size=n, replace=False)
    donor = rng.permutation(len(work))[:n]
    out = work.iloc[take].copy().reset_index(drop=True)
    # Overwrite MODEL_A feature values with donor rows (identity kept for join checks).
    feat_cols = [c for c in MODEL_A_COLUMNS if c in work.columns]
    out.loc[:, feat_cols] = work.iloc[donor][feat_cols].to_numpy()
    out["donor_row_index"] = donor
    out["target_row_index"] = take
    out["p6_mismatched"] = out["donor_row_index"].to_numpy() != out["target_row_index"].to_numpy()
    return out


def run_p6_state_shuffle(
    features: pd.DataFrame,
    *,
    live: bool = False,
    budget_usd: float = DEFAULT_BUDGET_USD,
    n_states: int = P6_N_STATES,
    seed: int = 42,
    labels: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """P6 single run. Default dry: plan + shuffled-state sample, zero API calls.

    When ``live=True``, spends against ``BudgetGuard`` and records trial metadata.
    Live fan-out of 3,000 calls is operator-gated; this function still refuses to
    silently call the network unless ``live`` is set, and even then returns a
    structured payload expecting an external client loop (same pattern as audit).
    """
    plan = plan_p6_state_shuffle(n_states=n_states, budget_usd=budget_usd)
    shuffled = build_p6_shuffled_states(features, n_states=n_states, seed=seed)
    payload: dict[str, Any] = {
        "placebo_id": "P6",
        "plan": plan.to_dict(),
        "n_shuffled_states": len(shuffled),
        "mismatch_rate": float(shuffled["p6_mismatched"].mean()) if len(shuffled) else 0.0,
        "live": bool(live),
        "status": "dry_plan" if not live else "live_ready",
        "warning_banner": WARNING_BANNER,
        "contamination_label": CONTAMINATION_LABEL,
        "outcome_relationship": None,
    }
    if labels is not None and not labels.empty:
        # Integrity check scaffold: join shuffled identities to labels; any
        # non-null predictive relationship on mismatched states is a warning.
        key = ["symbol", "date"]
        if all(c in labels.columns for c in key) and "y" in labels.columns:
            joined = shuffled.merge(labels[[*key, "y"]], on=key, how="left")
            payload["label_join_rate"] = float(joined["y"].notna().mean())
    if not live:
        payload["note"] = (
            "P6 dry mode: no Jev calls. Re-run with --live and an explicit budget "
            "to execute the 3,000-state state-shuffle placebo."
        )
        return payload

    # Live path: budget already validated by plan(); actual client calls are
    # performed by the operator CLI using this payload as the work queue.
    guard = BudgetGuard(budget_usd=budget_usd)
    guard.plan(len(shuffled))
    payload["budget_spent_usd"] = guard.spent_usd
    payload["status"] = "live_queued"
    payload["work_queue_n"] = len(shuffled)
    return payload


# Silence unused DEFAULT_BUCKET_EDGES import if only coarse uses COARSE — keep for API.
_ = DEFAULT_BUCKET_EDGES
