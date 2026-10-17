"""Execute missing Phase 10 live items (P6 + paraphrase) and offline gaps.

Does not modify the frozen research qset. All historical Jev outputs are
``potentially_training_contaminated``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jevlab.backtest.costs import CostModel
from jevlab.backtest.fast import run_fast
from jevlab.config.loader import ResearchConfigs, load_all_configs
from jevlab.config.models import RiskConfig
from jevlab.cv.walkforward import Fold, make_folds
from jevlab.evaluation.placebo import CONTAMINATION_LABEL, delta_log_loss_gain, sharpe_from_equity, train_b_oos
from jevlab.evaluation.predictive import binary_log_loss, daily_rank_ic_summary
from jevlab.jev.audit import (
    BudgetGuard,
    BudgetExceededError,
    assert_audit_dates_allowed,
)
from jevlab.jev.cache import JevCache, request_hash
from jevlab.jev.client import JevClient
from jevlab.jev.parse import JEV_FEATURE_COLUMNS, parse_response
from jevlab.jev.privacy import assert_state_private, validate_state_privacy
from jevlab.jev.qset import QuestionSet, compute_qset_hash, load_qset
from jevlab.jev.state_builder import build_states_frame
from jevlab.jev.vocabulary import CLOSED_VOCABULARY
from jevlab.models.column_sets import column_set_registry
from jevlab.models.pipelines import prepare_model_frame, run_column_set_logistic
from jevlab.targets.labels import build_labels
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger
from jevlab.utils.paths import ProjectPaths, get_project_root

log = get_logger(__name__)

PINNED_MODEL: str = "jev-1.13.0"


@dataclass
class LiveCallStats:
    n_attempted: int = 0
    n_ok: int = 0
    n_cache_hit: int = 0
    n_failed: int = 0
    n_retries: int = 0
    input_tokens: int = 0
    spent_usd: float = 0.0
    resolved_models: list[str] = field(default_factory=list)


def _checkpoint_db(path: Path) -> Path:
    """Ensure checkpoint schema exists; return path (connections opened per call)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path), timeout=60) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS phase10_live_jobs (
                job_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model_resolved TEXT,
                error TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
    return path


def _job_done(db_path: Path, job_key: str) -> bool:
    with sqlite3.connect(str(db_path), timeout=60) as conn:
        row = conn.execute(
            "SELECT status FROM phase10_live_jobs WHERE job_key=? AND status='ok'",
            (job_key,),
        ).fetchone()
    return row is not None


def _mark_job(
    db_path: Path,
    job_key: str,
    status: str,
    *,
    model_resolved: str | None = None,
    error: str | None = None,
) -> None:
    with sqlite3.connect(str(db_path), timeout=60) as conn:
        conn.execute(
            """
            INSERT INTO phase10_live_jobs(job_key, status, model_resolved, error, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(job_key) DO UPDATE SET
                status=excluded.status,
                model_resolved=excluded.model_resolved,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            (job_key, status, model_resolved, error),
        )
        conn.commit()


def build_p6_mismatched_states(
    features: pd.DataFrame,
    *,
    n_states: int = 3000,
    seed: int = 42,
    bucket_edges: tuple[float, ...] | list[float],
) -> pd.DataFrame:
    """Permute rows before bucketing: build states on donor features, keep target IDs.

    Sample only from rows that already admit a ``state_v1`` (warm history), permute
    MODEL_A feature values from donor rows onto targets, then rebuild states on the
    patched panel so bucketing happens *after* the permutation.
    """
    assert_audit_dates_allowed(features["date"])
    from jevlab.features.columns import MODEL_A_COLUMNS

    true_states = build_states_frame(features, edges=tuple(bucket_edges))
    true_states["date"] = pd.to_datetime(true_states["date"]).dt.date
    work = features.reset_index(drop=True).copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    work["_row_idx"] = np.arange(len(work), dtype=int)
    # Restrict to rows with buildable states
    eligible = work.merge(true_states[["symbol", "date"]], on=["symbol", "date"], how="inner")
    eligible_idx = eligible["_row_idx"].to_numpy()
    if len(eligible_idx) < n_states:
        raise ValueError(f"only {len(eligible_idx)} state-capable rows; need {n_states}")
    rng = np.random.default_rng(seed)
    take_pos = rng.choice(len(eligible_idx), size=n_states, replace=False)
    # Donors also from state-capable rows
    donor_pos = rng.choice(len(eligible_idx), size=n_states, replace=False)
    same = donor_pos == take_pos
    if same.any():
        donor_pos = donor_pos.copy()
        for i in np.where(same)[0]:
            donor_pos[i] = (donor_pos[i] + 1) % len(eligible_idx)

    target_row_index = eligible_idx[take_pos]
    donor_row_index = eligible_idx[donor_pos]
    feat_cols = [c for c in MODEL_A_COLUMNS if c in work.columns]
    # Patch targets with donor feature values (permute before bucketing)
    patched = work.copy()
    patched.loc[target_row_index, feat_cols] = work.loc[donor_row_index, feat_cols].to_numpy()
    mismatched_states = build_states_frame(patched, edges=tuple(bucket_edges))
    mismatched_states["date"] = pd.to_datetime(mismatched_states["date"]).dt.date

    meta = work.loc[target_row_index, ["symbol", "date"]].copy().reset_index(drop=True)
    meta["target_row_index"] = target_row_index
    meta["donor_row_index"] = donor_row_index
    meta["p6_mismatched"] = meta["donor_row_index"].to_numpy() != meta["target_row_index"].to_numpy()
    out = meta.merge(
        mismatched_states.rename(columns={"state": "mismatched_state"}),
        on=["symbol", "date"],
        how="left",
    )
    out = out.merge(
        true_states.rename(columns={"state": "true_state"}),
        on=["symbol", "date"],
        how="left",
    )
    return out


def _infer_one(
    *,
    client: JevClient,
    cache: JevCache,
    qset: QuestionSet,
    route: str,
    model_id: str,
    state: dict[str, str],
    state_schema_version: str,
) -> tuple[dict[str, float], str, int, bool]:
    """Return (features, model_resolved, input_tokens, cache_hit)."""
    from jevlab.jev.schemas import SystemOneResponse

    violations = validate_state_privacy(state, CLOSED_VOCABULARY)
    if violations:
        raise ValueError(f"privacy violations: {violations}")
    assert_state_private(state, CLOSED_VOCABULARY)
    questions = qset.as_request_questions()
    cacheable = qset.to_cacheable_dict()
    rh = request_hash(route=route, model_requested=model_id, state=state, questions=cacheable)
    hit = cache.get(rh)
    if hit is not None:
        resp = SystemOneResponse.model_validate(hit.response)
        feats = parse_response(resp)
        resolved = str(resp.model or hit.model_resolved or model_id)
        return feats, resolved, 0, True
    resp = client.infer(state, questions, model=model_id)
    resolved = str(resp.model)
    if resolved != model_id:
        raise RuntimeError(f"model_resolved mismatch: {resolved} != {model_id}")
    feats = parse_response(resp)
    usage = resp.usage
    tokens = int(usage.input_tokens or 0) if usage is not None else 0
    try:
        cache.put(
            req_hash=rh,
            route=route,
            model_requested=model_id,
            model_resolved=resolved,
            purpose="research",
            qset_version=qset.qset_version,
            state_schema_version=state_schema_version,
            response=resp.model_dump(mode="json"),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("cache.put_failed", error=str(exc))
    return feats, resolved, tokens, False


def execute_p6_live(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    configs: ResearchConfigs,
    paths: ProjectPaths,
    n_states: int = 3000,
    seed: int = 42,
    budget_usd: float = 25.0,
    max_workers: int = 4,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run real P6: 3000 mismatched-state Jev calls + integrity evaluation."""
    out_dir = Path(out_dir or paths.reports / "robustness_report" / "p6_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _checkpoint_db(out_dir / "p6_checkpoint.sqlite")  # path
    qset = load_qset(
        paths.root / configs.jev.qset_path,
        expected_hash=configs.jev.qset_hash,
        expected_purpose="research",
    )
    model_id = configs.jev.model_id
    if model_id != PINNED_MODEL:
        raise RuntimeError(f"pinned model must be {PINNED_MODEL}, got {model_id}")
    panel = build_p6_mismatched_states(
        features,
        n_states=n_states,
        seed=seed,
        bucket_edges=tuple(configs.jev.bucket_edges),
    )
    panel = panel.loc[panel["mismatched_state"].notna()].reset_index(drop=True)
    # Join original outcomes (target identity)
    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"]).dt.date
    panel = panel.merge(lab[["symbol", "date", "y", "X"]], on=["symbol", "date"], how="left")

    client = JevClient(config=configs.jev)
    cache = JevCache(paths.data_jev_cache if hasattr(paths, "data_jev_cache") else paths.root / "data" / "jev_cache")
    # ProjectPaths may use data_jev_cache or similar
    cache_dir = Path(configs.jev.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = paths.root / cache_dir
    cache = JevCache(cache_dir)

    guard = BudgetGuard(budget_usd=budget_usd)
    # Plan only uncached / not-yet-done calls after resume scan
    stats = LiveCallStats()
    lock = threading.Lock()
    ckpt_lock = threading.Lock()
    rows_out: list[dict[str, Any]] = []

    # Resume: load prior ok rows from parquet if present
    prior_path = out_dir / "p6_responses.parquet"
    done_keys: set[str] = set()
    if prior_path.is_file():
        prior = pd.read_parquet(prior_path)
        rows_out.extend([r for r in prior.to_dict(orient="records") if r.get("status") == "ok"])
        done_keys = {
            f"{r['symbol']}|{r['date']}|{r['donor_row_index']}" for r in rows_out
        }

    n_resume = len(done_keys)
    n_plan = max(0, len(panel) - n_resume)
    if n_plan:
        guard.plan(n_plan)

    def _work(i: int) -> dict[str, Any]:
        row = panel.iloc[i]
        job_key = f"p6|{row['symbol']}|{row['date']}|{int(row['donor_row_index'])}"
        with ckpt_lock:
            already = job_key in done_keys or _job_done(ckpt, job_key)
        if already:
            return {"job_key": job_key, "status": "skipped_resume"}
        state = row["mismatched_state"]
        if not isinstance(state, dict):
            raise TypeError("mismatched_state must be dict")
        state_s = {str(k): str(v) for k, v in state.items()}
        last_err: Exception | None = None
        for attempt in range(3):
            with lock:
                stats.n_attempted += 1
                if attempt:
                    stats.n_retries += 1
            try:
                feats, resolved, tokens, cache_hit = _infer_one(
                    client=client,
                    cache=cache,
                    qset=qset,
                    route=configs.jev.route,
                    model_id=model_id,
                    state=state_s,
                    state_schema_version=configs.jev.state_schema_version,
                )
                with lock:
                    stats.n_ok += 1
                    if cache_hit:
                        stats.n_cache_hit += 1
                    else:
                        t = tokens or guard.tokens_per_call
                        stats.spent_usd += guard.record_usage(t)
                        stats.input_tokens += t
                    if resolved not in stats.resolved_models:
                        stats.resolved_models.append(resolved)
                with ckpt_lock:
                    _mark_job(ckpt, job_key, "ok", model_resolved=resolved)
                return {
                    "symbol": row["symbol"],
                    "date": str(row["date"]),
                    "donor_row_index": int(row["donor_row_index"]),
                    "target_row_index": int(row["target_row_index"]),
                    "p6_mismatched": bool(row["p6_mismatched"]),
                    "y": row.get("y"),
                    "X": row.get("X"),
                    "model_resolved": resolved,
                    "cache_hit": cache_hit,
                    "status": "ok",
                    **{c: feats.get(c) for c in JEV_FEATURE_COLUMNS},
                }
            except BudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2**attempt, 8))
        with lock:
            stats.n_failed += 1
        with ckpt_lock:
            _mark_job(ckpt, job_key, "error", error=str(last_err)[:500] if last_err else "unknown")
        return {
            "symbol": row["symbol"],
            "date": str(row["date"]),
            "donor_row_index": int(row["donor_row_index"]),
            "status": "error",
            "error": str(last_err)[:500] if last_err else "unknown",
        }

    pending_idx = [
        i
        for i in range(len(panel))
        if f"p6|{panel.iloc[i]['symbol']}|{panel.iloc[i]['date']}|{int(panel.iloc[i]['donor_row_index'])}"
        not in done_keys
    ]
    log.info("p6.live.start", n_total=len(panel), n_pending=len(pending_idx), workers=max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_work, i): i for i in pending_idx}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            if rec.get("status") == "ok":
                rows_out.append(rec)
            done += 1
            if done % 50 == 0 or done == len(pending_idx):
                log.info("p6.live.progress", done=done, pending=len(pending_idx), ok=stats.n_ok, fail=stats.n_failed)
                pd.DataFrame(rows_out).to_parquet(prior_path, index=False)

    result_df = pd.DataFrame(rows_out)
    if not result_df.empty:
        result_df.to_parquet(prior_path, index=False)
        result_df.to_csv(out_dir / "p6_responses.csv", index=False)

    # Integrity: mismatched Jev vs original-row outcomes
    ok = result_df.loc[result_df.get("status", "ok") == "ok"].copy() if not result_df.empty else result_df
    outcome: dict[str, Any] = {"n_ok": int(len(ok))}
    pipeline_warning = False
    if not ok.empty and ok["y"].notna().sum() >= 50:
        y = ok["y"].astype(float).to_numpy()
        # Use j_dislocation / j_pullback_p as scores
        for col in ("j_dislocation", "j_pullback_p", "j_regime", "j_setup_conf"):
            if col not in ok.columns:
                continue
            s = ok[col].astype(float).to_numpy()
            mask = np.isfinite(y) & np.isfinite(s)
            if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
                continue
            auc = float(roc_auc_score(y[mask], s[mask]))
            # rank-IC vs X if present
            ic = float("nan")
            if "X" in ok.columns:
                ic = float(ok.loc[mask, col].corr(ok.loc[mask, "X"], method="spearman"))
            outcome[f"auc_{col}"] = auc
            outcome[f"ic_{col}"] = ic
            # Warning if predictive vs original outcomes (AUC away from 0.5)
            if abs(auc - 0.5) > 0.05:
                pipeline_warning = True
        outcome["label_coverage"] = float(ok["y"].notna().mean())
    else:
        outcome["label_coverage"] = 0.0

    # Real-state comparison on same keys if canonical jev_features exist
    jev_path = paths.data_features / "jev" / configs.jev.qset_version / "jev_features.parquet"
    real_comp: dict[str, Any] = {}
    if jev_path.is_file() and not ok.empty:
        canon = pd.read_parquet(jev_path)
        canon["date"] = pd.to_datetime(canon["date"]).dt.date
        ok["date"] = pd.to_datetime(ok["date"]).dt.date
        m = ok.merge(
            canon[["symbol", "date", *JEV_FEATURE_COLUMNS]],
            on=["symbol", "date"],
            how="inner",
            suffixes=("_p6", "_real"),
        )
        corrs = {}
        for col in JEV_FEATURE_COLUMNS:
            a, b = f"{col}_p6", f"{col}_real"
            if a in m.columns and b in m.columns:
                corrs[col] = float(m[a].corr(m[b]))
        real_comp = {"n_overlap": len(m), "feature_corr_vs_real_state": corrs}

    n_ok_final = int(len(ok))
    payload = {
        "placebo_id": "P6",
        "status": "complete" if stats.n_failed == 0 and n_ok_final >= int(n_states * 0.95) else "incomplete",
        "live": True,
        "n_candidate_states": int(len(panel)),
        "n_ok": n_ok_final,
        "n_failed": int(stats.n_failed),
        "n_cache_hit": int(stats.n_cache_hit),
        "n_retries": int(stats.n_retries),
        "n_resume_ok": int(n_resume),
        "mismatch_rate": float(panel["p6_mismatched"].mean()) if len(panel) else 0.0,
        "api_calls_attempted": int(stats.n_attempted),
        "input_tokens": int(stats.input_tokens),
        "spent_usd": float(stats.spent_usd),
        "requested_model": model_id,
        "resolved_models": stats.resolved_models,
        "outcome_relationship": outcome,
        "real_state_comparison": real_comp,
        "pipeline_integrity_warning": bool(pipeline_warning),
        "contamination_label": CONTAMINATION_LABEL,
        "note": (
            "P6 is a pipeline-integrity check. A predictive relationship between "
            "mismatched-state Jev outputs and the original row's outcomes is a warning "
            "signal for join/leakage bugs — not proof about training contamination."
        ),
    }
    (out_dir / "p6_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def execute_paraphrase_live(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    configs: ResearchConfigs,
    paths: ProjectPaths,
    n_states: int = 3000,
    seed: int = 42,
    budget_usd: float = 25.0,
    max_workers: int = 4,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """2 paraphrased qsets × n_states new Jev calls (exploratory; frozen qset untouched)."""
    out_dir = Path(out_dir or paths.reports / "robustness_report" / "paraphrase_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    para_paths = [
        paths.root / "config" / "paraphrase" / "qset_paraphrase_1.json",
        paths.root / "config" / "paraphrase" / "qset_paraphrase_2.json",
    ]
    variants: list[dict[str, Any]] = []
    # Sample states from full panel (dev dates only)
    assert_audit_dates_allowed(features["date"])
    states = build_states_frame(features, edges=tuple(configs.jev.bucket_edges))
    states["date"] = pd.to_datetime(states["date"]).dt.date
    rng = np.random.default_rng(seed)
    n = min(n_states, len(states))
    take = rng.choice(len(states), size=n, replace=False)
    sample = states.iloc[take].reset_index(drop=True)
    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"]).dt.date
    sample = sample.merge(lab[["symbol", "date", "y", "X"]], on=["symbol", "date"], how="left")

    # Canonical features for correlation
    jev_path = paths.data_features / "jev" / configs.jev.qset_version / "jev_features.parquet"
    canon = pd.read_parquet(jev_path) if jev_path.is_file() else None
    if canon is not None:
        canon["date"] = pd.to_datetime(canon["date"]).dt.date

    client = JevClient(config=configs.jev)
    cache_dir = Path(configs.jev.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = paths.root / cache_dir
    cache = JevCache(cache_dir)
    model_id = configs.jev.model_id
    guard = BudgetGuard(budget_usd=budget_usd)
    # 2 variants × n states
    guard.plan(2 * n)

    frozen_hash = configs.jev.qset_hash

    for ppath in para_paths:
        qset = load_qset(ppath, expected_purpose="research")
        # Must NOT equal frozen research hash
        if qset.content_hash == frozen_hash:
            raise RuntimeError("paraphrase qset unexpectedly matches frozen hash")
        vdir = out_dir / qset.qset_version
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "qset_copy.json").write_text(ppath.read_text(encoding="utf-8"), encoding="utf-8")
        (vdir / "qset_hash.txt").write_text(qset.content_hash + "\n", encoding="utf-8")
        ckpt = _checkpoint_db(vdir / "checkpoint.sqlite")
        prior_path = vdir / "responses.parquet"
        rows_out: list[dict[str, Any]] = []
        done_keys: set[str] = set()
        if prior_path.is_file():
            prior = pd.read_parquet(prior_path)
            rows_out = prior.to_dict(orient="records")
            done_keys = {f"{r['symbol']}|{r['date']}" for r in rows_out if r.get("status") == "ok"}

        stats = LiveCallStats()
        lock = threading.Lock()

        def _work(i: int, q: QuestionSet = qset, ver: str = qset.qset_version) -> dict[str, Any]:
            row = sample.iloc[i]
            job_key = f"para|{ver}|{row['symbol']}|{row['date']}"
            if f"{row['symbol']}|{row['date']}" in done_keys or _job_done(ckpt, job_key):
                return {"status": "skipped_resume"}
            state = row["state"]
            state_s = {str(k): str(v) for k, v in dict(state).items()}
            with lock:
                stats.n_attempted += 1
            try:
                feats, resolved, tokens, cache_hit = _infer_one(
                    client=client,
                    cache=cache,
                    qset=q,
                    route=configs.jev.route,
                    model_id=model_id,
                    state=state_s,
                    state_schema_version=configs.jev.state_schema_version,
                )
                with lock:
                    stats.n_ok += 1
                    if cache_hit:
                        stats.n_cache_hit += 1
                    else:
                        t = tokens or guard.tokens_per_call
                        stats.spent_usd += guard.record_usage(t)
                        stats.input_tokens += t
                    if resolved not in stats.resolved_models:
                        stats.resolved_models.append(resolved)
                _mark_job(ckpt, job_key, "ok", model_resolved=resolved)
                return {
                    "symbol": row["symbol"],
                    "date": str(row["date"]),
                    "y": row.get("y"),
                    "X": row.get("X"),
                    "model_resolved": resolved,
                    "cache_hit": cache_hit,
                    "status": "ok",
                    **{c: feats.get(c) for c in JEV_FEATURE_COLUMNS},
                }
            except Exception as exc:  # noqa: BLE001
                with lock:
                    stats.n_failed += 1
                _mark_job(ckpt, job_key, "error", error=str(exc)[:500])
                return {"symbol": row["symbol"], "date": str(row["date"]), "status": "error", "error": str(exc)[:500]}

        pending = [i for i in range(len(sample)) if f"{sample.iloc[i]['symbol']}|{sample.iloc[i]['date']}" not in done_keys]
        log.info("paraphrase.live.start", version=qset.qset_version, n=len(pending))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_work, i) for i in pending]
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("status") == "ok":
                    rows_out.append(rec)
                done += 1
                if done % 50 == 0 or done == len(pending):
                    log.info("paraphrase.progress", version=qset.qset_version, done=done, n=len(pending))
                    pd.DataFrame(rows_out).to_parquet(prior_path, index=False)

        df = pd.DataFrame(rows_out)
        if not df.empty:
            df.to_parquet(prior_path, index=False)
        # Correlations vs canonical + subset A→variant-B
        corrs: dict[str, float] = {}
        pred: dict[str, Any] = {}
        ab_delta: dict[str, Any] = {}
        d2 = df.copy() if not df.empty else df
        if not d2.empty:
            d2["date"] = pd.to_datetime(d2["date"]).dt.date
        if canon is not None and not d2.empty:
            m = d2.merge(canon, on=["symbol", "date"], how="inner", suffixes=("_para", "_canon"))
            for col in JEV_FEATURE_COLUMNS:
                a, b = f"{col}_para", f"{col}_canon"
                if a in m.columns and b in m.columns:
                    corrs[col] = float(m[a].corr(m[b]))
            y = d2["y"].astype(float)
            for col in ("j_dislocation", "j_pullback_p", "j_regime", "j_setup_conf"):
                if col in d2.columns:
                    s = d2[col].astype(float)
                    mask = y.notna() & s.notna()
                    if int(mask.sum()) >= 50 and y[mask].nunique() > 1:
                        pred[f"auc_{col}"] = float(roc_auc_score(y[mask], s[mask]))
                        if "X" in d2.columns:
                            pred[f"ic_{col}"] = float(
                                s[mask].corr(d2.loc[mask, "X"], method="spearman")
                            )
        # Exploratory A→variant-B on the paraphrase subset (same keys)
        if not d2.empty and canon is not None:
            try:
                from jevlab.experiments.runner import _join_jev_features
                from jevlab.models.controls import ensure_placebo_columns

                feat = ensure_placebo_columns(_join_jev_features(features, canon))
                frame = prepare_model_frame(feat, labels, extra_columns=list(JEV_FEATURE_COLUMNS))
                frame["date"] = pd.to_datetime(frame["date"]).dt.date
                para_cols = [c for c in JEV_FEATURE_COLUMNS if c in d2.columns]
                sub = frame.merge(
                    d2[["symbol", "date", *para_cols]],
                    on=["symbol", "date"],
                    how="inner",
                    suffixes=("_canon", "_para"),
                )
                for col in JEV_FEATURE_COLUMNS:
                    pc = f"{col}_para"
                    if pc in sub.columns:
                        sub[col] = sub[pc]
                    elif col not in sub.columns and f"{col}_canon" in sub.columns:
                        sub[col] = sub[f"{col}_canon"]
                sessions = sorted(sub["date"].unique().tolist())
                folds = make_folds(sessions, embargo_sessions=5)
                if folds and len(sub) >= 200:
                    C = list(configs.strategy.logistic_C_grid)
                    reg = column_set_registry()
                    preds_a = run_column_set_logistic(
                        sub,
                        folds,
                        reg["A"],
                        sessions=sessions,
                        C_grid=C,
                        seed=seed,
                        embargo_sessions=5,
                    ).predictions
                    preds_b = train_b_oos(
                        sub, folds, sessions=sessions, C_grid=C, seed=seed, embargo_sessions=5
                    )
                    ab_delta = {
                        "delta_log_loss_gain": delta_log_loss_gain(preds_a, preds_b),
                        "n_oos": len(preds_a),
                        "n_folds": len(folds),
                    }
                    sub_c = frame.merge(d2[["symbol", "date"]], on=["symbol", "date"], how="inner")
                    preds_b_c = train_b_oos(
                        sub_c, folds, sessions=sessions, C_grid=C, seed=seed, embargo_sessions=5
                    )
                    ab_delta["canonical_B_delta_log_loss_gain"] = delta_log_loss_gain(
                        preds_a, preds_b_c
                    )
            except Exception as exc:  # noqa: BLE001
                ab_delta = {"error": str(exc)[:300]}
        variants.append(
            {
                "qset_version": qset.qset_version,
                "qset_hash": qset.content_hash,
                "qset_path": str(ppath),
                "frozen_qset_hash_untouched": frozen_hash,
                "n_ok": int(len(d2) if not d2.empty else 0),
                "n_failed": int(stats.n_failed),
                "n_cache_hit": int(stats.n_cache_hit),
                "spent_usd": float(stats.spent_usd),
                "requested_model": model_id,
                "resolved_models": stats.resolved_models,
                "feature_corr_vs_canonical": corrs,
                "subset_predictive": pred,
                "a_to_variant_b": ab_delta,
                "contamination_label": CONTAMINATION_LABEL,
            }
        )

    summary = {
        "status": "complete",
        "n_states": n,
        "n_paraphrases": 2,
        "variants": variants,
        "frozen_qset_hash": frozen_hash,
        "contamination_label": CONTAMINATION_LABEL,
        "materially_wording_dependent": any(
            (min((v.get("feature_corr_vs_canonical") or {}).values()) if v.get("feature_corr_vs_canonical") else 1.0)
            < 0.7
            for v in variants
        ),
    }
    (out_dir / "paraphrase_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def horizon_sweep(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    configs: ResearchConfigs,
    jev_features: pd.DataFrame | None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Exploratory horizon ±1 with relabeling (canonical horizon=3)."""
    from jevlab.experiments.runner import _join_jev_features
    from jevlab.models.controls import ensure_placebo_columns
    from jevlab.jev.parse import JEV_FEATURE_COLUMNS as JCOLS

    rows: list[dict[str, Any]] = []
    scenario = next(c for c in configs.backtest.cost_scenarios if c.name == "normal")
    elig = features.loc[:, ["symbol", "date", "eligible"]]
    for h in (2, 3, 4):
        labels = build_labels(
            bars,
            eligible=elig,
            universe_symbols=list(configs.universe.symbols),
            cost_bps_per_side=scenario.cost_bps_per_side,
            sell_fee_bps=configs.backtest.sell_fee_bps,
            per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
            horizon_sessions=h,
        )
        feat = ensure_placebo_columns(_join_jev_features(features, jev_features))
        frame = prepare_model_frame(feat, labels, extra_columns=list(JCOLS))
        sessions = sorted(frame["date"].unique().tolist())
        folds = make_folds(sessions, embargo_sessions=5)
        if not folds:
            rows.append({"horizon": h, "status": "no_folds"})
            continue
        C = list(configs.strategy.logistic_C_grid)
        reg = column_set_registry()
        preds_a = run_column_set_logistic(
            frame, folds, reg["A"], sessions=sessions, C_grid=C, seed=seed, embargo_sessions=5
        ).predictions
        preds_b = train_b_oos(frame, folds, sessions=sessions, C_grid=C, seed=seed, embargo_sessions=5)
        rows.append(
            {
                "horizon": h,
                "canonical": h == 3,
                "exploratory": h != 3,
                "n_oos": len(preds_a),
                "delta_log_loss_gain": delta_log_loss_gain(preds_a, preds_b),
                "log_loss_a": binary_log_loss(
                    preds_a["y"].to_numpy(dtype=float), preds_a["prob"].to_numpy(dtype=float)
                ),
                "log_loss_b": binary_log_loss(
                    preds_b["y"].to_numpy(dtype=float), preds_b["prob"].to_numpy(dtype=float)
                ),
                "contamination_label": CONTAMINATION_LABEL,
            }
        )
    return rows


def delayed_execution_full(
    bars: pd.DataFrame,
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    *,
    cost: CostModel,
    risk_config: RiskConfig,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Canonical open t+1, delayed fill t+1 close, and exploratory open t+2."""
    rows: list[dict[str, Any]] = []
    specs = [
        ("open_t1", "open_t1", 1),
        ("close_t1", "delayed", 1),
        ("open_t2", "open_t1", 2),
    ]
    for name, mode, lag in specs:
        ra = run_fast(
            bars,
            preds_a,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=0,
            fill_mode=mode,  # type: ignore[arg-type]
            entry_lag_sessions=lag,
            apply_risk=False,
        )
        rb = run_fast(
            bars,
            preds_b,
            cost=cost,
            risk_config=risk_config,
            top_k=top_k,
            phase=0,
            fill_mode=mode,  # type: ignore[arg-type]
            entry_lag_sessions=lag,
            apply_risk=False,
        )
        rows.append(
            {
                "fill_mode": name,
                "entry_lag_sessions": lag,
                "delta_sharpe": sharpe_from_equity(rb.equity_curve) - sharpe_from_equity(ra.equity_curve),
                "sharpe_a": sharpe_from_equity(ra.equity_curve),
                "sharpe_b": sharpe_from_equity(rb.equity_curve),
            }
        )
    return rows
