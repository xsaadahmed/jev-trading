"""Feature pipeline: as_of-filtered inputs → 20 features + eligibility (§6 / §6.1)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from jevlab.data.availability import as_of, decision_time
from jevlab.features.columns import (
    FEATURE_VERSION,
    MODEL_A_COLUMNS,
    STORED_FEATURE_COLUMNS,
)
from jevlab.features.market_context import compute_spy_context, compute_vix_lag1_pct_252
from jevlab.features.relative import add_relative_and_xs_ranks
from jevlab.features.technical import compute_symbol_technicals
from jevlab.targets.eligibility import attach_eligible
from jevlab.utils.hashing import logical_table_digest
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

FEATURE_PRIMARY_KEY: tuple[str, ...] = ("symbol", "date", "feature_version")

OUTPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "feature_version",
    "decision_time",
    *STORED_FEATURE_COLUMNS,
    "eligible",
)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    if "available_at" not in out.columns:
        raise ValueError("bars require available_at; assign via data.availability")
    out["available_at"] = pd.to_datetime(out["available_at"], utc=True)
    return out


def _normalize_vix(vix: pd.DataFrame) -> pd.DataFrame:
    out = vix.copy()
    date_col = "obs_date" if "obs_date" in out.columns else "date"
    if date_col != "obs_date":
        out = out.rename(columns={date_col: "obs_date"})
    out["obs_date"] = pd.to_datetime(out["obs_date"]).dt.date
    if "available_at" not in out.columns:
        raise ValueError("vix requires available_at; assign via data.availability")
    out["available_at"] = pd.to_datetime(out["available_at"], utc=True)
    return out


def build_features_for_session(
    bars: pd.DataFrame,
    vix: pd.DataFrame,
    session: date,
    *,
    symbols: list[str],
    spy_symbol: str = "SPY",
    feature_version: str = FEATURE_VERSION,
) -> pd.DataFrame:
    """Build feature rows for one decision session using ``as_of(decision_time)`` only.

    Time semantics: decision after close of ``session`` at 18:30 ET; every input row
    satisfies ``available_at <= decision_time``. VIX uses the previous session only.
    """
    dt = decision_time(session)
    bars_t = as_of(_normalize_bars(bars), dt)
    vix_t = as_of(_normalize_vix(vix), dt)

    # Restrict to history ending at this session (bars for session are available).
    bars_t = bars_t.loc[bars_t["date"] <= session].copy()
    if bars_t.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    tech_parts: list[pd.DataFrame] = []
    for sym in symbols:
        sym_bars = bars_t.loc[bars_t["symbol"] == sym]
        if sym_bars.empty:
            continue
        tech = compute_symbol_technicals(sym_bars)
        # Keep only the row for this session.
        tech = tech.loc[tech["date"] == session]
        if not tech.empty:
            tech_parts.append(tech)

    if not tech_parts:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    panel = pd.concat(tech_parts, ignore_index=True)

    spy_bars = bars_t.loc[bars_t["symbol"] == spy_symbol]
    spy_ctx = compute_spy_context(spy_bars)
    spy_row = spy_ctx.loc[spy_ctx["date"] == session]
    if spy_row.empty:
        panel["spy_ret_5d"] = float("nan")
        panel["spy_px_vs_sma200"] = float("nan")
    else:
        panel["spy_ret_5d"] = float(spy_row.iloc[0]["spy_ret_5d"])
        panel["spy_px_vs_sma200"] = float(spy_row.iloc[0]["spy_px_vs_sma200"])

    vix_feat = compute_vix_lag1_pct_252(vix_t, [session])
    panel["vix_lag1_pct_252"] = float(vix_feat.iloc[0]["vix_lag1_pct_252"])

    panel = add_relative_and_xs_ranks(panel)
    panel = attach_eligible(panel)
    panel["feature_version"] = feature_version
    panel["decision_time"] = dt.tz_convert("UTC")

    return panel.loc[:, list(OUTPUT_COLUMNS)]


def build_feature_store(
    bars: pd.DataFrame,
    vix: pd.DataFrame,
    *,
    symbols: list[str],
    sessions: list[date] | None = None,
    spy_symbol: str = "SPY",
    feature_version: str = FEATURE_VERSION,
) -> pd.DataFrame:
    """Build the full feature panel for many sessions (each via ``as_of``).

    If ``sessions`` is None, uses the sorted union of bar dates present for all
    symbols (intersection is safer for xs ranks — use dates where any symbol has a
    bar; xs rank uses available symbols that day).
    """
    bars_n = _normalize_bars(bars)
    vix_n = _normalize_vix(vix)
    if sessions is None:
        sessions = sorted({d for d in bars_n["date"].tolist()})

    parts: list[pd.DataFrame] = []
    for i, session in enumerate(sessions):
        part = build_features_for_session(
            bars_n,
            vix_n,
            session,
            symbols=symbols,
            spy_symbol=spy_symbol,
            feature_version=feature_version,
        )
        if not part.empty:
            parts.append(part)
        if (i + 1) % 250 == 0:
            log.info("features.built", sessions_done=i + 1, sessions_total=len(sessions))

    if not parts:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    log.info("features.built", rows=len(out), feature_version=feature_version)
    return out


def write_feature_store(features: pd.DataFrame, root: Path) -> list[Path]:
    """Write hive-partitioned parquet under ``root`` (e.g. data/features/v1)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if features.empty:
        return []
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["year"] = [d.year for d in frame["date"]]
    written: list[Path] = []
    for (symbol, year), group in frame.groupby(["symbol", "year"], sort=True):
        path = root / f"symbol={symbol}" / f"year={int(str(year))}" / "part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = group.drop(columns=["year"]).sort_values("date").reset_index(drop=True)
        out.to_parquet(path, index=False)
        written.append(path)
    return written


def read_feature_store(root: Path) -> pd.DataFrame:
    """Load a hive feature store written by ``write_feature_store``."""
    root = Path(root)
    paths = sorted(root.glob("symbol=*/year=*/part.parquet"))
    if not paths:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    parts = [pd.read_parquet(p) for p in paths]
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    if "decision_time" in out.columns:
        out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_store_digest(features: pd.DataFrame) -> dict[str, str]:
    """Logical content digest of the feature table (§20)."""
    cols = [c for c in OUTPUT_COLUMNS if c in features.columns]
    return logical_table_digest(features.loc[:, cols], primary_key=FEATURE_PRIMARY_KEY)


def assert_model_a_columns(features: pd.DataFrame) -> None:
    """Raise if Model A columns are missing or the count is not 17."""
    missing = [c for c in MODEL_A_COLUMNS if c not in features.columns]
    if missing:
        raise AssertionError(f"MODEL_A_COLUMNS missing from features: {missing}")
    if len(MODEL_A_COLUMNS) != 17:
        raise AssertionError("MODEL_A_COLUMNS must contain exactly 17 names")


def model_a_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Return the Model A design matrix columns in canonical order."""
    assert_model_a_columns(features)
    return features.loc[:, list(MODEL_A_COLUMNS)]


def build_features_from_store_paths(
    *,
    bars: pd.DataFrame,
    vix: pd.DataFrame,
    symbols: list[str],
    output_dir: Path,
    sessions: list[date] | None = None,
) -> dict[str, Any]:
    """Build, write, and digest a feature store; return summary metadata."""
    features = build_feature_store(bars, vix, symbols=symbols, sessions=sessions)
    paths = write_feature_store(features, output_dir)
    digest = (
        feature_store_digest(features)
        if not features.empty
        else {
            "content_sha256": "",
            "schema_sha256": "",
        }
    )
    return {
        "rows": len(features),
        "paths": [str(p) for p in paths],
        "digest": digest,
        "feature_version": FEATURE_VERSION,
    }
