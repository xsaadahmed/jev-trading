"""Load pre-2024 development artifacts without touching sealed partitions.

Never globs ``year=2024`` / ``year=2025`` or ``data/sealed/``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from jevlab.cv.walkforward import DEV_MAX_DATE
from jevlab.features.columns import FEATURE_VERSION, MODEL_A_COLUMNS
from jevlab.features.pipeline import OUTPUT_COLUMNS
from jevlab.utils.paths import ProjectPaths, get_project_root

DEV_YEAR_MIN = 2015
DEV_YEAR_MAX = 2023  # inclusive; 2024-2025 never opened
SEALED_MAX_DATE = date(2025, 12, 31)


class SealedPathError(PermissionError):
    """Raised if a caller attempts a sealed-era path."""


def _reject_sealed_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if "year=2024" in text or "year=2025" in text:
        raise SealedPathError(f"refusing sealed-year path: {path}")
    if "/sealed/" in text or text.endswith("/sealed"):
        raise SealedPathError(f"refusing sealed directory path: {path}")


def iter_dev_year_paths(root: Path, pattern_prefix: str) -> list[Path]:
    """Glob only years 2015-2023 under ``root``."""
    root = Path(root)
    paths: list[Path] = []
    for year in range(DEV_YEAR_MIN, DEV_YEAR_MAX + 1):
        matched = sorted(root.glob(f"{pattern_prefix}year={year}/**/part.parquet"))
        if not matched:
            matched = sorted(root.glob(f"{pattern_prefix}year={year}/part.parquet"))
        for p in matched:
            _reject_sealed_path(p)
            paths.append(p)
    return sorted(set(paths))


def read_dev_feature_store(root: Path | None = None) -> pd.DataFrame:
    """Load Model A feature hive for years ≤ 2023 only."""
    paths_obj = ProjectPaths.from_root(get_project_root())
    feat_root = Path(root) if root is not None else paths_obj.data_features / FEATURE_VERSION
    files = iter_dev_year_paths(feat_root, "symbol=*/")
    if not files:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    parts = [pd.read_parquet(p) for p in files]
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.loc[out["date"] <= DEV_MAX_DATE].reset_index(drop=True)
    if out["date"].map(lambda d: d > DEV_MAX_DATE).any():
        raise SealedPathError("dev feature frame still contains post-2023 dates")
    missing = [c for c in MODEL_A_COLUMNS if c not in out.columns]
    if missing:
        raise AssertionError(f"MODEL_A_COLUMNS missing: {missing}")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def read_dev_bars(*, vendor: str, symbols: list[str] | None = None) -> pd.DataFrame:
    """Load processed bars for years ≤ 2023 only."""
    from jevlab.config.loader import load_all_configs
    from jevlab.data.downloader import default_store

    paths = ProjectPaths.from_root(get_project_root())
    configs = load_all_configs(paths.config)
    store = default_store(paths, configs.data)
    files = iter_dev_year_paths(store.bars_dir, "symbol=*/")
    frames: list[pd.DataFrame] = []
    wanted = {s.upper() for s in symbols} if symbols is not None else None
    for path in files:
        parts = {p.split("=")[0]: p.split("=")[1] for p in path.parts if "=" in p}
        if wanted is not None and parts.get("symbol") not in wanted:
            continue
        if parts.get("vendor") != vendor:
            continue
        _reject_sealed_path(path)
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.loc[out["date"] <= DEV_MAX_DATE].reset_index(drop=True)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def read_dev_vix(root: Path | None = None) -> pd.DataFrame:
    """Load FRED VIX for years ≤ 2023 only. Never opens 2024-2025 partitions."""
    from jevlab.config.loader import load_all_configs
    from jevlab.cv.walkforward import DEV_MAX_DATE as _DEV_MAX
    from jevlab.data.downloader import default_store

    paths_obj = ProjectPaths.from_root(root or get_project_root())
    configs = load_all_configs(paths_obj.config)
    store = default_store(paths_obj, configs.data)
    series_id = configs.data.vix_series_id
    files = iter_dev_year_paths(store.external_dir / f"series_id={series_id}", "")
    if not files:
        return pd.DataFrame(columns=["series_id", "obs_date", "value"])
    parts = [pd.read_parquet(p) for p in files]
    out = pd.concat(parts, ignore_index=True)
    date_col = "obs_date" if "obs_date" in out.columns else "date"
    out[date_col] = pd.to_datetime(out[date_col]).dt.date
    out = out.loc[out[date_col] <= _DEV_MAX].reset_index(drop=True)
    if date_col != "obs_date":
        out = out.rename(columns={date_col: "obs_date"})
    return out.sort_values(["obs_date"]).reset_index(drop=True)


def read_canonical_oos_a(path: Path | None = None) -> pd.DataFrame:
    """Load V1 Phase-8 Model A OOS predictions (development period only)."""
    paths = ProjectPaths.from_root(get_project_root())
    pred_path = path or (paths.reports / "phase8_lineup" / "oos_A.parquet")
    _reject_sealed_path(pred_path)
    if not pred_path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(pred_path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.loc[frame["date"] <= DEV_MAX_DATE].reset_index(drop=True)


def assert_frame_pre_sealed(frame: pd.DataFrame, *, date_col: str = "date") -> None:
    if frame.empty:
        return
    dates = pd.to_datetime(frame[date_col]).dt.date
    if (dates > DEV_MAX_DATE).any():
        raise SealedPathError("frame contains dates after 2023-12-29")


def assert_frame_date_window(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    allow_sealed_dates: bool = False,
) -> None:
    """Pre-2024 by default; authorized sealed path may include through 2025-12-31."""
    if not allow_sealed_dates:
        assert_frame_pre_sealed(frame, date_col=date_col)
        return
    if frame.empty:
        return
    dates = pd.to_datetime(frame[date_col]).dt.date
    if (dates > SEALED_MAX_DATE).any():
        raise SealedPathError("frame contains dates after sealed window 2025-12-31")


def v1_canonical_artifact_paths() -> dict[str, Path]:
    reports = ProjectPaths.from_root(get_project_root()).reports
    return {
        "oos_A": reports / "phase8_lineup" / "oos_A.parquet",
        "signals_A": reports / "phase8_lineup" / "signals_A.parquet",
        "comparison": reports / "phase8_lineup" / "comparison_table.csv",
        "phase3_metrics": reports / "phase3_baselines" / "metrics.json",
        "phase3_oos_A": reports / "phase3_baselines" / "oos_A.parquet",
    }
