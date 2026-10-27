"""Typed production sealed payload. Anonymous ``frames`` never enter analytics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.v2.candidate import V2_UNIVERSE
from jevlab.v2.dev_io import SEALED_MAX_DATE
from jevlab.v2.training import SEALED_T0

BAR_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"symbol", "date", "adj_open", "adj_close", "raw_close", "raw_volume"}
)
BAR_ALLOWED_COLUMNS: frozenset[str] = BAR_REQUIRED_COLUMNS | frozenset(
    {
        "raw_open",
        "raw_high",
        "raw_low",
        "adj_high",
        "adj_low",
        "adj_volume",
        "div_cash",
        "split_factor",
        "available_at",
        "vendor",
        "retrieved_at",
        "row_origin",
        "row_source",
    }
)
VIX_REQUIRED_COLUMNS: frozenset[str] = frozenset({"value"})
VIX_ALLOWED_COLUMNS: frozenset[str] = frozenset(
    {"obs_date", "date", "value", "available_at", "series_id", "retrieved_at", "row_origin"}
)
AUDIT_ONLY_COLUMNS: tuple[str, ...] = (
    "row_origin",
    "row_source",
    "decision_origin",
    "used_historical_context",
    "target_origin",
    "sealed_source",
)
HISTORICAL_ORIGIN = "historical_context"
SEALED_ORIGIN = "sealed_evaluation"


class SealedPayloadError(AssertionError):
    """Sealed payload could not be classified or violated the source contract."""


class EmptySealedScoringError(AssertionError):
    """Seal was opened but zero sealed observations reached scoring."""


@dataclass
class ProductionSealedPayload:
    """Explicit sealed inputs required by frozen V1/V2 feature semantics."""

    bars: pd.DataFrame
    vix: pd.DataFrame
    configured_path: str = ""
    n_files: int = 0
    n_raw_frames: int = 0
    n_classified_frames: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_bar_rows(self) -> int:
        return len(self.bars)

    @property
    def n_vix_rows(self) -> int:
        return len(self.vix)


@dataclass
class SealedDataFlow:
    """Independent data-flow oracle for pseudo-sealed rehearsal."""

    raw_rows: int = 0
    classified_rows: int = 0
    combined_history_sealed_rows: int = 0
    feature_builder_input_rows: int = 0
    valid_feature_rows: int = 0
    candidate_scoring_rows: int = 0
    first_candidate_date: date | None = None
    last_candidate_date: date | None = None
    n_predictions: int = 0
    n_completed_trades: int = 0
    causal_feature_builder_invoked: bool = False
    historical_only_fallback: bool = False
    unknown_frames: int = 0
    canary_date: date | None = None
    canary_seen_in_combined: bool = False
    canary_seen_in_features: bool = False
    canary_seen_in_scoring: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_rows": self.raw_rows,
            "classified_rows": self.classified_rows,
            "combined_history_sealed_rows": self.combined_history_sealed_rows,
            "feature_builder_input_rows": self.feature_builder_input_rows,
            "valid_feature_rows": self.valid_feature_rows,
            "candidate_scoring_rows": self.candidate_scoring_rows,
            "first_candidate_date": (
                self.first_candidate_date.isoformat() if self.first_candidate_date else None
            ),
            "last_candidate_date": (
                self.last_candidate_date.isoformat() if self.last_candidate_date else None
            ),
            "n_predictions": self.n_predictions,
            "n_completed_trades": self.n_completed_trades,
            "causal_feature_builder_invoked": self.causal_feature_builder_invoked,
            "historical_only_fallback": self.historical_only_fallback,
            "unknown_frames": self.unknown_frames,
            "canary_date": self.canary_date.isoformat() if self.canary_date else None,
            "canary_seen_in_combined": self.canary_seen_in_combined,
            "canary_seen_in_features": self.canary_seen_in_features,
            "canary_seen_in_scoring": self.canary_seen_in_scoring,
        }


def strip_audit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop provenance columns so they cannot become model features."""
    drop = [c for c in AUDIT_ONLY_COLUMNS if c in frame.columns]
    if not drop:
        return frame
    return frame.drop(columns=drop)


def as_production_sealed_payload(
    bars: pd.DataFrame,
    vix: pd.DataFrame | None = None,
    *,
    configured_path: str = "pseudo-sealed",
    t0: date | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap classified market inputs in the same structure as ProductionSealedSource."""
    frames: list[pd.DataFrame] = [bars.copy()]
    if vix is not None and not vix.empty:
        frames.append(vix.copy())
    payload: dict[str, Any] = {
        "configured_path": configured_path,
        "n_files": len(frames),
        "frames": frames,
    }
    if t0 is not None:
        payload["t0"] = t0
    if extras:
        payload.update(dict(extras))
    return payload


def _as_date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column]).dt.date


def _frame_from_payload(value: Any) -> pd.DataFrame:
    frame = pd.DataFrame(value)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if "obs_date" in frame.columns:
        frame["obs_date"] = pd.to_datetime(frame["obs_date"]).dt.date
    return frame


def _classify_one_frame(frame: pd.DataFrame) -> str:
    cols = set(frame.columns)
    if cols >= BAR_REQUIRED_COLUMNS:
        unexpected = cols - BAR_ALLOWED_COLUMNS
        if unexpected:
            raise SealedPayloadError(
                f"sealed bar frame has unexpected columns: {sorted(unexpected)}"
            )
        return "bars"
    if "value" in cols and (("obs_date" in cols) or ("date" in cols)) and "adj_open" not in cols:
        unexpected = cols - VIX_ALLOWED_COLUMNS
        if unexpected:
            raise SealedPayloadError(
                f"sealed vix frame has unexpected columns: {sorted(unexpected)}"
            )
        if not (cols >= VIX_REQUIRED_COLUMNS):
            raise SealedPayloadError("sealed vix frame missing required value column")
        return "vix"
    if all(c in cols for c in MODEL_A_COLUMNS):
        raise SealedPayloadError(
            "sealed payload supplied precomputed Model A features; "
            "r4 requires causal construction from classified market bars"
        )
    raise SealedPayloadError(f"unrecognized sealed frame columns: {sorted(cols)}")


def _validate_bars(
    bars: pd.DataFrame,
    *,
    allowed_symbols: frozenset[str],
    min_date: date | None,
    max_date: date | None,
) -> pd.DataFrame:
    work = bars.copy()
    if work.empty:
        return work
    work["date"] = _as_date_series(work, "date")
    work["symbol"] = work["symbol"].astype(str)
    missing_price = [
        c for c in ("adj_open", "adj_close", "raw_close", "raw_volume") if c not in work.columns
    ]
    if missing_price:
        raise SealedPayloadError(f"sealed bars missing required price fields: {missing_price}")
    symbols = set(work["symbol"].unique().tolist())
    unexpected = sorted(symbols - set(allowed_symbols))
    if unexpected:
        raise SealedPayloadError(f"sealed bars contain unexpected symbols: {unexpected}")
    if min_date is not None and (work["date"] < min_date).any():
        raise SealedPayloadError("sealed bars contain dates before the evaluation window")
    if max_date is not None and (work["date"] > max_date).any():
        raise SealedPayloadError("sealed bars violate the expected date/session schema")
    dup = work.duplicated(subset=["symbol", "date"], keep=False)
    if bool(dup.any()):
        raise SealedPayloadError("sealed bars contain duplicate symbol/date keys")
    return work.reset_index(drop=True)


def _validate_vix(
    vix: pd.DataFrame,
    *,
    min_date: date | None,
    max_date: date | None,
) -> pd.DataFrame:
    work = vix.copy()
    if work.empty:
        return work
    if "obs_date" not in work.columns and "date" in work.columns:
        work = work.rename(columns={"date": "obs_date"})
    work["obs_date"] = _as_date_series(work, "obs_date")
    if min_date is not None and (work["obs_date"] < min_date).any():
        raise SealedPayloadError("sealed vix contains dates before the evaluation window")
    if max_date is not None and (work["obs_date"] > max_date).any():
        raise SealedPayloadError("sealed vix violates the expected date/session schema")
    dup = work.duplicated(subset=["obs_date"], keep=False)
    if bool(dup.any()):
        raise SealedPayloadError("sealed vix contains duplicate observation dates")
    return work.reset_index(drop=True)


def classify_production_sealed_payload(
    payload: Mapping[str, Any],
    *,
    allowed_symbols: tuple[str, ...] | list[str] | None = None,
    min_date: date | None = None,
    max_date: date | None = None,
    require_real_window: bool = False,
) -> ProductionSealedPayload:
    """Classify every loaded frame into an explicit role. Fail closed on unknowns."""
    if payload.get("fixture") and "frames" not in payload and "bars" not in payload:
        raise ValueError(
            "payload is a fixture stub; production computation requires "
            "features, labels/targets, and bars"
        )
    symbols = frozenset(allowed_symbols or V2_UNIVERSE)
    window_min = min_date
    window_max = max_date
    if require_real_window:
        window_min = SEALED_T0 if window_min is None else window_min
        window_max = SEALED_MAX_DATE if window_max is None else window_max
    raw_frames: list[Any]
    if "frames" in payload:
        raw_frames = list(payload["frames"])
    elif "bars" in payload:
        raw_frames = [payload["bars"]]
        if payload.get("vix") is not None:
            raw_frames.append(payload["vix"])
    else:
        raise SealedPayloadError(
            "sealed payload has no frames to classify; refusing anonymous bag"
        )
    if not raw_frames:
        raise SealedPayloadError("sealed payload frames list is empty")

    bar_parts: list[pd.DataFrame] = []
    vix_parts: list[pd.DataFrame] = []
    classified = 0
    for raw in raw_frames:
        frame = _frame_from_payload(raw)
        if frame.empty:
            raise SealedPayloadError("sealed payload contains an empty frame")
        kind = _classify_one_frame(frame)
        classified += 1
        if kind == "bars":
            bar_parts.append(frame)
        else:
            vix_parts.append(frame)

    bars = (
        pd.concat(bar_parts, ignore_index=True)
        if bar_parts
        else pd.DataFrame(columns=list(BAR_REQUIRED_COLUMNS))
    )
    vix = (
        pd.concat(vix_parts, ignore_index=True)
        if vix_parts
        else pd.DataFrame(columns=["obs_date", "value"])
    )
    bars = _validate_bars(bars, allowed_symbols=symbols, min_date=window_min, max_date=window_max)
    vix = _validate_vix(vix, min_date=window_min, max_date=window_max)
    extras = {
        k: v
        for k, v in dict(payload).items()
        if k not in {"frames", "bars", "vix", "features", "labels"}
    }
    return ProductionSealedPayload(
        bars=bars,
        vix=vix,
        configured_path=str(payload.get("configured_path", "")),
        n_files=int(payload.get("n_files", len(raw_frames))),
        n_raw_frames=len(raw_frames),
        n_classified_frames=classified,
        extras=extras,
    )


def assert_sealed_scoring_nonempty(
    features: pd.DataFrame,
    *,
    t0: date,
    max_date: date = SEALED_MAX_DATE,
) -> None:
    """Hard execution-integrity failure if the opened seal contributed nothing."""
    if features.empty:
        raise EmptySealedScoringError(
            "zero sealed observations reached feature construction after the seal was opened"
        )
    dates = pd.to_datetime(features["date"]).dt.date
    if int((dates >= t0).sum()) == 0:
        raise EmptySealedScoringError(
            "zero sealed observations reached candidate scoring; refusing historical-only fallback"
        )
    if bool((dates < t0).any()):
        raise EmptySealedScoringError(
            "historical context rows were presented as sealed scoring observations"
        )
    if bool((dates > max_date).any()):
        raise EmptySealedScoringError("sealed scoring rows extend past 2025-12-31")
    if t0 == SEALED_T0 and int((dates >= date(2024, 1, 2)).sum()) == 0:
        raise EmptySealedScoringError("real sealed path produced no decision dates >= 2024-01-02")
