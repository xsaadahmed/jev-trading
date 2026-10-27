"""Production sealed-data source and allowed pre-sealed training source.

Construction of ``ProductionSealedSource`` does not read sealed contents.
``PreSealedTrainingSource`` may read pre-2024 stores without consuming the seal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevlab.cv.sealed_guard import SealedAccessError
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.v2.dev_io import iter_dev_year_paths, read_dev_bars, read_dev_feature_store

SEALED_SCORE_YEARS: tuple[int, ...] = (2024, 2025)


class PreSealedTrainingSource:
    """Allowed pre-2024 training/context. Does not consume the seal."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        use_dev_stores: bool = False,
        vendor: str | None = None,
    ) -> None:
        self.payload = payload
        self.use_dev_stores = use_dev_stores
        self.vendor = vendor
        self.training_context_reads = 0
        self.sealed_underlying_reads = 0
        self.filesystem_reads = 0

    def load(self) -> dict[str, Any]:
        self.training_context_reads += 1
        if self.payload is not None:
            return dict(self.payload)
        if not self.use_dev_stores:
            raise SealedAccessError("PreSealedTrainingSource has no payload")
        features = read_dev_feature_store()
        from jevlab.config.loader import load_all_configs
        from jevlab.v2.candidate import V2_UNIVERSE

        vendor = self.vendor or load_all_configs().data.primary_vendor
        bars = read_dev_bars(vendor=vendor, symbols=list(V2_UNIVERSE))
        self.filesystem_reads += 1
        labels = features
        if not bars.empty and "y" not in features.columns:
            from jevlab.targets.labels import build_labels

            configs = load_all_configs()
            elig_cols = [c for c in ("symbol", "date", "eligible") if c in features.columns]
            elig = features.loc[:, elig_cols]
            labels = build_labels(
                bars,
                eligible=elig,
                universe_symbols=list(V2_UNIVERSE),
                cost_bps_per_side=2.5,
                sell_fee_bps=configs.backtest.sell_fee_bps,
                per_symbol_cost_multiplier=dict(configs.backtest.per_symbol_cost_multiplier),
            )
        return {
            "features": features,
            "labels": labels,
            "bars": bars,
            "pre_sealed": True,
        }

    def source_manifest_digest(self, *, root: Path | None = None) -> str:
        if self.payload is not None:
            keys = sorted(str(k) for k in self.payload)
            n_feat = 0
            n_bars = 0
            if "features" in self.payload:
                n_feat = int(getattr(self.payload["features"], "__len__", lambda: 0)())
            if "bars" in self.payload:
                n_bars = int(getattr(self.payload["bars"], "__len__", lambda: 0)())
            return sha256_of(
                canonical_json(
                    {"kind": "payload", "keys": keys, "n_feat": n_feat, "n_bars": n_bars}
                )
            )
        from jevlab.features.columns import FEATURE_VERSION
        from jevlab.utils.paths import ProjectPaths, get_project_root

        paths = ProjectPaths.from_root(root or get_project_root())
        feat_root = paths.data_features / FEATURE_VERSION
        files = iter_dev_year_paths(feat_root, "symbol=*/")
        manifest = [
            {"rel": str(p).replace("\\", "/"), "size": int(p.stat().st_size)} for p in files
        ]
        return sha256_of(canonical_json(manifest))


class ProductionSealedSource:
    """Adapter for the configured sealed store.

    ``__init__`` stores the path string only. It must not list, glob, stat-walk,
    or read files. ``load()`` may read only after ``authorize_for_atomic_path``.
    """

    def __init__(self, configured_path: Path | str) -> None:
        self.configured_path = Path(configured_path)
        self.filesystem_reads = 0
        self._authorized = False

    def configured_path_string(self) -> str:
        return str(self.configured_path)

    def authorize_for_atomic_path(self) -> None:
        """Called by the atomic runner after preconditions + SEAL token pass."""
        self._authorized = True

    def load(self) -> dict[str, Any]:
        if not self._authorized:
            raise SealedAccessError(
                "ProductionSealedSource.load requires atomic-path authorization "
                "after preconditions and human SEAL token"
            )
        return self._read_authorized()

    def _read_authorized(self) -> dict[str, Any]:
        """Read 2024-2025 partitions. Must not run during the pre-seal freeze."""
        import pandas as pd

        self.filesystem_reads += 1
        root = self.configured_path
        files: list[Path] = []
        for year in SEALED_SCORE_YEARS:
            files.extend(sorted(root.glob(f"**/*year={year}*/**/*.parquet")))
            files.extend(sorted(root.glob(f"**/*{year}*.parquet")))
        unique = sorted(set(files))
        frames = [pd.read_parquet(p) for p in unique]
        return {
            "configured_path": str(root),
            "n_files": len(unique),
            "frames": frames,
        }


class CountingSealedSource:
    """Test double: records load() calls; never touches a real store."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.loads = 0
        self.filesystem_reads = 0
        self.sealed_underlying_reads = 0
        self.payload = payload if payload is not None else {"fixture": True}
        self._authorized = False

    def authorize_for_atomic_path(self) -> None:
        self._authorized = True

    def load(self) -> dict[str, Any]:
        self.loads += 1
        self.filesystem_reads += 1
        self.sealed_underlying_reads += 1
        return dict(self.payload)


class CountingPreSealedSource(PreSealedTrainingSource):
    """Test double for the allowed training/context source."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        super().__init__(payload=payload if payload is not None else {"fixture": True})
        self.loads = 0

    def load(self) -> dict[str, Any]:
        self.loads += 1
        return super().load()
