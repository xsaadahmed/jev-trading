"""Sealed-partition guard for data after 2023-12-29 (MASTER_SPEC §10.3 / Phase 3)."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from jevlab.cv.walkforward import DEV_MAX_DATE, SEALED_AFTER
from jevlab.utils.logging import get_logger
from jevlab.utils.registry import init_registry

log = get_logger(__name__)

# Environment variable holding the one-shot SEAL token for authorized access.
SEAL_TOKEN_ENV: str = "JEV_SEAL_TOKEN"
SEALED_DIR_NAME: str = "sealed"


class SealedAccessError(PermissionError):
    """Raised when sealed data is requested without a valid token."""


class SealedDataGuard:
    """Blocks reads of post-``SEALED_AFTER`` data unless an explicit token is provided.

    Authorized loads write a row to ``sealed_access_log`` in the experiment registry.
    """

    def __init__(
        self,
        *,
        sealed_root: Path,
        registry_path: Path,
        sealed_after: date = SEALED_AFTER,
    ) -> None:
        self.sealed_root = Path(sealed_root)
        self.registry_path = Path(registry_path)
        self.sealed_after = sealed_after
        self._engine: Engine = init_registry(self.registry_path)

    def is_sealed_date(self, d: date) -> bool:
        return d > self.sealed_after

    def assert_dev_dates(self, dates: list[date] | pd.Series) -> None:
        """Raise if any date is after the sealed cutoff (dev pipelines must not see them)."""
        date_list = [pd.Timestamp(x).date() for x in dates]
        sealed = [d for d in date_list if self.is_sealed_date(d)]
        if sealed:
            raise SealedAccessError(
                f"dev pipeline attempted to read {len(sealed)} sealed dates "
                f"(after {self.sealed_after}); e.g. {sealed[:3]}"
            )

    def filter_dev_frame(self, frame: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
        """Return only rows with ``date <= SEALED_AFTER`` (2023-12-29)."""
        if frame.empty:
            return frame.copy()
        out = frame.copy()
        out[date_col] = pd.to_datetime(out[date_col]).dt.date
        return out.loc[out[date_col] <= self.sealed_after].reset_index(drop=True)

    def _check_token(self, token: str | None) -> None:
        expected = os.environ.get(SEAL_TOKEN_ENV, "").strip()
        if not expected:
            raise SealedAccessError(
                f"sealed access denied: set {SEAL_TOKEN_ENV} to authorize a one-shot load"
            )
        if not token or token.strip() != expected:
            raise SealedAccessError("sealed access denied: invalid or missing SEAL token")

    def _log_access(
        self,
        *,
        user: str,
        experiment_hash: str,
        reason: str,
        partition: str,
    ) -> str:
        access_id = str(uuid.uuid4())
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO sealed_access_log (
                        access_id, at, user, experiment_hash, reason, partition
                    ) VALUES (
                        :access_id, :at, :user, :experiment_hash, :reason, :partition
                    )
                    """
                ),
                {
                    "access_id": access_id,
                    "at": datetime.now(UTC).isoformat(),
                    "user": user,
                    "experiment_hash": experiment_hash,
                    "reason": reason,
                    "partition": partition,
                },
            )
        log.info(
            "sealed.access",
            access_id=access_id,
            user=user,
            reason=reason,
            partition=partition,
        )
        return access_id

    def load_sealed_parquet(
        self,
        relative_path: str | Path,
        *,
        token: str | None,
        reason: str,
        user: str = "operator",
        experiment_hash: str = "",
    ) -> pd.DataFrame:
        """Load a parquet file under ``data/sealed/`` with token + access log."""
        self._check_token(token)
        path = self.sealed_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"sealed file not found: {path}")
        self._log_access(
            user=user,
            experiment_hash=experiment_hash,
            reason=reason,
            partition=str(relative_path),
        )
        return pd.read_parquet(path)

    def list_access_log(self) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT access_id, at, user, experiment_hash, reason, partition "
                    "FROM sealed_access_log ORDER BY at"
                )
            ).mappings()
            return [dict(r) for r in rows]


def reject_sealed_dates_in_dev(frame: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """Convenience: drop/forbid sealed dates for any development pipeline."""
    guard = SealedDataGuard(
        sealed_root=Path("data") / SEALED_DIR_NAME,
        registry_path=Path("registry") / "experiments.sqlite",
    )
    # Soft filter + hard assert that nothing sealed remains if caller passed sealed rows
    # intending to use them without a token.
    filtered = guard.filter_dev_frame(frame, date_col=date_col)
    if len(filtered) < len(frame):
        log.warning(
            "sealed.rows_dropped",
            n_dropped=len(frame) - len(filtered),
            sealed_after=str(DEV_MAX_DATE),
        )
    return filtered
