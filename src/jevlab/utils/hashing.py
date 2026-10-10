"""Canonical JSON serialization and logical table digests (MASTER_SPEC §20)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _dump_canonical(obj: Any) -> str:
    """Recursively serialize with sorted keys, no whitespace, fixed float rule."""
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (np.bool_,)):
        return "true" if bool(obj) else "false"
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return str(int(obj))
    if isinstance(obj, (float, np.floating)):
        f64 = float(np.float64(obj))
        if not np.isfinite(f64):
            raise ValueError(f"non-finite float not allowed in canonical_json: {obj!r}")
        return repr(f64)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(obj)
        if ts.tzinfo is None:
            raise ValueError("timestamps in canonical_json must be timezone-aware (UTC)")
        return json.dumps(ts.tz_convert("UTC").isoformat(), ensure_ascii=False)
    if isinstance(obj, Mapping):
        parts = [
            f"{json.dumps(str(k), ensure_ascii=False)}:{_dump_canonical(obj[k])}"
            for k in sorted(obj.keys(), key=str)
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_dump_canonical(v) for v in obj) + "]"
    raise TypeError(f"unsupported type for canonical_json: {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` with sorted keys, no whitespace, fixed float formatting.

    Floats are formatted via ``repr(float64(...))``. Keys are sorted recursively.
    Separators are commas and colons with no spaces.
    """
    return _dump_canonical(obj)


def sha256_of(data: str | bytes) -> str:
    """Return the hex sha256 digest of a string (UTF-8) or bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a column dtype per §20: float64, int64, UTC timestamps, UTF-8 strings."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_float_dtype(series):
        return series.astype("float64")
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    return series.map(lambda v: None if pd.isna(v) else str(v)).astype("string")


def _serialize_cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return "null"
        f64 = float(np.float64(value))
        if not np.isfinite(f64):
            raise ValueError(f"non-finite float in table digest: {value!r}")
        return repr(f64)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    return str(value)


def logical_table_digest(
    table: pd.DataFrame,
    *,
    primary_key: Sequence[str],
) -> dict[str, str]:
    """Compute logical content digests for a table (MASTER_SPEC §20).

    Steps: select schema columns in sorted order → normalize dtypes → sort by
    primary key → serialize rows canonically (floats via float64 ``repr``) →
    sha256, plus a hash of the schema.

    Returns
    -------
    dict with keys ``content_sha256`` and ``schema_sha256``.
    """
    if table.empty and len(table.columns) == 0:
        schema_desc: list[tuple[str, str]] = []
        schema_sha = sha256_of(canonical_json(schema_desc))
        content_sha = sha256_of("")
        return {"content_sha256": content_sha, "schema_sha256": schema_sha}

    missing_pk = [c for c in primary_key if c not in table.columns]
    if missing_pk:
        raise ValueError(f"primary_key columns missing from table: {missing_pk}")

    columns_sorted = sorted(table.columns.tolist())
    work = table.loc[:, columns_sorted].copy()
    for col in columns_sorted:
        work[col] = _normalize_series(work[col])

    work = work.sort_values(list(primary_key), kind="mergesort").reset_index(drop=True)

    schema_desc = [(col, str(work[col].dtype)) for col in columns_sorted]
    schema_sha = sha256_of(canonical_json(schema_desc))

    lines: list[str] = []
    for row in work.itertuples(index=False, name=None):
        cells = [_serialize_cell(v) for v in row]
        lines.append("\t".join(cells))
    payload = "\n".join(lines)
    content_sha = sha256_of(payload)
    return {"content_sha256": content_sha, "schema_sha256": schema_sha}
