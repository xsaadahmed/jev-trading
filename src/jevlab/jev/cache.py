"""Immutable Jev response cache: SQLite index + zstd JSONL shards (Phase 5).

``request_hash = sha256(canonical_json({route, model_requested, state, questions}))``.
Repeat calls are stored as separate attempts (determinism audits).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import zstandard as zstd

from jevlab.utils.hashing import canonical_json, sha256_of

Purpose = Literal["research", "audit"]


def request_hash(
    *,
    route: str,
    model_requested: str,
    state: Any,
    questions: Any,
) -> str:
    payload = {
        "route": route,
        "model_requested": model_requested,
        "state": state,
        "questions": questions,
    }
    return sha256_of(canonical_json(payload))


@dataclass(frozen=True, slots=True)
class CacheRecord:
    request_hash: str
    attempt: int
    route: str
    model_requested: str
    model_resolved: str | None
    purpose: Purpose
    qset_version: str
    state_schema_version: str
    response: dict[str, Any]
    created_at: str
    shard_path: str


class JevCache:
    """Append-only cache. Hits return the first attempt unless ``attempt`` is set."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards_dir = self.root / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jev_cache (
                    request_hash TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    route TEXT NOT NULL,
                    model_requested TEXT NOT NULL,
                    model_resolved TEXT,
                    purpose TEXT NOT NULL,
                    qset_version TEXT NOT NULL,
                    state_schema_version TEXT NOT NULL,
                    shard_path TEXT NOT NULL,
                    byte_offset INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (request_hash, attempt)
                )
                """
            )
            conn.commit()

    def get(
        self,
        req_hash: str,
        *,
        attempt: int | None = None,
    ) -> CacheRecord | None:
        with self._connect() as conn:
            if attempt is None:
                row = conn.execute(
                    """
                    SELECT * FROM jev_cache
                    WHERE request_hash = ?
                    ORDER BY attempt ASC
                    LIMIT 1
                    """,
                    (req_hash,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT * FROM jev_cache
                    WHERE request_hash = ? AND attempt = ?
                    """,
                    (req_hash, attempt),
                ).fetchone()
        if row is None:
            return None
        response = self._read_shard(row["shard_path"], int(row["byte_offset"]))
        return CacheRecord(
            request_hash=row["request_hash"],
            attempt=int(row["attempt"]),
            route=row["route"],
            model_requested=row["model_requested"],
            model_resolved=row["model_resolved"],
            purpose="audit" if str(row["purpose"]) == "audit" else "research",
            qset_version=row["qset_version"],
            state_schema_version=row["state_schema_version"],
            response=response,
            created_at=row["created_at"],
            shard_path=row["shard_path"],
        )

    def next_attempt(self, req_hash: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(attempt), 0) AS m FROM jev_cache WHERE request_hash = ?",
                (req_hash,),
            ).fetchone()
        return int(row["m"]) + 1

    def put(
        self,
        *,
        req_hash: str,
        route: str,
        model_requested: str,
        model_resolved: str | None,
        purpose: Purpose,
        qset_version: str,
        state_schema_version: str,
        response: dict[str, Any],
        attempt: int | None = None,
    ) -> CacheRecord:
        """Append a response. Never overwrites an existing (hash, attempt)."""
        att = attempt if attempt is not None else self.next_attempt(req_hash)
        existing = self.get(req_hash, attempt=att)
        if existing is not None:
            raise ValueError(f"immutable cache: ({req_hash}, attempt={att}) already exists")
        shard_name = f"{datetime.now(UTC).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}.jsonl.zst"
        shard_path = self.shards_dir / shard_name
        offset = self._append_shard(shard_path, response)
        created = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jev_cache (
                    request_hash, attempt, route, model_requested, model_resolved,
                    purpose, qset_version, state_schema_version, shard_path,
                    byte_offset, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req_hash,
                    att,
                    route,
                    model_requested,
                    model_resolved,
                    purpose,
                    qset_version,
                    state_schema_version,
                    str(shard_path.relative_to(self.root)),
                    offset,
                    created,
                ),
            )
            conn.commit()
        return CacheRecord(
            request_hash=req_hash,
            attempt=att,
            route=route,
            model_requested=model_requested,
            model_resolved=model_resolved,
            purpose=purpose,
            qset_version=qset_version,
            state_schema_version=state_schema_version,
            response=response,
            created_at=created,
            shard_path=str(shard_path.relative_to(self.root)),
        )

    def _append_shard(self, path: Path, response: dict[str, Any]) -> int:
        line = (json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(line)
        # Store one compressed frame per append with a length prefix for seeking.
        # Format: uint64 big-endian length + zstd frame (simple, immutable).
        header = len(compressed).to_bytes(8, "big")
        offset = path.stat().st_size if path.exists() else 0
        with path.open("ab") as fh:
            fh.write(header)
            fh.write(compressed)
        return offset

    def _read_shard(self, rel_path: str, offset: int) -> dict[str, Any]:
        path = self.root / rel_path
        with path.open("rb") as fh:
            fh.seek(offset)
            header = fh.read(8)
            if len(header) != 8:
                raise OSError(f"corrupt shard header at {path}:{offset}")
            n = int.from_bytes(header, "big")
            blob = fh.read(n)
        raw = zstd.ZstdDecompressor().decompress(blob)
        loaded: Any = json.loads(raw.decode("utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError(f"shard payload must be object, got {type(loaded)!r}")
        return loaded
