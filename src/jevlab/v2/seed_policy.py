"""V2 protocol revision 1 random-null child-seed derivation.

The original additive formula is retained only as an audit artifact.
Execution uses SHA-256 mixing of (namespace, base_seed, draw_id, phase).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from jevlab.config.v2_research import (
    CHILD_SEED_BYTES,
    CHILD_SEED_FORMULA_R1,
    EXPECTED_BASE_SEED,
    NUMPY_PCG64_API,
    PROTOCOL_REVISION,
    SEED_CANONICAL_STRING,
    SEED_NAMESPACE_R1,
)
from jevlab.v2.candidate import V2_PHASES
from jevlab.v2.statistics import FROZEN_N

PROTOCOL_REVISION_R1: int = PROTOCOL_REVISION
BASE_SEED: int = EXPECTED_BASE_SEED
assert BASE_SEED == 15224036858585574915
assert SEED_CANONICAL_STRING == "v2-baseline-validation-random-null-v1"
assert SEED_NAMESPACE_R1 == "v2-baseline-validation-random-null-r1"

DRAW_ID_MIN: int = 0
DRAW_ID_MAX: int = FROZEN_N - 1  # 1999


def child_canonical_string(
    draw_id: int,
    phase: int,
    *,
    namespace: str = SEED_NAMESPACE_R1,
    base_seed: int = BASE_SEED,
) -> str:
    """Exact UTF-8 canonical string for one (draw_id, phase) stream."""
    return f"{namespace}|{base_seed}|{draw_id}|{phase}"


def child_seed(
    draw_id: int,
    phase: int,
    *,
    namespace: str = SEED_NAMESPACE_R1,
    base_seed: int = BASE_SEED,
) -> int:
    """Unsigned big-endian integer from the first 16 bytes of SHA-256."""
    if draw_id < DRAW_ID_MIN or draw_id > DRAW_ID_MAX:
        raise ValueError(f"draw_id must be in [{DRAW_ID_MIN}, {DRAW_ID_MAX}]")
    if phase not in V2_PHASES:
        raise ValueError(f"phase must be in {V2_PHASES}")
    canonical = child_canonical_string(draw_id, phase, namespace=namespace, base_seed=base_seed)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:CHILD_SEED_BYTES], byteorder="big", signed=False)


def rng_for(
    draw_id: int,
    phase: int,
    *,
    namespace: str = SEED_NAMESPACE_R1,
    base_seed: int = BASE_SEED,
) -> np.random.Generator:
    """``Generator(PCG64(child_seed))`` with the 128-bit SHA-256 child seed."""
    return np.random.Generator(
        np.random.PCG64(child_seed(draw_id, phase, namespace=namespace, base_seed=base_seed))
    )


def enumerate_child_seeds(
    *,
    n: int = FROZEN_N,
    phases: tuple[int, ...] = V2_PHASES,
) -> list[tuple[int, int, int]]:
    """All (draw_id, phase, child_seed) triples in canonical order."""
    rows: list[tuple[int, int, int]] = []
    for draw_id in range(0, n):
        for phase in phases:
            rows.append((draw_id, phase, child_seed(draw_id, phase)))
    return rows


def seed_mapping_digest(*, n: int = FROZEN_N) -> tuple[str, int]:
    """Canonical digest of the 6000-stream mapping; returns (digest, n_unique)."""
    from jevlab.utils.hashing import canonical_json, sha256_of

    rows = enumerate_child_seeds(n=n)
    unique = {c for _d, _p, c in rows}
    payload: list[dict[str, Any]] = [
        {"draw_id": d, "phase": p, "child_seed": c} for d, p, c in rows
    ]
    return sha256_of(canonical_json(payload)), len(unique)


def seed_policy_dict() -> dict[str, Any]:
    return {
        "protocol_revision": PROTOCOL_REVISION_R1,
        "seed_canonical_string": SEED_CANONICAL_STRING,
        "seed_namespace": SEED_NAMESPACE_R1,
        "base_seed": BASE_SEED,
        "child_seed_formula": CHILD_SEED_FORMULA_R1,
        "generator": "numpy.random.PCG64",
        "numpy_api": NUMPY_PCG64_API,
        "n": FROZEN_N,
        "draw_id_indexing": "zero_based",
        "child_seed_bytes": CHILD_SEED_BYTES,
    }
