"""Revision-1 child-seed uniqueness invariant. STOP if unique != 6000."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from jevlab.config.v2_research import SEED_NAMESPACE_R1
from jevlab.v2.candidate import V2_PHASES
from jevlab.v2.seed_policy import (
    BASE_SEED,
    child_canonical_string,
    child_seed,
    enumerate_child_seeds,
    seed_mapping_digest,
    seed_policy_dict,
)
from jevlab.v2.statistics import FROZEN_N

ORIGINAL_ADDITIVE_FORMULA = "(base_seed + draw_id + phase) mod 2**64"
MOD_64: int = 2**64


def original_additive_child_seed(draw_id: int, phase: int, *, base_seed: int = BASE_SEED) -> int:
    """Audit-only reconstruction of the original colliding formula (1-based draw_id)."""
    return int((base_seed + int(draw_id) + int(phase)) % MOD_64)


def audit_original_additive_collisions(
    *,
    n: int = FROZEN_N,
    phases: tuple[int, ...] = V2_PHASES,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Enumerate the original additive formula. Does not implement revision 1."""
    by_seed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for draw_id in range(1, n + 1):
        for phase in phases:
            child = original_additive_child_seed(draw_id, phase, base_seed=base_seed)
            by_seed[child].append((draw_id, phase))
    n_pairs = n * len(phases)
    n_unique = len(by_seed)
    colliding_seeds = {k: v for k, v in by_seed.items() if len(v) > 1}
    return {
        "formula": ORIGINAL_ADDITIVE_FORMULA,
        "protocol_revision": 0,
        "n_requested_streams": n_pairs,
        "n_unique_child_seeds": n_unique,
        "collision_seed_count": len(colliding_seeds),
        "material_violation_of_distinct_draw_phase_streams": n_unique != n_pairs,
        "superseded_by": "protocol_revision_1",
    }


def audit_revision1_uniqueness(
    *,
    n: int = FROZEN_N,
    phases: tuple[int, ...] = V2_PHASES,
) -> dict[str, Any]:
    """Enumerate all revision-1 streams. Raises if uniqueness fails."""
    rows = enumerate_child_seeds(n=n, phases=phases)
    by_seed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_pair: dict[tuple[int, int], int] = {}
    for draw_id, phase, child in rows:
        by_seed[child].append((draw_id, phase))
        key = (draw_id, phase)
        if key in by_pair:
            raise AssertionError(f"duplicate (draw_id, phase): {key}")
        by_pair[key] = child
    n_pairs = n * len(phases)
    n_unique = len(by_seed)
    colliding = {k: v for k, v in by_seed.items() if len(v) > 1}
    mapping_digest, digest_unique = seed_mapping_digest(n=n)
    if digest_unique != n_unique:
        raise AssertionError("mapping digest unique count drifted")
    if n_unique != n_pairs:
        raise AssertionError(
            f"STOP: unique_child_seeds={n_unique} != requested={n_pairs}"
        )
    if colliding:
        raise AssertionError(f"STOP: child seed maps to multiple pairs: {len(colliding)}")
    if len(by_pair) != n_pairs:
        raise AssertionError("STOP: (draw_id, phase) mapping is not 1-1")
    return {
        "protocol_revision": 1,
        "namespace": SEED_NAMESPACE_R1,
        "base_seed": BASE_SEED,
        "n_draws": n,
        "phases": list(phases),
        "n_requested_streams": n_pairs,
        "n_unique_child_seeds": n_unique,
        "collision_count": 0,
        "seed_mapping_digest": mapping_digest,
        "canonical_example": child_canonical_string(0, 0),
        "child_seed_0_0": child_seed(0, 0),
        "each_pair_one_seed": True,
        "each_seed_one_pair": True,
        "disposition": "UNIQUE_OK",
        "policy": seed_policy_dict(),
    }
