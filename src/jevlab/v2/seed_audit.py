"""Audit frozen child-seed uniqueness. Does not change the frozen formula."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from jevlab.v2.candidate import V2_PHASES
from jevlab.v2.null_engine import BASE_SEED, MOD_64, child_seed
from jevlab.v2.statistics import FROZEN_N

# Frozen formula, unchanged:
# child = (base_seed + draw_id + phase) mod 2**64
# draw_id is 1-based; phase in {0,1,2}.


def audit_child_seed_collisions(
    *,
    n: int = FROZEN_N,
    phases: tuple[int, ...] = V2_PHASES,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Enumerate all (draw, phase) child seeds under the frozen additive rule.

    Does not modify the formula. Reports uniqueness / collisions for the
    2000 x 3 confirmatory streams.
    """
    by_seed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pairs: list[tuple[int, int, int]] = []
    for draw_id in range(1, n + 1):
        for phase in phases:
            child = child_seed(draw_id, phase, base_seed=base_seed)
            expected = int((base_seed + draw_id + phase) % MOD_64)
            if child != expected:
                raise AssertionError("child_seed drifted from frozen additive formula")
            by_seed[child].append((draw_id, phase))
            pairs.append((draw_id, phase, child))

    n_pairs = n * len(phases)
    n_unique = len(by_seed)
    colliding_seeds = {k: v for k, v in by_seed.items() if len(v) > 1}
    unique_pairs = [v[0] for v in by_seed.values() if len(v) == 1]
    examples = [{"child_seed": k, "pairs": v} for k, v in list(colliding_seeds.items())[:5]]
    # Canonical additive collision: (r, 1) == (r+1, 0).
    additive_example = {
        "draw_r_phase_1": child_seed(1, 1, base_seed=base_seed),
        "draw_rplus1_phase_0": child_seed(2, 0, base_seed=base_seed),
        "equal": child_seed(1, 1, base_seed=base_seed) == child_seed(2, 0, base_seed=base_seed),
    }
    material = n_unique != n_pairs
    return {
        "formula": "(base_seed + draw_id + phase) mod 2**64",
        "encoding": "plain integer addition; no extra mixing of (draw_id, phase)",
        "n_draws": n,
        "phases": list(phases),
        "n_requested_streams": n_pairs,
        "n_unique_child_seeds": n_unique,
        "collision_seed_count": len(colliding_seeds),
        "colliding_pair_count": n_pairs - len(unique_pairs),
        "unique_pair_count": len(unique_pairs),
        "unique_pairs": unique_pairs,
        "examples": examples,
        "additive_collision_example": additive_example,
        "same_phase_draws_distinct": all(
            child_seed(1, p, base_seed=base_seed) != child_seed(2, p, base_seed=base_seed)
            for p in phases
        ),
        "same_draw_phases_distinct": all(
            child_seed(1, 0, base_seed=base_seed) != child_seed(1, p, base_seed=base_seed)
            for p in phases
            if p != 0
        ),
        "material_violation_of_distinct_draw_phase_streams": material,
        "formula_unchanged": True,
        "disposition": "BLOCK_FREEZE" if material else "UNIQUE_OK",
        "n_enumerated_pairs": len(pairs),
    }
