"""Frozen child-seed collision audit. Does not change the formula."""

from __future__ import annotations

from jevlab.v2.null_engine import child_seed
from jevlab.v2.seed_audit import audit_child_seed_collisions
from jevlab.v2.statistics import FROZEN_N


def test_frozen_additive_formula_has_draw_phase_collisions() -> None:
    audit = audit_child_seed_collisions(n=FROZEN_N)
    assert audit["formula_unchanged"] is True
    assert audit["n_requested_streams"] == 6000
    assert child_seed(1, 1) == child_seed(2, 0)
    assert child_seed(1, 2) == child_seed(3, 0)
    assert audit["same_phase_draws_distinct"] is True
    assert audit["same_draw_phases_distinct"] is True
    assert audit["n_unique_child_seeds"] == 2002
    assert audit["collision_seed_count"] == 2000
    assert audit["material_violation_of_distinct_draw_phase_streams"] is True
    assert audit["disposition"] == "BLOCK_FREEZE"
