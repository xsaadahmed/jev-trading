"""Revision-1 child-seed uniqueness and mapping determinism (no sealed data)."""

from __future__ import annotations

import hashlib
import subprocess
import sys

from jevlab.config.v2_research import (
    CHILD_SEED_BYTES,
    EXPECTED_BASE_SEED,
    SEED_NAMESPACE_R1,
)
from jevlab.v2.hashes import FROZEN_SEED_MAPPING_DIGEST
from jevlab.v2.seed_audit import (
    audit_original_additive_collisions,
    audit_revision1_uniqueness,
)
from jevlab.v2.seed_policy import (
    BASE_SEED,
    child_canonical_string,
    child_seed,
    seed_mapping_digest,
)
from jevlab.v2.statistics import FROZEN_N


def test_original_additive_formula_collision_is_preserved_as_audit() -> None:
    audit = audit_original_additive_collisions(n=FROZEN_N)
    assert audit["n_requested_streams"] == 6000
    assert audit["n_unique_child_seeds"] == 2002
    assert audit["material_violation_of_distinct_draw_phase_streams"] is True
    assert audit["superseded_by"] == "protocol_revision_1"


def test_revision1_unique_child_seeds_are_6000() -> None:
    audit = audit_revision1_uniqueness(n=FROZEN_N)
    assert audit["n_requested_streams"] == 6000
    assert audit["n_unique_child_seeds"] == 6000
    assert audit["collision_count"] == 0
    assert audit["disposition"] == "UNIQUE_OK"
    digest, n_unique = seed_mapping_digest(n=FROZEN_N)
    assert n_unique == 6000
    assert digest == audit["seed_mapping_digest"]
    digest2, n_unique2 = seed_mapping_digest(n=FROZEN_N)
    assert digest2 == digest
    assert n_unique2 == 6000
    assert digest == FROZEN_SEED_MAPPING_DIGEST


def test_revision1_child_seed_is_sha256_first_16_bytes() -> None:
    canonical = child_canonical_string(0, 0)
    assert canonical == f"{SEED_NAMESPACE_R1}|{EXPECTED_BASE_SEED}|0|0"
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    expected = int.from_bytes(digest[:CHILD_SEED_BYTES], byteorder="big", signed=False)
    assert child_seed(0, 0) == expected
    assert child_seed(0, 0).bit_length() <= 128
    assert child_seed(0, 0) != (BASE_SEED + 0 + 0) % 2**64
    assert child_seed(0, 1) != child_seed(1, 0)
    assert child_seed(1, 1) != child_seed(2, 0)


def test_child_seed_deterministic_across_process_restarts() -> None:
    code = (
        "from jevlab.v2.seed_policy import child_seed, seed_mapping_digest; "
        "d, n = seed_mapping_digest(); "
        "print(child_seed(0, 0)); print(child_seed(1999, 2)); print(d); print(n)"
    )
    a = subprocess.check_output([sys.executable, "-c", code], text=True)
    b = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert a == b
    lines = a.strip().splitlines()
    assert int(lines[0]) == child_seed(0, 0)
    assert int(lines[1]) == child_seed(1999, 2)
    assert int(lines[3]) == 6000
