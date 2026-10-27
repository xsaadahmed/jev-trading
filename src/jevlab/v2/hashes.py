"""Frozen V2 research hashes. Implementation must not drift these values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevlab.config.v2_research import PROTOCOL_REVISION, load_v2_research_bundle

# Original V2 design freeze (provenance; not execution-authoritative after amendment 001).
ORIGINAL_DESIGN_COMMIT = "6f4f815a7551244673cb8f0a853f238ad11b8544"
ORIGINAL_TAG_PREREG = "prereg-v2-baseline"
ORIGINAL_TAG_DESIGN = "v2-design-frozen"
ORIGINAL_RESEARCH_CONFIG_HASH = "96947d9901b5366246ab1073af7ec17a83cf2bd074ecb0f32f66fcad21a8fe1c"
ORIGINAL_RANDOM_NULL_HASH = "d8661aa1fcaf42ce239865049a9707d41dee2c598faf4de06de7630cfead977e"

FROZEN_COMMIT = ORIGINAL_DESIGN_COMMIT
FROZEN_TAG_PREREG = ORIGINAL_TAG_PREREG
FROZEN_TAG_DESIGN = ORIGINAL_TAG_DESIGN
FROZEN_TAG_PREREG_R1 = "prereg-v2-baseline-r1"
FROZEN_TAG_DESIGN_R1 = "v2-design-frozen-r1"
FROZEN_PRESEAL_IMPL_TAG_R1 = "v2-preseal-implementation-r1"
FROZEN_PRESEAL_IMPL_TAG_R2 = "v2-preseal-implementation-r2"
FROZEN_PRESEAL_IMPL_TAG = "v2-preseal-implementation-r3"
FROZEN_IMPLEMENTATION_REVISION = 3
FROZEN_PROTOCOL_REVISION = PROTOCOL_REVISION
R2_EXECUTION_COMMIT = "8197cd906752c40f18129fa372e56b6013fc1079"

# Execution-authoritative revision-1 hashes.
FROZEN_RESEARCH_CONFIG_HASH = "e436ed977a02a39a0a280d78dcad74fa67e3ff1276f18a3430f249a9aeaf13ce"
FROZEN_CANDIDATE_HASH = "aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28"
FROZEN_UNIVERSE_HASH = "73d3fcff3c0b7df72871e6ab72085c3e4afba998ce1f66e31e06705b36d473fb"
FROZEN_FEATURE_SET_HASH = "cf4e9dab9b20d33cb5c51ecb5cf4c2bdad3282016afbe34e2a6ce9d94e7ad396"
FROZEN_TARGET_HASH = "baf60d37d71211e94a2bfeff0027d8a1b4a2e131cf1023e03ce1d289acf35381"
FROZEN_STRATEGY_HASH = "ce46464d3d33bd20e96c6f429ebad60a991dc284b331f769c457f121ea43b521"
FROZEN_COST_HASH = "74114006e731780591dd2a0b8ea9110f1261fe99fe9e8bb7231d326b7b529329"
FROZEN_RANDOM_NULL_HASH = "9365e46d2ffc657d2b078cd851b90bac8f8ace3357313743769efdabf6498b2e"
FROZEN_SEED_MAPPING_DIGEST = "5c773811984b2fea232889295560f3ded3e6db79ca8e949ee5518933f88e58f9"
FROZEN_ACCEPTANCE_GATE_HASH = "5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be"
FROZEN_ELIGIBILITY_HASH = "b97cb39dd99e65f9c6ede7a2a34b2e5a2cc55b5dec196117c42f4375c1ece420"
FROZEN_TRAINING_HASH = "efb523fdec83fcca00c2caea0753188f168f237a8af0f7736af508ef5bccbee4"

FROZEN_SECTION_HASHES: dict[str, str] = {
    "candidate": FROZEN_CANDIDATE_HASH,
    "universe": FROZEN_UNIVERSE_HASH,
    "feature_set": FROZEN_FEATURE_SET_HASH,
    "target": FROZEN_TARGET_HASH,
    "eligibility": FROZEN_ELIGIBILITY_HASH,
    "training": FROZEN_TRAINING_HASH,
    "strategy": FROZEN_STRATEGY_HASH,
    "costs": FROZEN_COST_HASH,
    "random_null": FROZEN_RANDOM_NULL_HASH,
    "acceptance_gates": FROZEN_ACCEPTANCE_GATE_HASH,
}


class FrozenHashError(AssertionError):
    """V2 live hashes diverged from the frozen design."""


def live_section_hashes(*, root: Path | None = None) -> dict[str, str]:
    bundle = load_v2_research_bundle(root=root)
    return {
        "research_config_hash": bundle.research_config_hash,
        **{k: bundle.section_hashes[k] for k in FROZEN_SECTION_HASHES},
    }


def assert_frozen_research_hashes(*, root: Path | None = None) -> dict[str, str]:
    """Assert live V2 hashes equal the freeze-commit hashes.

    Raises FrozenHashError rather than silently adopting new semantics.
    """
    live = live_section_hashes(root=root)
    mismatches: list[str] = []
    if live["research_config_hash"] != FROZEN_RESEARCH_CONFIG_HASH:
        mismatches.append(
            f"research_config_hash live={live['research_config_hash']} "
            f"frozen={FROZEN_RESEARCH_CONFIG_HASH}"
        )
    for key, expected in FROZEN_SECTION_HASHES.items():
        got = live[key]
        if got != expected:
            mismatches.append(f"{key} live={got} frozen={expected}")
    if mismatches:
        raise FrozenHashError(
            "V2 research hashes drifted from freeze; do not change semantics. "
            + "; ".join(mismatches)
        )
    return live


def frozen_hash_payload() -> dict[str, Any]:
    return {
        "protocol_revision": FROZEN_PROTOCOL_REVISION,
        "original_design_commit": ORIGINAL_DESIGN_COMMIT,
        "original_research_config_hash": ORIGINAL_RESEARCH_CONFIG_HASH,
        "original_random_null_hash": ORIGINAL_RANDOM_NULL_HASH,
        "frozen_commit": FROZEN_COMMIT,
        "research_config_hash": FROZEN_RESEARCH_CONFIG_HASH,
        "seed_mapping_digest": FROZEN_SEED_MAPPING_DIGEST,
        "implementation_revision": FROZEN_IMPLEMENTATION_REVISION,
        **FROZEN_SECTION_HASHES,
    }
