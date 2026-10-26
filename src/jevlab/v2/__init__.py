"""V2 baseline-validation implementation (post-preregistration).

Preserves frozen research semantics. Does not access 2024-2025 sealed data
unless an authorized one-shot sealed command is explicitly invoked.
"""

from jevlab.v2.hashes import (
    FROZEN_ACCEPTANCE_GATE_HASH,
    FROZEN_CANDIDATE_HASH,
    FROZEN_COMMIT,
    FROZEN_RESEARCH_CONFIG_HASH,
    assert_frozen_research_hashes,
)

__all__ = [
    "FROZEN_ACCEPTANCE_GATE_HASH",
    "FROZEN_CANDIDATE_HASH",
    "FROZEN_COMMIT",
    "FROZEN_RESEARCH_CONFIG_HASH",
    "assert_frozen_research_hashes",
]
