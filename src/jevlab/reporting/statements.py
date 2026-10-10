"""Mandatory report statements (MASTER_SPEC §16.4)."""

from __future__ import annotations

# Exact strings required in every experiment report (§16.4).
HISTORICAL_CONTAMINATION: str = (
    "Results involving Jev features are potentially training-contaminated."
)
LIVE_EXECUTION_EXPERIMENT: str = (
    "This live phase is an execution experiment, not evidence of a profitable trading edge."
)
CONCLUSION_NO_VALUE: str = "Jev does not demonstrate useful predictive value."
CONCLUSION_UNDERPOWERED: str = "Inconclusive: study underpowered."

# Additional required phrasing when HISTORICAL_RESEARCH_PASS is granted (§16.1).
HISTORICAL_PASS_CONTAMINATION_CAVEAT: str = (
    "Historical Jev evidence is potentially training-contaminated; "
    "this status does not establish that the edge is genuine."
)

WARNING_SIGNAL_NOT_PROOF: str = "warning signal, not proof"

CONTAMINATION_LABEL: str = "potentially_training_contaminated"

# Pass verdict language — must never appear when any gate fails, and must not
# claim that live $10 trading is evidence of an edge.
CONCLUSION_PASS: str = (
    "Historical research gates all passed under pre-registered thresholds; "
    "status HISTORICAL_RESEARCH_PASS authorizes continuing research and dry-run "
    "infrastructure only. It does not authorize live capital deployment by itself, "
    "and it does not establish that any Jev edge is free of training contamination."
)

MANDATORY_STATEMENTS: tuple[str, ...] = (
    HISTORICAL_CONTAMINATION,
    LIVE_EXECUTION_EXPERIMENT,
)


def assert_mandatory_present(text: str) -> list[str]:
    """Return list of missing mandatory §16.4 strings (empty if complete)."""
    return [s for s in MANDATORY_STATEMENTS if s not in text]
