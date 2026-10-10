"""Experiment manifests, runners, acceptance artifacts, and forward registry.

Phase 13A provides forward INPUT capture and the immutable forward_registry.
This package must not import ``jevlab.brokers`` (shadow logging places no orders).
"""

from jevlab.experiments.acceptance import (
    HistoricalGateMetrics,
    evaluate_historical_gates,
    run_final_test,
)
from jevlab.experiments.artifacts import verify_acceptance, write_acceptance
from jevlab.experiments.forward_registry import (
    ForwardRegistryImmutableError,
    ForwardRegistryRow,
    compute_forward_start_date,
    read_deployment,
    read_forward_start_date,
    register_deployment,
)
from jevlab.experiments.runner import (
    LineupRunResult,
    build_comparison_table,
    comparison_table_markdown,
    run_model_set,
)

__all__ = [
    "ForwardRegistryImmutableError",
    "ForwardRegistryRow",
    "HistoricalGateMetrics",
    "LineupRunResult",
    "build_comparison_table",
    "comparison_table_markdown",
    "compute_forward_start_date",
    "evaluate_historical_gates",
    "read_deployment",
    "read_forward_start_date",
    "register_deployment",
    "run_final_test",
    "run_model_set",
    "verify_acceptance",
    "write_acceptance",
]
