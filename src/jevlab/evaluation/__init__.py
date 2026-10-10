"""Predictive/economic metrics, bootstrap inference, placebo and robustness suites."""

from jevlab.evaluation.bootstrap import bootstrap_ab_deltas, bootstrap_mean_difference
from jevlab.evaluation.metrics import (
    compute_economic_metrics,
    power_statement,
)
from jevlab.evaluation.metrics import (
    minimum_detectable_effect as minimum_detectable_effect_blocks,
)
from jevlab.evaluation.placebo import PLACEBO_N, empirical_pvalue, run_placebo_loop
from jevlab.evaluation.predictive import predictive_summary
from jevlab.evaluation.robustness import run_robustness_suite, write_robustness_report
from jevlab.evaluation.stats_summary import build_ab_stats_summary, write_stats_summary
from jevlab.evaluation.tests import (
    deflated_sharpe_ratio,
    diebold_mariano,
    forward_consistency_z,
    minimum_detectable_effect,
)

__all__ = [
    "PLACEBO_N",
    "bootstrap_ab_deltas",
    "bootstrap_mean_difference",
    "build_ab_stats_summary",
    "compute_economic_metrics",
    "deflated_sharpe_ratio",
    "diebold_mariano",
    "empirical_pvalue",
    "forward_consistency_z",
    "minimum_detectable_effect",
    "minimum_detectable_effect_blocks",
    "power_statement",
    "predictive_summary",
    "run_placebo_loop",
    "run_robustness_suite",
    "write_robustness_report",
    "write_stats_summary",
]
