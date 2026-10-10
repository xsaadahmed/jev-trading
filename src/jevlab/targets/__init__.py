"""Label construction and eligibility; sole home for functions that touch targets.

Eligibility helpers are imported eagerly (used by the feature pipeline). Label
constructors load lazily so Phase 13A forward-capture can use features without
pulling scoring/label code into memory.
"""

from __future__ import annotations

from typing import Any

from jevlab.targets.eligibility import attach_eligible, compute_eligible

__all__ = [
    "LABEL_VERSION",
    "assert_label_after_feature",
    "attach_eligible",
    "build_labels",
    "compute_eligible",
    "roundtrip_cost_dec",
]


def __getattr__(name: str) -> Any:
    if name in {
        "LABEL_VERSION",
        "assert_label_after_feature",
        "build_labels",
        "roundtrip_cost_dec",
    }:
        from jevlab.targets import labels as _labels

        return getattr(_labels, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
