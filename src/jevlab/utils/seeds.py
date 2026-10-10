"""Deterministic seed helpers for confirmatory and exploratory runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Bundle of explicit seeds recorded in experiment manifests."""

    master: int
    numpy: int
    python: int
    bootstrap: int
    placebo: int


def set_global_seeds(seed: int) -> SeedBundle:
    """Set Python and NumPy RNGs from an explicit integer seed.

    Does not touch LightGBM / sklearn; those receive seeds at construction time.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    python_seed = int(seed)
    numpy_seed = int(seed)
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    os.environ["PYTHONHASHSEED"] = str(python_seed)
    return SeedBundle(
        master=python_seed,
        numpy=numpy_seed,
        python=python_seed,
        bootstrap=python_seed + 1,
        placebo=python_seed + 2,
    )
