"""V2 p-value / Sharpe-advantage boundary tests (full precision)."""

from __future__ import annotations

import math

import pytest

from jevlab.v2.statistics import (
    FROZEN_N,
    MIN_SHARPE_ADVANTAGE,
    e1_pass,
    e2_pass,
    e3_pass,
    e4_pass,
    e5_pass,
    empirical_pvalue,
    evaluate_randomization,
)


def test_frozen_n_is_2000() -> None:
    assert FROZEN_N == 2000


def test_p_boundary_exact() -> None:
    # N=19 → denominator 20; k=0 → p=0.05 exactly.
    p = empirical_pvalue(n_ge=0, n=19)
    assert p == pytest.approx(0.05)
    assert e1_pass(p) is True
    p_fail = empirical_pvalue(n_ge=1, n=19)
    assert p_fail > 0.05
    assert e1_pass(p_fail) is False


def test_p_formula_n2000() -> None:
    p = empirical_pvalue(n_ge=99, n=2000)
    assert p == (1.0 + 99.0) / 2001.0
    assert p < 0.05
    assert e1_pass(p) is True
    p2 = empirical_pvalue(n_ge=100, n=2000)
    assert p2 == (1.0 + 100.0) / 2001.0
    assert p2 > 0.05
    assert e1_pass(p2) is False


def test_advantage_boundary() -> None:
    assert e2_pass(0.10) is True
    assert e2_pass(MIN_SHARPE_ADVANTAGE) is True
    assert e2_pass(math.nextafter(0.10, 0.0)) is False
    assert e2_pass(0.099999999999) is False


def test_cagr_and_dd_boundaries() -> None:
    assert e3_pass(0.0) is False
    assert e3_pass(1e-16) is True
    assert e4_pass(-1e-16) is False
    assert e5_pass(-0.35) is True
    assert e5_pass(-0.3500000001) is False
    assert e5_pass(-0.3500001) is False
    assert e5_pass(-0.80) is False
    assert e5_pass(float("nan")) is False
    assert e1_pass(float("inf")) is False
    assert e2_pass(float("nan")) is False
    assert e3_pass(float("inf")) is False
    assert e4_pass(float("-inf")) is False


def test_evaluate_randomization_nan_fail_closed() -> None:
    res = evaluate_randomization(float("nan"), [0.1, 0.2, 0.3], n=3)
    assert math.isnan(res.p_value)
    assert e1_pass(res.p_value) is False
