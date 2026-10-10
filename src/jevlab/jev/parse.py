"""Parse System One answers into Jev features (MASTER_SPEC §7.2).

Features produced:
``j_dislocation, j_downside_E, j_downside_p3, j_regime, j_pullback_p,
j_breakdown_p, j_sideways_p, j_setup_conf`` plus quality metric
``j_order_disagree`` (total variation distance between choice orders).
"""

from __future__ import annotations

from typing import Any

from jevlab.jev.schemas import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

JEV_FEATURE_COLUMNS: tuple[str, ...] = (
    "j_dislocation",
    "j_downside_E",
    "j_downside_p3",
    "j_regime",
    "j_pullback_p",
    "j_breakdown_p",
    "j_sideways_p",
    "j_setup_conf",
)

SETUP_OPTIONS: tuple[str, ...] = (
    "pullback_in_uptrend",
    "trend_breakdown",
    "sideways_noise",
    "extended_rally_pause",
)


def noul_confidence(p: float) -> float:
    """Stand-in confidence for Noul: ``|2p - 1|`` (§2.4)."""
    return abs(2.0 * float(p) - 1.0)


def choice_confidence(probabilities: dict[str, float]) -> float:
    """Choice confidence = ``(p_max - 1/n) / (1 - 1/n)``."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    p_max = max(probabilities.values()) if probabilities else 0.0
    return float((p_max - 1.0 / n) / (1.0 - 1.0 / n))


def score_confidence(probabilities: dict[str, float]) -> float:
    """Ordinal MAD-from-mode confidence (§2.4 / TypeSafe docs).

    ``confidence = max(0, 1 - d / MAD_unif)`` where
    ``d = sum_i p_i * |i - m|``, ``MAD_unif = (1/n) * sum_i |i - (n-1)/2|``,
    levels numbered ``0..n-1``, ``m`` = argmax level.
    """
    if not probabilities:
        return 0.0
    items = sorted(((int(k), float(v)) for k, v in probabilities.items()), key=lambda kv: kv[0])
    n = len(items)
    levels = [i for i, _ in items]
    # Map to dense 0..n-1 by order of keys
    index_by_level = {lev: idx for idx, lev in enumerate(levels)}
    m_level = max(items, key=lambda kv: kv[1])[0]
    m = index_by_level[m_level]
    d = 0.0
    for lev, p in items:
        d += p * abs(index_by_level[lev] - m)
    center = (n - 1) / 2.0
    mad_unif = sum(abs(i - center) for i in range(n)) / float(n)
    if mad_unif <= 0:
        return 1.0
    return float(max(0.0, 1.0 - d / mad_unif))


def score_expectation(probabilities: dict[str, float]) -> float:
    """Probability-weighted level index (0-based)."""
    total = 0.0
    for k, p in probabilities.items():
        total += int(k) * float(p)
    return float(total)


def score_p_ge(probabilities: dict[str, float], threshold: int = 3) -> float:
    """``P(level >= threshold)``."""
    return float(sum(float(p) for k, p in probabilities.items() if int(k) >= threshold))


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance ``0.5 * sum_i |p_i - q_i|`` over union of keys."""
    keys = set(p) | set(q)
    return 0.5 * float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def average_choice_probs(
    a: dict[str, float],
    b: dict[str, float],
) -> dict[str, float]:
    """Equal-weight average of two choice distributions (order-bias mitigation)."""
    keys = set(a) | set(b)
    return {k: 0.5 * (float(a.get(k, 0.0)) + float(b.get(k, 0.0))) for k in keys}


def parse_response(response: SystemOneResponse | dict[str, Any]) -> dict[str, float]:
    """Parse a qset_v1 response into the 8 features + ``j_order_disagree``."""
    resp = (
        response
        if isinstance(response, SystemOneResponse)
        else SystemOneResponse.model_validate(response)
    )
    dislocation = NoulAnswer.model_validate(resp.answers["dislocation"])
    regime = NoulAnswer.model_validate(resp.answers["regime_supportive"])
    downside = ScoreAnswer.model_validate(resp.answers["downside_risk"])
    setup_a = ChoiceAnswer.model_validate(resp.answers["setup_type_a"])
    setup_b = ChoiceAnswer.model_validate(resp.answers["setup_type_b"])

    avg = average_choice_probs(setup_a.probabilities, setup_b.probabilities)
    conf = choice_confidence(avg)
    # Recompute score confidence from documented formula (ignore vendor field).
    _ = score_confidence(downside.probabilities)

    return {
        "j_dislocation": float(dislocation.noul),
        "j_downside_E": score_expectation(downside.probabilities),
        "j_downside_p3": score_p_ge(downside.probabilities, 3),
        "j_regime": float(regime.noul),
        "j_pullback_p": float(avg.get("pullback_in_uptrend", 0.0)),
        "j_breakdown_p": float(avg.get("trend_breakdown", 0.0)),
        "j_sideways_p": float(avg.get("sideways_noise", 0.0)),
        "j_setup_conf": conf,
        "j_order_disagree": total_variation_distance(setup_a.probabilities, setup_b.probabilities),
        "j_noul_dislocation_conf": noul_confidence(dislocation.noul),
        "j_noul_regime_conf": noul_confidence(regime.noul),
        "j_downside_conf": score_confidence(downside.probabilities),
    }
