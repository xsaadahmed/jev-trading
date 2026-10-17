"""Rule-based conclusion generator (MASTER_SPEC §16.4 / Phase 11).

Produces exactly one of three fixed verdicts. Never emits pass language when any
gate fails. Never implies that live $10 trading is evidence of an edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from jevlab.reporting.statements import (
    CONCLUSION_NO_VALUE,
    CONCLUSION_PASS,
    CONCLUSION_UNDERPOWERED,
    LIVE_EXECUTION_EXPERIMENT,
)

VerdictKind = Literal["pass", "no_value", "underpowered"]


@dataclass(frozen=True, slots=True)
class ConclusionResult:
    kind: VerdictKind
    text: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "reasons": list(self.reasons)}


def _gate_map(gates: Mapping[str, Any] | None) -> dict[str, bool | None]:
    """Normalize gate payload to {gate_id: passed|None}."""
    out: dict[str, bool | None] = {}
    if not gates:
        return out
    # Accept {"G4": true}, {"G4": {"passed": true}}, or list of {id, passed}
    if isinstance(gates, list):
        for item in gates:
            if not isinstance(item, Mapping):
                continue
            gid = str(item.get("id") or item.get("gate") or "")
            if not gid:
                continue
            if "passed" in item:
                if item["passed"] is None:
                    out[gid] = None
                else:
                    out[gid] = bool(item["passed"])
            elif "status" in item:
                out[gid] = _status_to_passed(str(item["status"]))
            else:
                out[gid] = None
        return out
    for k, v in gates.items():
        if isinstance(v, bool):
            out[str(k)] = v
        elif isinstance(v, Mapping):
            if "passed" in v:
                if v["passed"] is None:
                    out[str(k)] = None
                else:
                    out[str(k)] = bool(v["passed"])
            elif "status" in v:
                out[str(k)] = _status_to_passed(str(v["status"]))
            else:
                out[str(k)] = None
        elif v is None:
            out[str(k)] = None
        else:
            out[str(k)] = bool(v)
    return out


def _status_to_passed(status: str) -> bool | None:
    st = status.strip().lower().replace(" ", "_")
    if st in {"pass", "passed", "ok", "true"}:
        return True
    if st in {
        "fail",
        "failed",
        "false",
        "no",
    }:
        return False
    if st in {
        "not_evaluated",
        "n/a",
        "na",
        "unevaluable",
        "skipped",
        "deferred",
    }:
        return None
    return None


def generate_conclusion(
    *,
    gates: Mapping[str, Any] | list[Any] | None = None,
    underpowered: bool | None = None,
    power: Mapping[str, Any] | None = None,
    force_kind: VerdictKind | None = None,
) -> ConclusionResult:
    """Derive the report conclusion from gate results and power metadata.

    Precedence:
    1. ``force_kind`` (tests / explicit operator override in fixture only)
    2. Any gate with ``passed is False`` → ``CONCLUSION_NO_VALUE``
    3. ``underpowered`` True, or power.flag / observed < MDE → ``CONCLUSION_UNDERPOWERED``
    4. All evaluated gates True and at least one gate present → ``CONCLUSION_PASS``
    5. Otherwise underpowered (insufficient to claim no-value or pass)
    """
    if force_kind == "pass":
        return ConclusionResult("pass", CONCLUSION_PASS, ("forced",))
    if force_kind == "no_value":
        return ConclusionResult("no_value", CONCLUSION_NO_VALUE, ("forced",))
    if force_kind == "underpowered":
        return ConclusionResult("underpowered", CONCLUSION_UNDERPOWERED, ("forced",))

    gmap = _gate_map(gates)
    failed = sorted(gid for gid, passed in gmap.items() if passed is False)
    if failed:
        return ConclusionResult(
            "no_value",
            CONCLUSION_NO_VALUE,
            tuple(f"gate_failed:{gid}" for gid in failed),
        )

    under = bool(underpowered) if underpowered is not None else False
    reasons: list[str] = []
    if power:
        if (
            power.get("underpowered") is True
            or str(power.get("verdict", "")).lower() == "underpowered"
        ):
            under = True
            reasons.append("power.underpowered")
        obs = power.get("observed_effect")
        mde = power.get("mde")
        try:
            if obs is not None and mde is not None and float(obs) < float(mde):
                under = True
                reasons.append("observed_effect_below_mde")
        except (TypeError, ValueError):
            pass

    if under:
        return ConclusionResult(
            "underpowered",
            CONCLUSION_UNDERPOWERED,
            tuple(reasons) or ("underpowered_flag",),
        )

    passed = [gid for gid, p in gmap.items() if p is True]
    if passed and not failed and all(p is not False for p in gmap.values()):
        # Require no explicit failures; unevaluable (None) does not count as pass.
        if any(p is None for p in gmap.values()):
            return ConclusionResult(
                "underpowered",
                CONCLUSION_UNDERPOWERED,
                ("unevaluable_gates_present",),
            )
        return ConclusionResult(
            "pass",
            CONCLUSION_PASS,
            tuple(f"gate_passed:{gid}" for gid in sorted(passed)),
        )

    return ConclusionResult(
        "underpowered",
        CONCLUSION_UNDERPOWERED,
        ("insufficient_gate_evidence",),
    )


def assert_conclusion_safety(text: str, *, kind: VerdictKind) -> None:
    """Raise if conclusion text violates Phase 11 safety rules."""
    if kind != "pass":
        lowered = text.lower()
        forbidden = (
            "historical_research_pass",
            "gates all passed",
            "all gates passed",
            "demonstrates useful",
            "edge is proven",
            "edge is genuine",
        )
        for token in forbidden:
            if token in lowered:
                raise AssertionError(f"pass language forbidden for kind={kind}: {token}")
    # Never imply $10 live is evidence of an edge.
    if "evidence of a profitable" in text.lower() and "not evidence" not in text.lower():
        raise AssertionError("must not imply live $10 is evidence of an edge")
    if kind == "no_value" and text != CONCLUSION_NO_VALUE:
        raise AssertionError("no_value verdict must be exact CONCLUSION_NO_VALUE string")
    if kind == "underpowered" and text != CONCLUSION_UNDERPOWERED:
        raise AssertionError("underpowered verdict must be exact CONCLUSION_UNDERPOWERED string")
    _ = LIVE_EXECUTION_EXPERIMENT  # referenced for documentation coupling
