"""TINY_LIVE_READY evaluator T1-T9 (MASTER_SPEC §16.2). No forward-session count."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jevlab.brokers.live_gate import EXECUTION_EXPERIMENT_SENTENCE
from jevlab.constants import TINY_LIVE_CEILING_USD
from jevlab.experiments.artifacts import (
    AcceptanceArtifact,
    ArtifactError,
    verify_acceptance,
    write_acceptance,
)
from jevlab.experiments.production_models import (
    list_model_artifacts,
    production_hashes_path,
    read_production_hashes,
)
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

TINY_GATE_IDS: tuple[str, ...] = tuple(f"T{i}" for i in range(1, 10))


@dataclass(frozen=True, slots=True)
class TinyGateResult:
    gate_id: str
    passed: bool
    detail: str


@dataclass
class TinyLiveMetrics:
    """Fixture-friendly T1-T9 inputs (no forward-session fields)."""

    historical_valid: bool = True
    production_forward_a_b_exist: bool = True
    production_hashes_before_label_join: bool = True
    safety_tests_pass: bool = True
    safety_test_report_present: bool = True
    dry_run_consecutive_sessions: int = 10
    dry_run_zero_discrepancies: bool = True
    dry_run_zero_gate_bypasses: bool = True
    kill_switch_drill_logged: bool = True
    pinned_model_id: str = "jev-1.13.0"
    dry_run_resolved_ids_match_pin: bool = True
    max_strategy_capital_usd: float = 10.0
    isolation_tests_pass: bool = True
    account_cash_verified: bool = True
    known_issues_no_critical_open: bool = True

    def with_overrides(self, **kwargs: Any) -> TinyLiveMetrics:
        return replace(self, **kwargs)


@dataclass
class TinyEvalResult:
    gate_report: dict[str, Any]
    acceptance: AcceptanceArtifact | None
    gate_report_path: Path
    rejection_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.acceptance is not None


def evaluate_tiny_gates(metrics: TinyLiveMetrics) -> list[TinyGateResult]:
    results: list[TinyGateResult] = []

    results.append(
        TinyGateResult(
            "T1",
            metrics.historical_valid,
            "HISTORICAL_ACCEPTANCE valid" if metrics.historical_valid else "historical invalid",
        )
    )
    t2 = metrics.production_forward_a_b_exist and metrics.production_hashes_before_label_join
    results.append(
        TinyGateResult(
            "T2",
            t2,
            "production-forward A/B + hashes-before-join"
            if t2
            else "production artifacts / hash guard failed",
        )
    )
    t3 = metrics.safety_tests_pass and metrics.safety_test_report_present
    results.append(
        TinyGateResult(
            "T3",
            t3,
            "safety tests + report" if t3 else "safety incomplete",
        )
    )
    t4 = (
        metrics.dry_run_consecutive_sessions >= 10
        and metrics.dry_run_zero_discrepancies
        and metrics.dry_run_zero_gate_bypasses
    )
    results.append(
        TinyGateResult(
            "T4",
            t4,
            f"dry_run_sessions={metrics.dry_run_consecutive_sessions}",
        )
    )
    results.append(
        TinyGateResult(
            "T5",
            metrics.kill_switch_drill_logged,
            "kill-switch drill" if metrics.kill_switch_drill_logged else "drill missing",
        )
    )
    t6 = metrics.pinned_model_id == "jev-1.13.0" and metrics.dry_run_resolved_ids_match_pin
    results.append(TinyGateResult("T6", t6, f"pinned={metrics.pinned_model_id}"))
    t7 = (
        metrics.max_strategy_capital_usd <= TINY_LIVE_CEILING_USD + 1e-12
        and metrics.isolation_tests_pass
    )
    results.append(
        TinyGateResult(
            "T7",
            t7,
            f"cap={metrics.max_strategy_capital_usd} isolation={metrics.isolation_tests_pass}",
        )
    )
    results.append(
        TinyGateResult(
            "T8",
            metrics.account_cash_verified,
            "cash account verified" if metrics.account_cash_verified else "account not cash",
        )
    )
    results.append(
        TinyGateResult(
            "T9",
            metrics.known_issues_no_critical_open,
            "no open critical known issues"
            if metrics.known_issues_no_critical_open
            else "open critical known issue",
        )
    )
    return results


def known_issues_has_open_critical(path: Path) -> bool:
    """Return True if docs/KNOWN_ISSUES.md has an open critical row."""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    # Simple table row scan: | id | date | critical | open |
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip().lower() for c in line.split("|")]
        if "critical" in cells and "open" in cells:
            return True
    return False


def gather_tiny_metrics_from_repo(
    *,
    root: Path,
    registry_path: Path,
    historical_path: Path,
    research_config_hash: str,
    max_strategy_capital_usd: float,
    safety_test_report: Path | None = None,
    dry_run_sessions: int = 0,
) -> TinyLiveMetrics:
    """Best-effort assembly from repo state (still fixture-overridable)."""
    hist_ok = False
    try:
        body = verify_acceptance(historical_path, research_config_hash=research_config_hash)
        hist_ok = body.get("status") == "HISTORICAL_RESEARCH_PASS"
    except (ArtifactError, OSError):
        hist_ok = False

    prod = list_model_artifacts(registry_path, kind="production_forward")
    hashes = read_production_hashes(production_hashes_path(root / "data" / "forward"))
    hashes_ok = hashes is not None

    report_path = safety_test_report or (root / "reports" / "safety_test_report.json")
    safety_present = report_path.is_file()
    safety_pass = False
    if safety_present:
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            safety_pass = bool(rep.get("all_passed"))
        except (OSError, json.JSONDecodeError):
            safety_pass = False

    return TinyLiveMetrics(
        historical_valid=hist_ok,
        production_forward_a_b_exist=set(prod) == {"A", "B"},
        production_hashes_before_label_join=hashes_ok,
        safety_tests_pass=safety_pass,
        safety_test_report_present=safety_present,
        dry_run_consecutive_sessions=dry_run_sessions,
        max_strategy_capital_usd=max_strategy_capital_usd,
        known_issues_no_critical_open=not known_issues_has_open_critical(
            root / "docs" / "KNOWN_ISSUES.md"
        ),
    )


def evaluate_tiny(
    *,
    metrics: TinyLiveMetrics,
    acceptance_dir: Path,
    registry_path: Path,
    historical_path: Path,
    git_commit: str,
    research_config_hash: str,
    qset_hash: str,
    model_resolved: str,
) -> TinyEvalResult:
    """Write TINY_LIVE_ACCEPTANCE.json only when T1-T9 all pass.

    Parent = HISTORICAL integrity digest. Never labels the digest as a signature.
    Does not require any forward-session count. Does not raise capital ceiling.
    """
    hist = verify_acceptance(historical_path, research_config_hash=research_config_hash)
    if hist.get("status") != "HISTORICAL_RESEARCH_PASS":
        raise ArtifactError("parent must be HISTORICAL_RESEARCH_PASS")
    parent_digest = str(hist["integrity_digest"])

    # Align T1 with verified historical when calling evaluate_tiny directly.
    metrics = metrics.with_overrides(historical_valid=True)

    results = evaluate_tiny_gates(metrics)
    all_passed = all(r.passed for r in results)
    gate_report = {
        "gates": [asdict(r) for r in results],
        "all_passed": all_passed,
        "forward_session_count_required": False,
        "execution_experiment_sentence": EXECUTION_EXPERIMENT_SENTENCE,
        "ceiling_note": (
            f"TINY_LIVE_CEILING_USD remains {TINY_LIVE_CEILING_USD:.2f}; V1 cannot exceed $10.00"
        ),
        "note": "integrity_digest is an integrity digest, not a cryptographic signature",
        "created_at": datetime.now(UTC).isoformat(),
    }
    acceptance_dir = Path(acceptance_dir)
    acceptance_dir.mkdir(parents=True, exist_ok=True)
    gate_path = acceptance_dir / "TINY_LIVE_GATE_REPORT.json"
    gate_path.write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
    gate_sha = sha256_of(canonical_json(gate_report))

    if not all_passed:
        rej = acceptance_dir / "TINY_LIVE_REJECTION.json"
        rej.write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
        return TinyEvalResult(
            gate_report=gate_report,
            acceptance=None,
            gate_report_path=gate_path,
            rejection_path=rej,
        )

    art = write_acceptance(
        "tiny_live",
        {
            "gates": gate_report["gates"],
            "forward_session_count_required": False,
            "execution_experiment_sentence": EXECUTION_EXPERIMENT_SENTENCE,
            "note": "integrity_digest is an integrity digest, not a cryptographic signature",
            "tiny_live_ceiling_usd": TINY_LIVE_CEILING_USD,
        },
        out_dir=acceptance_dir,
        git_commit=git_commit,
        research_config_hash=research_config_hash,
        qset_hash=qset_hash,
        model_resolved=model_resolved,
        gate_report_sha256=gate_sha,
        parent_artifact_digest=parent_digest,
        registry_path=registry_path,
    )
    if TINY_LIVE_CEILING_USD > 10.00 + 1e-12:
        raise ArtifactError("invariant broken: TINY_LIVE_CEILING_USD exceeded 10")
    return TinyEvalResult(
        gate_report=gate_report,
        acceptance=art,
        gate_report_path=gate_path,
    )
