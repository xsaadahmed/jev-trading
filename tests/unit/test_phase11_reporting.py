"""Phase 11: manifest-only report builder, verdicts, mandatory statements."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevlab.reporting.build_report import build_report
from jevlab.reporting.conclusion import assert_conclusion_safety, generate_conclusion
from jevlab.reporting.statements import (
    CONCLUSION_NO_VALUE,
    CONCLUSION_PASS,
    CONCLUSION_UNDERPOWERED,
    HISTORICAL_CONTAMINATION,
    LIVE_EXECUTION_EXPERIMENT,
    WARNING_SIGNAL_NOT_PROOF,
    assert_mandatory_present,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "phase11"


def test_mandatory_statements_constant() -> None:
    assert "potentially training-contaminated" in HISTORICAL_CONTAMINATION
    assert "execution experiment" in LIVE_EXECUTION_EXPERIMENT
    assert "not evidence of a profitable trading edge" in LIVE_EXECUTION_EXPERIMENT


def test_verdict_no_value() -> None:
    c = generate_conclusion(gates={"G4": False, "G5": True})
    assert c.kind == "no_value"
    assert c.text == CONCLUSION_NO_VALUE
    assert_conclusion_safety(c.text, kind=c.kind)


def test_verdict_underpowered() -> None:
    c = generate_conclusion(gates={"G4": True}, underpowered=True)
    assert c.kind == "underpowered"
    assert c.text == CONCLUSION_UNDERPOWERED
    assert_conclusion_safety(c.text, kind=c.kind)


def test_verdict_underpowered_via_mde() -> None:
    c = generate_conclusion(
        gates={"G4": True, "G5": True},
        power={"observed_effect": 0.05, "mde": 0.25},
    )
    assert c.kind == "underpowered"
    assert c.text == CONCLUSION_UNDERPOWERED


def test_verdict_pass() -> None:
    gates = {f"G{i}": True for i in range(1, 11)}
    c = generate_conclusion(gates=gates, underpowered=False)
    assert c.kind == "pass"
    assert c.text == CONCLUSION_PASS
    assert "HISTORICAL_RESEARCH_PASS" in c.text
    assert "does not establish" in c.text.lower() or "contamination" in c.text.lower()
    # Must not claim $10 live is evidence of edge
    assert "evidence of a profitable trading edge" not in c.text or "not evidence" in c.text
    assert_conclusion_safety(c.text, kind=c.kind)


def test_pass_forbidden_when_any_gate_fails() -> None:
    c = generate_conclusion(gates={"G1": True, "G4": False})
    assert c.kind != "pass"
    assert "passed" not in c.text.lower() or c.text == CONCLUSION_NO_VALUE
    assert c.text == CONCLUSION_NO_VALUE


def test_unevaluable_gate_is_not_pass() -> None:
    c = generate_conclusion(gates={"G4": True, "G10": None})
    assert c.kind == "underpowered"
    assert c.text == CONCLUSION_UNDERPOWERED


def test_golden_render_full_manifest(tmp_path: Path) -> None:
    manifest = FIXTURES / "phase11_manifest_full.json"
    report = build_report(manifest, out_dir=tmp_path / "phase11_golden")
    assert (tmp_path / "phase11_golden" / "report.html").is_file()
    assert (tmp_path / "phase11_golden" / "report.md").is_file()
    md = report.markdown
    html = report.html
    for required in (HISTORICAL_CONTAMINATION, LIVE_EXECUTION_EXPERIMENT):
        assert required in md
        assert required in html
    assert WARNING_SIGNAL_NOT_PROOF in md
    assert report.conclusion.kind == "pass"
    assert report.conclusion.text == CONCLUSION_PASS
    # Section titles present
    for title in (
        "Strategy definition",
        "Dataset and splits",
        "Feature definitions",
        "Jev question definitions",
        "Equity and drawdown",
        "Monthly return",
        "Trade distribution",
        "Baseline comparison",
        "Jev comparison",
        "Calibration",
        "Placebo",
        "Cost sensitivity",
        "Robustness",
        "Statistical uncertainty",
        "Contamination-diagnostics",
        "Three-status table",
        "Final conclusion",
    ):
        assert title.lower() in md.lower()

    # Golden snapshot (markdown body without volatile paths)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "report_excerpt.md"
    excerpt = "\n".join(
        [
            report.conclusion.text,
            HISTORICAL_CONTAMINATION,
            LIVE_EXECUTION_EXPERIMENT,
            f"verdict={report.conclusion.kind}",
            f"experiment_id={report.experiment_id}",
        ]
    )
    if not golden_path.is_file():
        golden_path.write_text(excerpt + "\n", encoding="utf-8")
    expected = golden_path.read_text(encoding="utf-8").strip()
    assert excerpt.strip() == expected


def test_report_builds_with_missing_optional_sections(tmp_path: Path) -> None:
    manifest = FIXTURES / "phase11_manifest_minimal.json"
    report = build_report(manifest, out_dir=tmp_path / "minimal")
    assert report.conclusion.text == CONCLUSION_NO_VALUE
    assert HISTORICAL_CONTAMINATION in report.markdown
    assert LIVE_EXECUTION_EXPERIMENT in report.markdown
    # Optional sections marked missing but report still built
    assert "not present in manifest" in report.markdown.lower()
    missing = assert_mandatory_present(report.markdown)
    assert missing == []


def test_force_verdict_paths() -> None:
    assert generate_conclusion(force_kind="no_value").text == CONCLUSION_NO_VALUE
    assert generate_conclusion(force_kind="underpowered").text == CONCLUSION_UNDERPOWERED
    assert generate_conclusion(force_kind="pass").text == CONCLUSION_PASS


def test_assert_conclusion_safety_rejects_pass_language_on_failure() -> None:
    with pytest.raises(AssertionError):
        assert_conclusion_safety("All gates passed; HISTORICAL_RESEARCH_PASS", kind="no_value")
