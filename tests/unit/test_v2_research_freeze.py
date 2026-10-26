"""V2 research-design freeze consistency (no sealed data access)."""

from __future__ import annotations

import json

from jevlab.config.v2_research import (
    CHILD_SEED_FORMULA_R1,
    EXPECTED_BASE_SEED,
    EXPECTED_TRAINING_CUTOFF,
    PROTOCOL_REVISION,
    SEED_CANONICAL_STRING,
    SEED_NAMESPACE_R1,
    V2_MANIFEST_REL,
    build_freeze_manifest,
    load_v2_research_bundle,
)
from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.utils.paths import get_project_root


def test_v2_research_bundle_aligns_to_v1_model_a() -> None:
    bundle = load_v2_research_bundle()
    r = bundle.research
    assert r.study_id == "V2"
    assert r.candidate.jev_inputs is False
    assert list(bundle.features.model_a_columns) == list(MODEL_A_COLUMNS)
    assert len(MODEL_A_COLUMNS) == 17
    assert list(bundle.universe.symbols) == [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
    ]
    assert r.eligibility.rule == "adj_close > sma200"
    assert r.candidate.logistic_C_grid == [0.01, 0.1, 1.0]
    assert r.strategy_rules.top_k == 3
    assert r.strategy_rules.phases == [0, 1, 2]
    assert r.training.final_training_decision_date == EXPECTED_TRAINING_CUTOFF
    assert r.training.embargo_sessions == 5
    assert r.training.refit_policy == "fit_once_pre_sealed_no_quarterly_refit"
    assert r.protocol_revision == PROTOCOL_REVISION == 1
    assert r.random_null.n == 2000
    assert r.random_null.seed_canonical_string == SEED_CANONICAL_STRING
    assert r.random_null.seed_namespace == SEED_NAMESPACE_R1
    assert r.random_null.draw_id_indexing == "zero_based"
    assert r.random_null.child_seed_formula == CHILD_SEED_FORMULA_R1
    assert r.random_null.base_seed == EXPECTED_BASE_SEED
    assert r.acceptance.p_threshold == 0.05
    assert r.acceptance.min_sharpe_advantage == 0.10
    assert set(r.acceptance.hard_gates) == {
        "R1",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "PS1",
        "C1",
        "Z1",
    }
    assert r.forward.enabled_at_freeze is False
    assert r.seal.one_shot is True
    assert "null construction" in r.anti_overfitting


def test_v2_research_hash_stable_and_manifest_matches() -> None:
    root = get_project_root()
    bundle = load_v2_research_bundle(root=root)
    manifest_path = root / V2_MANIFEST_REL
    assert manifest_path.is_file(), "FREEZE_MANIFEST.json missing; run write_freeze_manifest"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    rebuilt = build_freeze_manifest(bundle)
    assert on_disk["research_config_hash"] == bundle.research_config_hash
    assert on_disk["section_hashes"] == rebuilt["section_hashes"]
    assert on_disk["locked"]["base_seed"] == EXPECTED_BASE_SEED
    assert on_disk["sealed_execution_authorized_by_this_freeze"] is False
    # Recompute twice - deterministic.
    assert load_v2_research_bundle(root=root).research_config_hash == bundle.research_config_hash


def test_v2_docs_exist() -> None:
    root = get_project_root()
    for rel in (
        "docs/V2_MASTER_SPEC.md",
        "docs/V2_PREREGISTRATION.md",
        "docs/V2_KNOWN_ISSUES.md",
        "docs/V2_AMENDMENT_001_SEED_STREAMS.md",
        "config/v2/research.yaml",
    ):
        assert (root / rel).is_file()


def test_no_v2_acceptance_or_result_artifacts() -> None:
    root = get_project_root()
    acc = root / "reports" / "acceptance"
    if acc.is_dir():
        bad = [
            p
            for p in acc.rglob("*")
            if p.is_file() and "V2" in p.name.upper() and p.name != ".gitkeep"
        ]
        assert not bad, f"unexpected V2 acceptance artifacts: {bad}"
