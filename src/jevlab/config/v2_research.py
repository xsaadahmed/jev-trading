"""V2 research-semantic config loader and canonical hashing.

Resolves V2 freeze inputs against committed V1 Model A sources. Does not
access sealed 2024–2025 data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from jevlab.config.loader import config_hash, load_yaml_config
from jevlab.config.models import BacktestConfig, FeaturesConfig, StrategyConfig, UniverseConfig
from jevlab.features.columns import FEATURE_VERSION, MODEL_A_COLUMNS
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.paths import get_project_root

V2_RESEARCH_REL = Path("config") / "v2" / "research.yaml"
V2_MANIFEST_REL = Path("config") / "v2" / "FREEZE_MANIFEST.json"
SEED_CANONICAL_STRING = "v2-baseline-validation-random-null-v1"
EXPECTED_BASE_SEED = int.from_bytes(
    hashlib.sha256(SEED_CANONICAL_STRING.encode("utf-8")).digest()[:8],
    byteorder="big",
)
EXPECTED_TRAINING_CUTOFF = "2023-12-22"
EXPECTED_T0 = "2024-01-02"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2SourceRefs(_Strict):
    universe: str
    features: str
    strategy: str
    backtest: str
    model_a_columns_module: str
    selection_module: str
    sizing_module: str
    labels_module: str
    embargo_module: str
    c_selection_module: str


class V2Candidate(_Strict):
    model_id: str
    model_class: str
    preprocessing: str
    logistic_C_grid: list[float]
    seed: int
    columns_must_equal_v1_MODEL_A_COLUMNS: bool
    jev_inputs: bool


class V2Target(_Strict):
    label_version: str
    definition: str
    R: str
    Rbar: str
    entry_lag_sessions: int
    horizon_sessions: int
    primary_label: str


class V2Eligibility(_Strict):
    rule: str
    sma_window: int


class V2Training(_Strict):
    sealed_score_start: str
    sealed_score_end: str
    t0: str
    embargo_sessions: int
    final_training_decision_date: str
    inner_cv_splits: int
    c_selection: str
    refit_policy: str
    forbid_train_on_sealed_years: bool


class V2StrategyRules(_Strict):
    top_k: int
    weighting: str
    shortage_rule: str
    rebalance_every_sessions: int
    phases: list[int]
    acceptance_phase_agg: str
    fill_convention: str
    fractional_shares: bool
    initial_capital_usd: float


class V2Costs(_Strict):
    primary_scenario: str
    normal_cost_bps_per_side: float
    pessimistic_cost_bps_per_side: float
    sell_fee_bps: float


class V2RandomNull(_Strict):
    name: str
    n: int
    randomizes: str
    retains: list[str]
    seed_canonical_string: str
    base_seed: int
    child_seed_formula: str
    generator: str
    draw_id_indexing: str


class V2Acceptance(_Strict):
    primary_statistic: str
    p_value_formula: str
    p_threshold: float
    min_sharpe_advantage: float
    hard_gates: dict[str, str]
    predictive_metrics_role: str
    robustness_role: str
    context_benchmarks_role: str
    context_benchmarks: list[str]


class V2Seal(_Strict):
    one_shot: bool
    second_look_flag: str
    second_look_effect: str
    require_clean_tree: bool
    require_sealed_access_count_zero_before: bool


class V2Forward(_Strict):
    enabled_at_freeze: bool
    reuse_v1_forward: bool
    clean_forward_requires: list[str]


class V2Docs(_Strict):
    master_spec: str
    preregistration: str
    known_issues: str


class V2ResearchFile(_Strict):
    study_id: str
    study_name: str
    parent_v1_tip: str
    branch: str
    docs: V2Docs
    source_refs: V2SourceRefs
    candidate: V2Candidate
    target: V2Target
    eligibility: V2Eligibility
    training: V2Training
    strategy_rules: V2StrategyRules
    costs: V2Costs
    random_null: V2RandomNull
    acceptance: V2Acceptance
    seal: V2Seal
    statuses: list[str]
    forward: V2Forward
    anti_overfitting: str


@dataclass(frozen=True, slots=True)
class V2ResearchBundle:
    """Resolved V2 research semantics + section hashes + aggregate hash."""

    research: V2ResearchFile
    universe: UniverseConfig
    features: FeaturesConfig
    strategy: StrategyConfig
    backtest: BacktestConfig
    section_payloads: dict[str, Any]
    section_hashes: dict[str, str]
    research_config_hash: str


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else get_project_root()


def load_v2_research_file(path: Path | str | None = None, *, root: Path | None = None) -> V2ResearchFile:
    """Load and validate ``config/v2/research.yaml``."""
    base = _root(root)
    cfg_path = Path(path) if path is not None else base / V2_RESEARCH_REL
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"V2 research config root must be a mapping: {cfg_path}")
    return V2ResearchFile.model_validate(raw)


def _assert_v1_alignment(
    research: V2ResearchFile,
    *,
    universe: UniverseConfig,
    features: FeaturesConfig,
    strategy: StrategyConfig,
    backtest: BacktestConfig,
) -> None:
    """Fail fast if V2 freeze drifts from committed V1 Model A semantics."""
    if research.study_id != "V2":
        raise ValueError(f"study_id must be V2, got {research.study_id}")
    if research.candidate.jev_inputs:
        raise ValueError("V2 candidate must not include Jev inputs")
    if list(features.model_a_columns) != list(MODEL_A_COLUMNS):
        raise ValueError("features.yaml model_a_columns != MODEL_A_COLUMNS")
    if features.feature_version != FEATURE_VERSION:
        raise ValueError(f"feature_version mismatch: {features.feature_version}")
    if research.candidate.columns_must_equal_v1_MODEL_A_COLUMNS:
        if research.candidate.logistic_C_grid != list(strategy.logistic_C_grid):
            raise ValueError("V2 C grid must match config/strategy.yaml")
    if research.candidate.logistic_C_grid != [0.01, 0.1, 1.0]:
        raise ValueError("V2 C grid must be [0.01, 0.1, 1.0]")
    if research.candidate.seed != strategy.seed:
        raise ValueError("V2 seed must match strategy.yaml seed")
    if research.eligibility.rule != strategy.eligibility_rule:
        raise ValueError("eligibility rule mismatch vs strategy.yaml")
    if research.eligibility.sma_window != features.eligibility_sma_window:
        raise ValueError("eligibility SMA window mismatch")
    if research.strategy_rules.top_k != strategy.top_k:
        raise ValueError("top_k mismatch")
    if research.strategy_rules.phases != list(strategy.phases):
        raise ValueError("phases mismatch")
    if research.strategy_rules.rebalance_every_sessions != strategy.rebalance_every_sessions:
        raise ValueError("rebalance_every_sessions mismatch")
    if research.target.entry_lag_sessions != strategy.entry_lag_sessions:
        raise ValueError("entry_lag_sessions mismatch")
    if research.target.horizon_sessions != strategy.horizon_sessions:
        raise ValueError("horizon_sessions mismatch")
    if research.strategy_rules.fill_convention != backtest.fill_convention:
        raise ValueError("fill_convention mismatch")
    if research.strategy_rules.initial_capital_usd != backtest.initial_capital_usd:
        raise ValueError("initial_capital_usd mismatch")
    if research.strategy_rules.fractional_shares != backtest.fractional_shares:
        raise ValueError("fractional_shares mismatch")
    if research.costs.sell_fee_bps != backtest.sell_fee_bps:
        raise ValueError("sell_fee_bps mismatch")
    by_name = {s.name: s.cost_bps_per_side for s in backtest.cost_scenarios}
    if by_name.get("normal") != research.costs.normal_cost_bps_per_side:
        raise ValueError("normal cost mismatch vs backtest.yaml")
    if by_name.get("pessimistic") != research.costs.pessimistic_cost_bps_per_side:
        raise ValueError("pessimistic cost mismatch vs backtest.yaml")
    if len(universe.symbols) != 13:
        raise ValueError("universe must have 13 symbols")
    if research.training.t0 != EXPECTED_T0:
        raise ValueError(f"T0 must be {EXPECTED_T0}")
    if research.training.final_training_decision_date != EXPECTED_TRAINING_CUTOFF:
        raise ValueError(f"training cutoff must be {EXPECTED_TRAINING_CUTOFF}")
    if research.training.embargo_sessions != 5:
        raise ValueError("embargo_sessions must be 5")
    if research.random_null.n != 2000:
        raise ValueError("random_null.n must be 2000")
    if research.random_null.seed_canonical_string != SEED_CANONICAL_STRING:
        raise ValueError("seed canonical string mismatch")
    if research.random_null.base_seed != EXPECTED_BASE_SEED:
        raise ValueError(
            f"base_seed mismatch: config={research.random_null.base_seed} "
            f"expected={EXPECTED_BASE_SEED}"
        )
    if research.acceptance.p_threshold != 0.05:
        raise ValueError("p_threshold must be 0.05")
    if research.acceptance.min_sharpe_advantage != 0.10:
        raise ValueError("min_sharpe_advantage must be 0.10")
    required_gates = {"R1", "E1", "E2", "E3", "E4", "E5", "PS1", "C1", "Z1"}
    if set(research.acceptance.hard_gates) != required_gates:
        raise ValueError(f"hard_gates must be exactly {sorted(required_gates)}")
    if research.forward.enabled_at_freeze:
        raise ValueError("forward must be disabled at freeze")
    if research.forward.reuse_v1_forward:
        raise ValueError("must not reuse V1 forward")
    anti = research.anti_overfitting.lower()
    for required in ("acceptance gate", "null construction", "2024", "new preregistered study"):
        if required not in anti:
            raise ValueError(f"anti_overfitting text missing required phrase: {required!r}")


def build_section_payloads(
    research: V2ResearchFile,
    *,
    universe: UniverseConfig,
    features: FeaturesConfig,
    strategy: StrategyConfig,
    backtest: BacktestConfig,
) -> dict[str, Any]:
    """Canonical research-semantic payloads included in the V2 research hash."""
    return {
        "candidate": {
            "model_id": research.candidate.model_id,
            "model_class": research.candidate.model_class,
            "preprocessing": research.candidate.preprocessing,
            "logistic_C_grid": list(research.candidate.logistic_C_grid),
            "seed": research.candidate.seed,
            "columns": list(MODEL_A_COLUMNS),
            "feature_version": FEATURE_VERSION,
            "jev_inputs": research.candidate.jev_inputs,
        },
        "universe": {
            "universe_version": universe.universe_version,
            "symbols": list(universe.symbols),
        },
        "feature_set": {
            "feature_version": features.feature_version,
            "model_a_columns": list(features.model_a_columns),
            "eligibility_sma_window": features.eligibility_sma_window,
        },
        "target": research.target.model_dump(mode="json"),
        "eligibility": research.eligibility.model_dump(mode="json"),
        "training": research.training.model_dump(mode="json"),
        "strategy": {
            **research.strategy_rules.model_dump(mode="json"),
            "strategy_id": strategy.strategy_id,
            "eligibility_rule": strategy.eligibility_rule,
            "logistic_C_grid": list(strategy.logistic_C_grid),
            "seed": strategy.seed,
            "entry_lag_sessions": strategy.entry_lag_sessions,
            "horizon_sessions": strategy.horizon_sessions,
        },
        "costs": {
            **research.costs.model_dump(mode="json"),
            "cost_scenarios": [s.model_dump(mode="json") for s in backtest.cost_scenarios],
            "per_symbol_cost_multiplier": dict(backtest.per_symbol_cost_multiplier),
            "commission_bps": backtest.commission_bps,
            "fill_convention": backtest.fill_convention,
        },
        "random_null": research.random_null.model_dump(mode="json"),
        "acceptance_gates": research.acceptance.model_dump(mode="json"),
        "seal": research.seal.model_dump(mode="json"),
        "statuses": list(research.statuses),
        "forward_policy": research.forward.model_dump(mode="json"),
        "anti_overfitting": research.anti_overfitting.strip(),
        "source_refs": research.source_refs.model_dump(mode="json"),
        "parent_v1_tip": research.parent_v1_tip,
    }


def load_v2_research_bundle(*, root: Path | None = None) -> V2ResearchBundle:
    """Load V2 research.yaml, align to V1 sources, and compute hashes."""
    base = _root(root)
    research = load_v2_research_file(root=base)
    universe = load_yaml_config(base / research.source_refs.universe, UniverseConfig)
    features = load_yaml_config(base / research.source_refs.features, FeaturesConfig)
    strategy = load_yaml_config(base / research.source_refs.strategy, StrategyConfig)
    backtest = load_yaml_config(base / research.source_refs.backtest, BacktestConfig)
    _assert_v1_alignment(
        research,
        universe=universe,
        features=features,
        strategy=strategy,
        backtest=backtest,
    )
    payloads = build_section_payloads(
        research,
        universe=universe,
        features=features,
        strategy=strategy,
        backtest=backtest,
    )
    section_hashes = {name: sha256_of(canonical_json(payload)) for name, payload in payloads.items()}
    # Aggregate over the research-semantic sections required by the freeze prompt.
    aggregate_keys = (
        "universe",
        "feature_set",
        "target",
        "eligibility",
        "candidate",
        "training",
        "strategy",
        "costs",
        "random_null",
        "acceptance_gates",
    )
    research_config_hash = sha256_of(
        canonical_json({k: section_hashes[k] for k in aggregate_keys})
    )
    return V2ResearchBundle(
        research=research,
        universe=universe,
        features=features,
        strategy=strategy,
        backtest=backtest,
        section_payloads=payloads,
        section_hashes=section_hashes,
        research_config_hash=research_config_hash,
    )


def build_freeze_manifest(bundle: V2ResearchBundle) -> dict[str, Any]:
    """JSON-serializable freeze manifest (hashes + locked constants)."""
    r = bundle.research
    return {
        "study_id": r.study_id,
        "study_name": r.study_name,
        "status": "RESEARCH_DESIGN_FROZEN",
        "parent_v1_tip": r.parent_v1_tip,
        "branch": r.branch,
        "docs": r.docs.model_dump(mode="json"),
        "research_config_hash": bundle.research_config_hash,
        "section_hashes": {
            "candidate": bundle.section_hashes["candidate"],
            "universe": bundle.section_hashes["universe"],
            "feature_set": bundle.section_hashes["feature_set"],
            "target": bundle.section_hashes["target"],
            "eligibility": bundle.section_hashes["eligibility"],
            "training": bundle.section_hashes["training"],
            "strategy": bundle.section_hashes["strategy"],
            "costs": bundle.section_hashes["costs"],
            "random_null": bundle.section_hashes["random_null"],
            "acceptance_gates": bundle.section_hashes["acceptance_gates"],
        },
        "v1_config_hashes": {
            "universe": config_hash(bundle.universe),
            "features": config_hash(bundle.features),
            "strategy": config_hash(bundle.strategy),
            "backtest": config_hash(bundle.backtest),
        },
        "locked": {
            "model_a_columns": list(MODEL_A_COLUMNS),
            "feature_version": FEATURE_VERSION,
            "t0": r.training.t0,
            "final_training_decision_date": r.training.final_training_decision_date,
            "embargo_sessions": r.training.embargo_sessions,
            "refit_policy": r.training.refit_policy,
            "random_null_n": r.random_null.n,
            "seed_canonical_string": r.random_null.seed_canonical_string,
            "base_seed": r.random_null.base_seed,
            "child_seed_formula": r.random_null.child_seed_formula,
            "primary_statistic": r.acceptance.primary_statistic,
            "p_threshold": r.acceptance.p_threshold,
            "min_sharpe_advantage": r.acceptance.min_sharpe_advantage,
            "hard_gates": r.acceptance.hard_gates,
        },
        "forward_enabled_at_freeze": r.forward.enabled_at_freeze,
        "sealed_execution_authorized_by_this_freeze": False,
    }


def write_freeze_manifest(path: Path | str | None = None, *, root: Path | None = None) -> dict[str, Any]:
    """Compute and write ``config/v2/FREEZE_MANIFEST.json``."""
    import json

    base = _root(root)
    bundle = load_v2_research_bundle(root=base)
    manifest = build_freeze_manifest(bundle)
    out = Path(path) if path is not None else base / V2_MANIFEST_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
