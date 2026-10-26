"""V2-4 pre-seal reproducibility report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jevlab.experiments.artifacts import compute_integrity_digest
from jevlab.utils.paths import ProjectPaths, get_project_root
from jevlab.v2.candidate import assert_v2_candidate_semantics
from jevlab.v2.dev_io import DEV_YEAR_MAX, v1_canonical_artifact_paths
from jevlab.v2.diagnostics import robustness_spec
from jevlab.v2.equivalence import SANITY_TARGETS, load_dev_inputs_or_none, run_equivalence
from jevlab.v2.hashes import (
    FROZEN_COMMIT,
    FROZEN_TAG_DESIGN,
    FROZEN_TAG_DESIGN_R1,
    FROZEN_TAG_PREREG,
    FROZEN_TAG_PREREG_R1,
    assert_frozen_research_hashes,
    frozen_hash_payload,
)
from jevlab.v2.rehearsal import REHEARSAL_LABEL, run_synthetic_rehearsal
from jevlab.v2.seed_policy import BASE_SEED, child_seed, seed_policy_dict
from jevlab.v2.statistics import FROZEN_N, MIN_SHARPE_ADVANTAGE, P_THRESHOLD
from jevlab.v2.training import EXPECTED_CUTOFF, SEALED_T0, assert_sealed_training_cutoff

PRESEAL_REL = Path("reports") / "v2_preseal"


def build_preseal_payload(*, run_equivalence_wf: bool = True) -> dict[str, Any]:
    hashes = assert_frozen_research_hashes()
    cutoff = assert_sealed_training_cutoff()
    assert_v2_candidate_semantics()
    features = labels = bars = None
    if run_equivalence_wf:
        features, labels, bars = load_dev_inputs_or_none()
    eq = run_equivalence(
        features=features,
        labels=labels,
        bars=bars,
        rerun_walkforward=run_equivalence_wf and features is not None and labels is not None,
    )
    rehearsal = run_synthetic_rehearsal(n_draws=8)
    from sqlalchemy import text

    from jevlab.utils.registry import init_registry

    paths = ProjectPaths.from_root(get_project_root())
    engine = init_registry(paths.registry / "experiments.sqlite")
    with engine.connect() as conn:
        sealed_n = int(conn.execute(text("SELECT COUNT(*) FROM sealed_access_log")).scalar_one())
    acc = paths.reports_acceptance / "V2_SEALED_PASS.json"
    payload = {
        "title": "V2 pre-seal reproducibility report",
        "created_at": datetime.now(UTC).isoformat(),
        "post_preregistration_implementation": True,
        "preserves_frozen_semantics": True,
        "A_frozen_research_hashes": {**frozen_hash_payload(), "live": hashes},
        "B_model_a_equivalence": eq.to_dict(),
        "C_v1_2019_2023_reconstruction": {
            "canonical_paths": {k: str(v) for k, v in v1_canonical_artifact_paths().items()},
            "sanity_targets_rounded": SANITY_TARGETS,
            "canonical_wins": True,
        },
        "D_ledger_fast_path": {
            "engine": "run_fast delegates to run_ledger; EQUITY_TOL_REL=1e-9",
            "checked_in_rehearsal": rehearsal.details.get("ledger_fast_ok"),
            "checked_in_2019_2023_equivalence": eq.checks.get("ledger_fast_equal"),
        },
        "E_random_null_reproducibility": {
            **seed_policy_dict(),
            "child_seed_draw1_phase0": child_seed(1, 0),
            "base_seed": BASE_SEED,
        },
        "F_null_uniformity": "see tests/unit/test_v2_null_engine.py",
        "G_acceptance_evaluator": {
            "p_threshold": P_THRESHOLD,
            "min_sharpe_advantage": MIN_SHARPE_ADVANTAGE,
            "n": FROZEN_N,
        },
        "H_one_shot_guard": "see tests/unit/test_v2_sealed_guard.py",
        "I_second_look": "see tests/unit/test_v2_sealed_guard.py",
        "J_atomic_path": "run_v2_final_test records steps without pause",
        "K_training_cutoff": {
            "t0": str(SEALED_T0),
            "embargo_sessions": 5,
            "final_training_decision_date": str(cutoff),
            "expected": str(EXPECTED_CUTOFF),
            "fit_once": True,
        },
        "L_no_sealed_access": {
            "sealed_access_count": sealed_n,
            "dev_year_max_loaded": DEV_YEAR_MAX,
            "v2_sealed_pass_exists": acc.is_file(),
        },
        "M_pseudo_sealed_rehearsal": rehearsal.to_dict(),
        "N_secret_scan": {"invoked_via_pytest_secret_scan": True},
        "O_known_issues": [
            (
                "Model A 2019-2023 walk-forward equivalence "
                + ("PASSED." if eq.passed else "did not fully pass or was not re-run.")
            ),
            "Rehearsal uses reduced N for software validation; frozen N remains 2000.",
            "V2_SEALED_PASS is not created from rehearsal.",
            "The real `jev v2 final-test` command is implemented but must not be "
            "executed against 2024-2025 in this phase.",
        ],
        "robustness_spec": robustness_spec(),
        "v2_design_tags": [FROZEN_TAG_PREREG, FROZEN_TAG_DESIGN],
        "v2_design_tags_r1": [FROZEN_TAG_PREREG_R1, FROZEN_TAG_DESIGN_R1],
        "v2_design_commit": FROZEN_COMMIT,
        "rehearsal_label": REHEARSAL_LABEL,
        "sealed_execution_authorized": False,
    }
    return payload


def write_preseal_report(*, root: Path | None = None) -> dict[str, Any]:
    root = root or get_project_root()
    payload = build_preseal_payload()
    digest = compute_integrity_digest(payload)
    payload_with = dict(payload)
    payload_with["integrity_digest"] = digest
    out_dir = root / PRESEAL_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "PRESEAL_REPRODUCIBILITY.json"
    md_path = out_dir / "PRESEAL_REPRODUCIBILITY.md"
    json_path.write_text(json.dumps(payload_with, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(payload_with), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "integrity_digest": digest}


def _to_markdown(payload: dict[str, Any]) -> str:
    eq = payload["B_model_a_equivalence"]
    hashes = payload["A_frozen_research_hashes"]
    lines = [
        "# V2 pre-seal reproducibility report",
        "",
        "Post-preregistration implementation preserving frozen semantics.",
        "This report is **not** V2 confirmatory evidence and does **not** authorize sealed access.",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- integrity_digest: `{payload['integrity_digest']}`",
        f"- frozen commit: `{payload['v2_design_commit']}`",
        f"- research_config_hash: `{hashes['research_config_hash']}`",
        "",
        "## A. Frozen research hashes",
        "",
        f"- research_config_hash: `{hashes['research_config_hash']}`",
        f"- candidate: `{hashes['candidate']}`",
        f"- universe: `{hashes['universe']}`",
        f"- feature_set: `{hashes['feature_set']}`",
        f"- target: `{hashes['target']}`",
        f"- strategy: `{hashes['strategy']}`",
        f"- costs: `{hashes['costs']}`",
        f"- random_null: `{hashes['random_null']}`",
        f"- acceptance_gates: `{hashes['acceptance_gates']}`",
        "",
        "Live hashes match freeze (asserted).",
        "",
        "## B. Exact Model A equivalence results",
        "",
        f"Available: `{eq['available']}`  ",
        f"Passed: `{eq['passed']}`  ",
        "Role: implementation equivalence only (not V2 confirmation).",
        "",
        f"Checks: `{eq.get('checks')}`",
        "",
        "## C. V1 2019-2023 reconstruction comparison",
        "",
        f"Canonical paths: `{payload['C_v1_2019_2023_reconstruction']['canonical_paths']}`",
        "",
        "Canonical artifacts win over rounded chat sanity targets.",
        "",
        "## D. Ledger vs fast-path equality",
        "",
        str(payload["D_ledger_fast_path"]),
        "",
        "## E. Random-null deterministic reproducibility",
        "",
        f"N={payload['E_random_null_reproducibility']['n']}; "
        f"base_seed={payload['E_random_null_reproducibility']['base_seed']}; "
        f"generator={payload['E_random_null_reproducibility']['generator']}",
        "",
        "## F. Null uniformity/sanity fixture results",
        "",
        str(payload["F_null_uniformity"]),
        "",
        "## G. Acceptance-evaluator gate tests",
        "",
        str(payload["G_acceptance_evaluator"]),
        "",
        "## H. One-shot guard tests",
        "",
        str(payload["H_one_shot_guard"]),
        "",
        "## I. Second-look invalidation tests",
        "",
        str(payload["I_second_look"]),
        "",
        "## J. Atomic-path test",
        "",
        str(payload["J_atomic_path"]),
        "",
        "## K. Training-cutoff test",
        "",
        f"T0={payload['K_training_cutoff']['t0']}; "
        f"cutoff={payload['K_training_cutoff']['final_training_decision_date']}; "
        f"fit_once={payload['K_training_cutoff']['fit_once']}",
        "",
        "## L. No-sealed-access verification",
        "",
        f"sealed_access_count={payload['L_no_sealed_access']['sealed_access_count']}; "
        f"dev_year_max={payload['L_no_sealed_access']['dev_year_max_loaded']}; "
        f"V2_SEALED_PASS exists={payload['L_no_sealed_access']['v2_sealed_pass_exists']}",
        "",
        "## M. Pseudo-sealed development rehearsal status",
        "",
        f"label=`{payload['rehearsal_label']}`; ok=`{payload['M_pseudo_sealed_rehearsal']['ok']}`",
        "",
        "Labeled `development_rehearsal_only`. Not V2 evidence.",
        "",
        "## N. Secret scan",
        "",
        "Invoked via `pytest -m secret_scan`.",
        "",
        "## O. Remaining known issues",
        "",
    ]
    for item in payload["O_known_issues"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("integrity_digest is an integrity digest, not a cryptographic signature.")
    lines.append("")
    return "\n".join(lines)
