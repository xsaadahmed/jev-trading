"""One-shot V2 sealed evaluation guard and atomic path.

The real 2024-2025 load happens only after every precondition passes, via an
injected ``SealedSource``. Tests inject fakes. The CLI must not be executed
for the real seal in V2-1...V2-4.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV, SealedAccessError, SealedDataGuard
from jevlab.experiments.acceptance import count_sealed_accesses
from jevlab.jev.preconditions import PreconditionError, git_tag_exists, git_working_tree_clean
from jevlab.targets.labels import build_labels
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.v2.acceptance import V2GateReport
from jevlab.v2.artifacts import atomic_write_json, invalidate_v2_sealed_pass, write_v2_sealed_pass
from jevlab.v2.candidate import V2_UNIVERSE
from jevlab.v2.causal import (
    build_causal_score_features,
    combine_history_and_sealed_bars,
    combine_history_and_sealed_vix,
    drop_unevaluable_end_window,
)
from jevlab.v2.dev_io import SEALED_MAX_DATE
from jevlab.v2.evaluate import (
    InjectedCrash,
    json_safe,
    payload_t0,
    score_sealed_study,
)
from jevlab.v2.hashes import (
    FROZEN_COMMIT,
    FROZEN_IMPLEMENTATION_REVISION,
    FROZEN_MODEL_LOGICAL_DIGEST,
    FROZEN_PRESEAL_IMPL_TAG,
    FROZEN_PRESEAL_IMPL_TAG_R1,
    FROZEN_PRESEAL_IMPL_TAG_R2,
    FROZEN_PRESEAL_IMPL_TAG_R3,
    FROZEN_TAG_DESIGN,
    FROZEN_TAG_DESIGN_R1,
    FROZEN_TAG_PREREG,
    FROZEN_TAG_PREREG_R1,
    FrozenHashError,
    assert_frozen_research_hashes,
)
from jevlab.v2.model_artifact import (
    SealedTestModelArtifact,
    assert_artifact_matches_freeze,
    load_sealed_test_model,
)
from jevlab.v2.payload import (
    EmptySealedScoringError,
    SealedDataFlow,
    SealedPayloadError,
    assert_sealed_scoring_nonempty,
    classify_production_sealed_payload,
    strip_audit_columns,
)
from jevlab.v2.rehearsal import REHEARSAL_LABEL
from jevlab.v2.sealed_source import PreSealedTrainingSource, ProductionSealedSource
from jevlab.v2.statistics import FROZEN_N
from jevlab.v2.training import SEALED_T0, FittedModelA, TrainingLeakError, logical_model_digest

SEAL_BROKEN_STAMP = "SEAL BROKEN"
SEAL_STATE_FILENAME = "v2_seal_state.json"
V2_SEAL_REASON = "v2-final-test-sealed-score"
VOID_AUTHORIZATION_PHRASES: frozenset[str] = frozenset(
    {
        "OPEN V2 REVISION 1 SEALED HOLDOUT ONCE",
        "OPEN V2 REVISION 1 SEALED HOLDOUT ONCE WITH IMPLEMENTATION R2",
        "OPEN V2 R1 IMPLEMENTATION R2 SEALED HOLDOUT ONCE",
        "OPEN V2 R1 IMPLEMENTATION R3 SEALED HOLDOUT ONCE",
    }
)
REQUIRED_AUTHORIZATION_PHRASE = "OPEN V2 R1 IMPLEMENTATION R4 SEALED HOLDOUT ONCE"
ATOMIC_STEPS: tuple[str, ...] = (
    "validate_preconditions",
    "append_sealed_access_record",
    "read_sealed_or_fixture",
    "classify_sealed_payload",
    "merge_historical_context",
    "causal_feature_build",
    "construct_features_labels",
    "load_prebuilt_model",
    "score_entire_period",
    "candidate_backtest_phase_0",
    "candidate_backtest_phase_1",
    "candidate_backtest_phase_2",
    "candidate_backtest",
    "run_random_nulls",
    "calculate_hard_gates",
    "calculate_predictive_diagnostics",
    "calculate_robustness_diagnostics",
    "generate_result_report",
    "conditionally_write_acceptance",
    "close_run",
)


class SealedSource(Protocol):
    def load(self) -> dict[str, Any]:
        """Return sealed features/labels/bars. Tests provide fakes."""
        ...


def _maybe_crash(fail_after: str | None, step: str) -> None:
    if fail_after is not None and fail_after == step:
        raise InjectedCrash(f"injected crash after {step}")


def assert_fresh_r4_authorization(
    authorization: str,
    *,
    implementation_tag: str = FROZEN_PRESEAL_IMPL_TAG,
) -> None:
    """Reject closed r1/r2/r3 phrases; require the exact r4 phrase for r4 only."""
    phrase = (authorization or "").strip()
    if phrase in VOID_AUTHORIZATION_PHRASES:
        raise PreconditionError(
            "prior human authorization is VOID/CLOSED; r1, r2, and r3 never consumed "
            "the seal and cannot be reused"
        )
    if phrase != REQUIRED_AUTHORIZATION_PHRASE:
        raise PreconditionError(
            "fresh explicit human authorization is required after "
            f"{FROZEN_PRESEAL_IMPL_TAG}; prior r1/r2/r3 authorizations are void"
        )
    if implementation_tag != FROZEN_PRESEAL_IMPL_TAG:
        raise PreconditionError(
            "r4 authorization phrase cannot authorize any implementation other "
            f"than {FROZEN_PRESEAL_IMPL_TAG}"
        )


def assert_fresh_r3_authorization(
    authorization: str,
    *,
    implementation_tag: str = FROZEN_PRESEAL_IMPL_TAG,
) -> None:
    """r3 authorization is VOID/CLOSED."""
    assert_fresh_r4_authorization(authorization, implementation_tag=implementation_tag)


def assert_fresh_r2_authorization(authorization: str) -> None:
    """r2 authorization is VOID/CLOSED."""
    assert_fresh_r4_authorization(authorization)


@dataclass
class AtomicStepLog:
    steps: list[str] = field(default_factory=list)
    paused: bool = False

    def record(self, name: str) -> None:
        if self.paused:
            raise RuntimeError("illegal pause in atomic sealed path")
        self.steps.append(name)


def _git(args: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def peel_tag(tag: str, *, cwd: Path) -> str:
    return _git(["rev-list", "-n", "1", tag], cwd=cwd)


def is_ancestor(commit: str, *, cwd: Path) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def v2_result_exists(
    root: Path,
    *,
    acceptance_dir: Path | None = None,
    seal_dir: Path | None = None,
) -> bool:
    acc = (acceptance_dir or (root / "reports" / "acceptance")) / "V2_SEALED_PASS.json"
    result = (seal_dir or (root / "reports" / "v2_sealed")) / "SEALED_RESULT.json"
    return acc.is_file() or result.is_file()


def read_seal_state(dir_path: Path) -> dict[str, Any]:
    path = dir_path / SEAL_STATE_FILENAME
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded)


def write_seal_state(dir_path: Path, state: dict[str, Any]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / SEAL_STATE_FILENAME).write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def assert_v2_final_test_preconditions(
    *,
    root: Path,
    registry_path: Path,
    seal_token: str,
    require_clean_tree: bool = True,
    require_git_tags: bool = True,
    require_preseal_tag: bool = True,
    require_hashes: bool = True,
    require_seal_token: bool = True,
    acceptance_dir: Path | None = None,
    seal_dir: Path | None = None,
) -> dict[str, str]:
    """Fail BEFORE any sealed read if a freeze/integrity precondition is unmet."""
    checks: dict[str, str] = {}
    if require_clean_tree and not git_working_tree_clean(cwd=root):
        raise PreconditionError("working tree is not clean")
    checks["working_tree"] = "clean"
    if require_git_tags:
        if not git_tag_exists(FROZEN_TAG_PREREG, cwd=root):
            raise PreconditionError(f"missing provenance tag {FROZEN_TAG_PREREG}")
        if not git_tag_exists(FROZEN_TAG_DESIGN, cwd=root):
            raise PreconditionError(f"missing provenance tag {FROZEN_TAG_DESIGN}")
        prereg = peel_tag(FROZEN_TAG_PREREG, cwd=root)
        design = peel_tag(FROZEN_TAG_DESIGN, cwd=root)
        if prereg != FROZEN_COMMIT or design != FROZEN_COMMIT:
            raise PreconditionError(
                f"original V2 design tags must peel to {FROZEN_COMMIT}; "
                f"got prereg={prereg} design={design}"
            )
        if prereg != design:
            raise PreconditionError("prereg-v2-baseline and v2-design-frozen diverge")
        checks["tags_original"] = FROZEN_COMMIT
        if not git_tag_exists(FROZEN_TAG_PREREG_R1, cwd=root):
            raise PreconditionError(
                f"missing execution tag {FROZEN_TAG_PREREG_R1} "
                "(original V2 design tags are provenance only)"
            )
        if not git_tag_exists(FROZEN_TAG_DESIGN_R1, cwd=root):
            raise PreconditionError(
                f"missing execution tag {FROZEN_TAG_DESIGN_R1} "
                "(original V2 design tags are provenance only)"
            )
        prereg_r1 = peel_tag(FROZEN_TAG_PREREG_R1, cwd=root)
        design_r1 = peel_tag(FROZEN_TAG_DESIGN_R1, cwd=root)
        if not prereg_r1 or not design_r1:
            raise PreconditionError("revision-1 design tags could not be peeled")
        if prereg_r1 != design_r1:
            raise PreconditionError("prereg-v2-baseline-r1 and v2-design-frozen-r1 diverge")
        checks["tags_r1"] = prereg_r1
        if not is_ancestor(prereg_r1, cwd=root):
            raise PreconditionError("HEAD is not a descendant of the revision-1 design freeze")
        checks["descendant_r1"] = "yes"
        if FROZEN_PRESEAL_IMPL_TAG in {
            FROZEN_PRESEAL_IMPL_TAG_R1,
            FROZEN_PRESEAL_IMPL_TAG_R2,
            FROZEN_PRESEAL_IMPL_TAG_R3,
        }:
            raise PreconditionError(
                "v2-preseal-implementation-r1/r2/r3 are audit artifacts and are not "
                "execution-capable after implementation r4"
            )
        if require_preseal_tag:
            if not git_tag_exists(FROZEN_PRESEAL_IMPL_TAG_R1, cwd=root):
                raise PreconditionError(f"missing audit tag {FROZEN_PRESEAL_IMPL_TAG_R1}")
            if not git_tag_exists(FROZEN_PRESEAL_IMPL_TAG_R2, cwd=root):
                raise PreconditionError(f"missing audit tag {FROZEN_PRESEAL_IMPL_TAG_R2}")
            if not git_tag_exists(FROZEN_PRESEAL_IMPL_TAG_R3, cwd=root):
                raise PreconditionError(f"missing audit tag {FROZEN_PRESEAL_IMPL_TAG_R3}")
            if not git_tag_exists(FROZEN_PRESEAL_IMPL_TAG, cwd=root):
                raise PreconditionError(
                    f"missing {FROZEN_PRESEAL_IMPL_TAG} (required for the real sealed run); "
                    f"{FROZEN_PRESEAL_IMPL_TAG_R1}, {FROZEN_PRESEAL_IMPL_TAG_R2}, and "
                    f"{FROZEN_PRESEAL_IMPL_TAG_R3} remain immutable audit artifacts"
                )
            r1_impl = peel_tag(FROZEN_PRESEAL_IMPL_TAG_R1, cwd=root)
            r2_impl = peel_tag(FROZEN_PRESEAL_IMPL_TAG_R2, cwd=root)
            r3_impl = peel_tag(FROZEN_PRESEAL_IMPL_TAG_R3, cwd=root)
            r4_impl = peel_tag(FROZEN_PRESEAL_IMPL_TAG, cwd=root)
            if not r4_impl:
                raise PreconditionError("r4 implementation tag could not be peeled")
            if r4_impl in {r1_impl, r2_impl, r3_impl}:
                raise PreconditionError("r4 tag must not move r1, r2, or r3")
            checks["preseal_tag_r1_audit"] = r1_impl
            checks["preseal_tag_r2_audit"] = r2_impl
            checks["preseal_tag_r3_audit"] = r3_impl
            checks["preseal_tag"] = r4_impl
            checks["implementation_revision"] = str(FROZEN_IMPLEMENTATION_REVISION)
        else:
            checks["preseal_tag"] = "skipped"
        checks["preseal_tag_r1_not_execution"] = FROZEN_PRESEAL_IMPL_TAG_R1
        checks["preseal_tag_r2_not_execution"] = FROZEN_PRESEAL_IMPL_TAG_R2
        checks["preseal_tag_r3_not_execution"] = FROZEN_PRESEAL_IMPL_TAG_R3
    if require_hashes:
        try:
            assert_frozen_research_hashes(root=root)
        except FrozenHashError as exc:
            raise PreconditionError(str(exc)) from exc
        checks["hashes"] = "match"
    n_access = count_sealed_accesses(registry_path, reason_prefix="")
    # count_sealed_accesses filters by reason prefix; also count raw table.
    from sqlalchemy import text

    from jevlab.utils.registry import init_registry

    engine = init_registry(registry_path)
    with engine.connect() as conn:
        raw = conn.execute(text("SELECT COUNT(*) FROM sealed_access_log")).scalar_one()
    if int(raw) != 0:
        raise PreconditionError(f"sealed access count is {raw}, required 0 before first run")
    _ = n_access
    checks["sealed_access_count"] = "0"
    if v2_result_exists(root, acceptance_dir=acceptance_dir, seal_dir=seal_dir):
        raise PreconditionError("prior V2 sealed result or acceptance artifact exists")
    checks["no_prior_v2_result"] = "ok"
    token_present = bool(os.environ.get(SEAL_TOKEN_ENV, "").strip())
    checks["seal_token_present"] = "yes" if token_present else "no"
    if require_seal_token:
        expected = os.environ.get(SEAL_TOKEN_ENV, "").strip()
        if not expected or not seal_token or seal_token.strip() != expected:
            raise SealedAccessError("sealed access denied: invalid or missing SEAL token")
        checks["seal_token"] = "ok"
    else:
        checks["seal_token"] = "skipped_preflight"
    return checks


@dataclass
class V2FinalTestResult:
    gate_report: V2GateReport
    acceptance_written: bool
    seal_broken: bool
    preconditions: dict[str, str]
    step_log: list[str]
    acceptance_path: Path | None = None
    result_path: Path | None = None
    data_flow: dict[str, Any] = field(default_factory=dict)


def run_v2_preflight(
    *,
    root: Path,
    registry_path: Path,
    configured_sealed_path: Path,
    require_clean_tree: bool = True,
    require_git_tags: bool = True,
    require_preseal_tag: bool = True,
    require_hashes: bool = True,
    training_source: PreSealedTrainingSource | None = None,
    model_artifact_path: Path | None = None,
    require_prebuilt_model: bool = False,
    model_freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Terminate before SEAL authorization and before any sealed read.

    May read pre-2024 training/context and the frozen model artifact.
    Must not read 2024-2025 sealed contents.
    """
    source = ProductionSealedSource(configured_sealed_path)
    checks = assert_v2_final_test_preconditions(
        root=root,
        registry_path=registry_path,
        seal_token="",
        require_clean_tree=require_clean_tree,
        require_git_tags=require_git_tags,
        require_preseal_tag=require_preseal_tag,
        require_hashes=require_hashes,
        require_seal_token=False,
    )
    training_reads = 0
    if training_source is not None:
        training_source.load()
        training_reads = int(training_source.training_context_reads)
        checks["training_context_reads"] = str(training_reads)
        checks["training_sealed_underlying_reads"] = str(training_source.sealed_underlying_reads)
        if training_source.sealed_underlying_reads != 0:
            raise AssertionError("preflight read sealed underlying via training source")
    if require_prebuilt_model:
        if model_artifact_path is None:
            raise PreconditionError("prebuilt sealed-test-model artifact is required")
        artifact = load_sealed_test_model(model_artifact_path)
        if model_freeze:
            assert_artifact_matches_freeze(artifact, expected=model_freeze)
        checks["prebuilt_model"] = artifact.integrity_digest
        checks["chosen_C"] = str(artifact.chosen_C)
        checks["model_logical_digest"] = str(artifact.manifest["fitted_model_digest"])
    exists = configured_sealed_path.exists()
    checks["configured_sealed_path"] = source.configured_path_string()
    checks["configured_path_exists"] = "yes" if exists else "no"
    checks["filesystem_reads"] = str(source.filesystem_reads)
    checks["sealed_underlying_reads"] = str(source.filesystem_reads)
    if source.filesystem_reads != 0:
        raise AssertionError("preflight performed a sealed filesystem read")
    return {
        "preflight": True,
        "sealed_read": False,
        "filesystem_reads": source.filesystem_reads,
        "training_context_reads": training_reads,
        "sealed_underlying_reads": source.filesystem_reads,
        "checks": checks,
    }


def run_v2_final_test(
    *,
    root: Path,
    registry_path: Path,
    acceptance_dir: Path,
    seal_dir: Path,
    seal_token: str,
    git_commit: str = "fixture",
    allow_second_look: bool = False,
    require_clean_tree: bool = False,
    require_git_tags: bool = False,
    require_preseal_tag: bool = False,
    require_hashes: bool = False,
    require_seal_token: bool = True,
    sealed_source: SealedSource | None = None,
    record_sealed_access: bool = True,
    fail_after: str | None = None,
    n: int = FROZEN_N,
    allow_reduced_n: bool = False,
    allow_sealed_dates: bool = False,
    result_label: str = "confirmatory",
    authorization: str = "",
    require_authorization: bool = False,
    t0: Any | None = None,
    run_robustness: bool = True,
    check_fast_first_null: bool = True,
    fitted_model: FittedModelA | None = None,
    model_artifact: SealedTestModelArtifact | None = None,
    model_artifact_path: Path | None = None,
    training_source: PreSealedTrainingSource | None = None,
    require_prebuilt_model: bool = False,
    model_freeze: dict[str, Any] | None = None,
    implementation_tag: str = FROZEN_PRESEAL_IMPL_TAG,
) -> V2FinalTestResult:
    """Atomic sealed path. Gate metrics come only from computed artifacts."""
    log = AtomicStepLog()
    log.record("validate_preconditions")
    _maybe_crash(fail_after, "validate_preconditions")
    if require_authorization:
        assert_fresh_r4_authorization(authorization, implementation_tag=implementation_tag)
    state = read_seal_state(seal_dir)
    seal_broken = bool(state.get("seal_broken"))
    if state.get("seal_opened") and not allow_second_look:
        raise PreconditionError(
            "V2 sealed evaluation already run; pass --allow-second-look "
            f"(will stamp '{SEAL_BROKEN_STAMP}' and never write V2_SEALED_PASS)"
        )

    loaded_artifact = model_artifact
    if model_artifact_path is not None and loaded_artifact is None:
        loaded_artifact = load_sealed_test_model(model_artifact_path)
        if model_freeze is not None:
            assert_artifact_matches_freeze(loaded_artifact, expected=model_freeze)
    if fitted_model is None and loaded_artifact is not None:
        fitted_model = loaded_artifact.fitted
    if require_prebuilt_model and fitted_model is None:
        raise PreconditionError("prebuilt sealed-test-model artifact is required")

    history_payload: dict[str, Any] | None = None
    if training_source is not None:
        history_payload = training_source.load()
        if training_source.sealed_underlying_reads != 0:
            raise TrainingLeakError("training source must not read sealed underlying")

    if not state.get("seal_opened"):
        preconditions = assert_v2_final_test_preconditions(
            root=root,
            registry_path=registry_path,
            seal_token=seal_token,
            require_clean_tree=require_clean_tree,
            require_git_tags=require_git_tags,
            require_preseal_tag=require_preseal_tag,
            require_hashes=require_hashes,
            require_seal_token=require_seal_token,
            acceptance_dir=acceptance_dir,
            seal_dir=seal_dir,
        )
    else:
        preconditions = {"second_look": "authorized"}
        expected = os.environ.get(SEAL_TOKEN_ENV, "").strip()
        if not expected or seal_token.strip() != expected:
            raise SealedAccessError("sealed access denied: invalid or missing SEAL token")

    if state.get("seal_opened") and allow_second_look:
        seal_broken = True
        state = {
            **state,
            "seal_broken": True,
            "stamp": SEAL_BROKEN_STAMP,
            "second_look_at": datetime.now(UTC).isoformat(),
        }
        write_seal_state(seal_dir, state)
        invalidate_v2_sealed_pass(acceptance_dir, reason=SEAL_BROKEN_STAMP)
        log.record("mark_seal_broken")

    log.record("append_sealed_access_record")
    if record_sealed_access and not state.get("seal_opened"):
        guard = SealedDataGuard(sealed_root=root / "data" / "sealed", registry_path=registry_path)
        guard._check_token(seal_token)
        guard._log_access(
            user="operator",
            experiment_hash=sha256_of(canonical_json({"study": "V2"})),
            reason=V2_SEAL_REASON,
            partition="v2-synthetic-or-authorized",
        )
        state = {
            "seal_opened": True,
            "seal_broken": seal_broken,
            "opened_at": datetime.now(UTC).isoformat(),
        }
        write_seal_state(seal_dir, state)
    elif not state.get("seal_opened"):
        state = {
            "seal_opened": True,
            "seal_broken": seal_broken,
            "opened_at": datetime.now(UTC).isoformat(),
            "record_sealed_access": False,
        }
        write_seal_state(seal_dir, state)
    _maybe_crash(fail_after, "append_sealed_access_record")

    log.record("read_sealed_or_fixture")
    if sealed_source is not None and hasattr(sealed_source, "authorize_for_atomic_path"):
        sealed_source.authorize_for_atomic_path()
    payload = sealed_source.load() if sealed_source is not None else {"fixture": True}
    _maybe_crash(fail_after, "read_sealed_or_fixture")

    from sqlalchemy import text

    from jevlab.utils.registry import init_registry

    engine = init_registry(registry_path)
    with engine.connect() as conn:
        access_count = int(
            conn.execute(text("SELECT COUNT(*) FROM sealed_access_log")).scalar_one()
        )
    if not record_sealed_access:
        access_count = 1 if not seal_broken else access_count

    study_t0 = t0 if t0 is not None else payload_t0(payload, default=SEALED_T0)
    flow = SealedDataFlow()
    if payload.get("fixture") and "frames" not in payload and "bars" not in payload:
        raise ValueError(
            "payload is a fixture stub; production computation requires "
            "features, labels/targets, and bars"
        )

    log.record("classify_sealed_payload")
    classified = classify_production_sealed_payload(
        payload,
        allowed_symbols=list(V2_UNIVERSE),
        min_date=None if study_t0 == SEALED_T0 else study_t0,
        max_date=SEALED_MAX_DATE if study_t0 == SEALED_T0 else None,
        require_real_window=study_t0 == SEALED_T0,
    )
    flow.raw_rows = classified.n_bar_rows + classified.n_vix_rows
    flow.classified_rows = classified.n_bar_rows + classified.n_vix_rows
    flow.unknown_frames = 0
    if classified.n_bar_rows == 0:
        raise EmptySealedScoringError(
            "classified sealed payload has zero market bars; refusing historical-only fallback"
        )
    _maybe_crash(fail_after, "classify_sealed_payload")

    log.record("merge_historical_context")
    if history_payload is None:
        raise EmptySealedScoringError(
            "r4 runner requires pre-sealed historical context plus sealed observations"
        )
    history_bars = pd.DataFrame(history_payload.get("bars", []))
    history_vix = pd.DataFrame(history_payload.get("vix", []))
    if history_bars.empty:
        raise EmptySealedScoringError("pre-sealed historical-context bars are empty")
    combined_bars = combine_history_and_sealed_bars(history_bars, classified.bars)
    combined_vix = combine_history_and_sealed_vix(history_vix, classified.vix)
    n_sealed_origin = int((combined_bars["row_origin"] == "sealed_evaluation").sum())
    flow.combined_history_sealed_rows = n_sealed_origin
    if n_sealed_origin == 0:
        raise EmptySealedScoringError(
            "historical/sealed merge contributed zero sealed rows; refusing fallback"
        )
    canary = classified.extras.get("canary_date")
    if canary is not None:
        canary_d = canary if isinstance(canary, date) else date.fromisoformat(str(canary))
        flow.canary_date = canary_d
        flow.canary_seen_in_combined = bool(
            (pd.to_datetime(combined_bars["date"]).dt.date == canary_d).any()
        )
    _maybe_crash(fail_after, "merge_historical_context")

    log.record("causal_feature_build")
    score_sessions = sorted(
        {d for d in pd.to_datetime(classified.bars["date"]).dt.date.tolist() if d >= study_t0}
    )
    if study_t0 == SEALED_T0:
        score_sessions = [d for d in score_sessions if d <= SEALED_MAX_DATE]
    if not score_sessions:
        raise EmptySealedScoringError(
            "no sealed decision dates reached causal feature construction"
        )
    present = set(classified.bars["symbol"].astype(str).unique().tolist()) | set(
        history_bars["symbol"].astype(str).unique().tolist()
    )
    symbols = [s for s in sorted(present) if s in set(V2_UNIVERSE)]
    if not symbols:
        raise SealedPayloadError("combined bars contain no allowed universe symbols")
    features = build_causal_score_features(
        history_bars=history_bars,
        sealed_bars=classified.bars,
        vix=combined_vix,
        history_vix=history_vix,
        sealed_vix=classified.vix,
        symbols=symbols,
        score_sessions=score_sessions,
    )
    flow.causal_feature_builder_invoked = True
    flow.feature_builder_input_rows = n_sealed_origin
    flow.valid_feature_rows = len(features)
    assert_sealed_scoring_nonempty(features, t0=study_t0)
    if flow.canary_date is not None:
        flow.canary_seen_in_features = bool(
            (pd.to_datetime(features["date"]).dt.date == flow.canary_date).any()
        )
    _maybe_crash(fail_after, "causal_feature_build")

    log.record("construct_features_labels")
    bars_for_labels = strip_audit_columns(combined_bars)
    labels = build_labels(
        bars_for_labels,
        eligible=features.loc[:, ["symbol", "date", "eligible"]],
        universe_symbols=symbols,
    )
    if labels.empty:
        raise EmptySealedScoringError("label construction produced zero rows from combined bars")
    labels = drop_unevaluable_end_window(labels, bars_for_labels)
    labels = labels.copy()
    labels["target_origin"] = "sealed_evaluation"
    model_features = strip_audit_columns(features)
    model_labels = strip_audit_columns(labels)
    frames = {
        "features": model_features,
        "labels": model_labels,
        "bars": bars_for_labels,
        "provenance_features": features,
        "combined_bars": combined_bars,
    }
    _maybe_crash(fail_after, "construct_features_labels")

    log.record("load_prebuilt_model")
    _maybe_crash(fail_after, "load_prebuilt_model")
    if fitted_model is None:
        raise TrainingLeakError(
            "r4 sealed runner requires a prebuilt Model A; refused to fit after sealed read"
        )
    live_digest = logical_model_digest(fitted_model)
    if study_t0 == SEALED_T0 and live_digest != FROZEN_MODEL_LOGICAL_DIGEST:
        raise TrainingLeakError(
            "prebuilt Model A logical digest drifted from the frozen sealed-test model"
        )
    training_meta = {}
    if loaded_artifact is not None:
        training_meta = {
            "n_2024_train": 0,
            "n_2025_train": 0,
            "training_row_digest": loaded_artifact.manifest["training_row_digest"],
            "preprocessing_digest": loaded_artifact.manifest["preprocessing_digest"],
            "model_digest": loaded_artifact.manifest["fitted_model_digest"],
            "training_source_manifest_digest": loaded_artifact.manifest[
                "training_source_manifest_digest"
            ],
        }
    evaluation = score_sealed_study(
        fitted=fitted_model,
        features=frames["features"],
        labels=frames["labels"],
        bars=frames["bars"],
        t0=study_t0,
        n=n,
        allow_reduced_n=allow_reduced_n,
        allow_sealed_dates=allow_sealed_dates,
        label=result_label,
        on_step=log.record,
        fail_after=fail_after,
        check_fast_first_null=check_fast_first_null,
        run_robustness=run_robustness,
        sealed_access_count=access_count,
        seal_broken=seal_broken,
        training_meta=training_meta,
        skip_construct_step=True,
    )
    pred_dates = pd.to_datetime(evaluation.predictions["date"]).dt.date
    flow.candidate_scoring_rows = int((pred_dates >= study_t0).sum())
    if "score" in evaluation.predictions.columns:
        flow.n_predictions = int(evaluation.predictions["score"].notna().sum())
    else:
        flow.n_predictions = len(evaluation.predictions)
    if flow.candidate_scoring_rows == 0:
        raise EmptySealedScoringError("zero sealed decision rows reached candidate scoring")
    scored = pred_dates.loc[pred_dates >= study_t0]
    flow.first_candidate_date = min(scored) if len(scored) else None
    flow.last_candidate_date = max(scored) if len(scored) else None
    if flow.canary_date is not None:
        flow.canary_seen_in_scoring = bool((pred_dates == flow.canary_date).any())
    flow.n_completed_trades = int(
        sum(int(v) for v in dict(evaluation.details.get("phase_round_trip_rows", {})).values())
    )
    flow.historical_only_fallback = False
    evaluation.details["data_flow"] = flow.to_dict()
    evaluation.details["causal_feature_builder_invoked"] = True
    evaluation.details["historical_only_fallback"] = False
    evaluation.details["production_payload_type"] = "ProductionSealedPayload"
    report = evaluation.gate_report
    result_path = seal_dir / "SEALED_RESULT.json"
    seal_dir.mkdir(parents=True, exist_ok=True)
    result_body = json_safe(
        {
            "label": result_label if not seal_broken else "diagnostics_only_seal_broken",
            "v2_confirmatory_evidence": result_label != REHEARSAL_LABEL and not seal_broken,
            "seal_broken": seal_broken,
            "gates": report.to_dict(),
            "atomic_steps": list(log.steps),
            "observed_mean_sharpe": evaluation.observed_mean_sharpe,
            "p_value": evaluation.test.p_value,
            "sharpe_advantage": evaluation.test.advantage,
            "details": evaluation.details,
        }
    )
    result_digest = sha256_of(canonical_json(result_body))
    result_body["integrity_digest"] = result_digest
    atomic_write_json(result_path, result_body)
    log.record("conditionally_write_acceptance")
    _maybe_crash(fail_after, "conditionally_write_acceptance")
    acceptance_path = None
    written = False
    can_write_pass = (
        not seal_broken
        and report.all_passed
        and result_label != REHEARSAL_LABEL
        and result_label != "development_rehearsal_only"
    )
    if can_write_pass:
        art = write_v2_sealed_pass(
            report,
            out_dir=acceptance_dir,
            git_commit=git_commit,
            sealed_result_digest=result_digest,
            training_artifact_digest=str(
                evaluation.details.get("training_row_digest", {}).get("content_sha256", "")
            ),
            random_null_artifact_digest=str(evaluation.details.get("null_vector_digest", "")),
        )
        acceptance_path = art.path
        written = True
    log.record("close_run")
    _maybe_crash(fail_after, "close_run")
    return V2FinalTestResult(
        gate_report=report,
        acceptance_written=written,
        seal_broken=seal_broken,
        preconditions=preconditions,
        step_log=list(log.steps),
        acceptance_path=acceptance_path,
        result_path=result_path,
        data_flow=flow.to_dict(),
    )


class ForbiddenRealSealedSource:
    """Production source - must not be constructed during V2-1...V2-4 tests."""

    def load(self) -> dict[str, Any]:
        raise SealedAccessError(
            "real 2024-2025 sealed load is not authorized in pre-seal implementation"
        )
