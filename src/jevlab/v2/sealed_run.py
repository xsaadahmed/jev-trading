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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV, SealedAccessError, SealedDataGuard
from jevlab.experiments.acceptance import count_sealed_accesses
from jevlab.jev.preconditions import PreconditionError, git_tag_exists, git_working_tree_clean
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.v2.acceptance import V2GateReport
from jevlab.v2.artifacts import atomic_write_json, invalidate_v2_sealed_pass, write_v2_sealed_pass
from jevlab.v2.evaluate import (
    InjectedCrash,
    construct_study_frames,
    evaluate_v2_study,
    json_safe,
    payload_t0,
)
from jevlab.v2.hashes import (
    FROZEN_COMMIT,
    FROZEN_PRESEAL_IMPL_TAG,
    FROZEN_PRESEAL_IMPL_TAG_R1,
    FROZEN_TAG_DESIGN,
    FROZEN_TAG_DESIGN_R1,
    FROZEN_TAG_PREREG,
    FROZEN_TAG_PREREG_R1,
    FrozenHashError,
    assert_frozen_research_hashes,
)
from jevlab.v2.rehearsal import REHEARSAL_LABEL
from jevlab.v2.sealed_source import ProductionSealedSource
from jevlab.v2.statistics import FROZEN_N
from jevlab.v2.training import SEALED_T0

SEAL_BROKEN_STAMP = "SEAL BROKEN"
SEAL_STATE_FILENAME = "v2_seal_state.json"
V2_SEAL_REASON = "v2-final-test-sealed-score"
VOID_AUTHORIZATION_PHRASES: frozenset[str] = frozenset(
    {
        "OPEN V2 REVISION 1 SEALED HOLDOUT ONCE",
    }
)
REQUIRED_AUTHORIZATION_PHRASE = "OPEN V2 REVISION 1 SEALED HOLDOUT ONCE WITH IMPLEMENTATION R2"
ATOMIC_STEPS: tuple[str, ...] = (
    "validate_preconditions",
    "append_sealed_access_record",
    "read_sealed_or_fixture",
    "construct_features_labels",
    "fit_once_pre_sealed",
    "archive_training_artifact",
    "score_entire_period",
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


def assert_fresh_r2_authorization(authorization: str) -> None:
    """Reject the closed r1 authorization; require a new r2 phrase."""
    phrase = (authorization or "").strip()
    if phrase in VOID_AUTHORIZATION_PHRASES:
        raise PreconditionError(
            "prior human authorization is VOID/CLOSED; implementation r1 never "
            "consumed the seal and cannot be reused"
        )
    if phrase != REQUIRED_AUTHORIZATION_PHRASE:
        raise PreconditionError(
            "fresh explicit human authorization is required after "
            f"{FROZEN_PRESEAL_IMPL_TAG}; the prior r1 authorization is void"
        )


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
        if FROZEN_PRESEAL_IMPL_TAG == FROZEN_PRESEAL_IMPL_TAG_R1:
            raise PreconditionError(
                "v2-preseal-implementation-r1 is provenance only and is not execution-capable"
            )
        if require_preseal_tag and not git_tag_exists(FROZEN_PRESEAL_IMPL_TAG, cwd=root):
            raise PreconditionError(
                f"missing {FROZEN_PRESEAL_IMPL_TAG} (required for the real sealed run); "
                f"{FROZEN_PRESEAL_IMPL_TAG_R1} is preserved as an incomplete audit artifact"
            )
        checks["preseal_tag"] = (
            "present" if git_tag_exists(FROZEN_PRESEAL_IMPL_TAG, cwd=root) else "skipped"
        )
        checks["preseal_tag_r1_not_execution"] = FROZEN_PRESEAL_IMPL_TAG_R1
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


def run_v2_preflight(
    *,
    root: Path,
    registry_path: Path,
    configured_sealed_path: Path,
    require_clean_tree: bool = True,
    require_git_tags: bool = True,
    require_preseal_tag: bool = True,
    require_hashes: bool = True,
) -> dict[str, Any]:
    """Terminate before SEAL authorization and before any sealed read.

    May check configured-path existence without listing contents.
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
    exists = configured_sealed_path.exists()
    checks["configured_sealed_path"] = source.configured_path_string()
    checks["configured_path_exists"] = "yes" if exists else "no"
    checks["filesystem_reads"] = str(source.filesystem_reads)
    if source.filesystem_reads != 0:
        raise AssertionError("preflight performed a sealed filesystem read")
    return {
        "preflight": True,
        "sealed_read": False,
        "filesystem_reads": source.filesystem_reads,
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
) -> V2FinalTestResult:
    """Atomic sealed path. Gate metrics come only from computed artifacts."""
    log = AtomicStepLog()
    log.record("validate_preconditions")
    _maybe_crash(fail_after, "validate_preconditions")
    if require_authorization:
        assert_fresh_r2_authorization(authorization)
    state = read_seal_state(seal_dir)
    seal_broken = bool(state.get("seal_broken"))
    if state.get("seal_opened") and not allow_second_look:
        raise PreconditionError(
            "V2 sealed evaluation already run; pass --allow-second-look "
            f"(will stamp '{SEAL_BROKEN_STAMP}' and never write V2_SEALED_PASS)"
        )

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

    frames = construct_study_frames(payload, allow_sealed_dates=allow_sealed_dates)
    study_t0 = t0 if t0 is not None else payload_t0(payload, default=SEALED_T0)
    evaluation = evaluate_v2_study(
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
    )
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
    )


class ForbiddenRealSealedSource:
    """Production source - must not be constructed during V2-1...V2-4 tests."""

    def load(self) -> dict[str, Any]:
        raise SealedAccessError(
            "real 2024-2025 sealed load is not authorized in pre-seal implementation"
        )
