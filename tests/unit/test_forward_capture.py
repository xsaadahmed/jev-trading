"""Phase 13A forward INPUT capture: registry, calendar, Jev failures, import boundary."""

from __future__ import annotations

import ast
import sys
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from jevlab.config.loader import load_all_configs
from jevlab.data.availability import decision_time
from jevlab.experiments.forward_capture import (
    is_session_complete,
    resolve_capture_session,
    run_forward_capture,
    session_dir,
)
from jevlab.experiments.forward_registry import (
    ForwardRegistryImmutableError,
    compute_forward_start_date,
    read_deployment,
    read_forward_start_date,
    register_deployment,
)
from jevlab.features.columns import FEATURE_VERSION, STORED_FEATURE_COLUMNS
from jevlab.jev.qset import load_qset
from jevlab.jev.schemas import SystemOneResponse
from jevlab.utils.paths import ProjectPaths, get_project_root
from tests.unit.feature_fixtures import make_sessions

PINNED = "jev-1.13.0"
ET_ZONE = ZoneInfo("America/New_York")


def _mock_response(model: str = PINNED) -> SystemOneResponse:
    return SystemOneResponse.model_validate(
        {
            "model": model,
            "answers": {
                "dislocation": {"type": "noul", "noul": 0.6},
                "regime_supportive": {"type": "noul", "noul": 0.55},
                "downside_risk": {
                    "type": "score",
                    "score": 2.0,
                    "legend": {
                        "0": "a",
                        "1": "b",
                        "2": "c",
                        "3": "d",
                        "4": "e",
                    },
                    "probabilities": {
                        "0": 0.1,
                        "1": 0.2,
                        "2": 0.3,
                        "3": 0.25,
                        "4": 0.15,
                    },
                    "confidence": 0.4,
                },
                "setup_type_a": {
                    "type": "choice",
                    "choice": "pullback_in_uptrend",
                    "probabilities": {
                        "pullback_in_uptrend": 0.4,
                        "trend_breakdown": 0.2,
                        "sideways_noise": 0.2,
                        "extended_rally_pause": 0.2,
                    },
                    "confidence": 0.2,
                },
                "setup_type_b": {
                    "type": "choice",
                    "choice": "pullback_in_uptrend",
                    "probabilities": {
                        "extended_rally_pause": 0.2,
                        "sideways_noise": 0.2,
                        "trend_breakdown": 0.2,
                        "pullback_in_uptrend": 0.4,
                    },
                    "confidence": 0.2,
                },
            },
        }
    )


class _FakeClient:
    def __init__(
        self,
        *,
        model: str = PINNED,
        error: Exception | None = None,
    ) -> None:
        self.model = model
        self.error = error
        self.calls = 0

    def infer(
        self,
        state: Any,
        questions: Any,
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _mock_response(self.model)


def _paths(tmp_path: Path) -> ProjectPaths:
    root = get_project_root()
    return ProjectPaths(
        root=root,
        config=root / "config",
        data=tmp_path / "data",
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        data_features=tmp_path / "data" / "features",
        data_jev_cache=tmp_path / "data" / "jev_cache",
        data_sealed=tmp_path / "data" / "sealed",
        data_forward=tmp_path / "data" / "forward",
        data_models=tmp_path / "data" / "models",
        registry=tmp_path / "registry",
        reports=tmp_path / "reports",
        reports_acceptance=tmp_path / "reports" / "acceptance",
        docs=root / "docs",
        tests=root / "tests",
    )


def _small_configs() -> Any:
    cfgs = load_all_configs(get_project_root() / "config")
    uni = cfgs.universe.model_copy(update={"symbols": ["SPY", "QQQ"]})
    return replace(cfgs, universe=uni)


def _synthetic_features(
    symbols: list[str],
    sessions: list[date],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Dense non-NaN feature panel so state trailing-252 percentiles are defined."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for j, sym in enumerate(symbols):
        for i, d in enumerate(sessions):
            row: dict[str, Any] = {
                "symbol": sym,
                "date": d,
                "feature_version": FEATURE_VERSION,
                "decision_time": decision_time(d).tz_convert("UTC"),
                "eligible": 1,
                "down_streak": float((i + j) % 5),
            }
            for col in STORED_FEATURE_COLUMNS:
                if col == "down_streak":
                    continue
                row[col] = float(rng.normal(0.0, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_fixture(
    n_sessions: int = 260,
) -> tuple[pd.DataFrame, pd.DataFrame, date]:
    """Return ``(history, session_features, target_session)`` for offline capture tests."""
    sessions = make_sessions(n_sessions, start=date(2018, 1, 2))
    symbols = ["SPY", "QQQ"]
    panel = _synthetic_features(symbols, sessions, seed=11)
    target = sessions[-1]
    hist = panel.loc[panel["date"] < target].reset_index(drop=True)
    session_feats = panel.loc[panel["date"] == target].reset_index(drop=True)
    return hist, session_feats, target


# --- registry -----------------------------------------------------------------


def test_registry_immutability_second_write_raises(tmp_path: Path) -> None:
    db = tmp_path / "experiments.sqlite"
    row1 = register_deployment(
        registry_path=db,
        jev_model_id=PINNED,
        qset_hash="abc",
        feature_version=FEATURE_VERSION,
        state_schema_version="state_v1",
        git_commit="deadbeef",
        deployment_timestamp=datetime(2026, 10, 7, 22, 45, tzinfo=UTC),
    )
    assert row1.forward_start_date == compute_forward_start_date(row1.deployment_timestamp)
    with pytest.raises(ForwardRegistryImmutableError):
        register_deployment(
            registry_path=db,
            jev_model_id=PINNED,
            qset_hash="abc",
            feature_version=FEATURE_VERSION,
            state_schema_version="state_v1",
            git_commit="cafebabe",
            deployment_timestamp=datetime(2026, 10, 8, 22, 45, tzinfo=UTC),
        )
    again = read_deployment(db)
    assert again is not None
    assert again.registry_id == row1.registry_id
    assert again.git_commit == "deadbeef"
    assert read_forward_start_date(db) == row1.forward_start_date


def test_forward_start_date_after_deployment_timestamp() -> None:
    # Monday 2026-10-05 22:45 UTC = 18:45 ET → decision_time(Mon)=18:30 ET is before
    deploy = datetime(2026, 10, 5, 22, 45, tzinfo=UTC)
    fwd = compute_forward_start_date(deploy)
    assert decision_time(fwd).tz_convert("UTC") > pd.Timestamp(deploy)
    assert fwd == date(2026, 10, 6)  # next NYSE session (Tue)


# --- calendar / gates ---------------------------------------------------------


def test_holiday_skip() -> None:
    thanksgiving = date(2024, 11, 28)
    now = datetime(2024, 11, 28, 19, 0, tzinfo=ET_ZONE)
    status, session, msg = resolve_capture_session(now, session=thanksgiving)
    assert status == "skipped_holiday"
    assert session == thanksgiving
    assert "not an NYSE" in msg

    status2, session2, _ = resolve_capture_session(now)
    assert status2 == "skipped_holiday"
    assert session2 == thanksgiving


def test_too_early_before_decision_time() -> None:
    session = date(2026, 10, 7)
    now = datetime(2026, 10, 7, 17, 0, tzinfo=ET_ZONE)
    status, day, _ = resolve_capture_session(now, session=session)
    assert status == "too_early"
    assert day == session


# --- capture pipeline ---------------------------------------------------------


def test_idempotent_per_date(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure_runtime_dirs()
    cfgs = _small_configs()
    hist, session_feats, target = _feature_fixture()
    qset = load_qset(get_project_root() / "config" / "qset_v1.json")
    client = _FakeClient()
    now = datetime.combine(target, datetime.min.time(), tzinfo=ET_ZONE).replace(hour=19, minute=0)

    r1 = run_forward_capture(
        paths=paths,
        configs=cfgs,
        now=now,
        session=target,
        client=client,
        download=False,
        feature_history=hist,
        session_features=session_feats,
        qset=qset,
        git_commit="testcommit1",
        deployment_timestamp=datetime(2026, 10, 8, 23, 0, tzinfo=UTC),
    )
    assert r1.status == "complete"
    assert r1.registry_written is True
    assert client.calls == 2
    assert is_session_complete(paths.data_forward, target)

    client2 = _FakeClient()
    r2 = run_forward_capture(
        paths=paths,
        configs=cfgs,
        now=now,
        session=target,
        client=client2,
        download=False,
        feature_history=hist,
        session_features=session_feats,
        qset=qset,
    )
    assert r2.status == "skipped_idempotent"
    assert client2.calls == 0
    dep = read_deployment(paths.registry / "experiments.sqlite")
    assert dep is not None
    assert dep.git_commit == "testcommit1"


def test_jev_failure_marks_day_incomplete(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure_runtime_dirs()
    cfgs = _small_configs()
    hist, session_feats, target = _feature_fixture()
    qset = load_qset(get_project_root() / "config" / "qset_v1.json")
    client = _FakeClient(error=RuntimeError("simulated API down"))
    now = datetime.combine(target, datetime.min.time(), tzinfo=ET_ZONE).replace(hour=19, minute=0)

    result = run_forward_capture(
        paths=paths,
        configs=cfgs,
        now=now,
        session=target,
        client=client,
        download=False,
        feature_history=hist,
        session_features=session_feats,
        qset=qset,
    )
    assert result.status == "incomplete"
    assert result.n_jev_failed >= 1
    assert result.registry_written is False
    assert read_deployment(paths.registry / "experiments.sqlite") is None
    assert not is_session_complete(paths.data_forward, target)
    out = session_dir(paths.data_forward, target)
    assert (out / "features.parquet").is_file()
    raw = (out / "jev_responses.jsonl").read_text(encoding="utf-8").strip()
    assert raw == ""
    fails = (out / "jev_failures.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(fails) >= 1


def test_model_id_mismatch_aborts_jev_part(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure_runtime_dirs()
    cfgs = _small_configs()
    hist, session_feats, target = _feature_fixture()
    qset = load_qset(get_project_root() / "config" / "qset_v1.json")
    client = _FakeClient(model="jev-9.9.9")
    now = datetime.combine(target, datetime.min.time(), tzinfo=ET_ZONE).replace(hour=19, minute=0)

    result = run_forward_capture(
        paths=paths,
        configs=cfgs,
        now=now,
        session=target,
        client=client,
        download=False,
        feature_history=hist,
        session_features=session_feats,
        qset=qset,
    )
    assert result.status == "incomplete"
    assert result.meta.get("model_mismatch") is True
    assert client.calls == 1
    assert result.n_jev_ok == 0
    assert result.registry_written is False
    out = session_dir(paths.data_forward, target)
    raw = (out / "jev_responses.jsonl").read_text(encoding="utf-8").strip()
    assert raw == ""


# --- import boundary / no scoring-labels --------------------------------------


_FORWARD_MODULES = (
    Path("src/jevlab/experiments/forward_capture.py"),
    Path("src/jevlab/experiments/forward_registry.py"),
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "jevlab.brokers",
    "jevlab.models",
    "jevlab.targets.labels",
    "jevlab.signals",
    "jevlab.backtest",
    "jevlab.portfolio",
    "jevlab.evaluation",
)

_FORBIDDEN_RUNTIME_PREFIXES = (
    "jevlab.brokers",
    "jevlab.models",
    "jevlab.targets.labels",
    "jevlab.signals",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_import_boundary_no_broker_module() -> None:
    root = get_project_root()
    for rel in _FORWARD_MODULES:
        imported = _imported_modules(root / rel)
        for name in imported:
            assert not name.startswith("jevlab.brokers"), f"{rel} imports {name}"

    before = {m for m in sys.modules if m.startswith("jevlab.brokers")}
    for m in list(sys.modules):
        if m.startswith("jevlab.experiments.forward"):
            del sys.modules[m]
    import jevlab.experiments.forward_capture as fc  # noqa: F401
    import jevlab.experiments.forward_registry as fr  # noqa: F401

    after = {m for m in sys.modules if m.startswith("jevlab.brokers")}
    assert after == before


def test_no_scoring_or_label_code_reachable() -> None:
    root = get_project_root()
    for rel in _FORWARD_MODULES:
        imported = _imported_modules(root / rel)
        for name in imported:
            for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
                assert not (name == forbidden or name.startswith(forbidden + ".")), (
                    f"{rel} imports forbidden {name}"
                )

    for m in list(sys.modules):
        if m.startswith("jevlab.experiments.forward"):
            del sys.modules[m]
    victims = [
        m
        for m in list(sys.modules)
        if any(m == p or m.startswith(p + ".") for p in _FORBIDDEN_RUNTIME_PREFIXES)
    ]
    saved = {m: sys.modules.pop(m) for m in victims}
    try:
        import jevlab.experiments.forward_capture  # noqa: F401

        leaked = [
            m
            for m in sys.modules
            if any(m == p or m.startswith(p + ".") for p in _FORBIDDEN_RUNTIME_PREFIXES)
        ]
        assert leaked == [], f"forward_capture pulled scoring/label/broker modules: {leaked}"
    finally:
        sys.modules.update(saved)


def test_capture_stores_timestamps_and_versions(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure_runtime_dirs()
    cfgs = _small_configs()
    hist, session_feats, target = _feature_fixture()
    qset = load_qset(get_project_root() / "config" / "qset_v1.json")
    now = datetime.combine(target, datetime.min.time(), tzinfo=ET_ZONE).replace(hour=19, minute=0)
    deploy_ts = datetime(2026, 10, 8, 23, 30, tzinfo=UTC)
    result = run_forward_capture(
        paths=paths,
        configs=cfgs,
        now=now,
        session=target,
        client=_FakeClient(),
        download=False,
        feature_history=hist,
        session_features=session_feats,
        qset=qset,
        git_commit="abc123",
        deployment_timestamp=deploy_ts,
    )
    assert result.status == "complete"
    assert result.forward_start_date == compute_forward_start_date(deploy_ts)
    feats = pd.read_parquet(session_dir(paths.data_forward, target) / "features.parquet")
    assert "decision_time" in feats.columns
    for col in STORED_FEATURE_COLUMNS:
        assert col in feats.columns
    assert pd.to_datetime(feats["decision_time"], utc=True).notna().all()
    status = (session_dir(paths.data_forward, target) / "status.json").read_text(encoding="utf-8")
    assert PINNED in status
    assert FEATURE_VERSION in status
    assert "state_v1" in status
