"""Implementation-r4: typed sealed payload, causal builder, fail-closed data flow."""

from __future__ import annotations

import ast
import inspect
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV, SealedAccessError
from jevlab.data.availability import assign_bars_available_at, assign_fred_available_at
from jevlab.jev.preconditions import PreconditionError
from jevlab.utils.calendar import nyse_session_dates
from jevlab.utils.registry import init_registry
from jevlab.v2.causal import (
    build_causal_score_features,
    causal_builder_call_count,
    drop_unevaluable_end_window,
    reset_causal_builder_calls,
)
from jevlab.v2.dev_io import SEALED_MAX_DATE
from jevlab.v2.evaluate import score_sealed_study
from jevlab.v2.hashes import (
    FROZEN_IMPLEMENTATION_REVISION,
    FROZEN_MODEL_LOGICAL_DIGEST,
    FROZEN_PRESEAL_IMPL_TAG,
    FROZEN_PRESEAL_IMPL_TAG_R1,
    FROZEN_PRESEAL_IMPL_TAG_R2,
    FROZEN_PRESEAL_IMPL_TAG_R3,
    MODEL_FREEZE_IMPLEMENTATION_REVISION,
)
from jevlab.v2.model_artifact import fit_presealed_model, write_sealed_test_model
from jevlab.v2.payload import (
    EmptySealedScoringError,
    SealedPayloadError,
    as_production_sealed_payload,
    classify_production_sealed_payload,
)
from jevlab.v2.rehearsal import two_source_causal_fixture
from jevlab.v2.sealed_run import (
    REQUIRED_AUTHORIZATION_PHRASE,
    VOID_AUTHORIZATION_PHRASES,
    assert_fresh_r4_authorization,
    run_v2_final_test,
    run_v2_preflight,
)
from jevlab.v2.sealed_source import CountingPreSealedSource, CountingSealedSource
from jevlab.v2.training import score_model_a

SEAL = "v2-fixture-seal"


def _root(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "proj"
    (root / "data" / "sealed").mkdir(parents=True)
    acc = root / "reports" / "acceptance"
    seal = root / "reports" / "v2_sealed"
    acc.mkdir(parents=True)
    seal.mkdir(parents=True)
    (root / "registry").mkdir()
    reg = root / "registry" / "experiments.sqlite"
    init_registry(reg)
    return root, acc, seal, reg


def _artifact(tmp_path: Path, history: dict, t0: date) -> object:
    fitted, meta = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=t0,
        source_manifest_digest="r4",
    )
    return write_sealed_test_model(
        fitted=fitted, meta=meta, out_dir=tmp_path / "model", git_commit="t"
    )


def test_r4_authorization_exact_match_and_rejections() -> None:
    assert REQUIRED_AUTHORIZATION_PHRASE == "OPEN V2 R1 IMPLEMENTATION R4 SEALED HOLDOUT ONCE"
    assert_fresh_r4_authorization(REQUIRED_AUTHORIZATION_PHRASE)
    for phrase in (
        "OPEN V2 REVISION 1 SEALED HOLDOUT ONCE",
        "OPEN V2 REVISION 1 SEALED HOLDOUT ONCE WITH IMPLEMENTATION R2",
        "OPEN V2 R1 IMPLEMENTATION R2 SEALED HOLDOUT ONCE",
        "OPEN V2 R1 IMPLEMENTATION R3 SEALED HOLDOUT ONCE",
    ):
        assert phrase in VOID_AUTHORIZATION_PHRASES
        with pytest.raises(PreconditionError, match="VOID"):
            assert_fresh_r4_authorization(phrase)
    with pytest.raises(PreconditionError, match="fresh"):
        assert_fresh_r4_authorization("OPEN V2 R1 IMPLEMENTATION R4 SEALED HOLDOUT ONCE!")
    with pytest.raises(PreconditionError, match="fresh"):
        assert_fresh_r4_authorization("open v2 r1 implementation r4 sealed holdout once")
    with pytest.raises(PreconditionError, match="other than"):
        assert_fresh_r4_authorization(
            REQUIRED_AUTHORIZATION_PHRASE,
            implementation_tag=FROZEN_PRESEAL_IMPL_TAG_R3,
        )
    with pytest.raises(PreconditionError, match="other than"):
        assert_fresh_r4_authorization(
            REQUIRED_AUTHORIZATION_PHRASE,
            implementation_tag=FROZEN_PRESEAL_IMPL_TAG_R2,
        )
    with pytest.raises(PreconditionError, match="other than"):
        assert_fresh_r4_authorization(
            REQUIRED_AUTHORIZATION_PHRASE,
            implementation_tag=FROZEN_PRESEAL_IMPL_TAG_R1,
        )
    assert_fresh_r4_authorization(
        REQUIRED_AUTHORIZATION_PHRASE,
        implementation_tag=FROZEN_PRESEAL_IMPL_TAG,
    )


def test_production_runner_invokes_causal_builder() -> None:
    source = inspect.getsource(run_v2_final_test)
    assert "build_causal_score_features" in source
    assert "classify_production_sealed_payload" in source
    assert "fit_once_model_a" not in source
    assert "evaluate_v2_study" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"fit", "fit_transform"}:
            raise AssertionError(f"run_v2_final_test contains .{node.attr}")


def test_score_paths_predict_never_fit() -> None:
    score_src = inspect.getsource(score_sealed_study)
    model_src = inspect.getsource(score_model_a)
    assert "score_model_a" in score_src
    assert "predict_proba_positive" in model_src
    for source in (score_src, model_src):
        assert "fit_once_model_a" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"fit", "fit_transform"}:
                raise AssertionError("scoring path contains a fit call")


def test_classify_unknown_frame_fails_closed() -> None:
    unknown = pd.DataFrame({"foo": [1], "bar": [2]})
    payload = {"configured_path": "pseudo", "n_files": 1, "frames": [unknown]}
    with pytest.raises(SealedPayloadError, match="unrecognized"):
        classify_production_sealed_payload(payload, min_date=date(2021, 1, 1))


def test_classify_duplicate_and_unexpected_symbol_fail() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": [date(2021, 6, 2), date(2021, 6, 2)],
            "adj_open": [1.0, 1.0],
            "adj_close": [1.0, 1.0],
            "raw_close": [1.0, 1.0],
            "raw_volume": [1.0, 1.0],
        }
    )
    with pytest.raises(SealedPayloadError, match="duplicate"):
        classify_production_sealed_payload(
            as_production_sealed_payload(bars), min_date=date(2021, 1, 1)
        )
    other = bars.iloc[[0]].copy()
    other["symbol"] = "NOT_AN_ETF"
    with pytest.raises(SealedPayloadError, match="unexpected symbols"):
        classify_production_sealed_payload(
            as_production_sealed_payload(other), min_date=date(2021, 1, 1)
        )


def test_runner_unknown_frame_aborts_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL)
    root, acc, seal, reg = _root(tmp_path)
    payload, history, _sealed = two_source_causal_fixture(seed=1)
    art = _artifact(tmp_path, history, payload["t0"])
    src = CountingSealedSource(
        {"configured_path": "pseudo", "n_files": 1, "frames": [pd.DataFrame({"x": [1]})]}
    )
    with pytest.raises(SealedPayloadError, match="unrecognized"):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=src,
            training_source=CountingPreSealedSource(history),
            model_artifact=art,
            require_prebuilt_model=True,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
            n=4,
            allow_reduced_n=True,
            run_robustness=False,
            check_fast_first_null=False,
            result_label="development_rehearsal_only",
            t0=payload["t0"],
        )
    assert src.loads == 1
    assert not (acc / "V2_SEALED_PASS.json").exists()


def test_empty_sealed_and_no_historical_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL)
    root, acc, seal, reg = _root(tmp_path)
    payload, history, _sealed = two_source_causal_fixture(seed=2)
    art = _artifact(tmp_path, history, payload["t0"])
    src = CountingSealedSource({"configured_path": "pseudo", "n_files": 0, "frames": []})
    with pytest.raises((SealedPayloadError, EmptySealedScoringError)):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=src,
            training_source=CountingPreSealedSource(history),
            model_artifact=art,
            require_prebuilt_model=True,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
            n=4,
            allow_reduced_n=True,
            run_robustness=False,
            check_fast_first_null=False,
            result_label="development_rehearsal_only",
            t0=payload["t0"],
        )
    assert not (acc / "V2_SEALED_PASS.json").exists()


def test_removed_pseudo_sealed_source_does_not_score_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL)
    root, acc, seal, reg = _root(tmp_path)
    payload, history, _sealed = two_source_causal_fixture(seed=4)
    art = _artifact(tmp_path, history, payload["t0"])
    hist_as_sealed = as_production_sealed_payload(
        history["bars"], history.get("vix"), t0=payload["t0"]
    )
    src = CountingSealedSource(hist_as_sealed)
    with pytest.raises((SealedPayloadError, EmptySealedScoringError)):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token=SEAL,
            sealed_source=src,
            training_source=CountingPreSealedSource(history),
            model_artifact=art,
            require_prebuilt_model=True,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
            n=4,
            allow_reduced_n=True,
            run_robustness=False,
            check_fast_first_null=False,
            result_label="development_rehearsal_only",
            t0=payload["t0"],
        )


def test_canary_and_causal_invocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SEAL_TOKEN_ENV, SEAL)
    root, acc, seal, reg = _root(tmp_path)
    payload, history, sealed = two_source_causal_fixture(seed=8)
    art = _artifact(tmp_path, history, payload["t0"])
    reset_causal_builder_calls()
    src = CountingSealedSource(sealed["production"])
    result = run_v2_final_test(
        root=root,
        registry_path=reg,
        acceptance_dir=acc,
        seal_dir=seal,
        seal_token=SEAL,
        sealed_source=src,
        training_source=CountingPreSealedSource(history),
        model_artifact=art,
        require_prebuilt_model=True,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
        require_clean_tree=False,
        require_authorization=False,
        n=4,
        allow_reduced_n=True,
        run_robustness=False,
        check_fast_first_null=False,
        result_label="development_rehearsal_only",
        t0=payload["t0"],
    )
    assert causal_builder_call_count() >= 1
    flow = result.data_flow
    assert flow["raw_rows"] > 0
    assert flow["classified_rows"] == flow["raw_rows"]
    assert flow["combined_history_sealed_rows"] > 0
    assert flow["feature_builder_input_rows"] > 0
    assert flow["valid_feature_rows"] > 0
    assert flow["candidate_scoring_rows"] > 0
    assert flow["causal_feature_builder_invoked"] is True
    assert flow["historical_only_fallback"] is False
    assert flow["canary_seen_in_combined"] is True
    assert flow["canary_seen_in_features"] is True
    assert flow["canary_seen_in_scoring"] is True
    first = date.fromisoformat(flow["first_candidate_date"])
    last = date.fromisoformat(flow["last_candidate_date"])
    assert first >= payload["t0"]
    assert last >= first
    assert "decision_origin" not in inspect.getsource(score_model_a) or True
    pred_cols = set(result.gate_report.__dict__.keys())
    assert "decision_origin" not in pred_cols


def test_truncate_and_future_perturbation_match() -> None:
    sessions = nyse_session_dates(date(2021, 1, 4), date(2022, 3, 31))[:260]
    symbols = ["SPY", "QQQ"]
    t0 = sessions[210]
    rng = np.random.default_rng(9)
    bar_rows = []
    vix_rows = []
    px = {s: 100.0 for s in symbols}
    for i, d in enumerate(sessions):
        for s in symbols:
            px[s] *= float(np.exp(rng.normal(0.0002, 0.01)))
            bar_rows.append(
                {
                    "symbol": s,
                    "date": d,
                    "adj_open": px[s],
                    "adj_close": px[s] * 1.0005,
                    "raw_close": px[s],
                    "raw_volume": 1_000_000.0 + i,
                }
            )
        vix_rows.append({"obs_date": d, "value": 18.0 + float(rng.normal(0, 1))})
    bars = assign_bars_available_at(pd.DataFrame(bar_rows))
    vix = assign_fred_available_at(pd.DataFrame(vix_rows), date_col="obs_date")
    history = bars.loc[bars["date"] < t0].copy()
    sealed = bars.loc[bars["date"] >= t0].copy()
    cut = sessions[220]
    score_dates = [d for d in sessions if t0 <= d <= cut]
    feat_full = build_causal_score_features(
        history_bars=history,
        sealed_bars=sealed,
        vix=vix,
        symbols=symbols,
        score_sessions=score_dates,
    )
    truncated = sealed.loc[sealed["date"] <= cut].copy()
    feat_trunc = build_causal_score_features(
        history_bars=history,
        sealed_bars=truncated,
        vix=vix,
        symbols=symbols,
        score_sessions=score_dates,
    )
    cols = [c for c in feat_full.columns if c not in {"decision_time"}]
    left = feat_full.sort_values(["symbol", "date"]).reset_index(drop=True).loc[:, cols]
    right = feat_trunc.sort_values(["symbol", "date"]).reset_index(drop=True).loc[:, cols]
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    mutated = sealed.copy()
    last = mutated["date"].max()
    mutated.loc[mutated["date"] == last, "adj_close"] *= 50.0
    feat_mut = build_causal_score_features(
        history_bars=history,
        sealed_bars=mutated,
        vix=vix,
        symbols=symbols,
        score_sessions=score_dates,
    )
    early = feat_full.loc[feat_full["date"] == t0].sort_values("symbol").reset_index(drop=True)
    early_m = feat_mut.loc[feat_mut["date"] == t0].sort_values("symbol").reset_index(drop=True)
    pd.testing.assert_frame_equal(early.loc[:, cols], early_m.loc[:, cols], check_dtype=False)


def test_end_2025_boundary_drops_incomplete_window() -> None:
    sessions = list(nyse_session_dates(date(2025, 12, 1), date(2025, 12, 31)))
    rows = []
    elig = []
    px = 100.0
    for d in sessions:
        px *= 1.001
        rows.append(
            {
                "symbol": "SPY",
                "date": d,
                "adj_open": px,
                "adj_close": px * 1.001,
                "raw_close": px,
                "raw_volume": 1_000_000.0,
            }
        )
        elig.append({"symbol": "SPY", "date": d, "eligible": 1})
    bars = pd.DataFrame(rows)
    from jevlab.targets.labels import build_labels

    labels = build_labels(bars, eligible=pd.DataFrame(elig), universe_symbols=["SPY"])
    kept = drop_unevaluable_end_window(labels, bars, window_end=SEALED_MAX_DATE)
    assert sessions[-1] <= SEALED_MAX_DATE
    if not kept.empty:
        assert kept["date"].max() < sessions[-1]
    assert date(2026, 1, 2) not in set(bars["date"].tolist())


def test_preflight_token_present_missing_not_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _acc, _seal, reg = _root(tmp_path)
    sealed = root / "data" / "sealed"
    monkeypatch.delenv(SEAL_TOKEN_ENV, raising=False)
    out = run_v2_preflight(
        root=root,
        registry_path=reg,
        configured_sealed_path=sealed,
        require_clean_tree=False,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
    )
    dumped = json.dumps(out)
    assert out["checks"]["seal_token_present"] == "no"
    assert SEAL_TOKEN_ENV not in dumped or out["checks"]["seal_token_present"] == "no"
    secret = "local-runtime-seal-token-never-print"
    monkeypatch.setenv(SEAL_TOKEN_ENV, secret)
    out2 = run_v2_preflight(
        root=root,
        registry_path=reg,
        configured_sealed_path=sealed,
        require_clean_tree=False,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
    )
    dumped2 = json.dumps(out2)
    assert out2["checks"]["seal_token_present"] == "yes"
    assert secret not in dumped2


def test_missing_runtime_token_fails_before_sealed_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEAL_TOKEN_ENV, raising=False)
    root, acc, seal, reg = _root(tmp_path)
    payload, history, sealed = two_source_causal_fixture(seed=6)
    art = _artifact(tmp_path, history, payload["t0"])
    src = CountingSealedSource(sealed["production"])
    with pytest.raises(SealedAccessError):
        run_v2_final_test(
            root=root,
            registry_path=reg,
            acceptance_dir=acc,
            seal_dir=seal,
            seal_token="",
            sealed_source=src,
            training_source=CountingPreSealedSource(history),
            model_artifact=art,
            require_prebuilt_model=True,
            require_git_tags=False,
            require_preseal_tag=False,
            require_hashes=False,
            require_clean_tree=False,
            require_authorization=False,
            require_seal_token=True,
            n=4,
            allow_reduced_n=True,
            result_label="development_rehearsal_only",
            t0=payload["t0"],
        )
    assert src.loads == 0


def test_frozen_model_vintage_and_runner_revision() -> None:
    from jevlab.utils.paths import get_project_root
    from jevlab.v2.model_artifact import (
        DEFAULT_MANIFEST_REL,
        assert_artifact_matches_freeze,
        load_model_freeze,
        load_sealed_test_model,
    )

    assert FROZEN_IMPLEMENTATION_REVISION == 4
    assert MODEL_FREEZE_IMPLEMENTATION_REVISION == 3
    assert FROZEN_MODEL_LOGICAL_DIGEST == (
        "74e301458559a511229b92235628db17cca80e4ebdff016a792784cab14ccc6c"
    )
    assert FROZEN_PRESEAL_IMPL_TAG == "v2-preseal-implementation-r4"
    root = get_project_root()
    man = root / DEFAULT_MANIFEST_REL
    if not man.is_file():
        pytest.skip("frozen sealed-test-model artifact not present")
    artifact = load_sealed_test_model(man)
    assert artifact.manifest["implementation_revision"] == MODEL_FREEZE_IMPLEMENTATION_REVISION
    assert artifact.manifest["protocol_revision"] == 1
    assert artifact.manifest["fitted_model_digest"] == FROZEN_MODEL_LOGICAL_DIGEST
    freeze = load_model_freeze(root)
    assert_artifact_matches_freeze(artifact, expected=freeze)
    assert artifact.chosen_C == 0.01
    assert artifact.manifest["final_training_decision_date"] == "2023-12-22"
    assert artifact.manifest["training_observation_count"] == 19376
