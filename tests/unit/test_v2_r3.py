"""Implementation-r3: prebuilt model, two sources, auth phrase, causal features."""

from __future__ import annotations

import ast
import inspect
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevlab.cv.sealed_guard import SEAL_TOKEN_ENV
from jevlab.data.availability import assign_bars_available_at, assign_fred_available_at
from jevlab.jev.preconditions import PreconditionError
from jevlab.targets.labels import build_labels
from jevlab.utils.calendar import nyse_session_dates
from jevlab.utils.registry import init_registry
from jevlab.v2.causal import build_causal_score_features, combine_history_and_sealed_bars
from jevlab.v2.dev_io import SEALED_MAX_DATE
from jevlab.v2.evaluate import score_sealed_study
from jevlab.v2.hashes import (
    FROZEN_ACCEPTANCE_GATE_HASH,
    FROZEN_CANDIDATE_HASH,
    FROZEN_PRESEAL_IMPL_TAG,
    FROZEN_PRESEAL_IMPL_TAG_R1,
    FROZEN_PRESEAL_IMPL_TAG_R2,
    FROZEN_RANDOM_NULL_HASH,
    FROZEN_RESEARCH_CONFIG_HASH,
    FROZEN_SEED_MAPPING_DIGEST,
    assert_frozen_research_hashes,
)
from jevlab.v2.model_artifact import (
    fit_presealed_model,
    load_sealed_test_model,
    write_sealed_test_model,
)
from jevlab.v2.rehearsal import split_preseal_and_score_payload, synthetic_pre2024_payload
from jevlab.v2.sealed_run import (
    REQUIRED_AUTHORIZATION_PHRASE,
    VOID_AUTHORIZATION_PHRASES,
    assert_fresh_r3_authorization,
    run_v2_preflight,
)
from jevlab.v2.sealed_source import CountingPreSealedSource, ProductionSealedSource
from jevlab.v2.training import EXPECTED_CUTOFF, logical_model_digest, preprocessing_logical_digest


def test_research_hashes_unchanged() -> None:
    live = assert_frozen_research_hashes()
    assert live["research_config_hash"] == FROZEN_RESEARCH_CONFIG_HASH
    assert live["candidate"] == FROZEN_CANDIDATE_HASH
    assert live["random_null"] == FROZEN_RANDOM_NULL_HASH
    assert live["acceptance_gates"] == FROZEN_ACCEPTANCE_GATE_HASH
    assert FROZEN_SEED_MAPPING_DIGEST == (
        "5c773811984b2fea232889295560f3ded3e6db79ca8e949ee5518933f88e58f9"
    )


def test_r3_authorization_exact_match_and_rejections() -> None:
    assert_fresh_r3_authorization(REQUIRED_AUTHORIZATION_PHRASE)
    assert REQUIRED_AUTHORIZATION_PHRASE == "OPEN V2 R1 IMPLEMENTATION R3 SEALED HOLDOUT ONCE"
    with pytest.raises(PreconditionError, match="VOID"):
        assert_fresh_r3_authorization("OPEN V2 REVISION 1 SEALED HOLDOUT ONCE")
    with pytest.raises(PreconditionError, match="VOID"):
        assert_fresh_r3_authorization(
            "OPEN V2 REVISION 1 SEALED HOLDOUT ONCE WITH IMPLEMENTATION R2"
        )
    with pytest.raises(PreconditionError, match="VOID"):
        assert_fresh_r3_authorization("OPEN V2 R1 IMPLEMENTATION R2 SEALED HOLDOUT ONCE")
    with pytest.raises(PreconditionError, match="fresh"):
        assert_fresh_r3_authorization("OPEN V2 R1 IMPLEMENTATION R3 SEALED HOLDOUT ONCE!")
    with pytest.raises(PreconditionError, match="fresh"):
        assert_fresh_r3_authorization("open v2 r1 implementation r3 sealed holdout once")
    with pytest.raises(PreconditionError, match="other than"):
        assert_fresh_r3_authorization(
            REQUIRED_AUTHORIZATION_PHRASE,
            implementation_tag=FROZEN_PRESEAL_IMPL_TAG_R2,
        )
    with pytest.raises(PreconditionError, match="other than"):
        assert_fresh_r3_authorization(
            REQUIRED_AUTHORIZATION_PHRASE,
            implementation_tag=FROZEN_PRESEAL_IMPL_TAG_R1,
        )
    assert_fresh_r3_authorization(
        REQUIRED_AUTHORIZATION_PHRASE,
        implementation_tag=FROZEN_PRESEAL_IMPL_TAG,
    )
    for phrase in VOID_AUTHORIZATION_PHRASES:
        with pytest.raises(PreconditionError, match="VOID"):
            assert_fresh_r3_authorization(phrase)


def test_score_sealed_study_contains_no_fit_call() -> None:
    source = inspect.getsource(score_sealed_study)
    assert "fit_once_model_a" not in source
    assert ".fit(" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "fit":
            raise AssertionError("score_sealed_study contains a .fit attribute")


def test_sealed_runner_source_does_not_fit_after_load() -> None:
    from jevlab.v2 import sealed_run

    source = inspect.getsource(sealed_run.run_v2_final_test)
    assert "fit_once_model_a" not in source
    assert "evaluate_v2_study" not in source
    assert "score_sealed_study" in source


def test_final_training_date_and_no_sealed_years() -> None:
    payload = synthetic_pre2024_payload(n_sessions=60, n_symbols=4, seed=2)
    history, _sealed = split_preseal_and_score_payload(payload)
    years = pd.to_datetime(history["features"]["date"]).dt.year
    assert int((years == 2024).sum()) == 0
    assert int((years == 2025).sum()) == 0
    fitted, meta = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=payload["t0"],
    )
    assert fitted.fit_count == 1
    assert meta["n_2024_train"] == 0
    assert meta["n_2025_train"] == 0
    assert fitted.cutoff == meta["cutoff"]
    assert date(2023, 12, 22) == EXPECTED_CUTOFF


def test_sealed_test_model_deterministic_rebuild(tmp_path: Path) -> None:
    payload = synthetic_pre2024_payload(n_sessions=56, n_symbols=4, seed=4)
    history, _ = split_preseal_and_score_payload(payload)
    a1, m1 = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=payload["t0"],
        source_manifest_digest="d",
    )
    a2, m2 = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=payload["t0"],
        source_manifest_digest="d",
    )
    assert a1.C == a2.C
    assert m1["training_row_digest"] == m2["training_row_digest"]
    assert m1["preprocessing_digest"] == m2["preprocessing_digest"]
    assert m1["model_logical_digest"] == m2["model_logical_digest"]
    assert logical_model_digest(a1) == logical_model_digest(a2)
    assert preprocessing_logical_digest(a1) == preprocessing_logical_digest(a2)
    coef1 = np.asarray(a1.pipeline.named_steps["clf"].coef_, dtype=float)
    coef2 = np.asarray(a2.pipeline.named_steps["clf"].coef_, dtype=float)
    assert np.max(np.abs(coef1 - coef2)) <= 1e-12
    art = write_sealed_test_model(fitted=a1, meta=m1, out_dir=tmp_path, git_commit="t")
    loaded = load_sealed_test_model(art.path)
    assert loaded.manifest["artifact_kind"] == "v2_sealed_test_model"
    assert loaded.manifest["protocol_revision"] == 1
    assert loaded.manifest["implementation_revision"] == 3
    assert loaded.manifest["contains_2024_2025"] is False
    assert loaded.fitted.C == a1.C
    assert logical_model_digest(loaded.fitted) == m1["model_logical_digest"]


def test_preflight_zero_sealed_reads_allows_training_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEAL_TOKEN_ENV, raising=False)
    root = tmp_path / "proj"
    sealed = root / "data" / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "year=2024").mkdir()
    (root / "registry").mkdir()
    (root / "reports" / "acceptance").mkdir(parents=True)
    (root / "reports" / "v2_sealed").mkdir(parents=True)
    init_registry(root / "registry" / "experiments.sqlite")
    payload = synthetic_pre2024_payload(n_sessions=48, n_symbols=4, seed=8)
    history, _ = split_preseal_and_score_payload(payload)
    fitted, meta = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=payload["t0"],
        source_manifest_digest="p",
    )
    art = write_sealed_test_model(
        fitted=fitted, meta=meta, out_dir=tmp_path / "model", git_commit="t"
    )
    train = CountingPreSealedSource(history)
    src = ProductionSealedSource(sealed)

    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("sealed glob/read during preflight")

    monkeypatch.setattr("pandas.read_parquet", boom)
    out = run_v2_preflight(
        root=root,
        registry_path=root / "registry" / "experiments.sqlite",
        configured_sealed_path=sealed,
        require_clean_tree=False,
        require_git_tags=False,
        require_preseal_tag=False,
        require_hashes=False,
        training_source=train,
        model_artifact_path=art.path,
        require_prebuilt_model=True,
        model_freeze={
            "fitted_model_digest": art.manifest["fitted_model_digest"],
            "preprocessing_digest": art.manifest["preprocessing_digest"],
            "training_row_digest": art.manifest["training_row_digest"],
            "chosen_C": art.manifest["chosen_C"],
            "training_source_manifest_digest": art.manifest["training_source_manifest_digest"],
        },
    )
    assert out["preflight"] is True
    assert out["sealed_underlying_reads"] == 0
    assert out["training_context_reads"] >= 1
    assert src.filesystem_reads == 0
    assert train.sealed_underlying_reads == 0


def test_model_freeze_mismatch_fails_closed(tmp_path: Path) -> None:
    payload = synthetic_pre2024_payload(n_sessions=48, n_symbols=4, seed=1)
    history, _ = split_preseal_and_score_payload(payload)
    fitted, meta = fit_presealed_model(
        features=history["features"],
        labels=history["labels"],
        bars=history["bars"],
        t0=payload["t0"],
    )
    art = write_sealed_test_model(fitted=fitted, meta=meta, out_dir=tmp_path, git_commit="t")
    from jevlab.v2.training import TrainingLeakError

    with pytest.raises(TrainingLeakError, match="freeze mismatch"):
        from jevlab.v2.model_artifact import assert_artifact_matches_freeze

        assert_artifact_matches_freeze(art, expected={"chosen_C": 999.0})


def test_end_of_2025_does_not_use_2026() -> None:
    sessions = [d for d in nyse_session_dates(date(2025, 12, 1), date(2025, 12, 31))]
    assert sessions[-1] <= SEALED_MAX_DATE
    rows = []
    elig_rows = []
    px = 100.0
    for d in sessions:
        px *= 1.001
        rows.append(
            {
                "symbol": "SPY",
                "date": d,
                "adj_open": px,
                "adj_close": px * 1.001,
            }
        )
        elig_rows.append({"symbol": "SPY", "date": d, "eligible": 1})
    bars = pd.DataFrame(rows)
    labels = build_labels(bars, eligible=pd.DataFrame(elig_rows), universe_symbols=["SPY"])
    labeled = set(pd.to_datetime(labels["date"]).dt.date.tolist()) if not labels.empty else set()
    assert all(d.year == 2025 for d in labeled)
    assert sessions[-1] not in labeled
    assert date(2026, 1, 2) not in set(bars["date"].tolist())


def test_causal_future_sealed_bar_does_not_change_earlier_feature() -> None:
    sessions = nyse_session_dates(date(2021, 1, 4), date(2022, 3, 31))[:260]
    symbols = ["SPY", "QQQ"]
    t0 = sessions[210]
    rng = np.random.default_rng(3)
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
    score_dates = [t0, sessions[211]]
    feat1 = build_causal_score_features(
        history_bars=history,
        sealed_bars=sealed,
        vix=vix,
        symbols=symbols,
        score_sessions=score_dates,
    )
    mutated = sealed.copy()
    last = mutated["date"].max()
    mutated.loc[mutated["date"] == last, "adj_close"] *= 50.0
    mutated.loc[mutated["date"] == last, "raw_close"] *= 50.0
    feat2 = build_causal_score_features(
        history_bars=history,
        sealed_bars=mutated,
        vix=vix,
        symbols=symbols,
        score_sessions=score_dates,
    )
    early = feat1.loc[feat1["date"] == t0].sort_values("symbol").reset_index(drop=True)
    early2 = feat2.loc[feat2["date"] == t0].sort_values("symbol").reset_index(drop=True)
    cols = [c for c in early.columns if c not in {"decision_time"}]
    pd.testing.assert_frame_equal(early.loc[:, cols], early2.loc[:, cols], check_dtype=False)
    combined = combine_history_and_sealed_bars(history, sealed)
    assert combined["date"].max() >= t0
    assert (history["date"] < t0).all()
