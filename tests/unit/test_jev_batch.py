"""Phase 7 bulk inference: resume, dedupe, budget, sealed, hash, reconcile."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from jevlab.jev.audit import BudgetExceededError, BudgetGuard
from jevlab.jev.batch_runner import (
    CONTAMINATION_TAG,
    SEALED_END,
    SEALED_START,
    UPPER_BOUND_STATES,
    BatchCheckpoint,
    ModelResolvedMismatchError,
    SealedBulkError,
    WorkItem,
    assert_no_sealed_dates,
    build_jev_features_table,
    dedupe_work_items,
    enumerate_feature_rows,
    interim_end_date,
    plan_batch,
    reconcile_row_counts,
    run_bulk_inference,
    write_jev_features,
)
from jevlab.jev.cache import JevCache, request_hash
from jevlab.jev.preconditions import (
    PreconditionError,
    assert_bulk_preconditions,
    parse_disposition_decision,
)
from jevlab.jev.qset import QSetHashMismatchError, load_qset
from jevlab.jev.schemas import SystemOneResponse
from jevlab.jev.state_builder import build_state_from_percentiles
from jevlab.jev.vocabulary import DEFAULT_BUCKET_EDGES, FIELD_PHRASE_BANK
from jevlab.utils.paths import get_project_root

PINNED = "jev-1.13.0"
ROUTE = "systemone"


def _mid_state() -> dict[str, str]:
    mid = 0.5
    percentiles = {
        "px_vs_sma200": mid,
        "px_vs_sma50": mid,
        "px_vs_sma20": mid,
        "ret_1d": mid,
        "ret_3d": mid,
        "ret_5d": mid,
        "ret_20d": mid,
        "rsi_3": mid,
        "rsi_14": mid,
        "rv_pct_252": mid,
        "vol_ratio_20": mid,
        "rel_ret_5d": mid,
        "spy_ret_5d": mid,
        "spy_px_vs_sma200": mid,
        "vix_lag1_pct_252": mid,
    }
    return build_state_from_percentiles(percentiles, down_streak=2, edges=DEFAULT_BUCKET_EDGES)


def _alt_state() -> dict[str, str]:
    # Different bucket → different request_hash
    high = 0.9
    percentiles = {
        "px_vs_sma200": high,
        "px_vs_sma50": high,
        "px_vs_sma20": high,
        "ret_1d": high,
        "ret_3d": high,
        "ret_5d": high,
        "ret_20d": high,
        "rsi_3": high,
        "rsi_14": high,
        "rv_pct_252": high,
        "vol_ratio_20": high,
        "rel_ret_5d": high,
        "spy_ret_5d": high,
        "spy_px_vs_sma200": high,
        "vix_lag1_pct_252": high,
    }
    return build_state_from_percentiles(percentiles, down_streak=0, edges=DEFAULT_BUCKET_EDGES)


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
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.2, "4": 0.2},
                    "legend": {str(i): str(i) for i in range(5)},
                    "confidence": 0.5,
                },
                "setup_type_a": {
                    "type": "choice",
                    "choice": "pullback_in_uptrend",
                    "probabilities": {
                        "pullback_in_uptrend": 0.6,
                        "trend_breakdown": 0.2,
                        "sideways_noise": 0.1,
                        "extended_rally_pause": 0.1,
                    },
                    "confidence": 0.5,
                },
                "setup_type_b": {
                    "type": "choice",
                    "choice": "trend_breakdown",
                    "probabilities": {
                        "pullback_in_uptrend": 0.4,
                        "trend_breakdown": 0.4,
                        "sideways_noise": 0.1,
                        "extended_rally_pause": 0.1,
                    },
                    "confidence": 0.2,
                },
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
    )


@pytest.fixture
def qset() -> Any:
    root = get_project_root()
    return load_qset(root / "config" / "qset_v1.json", expected_purpose="research")


def _item(
    qset: Any,
    *,
    symbol: str,
    day: date,
    state: dict[str, str],
) -> WorkItem:
    rh = request_hash(
        route=ROUTE,
        model_requested=PINNED,
        state=state,
        questions=qset.to_cacheable_dict(),
    )
    return WorkItem(symbol=symbol, date=day, state=state, request_hash=rh)


class CountingClient:
    def __init__(self, *, fail_after: int | None = None, model: str = PINNED) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.model = model
        self.seen_hashes: list[str] = []

    def infer(
        self,
        state: dict[str, str],
        questions: Any,
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        _ = questions, model
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated kill")
        return _mock_response(self.model)


class SelectiveFailClient:
    """Succeed for some states; fail for others (missing stay missing)."""

    def __init__(self, fail_phrases: set[str]) -> None:
        self.calls = 0
        self.fail_phrases = fail_phrases

    def infer(
        self,
        state: dict[str, str],
        questions: Any,
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        _ = questions, model
        self.calls += 1
        phrase = state.get("one_day_return", "")
        if phrase in self.fail_phrases:
            raise RuntimeError("forced miss")
        return _mock_response()


def test_sealed_dates_refused() -> None:
    with pytest.raises(SealedBulkError):
        assert_no_sealed_dates([date(2023, 12, 29), date(2024, 1, 2)])
    assert_no_sealed_dates([date(2023, 12, 29), date(2026, 1, 5)])
    assert date(2024, 1, 2) == SEALED_START
    assert date(2025, 12, 31) == SEALED_END


def test_enumerate_excludes_sealed() -> None:
    rows = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "date": [
                date(2023, 12, 28),
                date(2024, 6, 3),
                date(2025, 1, 2),
                date(2026, 1, 5),
            ],
            "ret_1d": [0.01, 0.01, 0.01, 0.01],
        }
    )
    out = enumerate_feature_rows(
        rows,
        forward_start=None,
        latest_completed_session=date(2026, 1, 10),
    )
    dates = set(out["date"].tolist())
    assert date(2023, 12, 28) in dates
    assert date(2026, 1, 5) in dates
    assert date(2024, 6, 3) not in dates
    assert date(2025, 1, 2) not in dates


def test_interim_end_uses_forward_start() -> None:
    # 2026-03-02 is a Monday; previous session is 2026-02-27
    end = interim_end_date(
        forward_start=date(2026, 3, 2),
        latest_completed_session=date(2026, 6, 1),
    )
    assert end is not None
    assert end < date(2026, 3, 2)
    end2 = interim_end_date(
        forward_start=None,
        latest_completed_session=date(2026, 2, 10),
    )
    assert end2 == date(2026, 2, 10)


def test_dedupe_by_request_hash(qset: Any) -> None:
    s = _mid_state()
    a = _item(qset, symbol="SPY", day=date(2020, 1, 2), state=s)
    b = _item(qset, symbol="QQQ", day=date(2020, 1, 3), state=s)  # same state → same hash
    c = _item(qset, symbol="IWM", day=date(2020, 1, 6), state=_alt_state())
    assert a.request_hash == b.request_hash
    assert a.request_hash != c.request_hash
    uniq = dedupe_work_items([a, b, c])
    assert len(uniq) == 2


def test_budget_guard_aborts_projected_spend(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    items = [
        _item(qset, symbol="SPY", day=date(2020, 1, 2), state=_mid_state()),
        _item(qset, symbol="QQQ", day=date(2020, 1, 3), state=_alt_state()),
    ]
    with pytest.raises(BudgetExceededError):
        plan_batch(items, cache, budget=BudgetGuard(budget_usd=0.0000001, tokens_per_call=800))
    plan = plan_batch(items, cache, budget=BudgetGuard(budget_usd=25.0))
    assert plan.n_to_call == 2
    assert plan.estimated_cost_usd > 0
    assert plan.upper_bound_states == UPPER_BOUND_STATES


def test_wrong_hash_refusal(tmp_path: Path) -> None:
    root = get_project_root()
    qset_src = root / "config" / "qset_v1.json"
    qset_copy = tmp_path / "qset_v1.json"
    qset_copy.write_text(qset_src.read_text(encoding="utf-8"), encoding="utf-8")
    disp = tmp_path / "contamination_disposition.md"
    disp.write_text(
        "- [x] `proceed`\n- [ ] `proceed_with_caveats`\n- [ ] `stop`\n",
        encoding="utf-8",
    )
    with pytest.raises((PreconditionError, QSetHashMismatchError)):
        assert_bulk_preconditions(
            root=tmp_path,
            qset_path=qset_copy,
            frozen_hash="0" * 64,
            disposition_path=disp,
            require_git_tags=False,
            require_clean_tree=False,
        )


def test_disposition_stop_blocks(tmp_path: Path) -> None:
    root = get_project_root()
    qset = load_qset(root / "config" / "qset_v1.json")
    disp = tmp_path / "contamination_disposition.md"
    disp.write_text(
        "- [ ] `proceed`\n- [ ] `proceed_with_caveats`\n- [x] `stop`\n",
        encoding="utf-8",
    )
    assert parse_disposition_decision(disp.read_text(encoding="utf-8")) == "stop"
    with pytest.raises(PreconditionError, match="stop"):
        assert_bulk_preconditions(
            root=tmp_path,
            qset_path=root / "config" / "qset_v1.json",
            frozen_hash=qset.content_hash,
            disposition_path=disp,
            require_git_tags=False,
            require_clean_tree=False,
        )


def test_kill_and_resume(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    ck = tmp_path / "ckpt.sqlite"
    out = tmp_path / "jev_out"
    items = [
        _item(qset, symbol="SPY", day=date(2020, 1, 2), state=_mid_state()),
        _item(qset, symbol="QQQ", day=date(2020, 1, 3), state=_alt_state()),
        _item(
            qset,
            symbol="IWM",
            day=date(2020, 1, 6),
            state=build_state_from_percentiles(
                {
                    k: 0.1
                    for k in [
                        "px_vs_sma200",
                        "px_vs_sma50",
                        "px_vs_sma20",
                        "ret_1d",
                        "ret_3d",
                        "ret_5d",
                        "ret_20d",
                        "rsi_3",
                        "rsi_14",
                        "rv_pct_252",
                        "vol_ratio_20",
                        "rel_ret_5d",
                        "spy_ret_5d",
                        "spy_px_vs_sma200",
                        "vix_lag1_pct_252",
                    ]
                },
                down_streak=1,
            ),
        ),
    ]

    class KillAfterOne:
        def __init__(self) -> None:
            self.calls = 0

        def infer(
            self,
            state: dict[str, str],
            questions: Any,
            *,
            model: str | None = None,
        ) -> SystemOneResponse:
            _ = state, questions, model
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("simulated kill")
            return _mock_response()

    killer = KillAfterOne()
    # Errors are recorded in the checkpoint (never imputed); run continues.
    run_bulk_inference(
        pd.DataFrame(),
        qset=qset,
        cache=cache,
        client=killer,
        route=ROUTE,
        model_id=PINNED,
        state_schema_version="state_v1",
        bucket_edges=DEFAULT_BUCKET_EDGES,
        out_dir=out,
        checkpoint_path=ck,
        dry_run=False,
        concurrency=1,
        work_items=items,
    )
    ckpt = BatchCheckpoint(ck)
    done_before = ckpt.completed_hashes()
    assert len(done_before) == 1
    assert killer.calls == 3

    client2 = CountingClient()
    result2 = run_bulk_inference(
        pd.DataFrame(),
        qset=qset,
        cache=cache,
        client=client2,
        route=ROUTE,
        model_id=PINNED,
        state_schema_version="state_v1",
        bucket_edges=DEFAULT_BUCKET_EDGES,
        out_dir=out,
        checkpoint_path=ck,
        dry_run=False,
        concurrency=1,
        work_items=items,
    )
    # Resume must not re-call completed hashes
    assert client2.calls == len(items) - len(done_before)
    assert result2.meta["reconciliation"]["reconciled"] is True
    assert result2.n_missing == 0


def test_dedupe_single_api_call(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    s = _mid_state()
    items = [
        _item(qset, symbol="SPY", day=date(2020, 1, 2), state=s),
        _item(qset, symbol="QQQ", day=date(2020, 1, 3), state=s),
    ]
    client = CountingClient()
    result = run_bulk_inference(
        pd.DataFrame(),
        qset=qset,
        cache=cache,
        client=client,
        route=ROUTE,
        model_id=PINNED,
        state_schema_version="state_v1",
        bucket_edges=DEFAULT_BUCKET_EDGES,
        out_dir=tmp_path / "out",
        checkpoint_path=tmp_path / "ck.sqlite",
        dry_run=False,
        concurrency=1,
        work_items=items,
    )
    assert client.calls == 1
    assert result.plan.n_unique_hashes == 1
    assert result.plan.n_total == 2
    # Both rows get features via shared hash
    assert result.meta["reconciliation"]["features_rows"] == 2
    assert result.n_missing == 0


def test_missing_responses_stay_missing(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    ok_state = _mid_state()
    bad_state = _alt_state()
    # SelectiveFailClient fails when one_day_return matches fail set
    fail_phrase = bad_state["one_day_return"]
    items = [
        _item(qset, symbol="SPY", day=date(2020, 1, 2), state=ok_state),
        _item(qset, symbol="QQQ", day=date(2020, 1, 3), state=bad_state),
    ]
    client = SelectiveFailClient({fail_phrase})
    result = run_bulk_inference(
        pd.DataFrame(),
        qset=qset,
        cache=cache,
        client=client,
        route=ROUTE,
        model_id=PINNED,
        state_schema_version="state_v1",
        bucket_edges=DEFAULT_BUCKET_EDGES,
        out_dir=tmp_path / "out",
        checkpoint_path=tmp_path / "ck.sqlite",
        dry_run=False,
        concurrency=1,
        work_items=items,
    )
    assert result.n_missing == 1
    assert result.meta["reconciliation"]["features_rows"] == 1
    assert result.meta["reconciliation"]["reconciled"] is True
    # No imputed row for the failed call
    feat = pd.read_parquet(result.features_path)
    assert len(feat) == 1
    assert feat.iloc[0]["symbol"] == "SPY"


def test_row_count_reconciliation_helper() -> None:
    r = reconcile_row_counts(100, 99, n_missing=1)
    assert r["reconciled"] is True
    bad = reconcile_row_counts(100, 99, n_missing=0)
    assert bad["reconciled"] is False


def test_model_resolved_mismatch_aborts(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    items = [_item(qset, symbol="SPY", day=date(2020, 1, 2), state=_mid_state())]
    client = CountingClient(model="jev-wrong")
    with pytest.raises(ModelResolvedMismatchError):
        run_bulk_inference(
            pd.DataFrame(),
            qset=qset,
            cache=cache,
            client=client,
            route=ROUTE,
            model_id=PINNED,
            state_schema_version="state_v1",
            bucket_edges=DEFAULT_BUCKET_EDGES,
            out_dir=tmp_path / "out",
            checkpoint_path=tmp_path / "ck.sqlite",
            dry_run=False,
            concurrency=1,
            work_items=items,
        )


def test_metadata_contamination_tag(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    items = [_item(qset, symbol="SPY", day=date(2020, 1, 2), state=_mid_state())]
    result = run_bulk_inference(
        pd.DataFrame(),
        qset=qset,
        cache=cache,
        client=CountingClient(),
        route=ROUTE,
        model_id=PINNED,
        state_schema_version="state_v1",
        bucket_edges=DEFAULT_BUCKET_EDGES,
        out_dir=tmp_path / "out",
        checkpoint_path=tmp_path / "ck.sqlite",
        dry_run=False,
        concurrency=1,
        work_items=items,
    )
    meta = (tmp_path / "out" / "metadata.json").read_text(encoding="utf-8")
    assert CONTAMINATION_TAG in meta
    assert result.meta["contamination_tag"] == CONTAMINATION_TAG


def test_dry_run_zero_calls_and_prints_plan(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    items = [
        _item(qset, symbol="SPY", day=date(2020, 1, 2), state=_mid_state()),
        _item(qset, symbol="QQQ", day=date(2020, 1, 3), state=_alt_state()),
    ]
    client = CountingClient()
    result = run_bulk_inference(
        pd.DataFrame(),
        qset=qset,
        cache=cache,
        client=client,
        route=ROUTE,
        model_id=PINNED,
        state_schema_version="state_v1",
        bucket_edges=DEFAULT_BUCKET_EDGES,
        out_dir=tmp_path / "out",
        checkpoint_path=tmp_path / "ck.sqlite",
        dry_run=True,
        work_items=items,
    )
    assert client.calls == 0
    assert result.meta["api_calls"] == 0
    assert result.plan.n_to_call == 2
    assert result.plan.upper_bound_states == UPPER_BOUND_STATES
    d = result.plan.to_dict()
    assert d["estimated_cost_usd"] > 0
    assert d["contamination_tag"] == CONTAMINATION_TAG


def test_skip_cache_hits(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    item = _item(qset, symbol="SPY", day=date(2020, 1, 2), state=_mid_state())
    cache.put(
        req_hash=item.request_hash,
        route=ROUTE,
        model_requested=PINNED,
        model_resolved=PINNED,
        purpose="research",
        qset_version=qset.qset_version,
        state_schema_version="state_v1",
        response=_mock_response().model_dump(mode="json"),
    )
    plan = plan_batch([item], cache)
    assert plan.n_cache_hits == 1
    assert plan.n_to_call == 0
    client = CountingClient()
    result = run_bulk_inference(
        pd.DataFrame(),
        qset=qset,
        cache=cache,
        client=client,
        route=ROUTE,
        model_id=PINNED,
        state_schema_version="state_v1",
        bucket_edges=DEFAULT_BUCKET_EDGES,
        out_dir=tmp_path / "out",
        checkpoint_path=tmp_path / "ck.sqlite",
        dry_run=False,
        concurrency=1,
        work_items=[item],
    )
    assert client.calls == 0
    assert result.n_missing == 0


def test_build_features_never_imputes(qset: Any, tmp_path: Path) -> None:
    cache = JevCache(tmp_path / "cache")
    items = [
        _item(qset, symbol="SPY", day=date(2020, 1, 2), state=_mid_state()),
        _item(qset, symbol="QQQ", day=date(2020, 1, 3), state=_alt_state()),
    ]
    # Only cache the first
    cache.put(
        req_hash=items[0].request_hash,
        route=ROUTE,
        model_requested=PINNED,
        model_resolved=PINNED,
        purpose="research",
        qset_version=qset.qset_version,
        state_schema_version="state_v1",
        response=_mock_response().model_dump(mode="json"),
    )
    feat = build_jev_features_table(
        items,
        cache=cache,
        qset_version=qset.qset_version,
        state_schema_version="state_v1",
        pinned_model=PINNED,
    )
    assert len(feat) == 1
    path = write_jev_features(feat, tmp_path / "out")
    assert path.exists()
    meta = (tmp_path / "out" / "metadata.json").read_text(encoding="utf-8")
    assert CONTAMINATION_TAG in meta


def test_upper_bound_cost_estimate_under_default_budget() -> None:
    """Spec ~28,600 states upper bound fits under default $25 at planning tokens."""
    guard = BudgetGuard(budget_usd=25.0, tokens_per_call=800)
    est = guard.estimate_usd(UPPER_BOUND_STATES)
    assert est < 25.0
    assert est == pytest.approx(UPPER_BOUND_STATES * 800 / 1_000_000 * 0.042)


def test_vocabulary_bank_used() -> None:
    # sanity: mid-state phrases come from the closed bank
    s = _mid_state()
    assert s["one_day_return"] in FIELD_PHRASE_BANK["one_day_return"]
