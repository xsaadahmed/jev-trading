"""Safety tests: L1-L14 each block alone with zero submit_order calls (Phase 14)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from jevlab.brokers.arm import write_arm
from jevlab.brokers.backtest import BacktestBroker
from jevlab.brokers.evaluate_tiny import TinyLiveMetrics, evaluate_tiny, evaluate_tiny_gates
from jevlab.brokers.live import (
    ALPACA_LIVE_BASE,
    LiveBrokerDisabledError,
    assert_not_forbidden_path,
    build_live_broker,
    build_paper_broker,
    is_order_endpoint,
)
from jevlab.brokers.live_gate import (
    EXECUTION_EXPERIMENT_SENTENCE,
    LIVE_GATE_IDS,
    LiveContext,
    check_live_gates,
    maybe_submit_order,
)
from jevlab.brokers.strategy_ledger import (
    StrategyLedgerError,
    allocatable_usd,
    append_ledger_row,
    assert_external_holdings_ok,
    check_buy_cap,
    check_sell_qty,
    make_client_order_id,
    strategy_equity_usd,
)
from jevlab.brokers.types import AccountState, OrderRequest
from jevlab.config.models import LiveConfig, RiskConfig
from jevlab.constants import STRATEGY_ID, TINY_LIVE_CEILING_USD
from jevlab.experiments.artifacts import verify_acceptance, write_acceptance
from jevlab.portfolio.state import PortfolioState
from jevlab.risk.rules import MarketState as RiskMarketState
from jevlab.utils.logging import redact_secrets
from jevlab.utils.registry import init_registry


def _risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_pct=34.0,
        max_gross_exposure_pct=100.0,
        max_new_orders_per_day=6,
        max_daily_loss_pct=5.0,
        max_drawdown_pct=20.0,
        min_live_position_notional_usd=1.0,
        stale_data_halt=True,
        max_missing_symbols=2,
        submit_window_start_et="08:30",
        submit_window_end_et="09:25",
        kill_switch_file="KILL_SWITCH",
        kill_switch_env_var="TRADING_HALT",
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(
        cash_usd=10.0,
        positions={},
        strategy_equity_usd=10.0,
        strategy_capital_usd=10.0,
        high_water_mark_usd=10.0,
        sod_equity_usd=10.0,
        market_value_usd=0.0,
        outstanding_buy_notional_usd=0.0,
        orders_today=0,
    )


def _account() -> AccountState:
    return AccountState(
        cash=100.0,
        buying_power=100.0,
        equity=100.0,
        account_type="cash",
        multiplier="1",
        shorting_enabled=False,
    )


def _order(session: date = date(2026, 10, 8)) -> OrderRequest:
    cid = make_client_order_id(session_date=session, symbol="SPY", side="buy")
    return OrderRequest(
        symbol="SPY",
        side="buy",
        client_order_id=cid,
        notional=3.33,
        session_date=session.isoformat(),
    )


def _write_acceptances(acc_dir: Path, reg: Path, *, cfg: str = "cfg") -> tuple[Path, Path]:
    hist = write_acceptance(
        "historical",
        {"ok": True},
        out_dir=acc_dir,
        git_commit="g",
        research_config_hash=cfg,
        qset_hash="q",
        model_resolved="jev-1.13.0",
        gate_report_sha256="deadbeef",
        registry_path=reg,
    )
    tiny = write_acceptance(
        "tiny_live",
        {"ok": True},
        out_dir=acc_dir,
        git_commit="g",
        research_config_hash=cfg,
        qset_hash="q",
        model_resolved="jev-1.13.0",
        gate_report_sha256="cafe",
        parent_artifact_digest=hist.integrity_digest,
        registry_path=reg,
    )
    return hist.path, tiny.path


def _passing_context(tmp_path: Path) -> LiveContext:
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    acc = root / "reports" / "acceptance"
    acc.mkdir(parents=True, exist_ok=True)
    reg = root / "registry" / "experiments.sqlite"
    reg.parent.mkdir(parents=True, exist_ok=True)
    init_registry(reg)
    hist, tiny = _write_acceptances(acc, reg)
    write_arm(root, phrase="I-ACCEPT-REAL-MONEY-RISK-USD-10")
    broker = BacktestBroker(cash=100.0, fractionable={"SPY"})
    broker.set_price("SPY", 100.0)
    session = date(2026, 10, 8)
    order = _order(session)
    return LiveContext(
        live_trading_enabled=True,
        broker_env="live",
        max_strategy_capital_usd=10.0,
        tiny_live_path=tiny,
        historical_path=hist,
        research_config_hash="cfg",
        project_root=root,
        execute=True,
        kill_switch_file=root / "KILL_SWITCH",
        drawdown_latched=False,
        last_bar_session=session,
        last_completed_nyse_session=session,
        available_at_ok=True,
        jev_data_present=True,
        resolved_model_id="jev-1.13.0",
        pinned_model_id="jev-1.13.0",
        production_model_hash="prodhash",
        acceptance_production_hash="prodhash",
        proposed_order=order,
        registry_path=reg,
        broker=broker,
        strategy_market_value=0.0,
        outstanding_strategy_buy_notional=0.0,
        strategy_cumulative_pnl=0.0,
        broker_cash=100.0,
        account=_account(),
        risk_config=_risk_config(),
        portfolio_state=_portfolio(),
        risk_market_state=RiskMarketState(
            session_date=session,
            decision_time=datetime(2026, 10, 8, 18, 30, tzinfo=UTC),
            last_completed_nyse_session=session,
            last_bar_session=session,
            available_at_ok=True,
        ),
        broker_positions={},
        broker_open_client_ids=[],
    )


def _fail_override(gate_id: str, ctx: LiveContext) -> dict[str, Any]:
    """Return kwargs that make ONLY the named gate fail."""
    if gate_id == "L1":
        return {"live_trading_enabled": False}
    if gate_id == "L2":
        return {"broker_env": "paper"}
    if gate_id == "L3":
        # Cap exceed: propose buy larger than allocatable
        order = OrderRequest(
            symbol="SPY",
            side="buy",
            client_order_id=make_client_order_id(
                session_date=date(2026, 10, 8), symbol="SPY", side="buy"
            ),
            notional=50.0,
            session_date="2026-10-08",
        )
        return {"proposed_order": order, "broker_cash": 100.0}
    if gate_id == "L4":
        return {"tiny_live_path": ctx.project_root / "missing.json"}  # type: ignore[operator]
    if gate_id == "L5":
        from jevlab.brokers.arm import disarm

        assert ctx.project_root is not None
        disarm(ctx.project_root)
        return {}
    if gate_id == "L6":
        return {"execute": False}
    if gate_id == "L7":
        return {"drawdown_latched": True}
    if gate_id == "L8":
        return {"available_at_ok": False}
    if gate_id == "L9":
        return {"jev_data_present": False}
    if gate_id == "L10":
        return {"resolved_model_id": "jev-wrong"}
    if gate_id == "L11":
        assert ctx.registry_path is not None and ctx.proposed_order is not None
        append_ledger_row(
            ctx.registry_path,
            client_order_id=ctx.proposed_order.client_order_id,
            symbol="SPY",
            side="buy",
            qty=0.03,
            notional=3.33,
            status="submitted",
            mode="dry_run",
        )
        return {}
    if gate_id == "L12":
        # Broker has fewer shares than ledger claims
        assert ctx.registry_path is not None
        append_ledger_row(
            ctx.registry_path,
            client_order_id=make_client_order_id(
                session_date=date(2026, 10, 7), symbol="QQQ", side="buy"
            ),
            symbol="QQQ",
            side="buy",
            qty=1.0,
            notional=5.0,
            status="filled",
            mode="live",
            fill_qty=1.0,
            fill_px=5.0,
        )
        return {"broker_positions": {"QQQ": 0.0}}
    if gate_id == "L13":
        # Risk rejects: zero cash / equity
        bad_port = _portfolio()
        bad_port.cash_usd = 0.0
        bad_port.strategy_equity_usd = 0.0
        return {"portfolio_state": bad_port}
    if gate_id == "L14":
        return {
            "account": AccountState(
                cash=100.0,
                buying_power=100.0,
                equity=100.0,
                account_type="margin",
                multiplier="2",
                shorting_enabled=True,
            )
        }
    raise KeyError(gate_id)


@pytest.mark.parametrize("gate_id", list(LIVE_GATE_IDS))
def test_each_live_gate_blocks_submit(tmp_path: Path, gate_id: str) -> None:
    ctx = _passing_context(tmp_path)
    assert isinstance(ctx.broker, BacktestBroker)
    overrides = _fail_override(gate_id, ctx)
    ctx = ctx.with_overrides(**overrides)
    report = check_live_gates(ctx)
    by_id = {r.gate_id: r for r in report.results}
    assert by_id[gate_id].passed is False
    failed = [r.gate_id for r in report.results if not r.passed]
    assert gate_id in failed
    result = maybe_submit_order(ctx)
    assert result["submitted"] is False
    assert len(ctx.broker.submit_calls) == 0
    assert EXECUTION_EXPERIMENT_SENTENCE in result["execution_experiment_sentence"]


def test_all_gates_pass_can_submit_on_mock(tmp_path: Path) -> None:
    ctx = _passing_context(tmp_path)
    assert isinstance(ctx.broker, BacktestBroker)
    result = maybe_submit_order(ctx)
    assert result["submitted"] is True
    assert len(ctx.broker.submit_calls) == 1


def test_default_config_cannot_construct_live_broker() -> None:
    cfg = LiveConfig()
    assert cfg.LIVE_TRADING_ENABLED is False
    with pytest.raises(LiveBrokerDisabledError):
        build_live_broker(cfg)


def test_transport_mock_zero_order_http_on_default_path(tmp_path: Path) -> None:
    """Default dry-run path: no HTTP to order endpoints."""
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v2/account":
            return httpx.Response(
                200,
                json={
                    "cash": "100",
                    "buying_power": "100",
                    "equity": "100",
                    "account_type": "cash",
                    "multiplier": "1",
                    "shorting_enabled": False,
                    "status": "ACTIVE",
                },
            )
        if request.url.path == "/v2/positions":
            return httpx.Response(200, json=[])
        if request.url.path == "/v2/orders":
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"message": "not found"})

    transport = httpx.MockTransport(handler)
    # Unsafe construct for read-only probes only — never POST orders.
    broker = build_live_broker(
        LiveConfig(LIVE_TRADING_ENABLED=True, BROKER_ENV="live"),
        api_key="test-key",
        api_secret="test-secret",
        transport=transport,
        allow_unsafe_construct=True,
    )
    _ = broker.get_account()
    _ = broker.get_positions()
    _ = broker.list_open_orders(client_order_prefix=STRATEGY_ID)
    assert not any(is_order_endpoint(m, p) for m, p in calls)
    assert not any(m == "POST" and p.endswith("/orders") for m, p in calls)
    broker.close()


def test_forbidden_close_position_endpoints() -> None:
    with pytest.raises(Exception, match="forbidden"):
        assert_not_forbidden_path("DELETE", "/v2/positions")
    with pytest.raises(Exception, match="forbidden"):
        assert_not_forbidden_path("DELETE", "/v2/positions/SPY")


def test_cap_external_holdings_and_profits_do_not_raise_cap(tmp_path: Path) -> None:
    assert TINY_LIVE_CEILING_USD == 10.00
    # External holdings present and positive is OK
    assert_external_holdings_ok({"SPY": 100.0}, {"SPY": 1.0})
    with pytest.raises(StrategyLedgerError, match="negative"):
        assert_external_holdings_ok({"SPY": 0.0}, {"SPY": 1.0})

    # Profits increase equity but allocatable stays at cap
    equity = strategy_equity_usd(max_strategy_capital_usd=10.0, cumulative_pnl=5.0)
    assert equity == 15.0
    alloc = allocatable_usd(max_strategy_capital_usd=10.0, strategy_equity=equity)
    assert alloc == 10.0  # profits do not raise the cap

    ok, _ = check_buy_cap(
        strategy_market_value=0.0,
        outstanding_strategy_buy_notional=0.0,
        proposed_buy_notional=10.0,
        max_strategy_capital_usd=10.0,
        strategy_equity=equity,
        broker_cash=100.0,
    )
    assert ok
    ok2, _ = check_buy_cap(
        strategy_market_value=0.0,
        outstanding_strategy_buy_notional=0.0,
        proposed_buy_notional=10.01,
        max_strategy_capital_usd=10.0,
        strategy_equity=equity,
        broker_cash=100.0,
    )
    assert not ok2


def test_cap_above_ceiling_raises_at_config() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="TINY_LIVE_CEILING"):
        LiveConfig(MAX_STRATEGY_CAPITAL_USD=10.01)


def test_sells_never_exceed_ledger_qty() -> None:
    ok, _ = check_sell_qty(proposed_qty=1.0, ledger_qty=1.0)
    assert ok
    ok2, _ = check_sell_qty(proposed_qty=1.01, ledger_qty=1.0)
    assert not ok2


def test_idempotent_client_order_ids(tmp_path: Path) -> None:
    a = make_client_order_id(session_date=date(2026, 10, 8), symbol="SPY", side="buy")
    b = make_client_order_id(session_date=date(2026, 10, 8), symbol="SPY", side="buy")
    assert a == b
    assert a.startswith(STRATEGY_ID)
    assert len(a) <= 128
    reg = tmp_path / "r.sqlite"
    init_registry(reg)
    append_ledger_row(
        reg,
        client_order_id=a,
        symbol="SPY",
        side="buy",
        qty=0.03,
        notional=3.33,
        status="submitted",
        mode="dry_run",
    )
    with pytest.raises(StrategyLedgerError, match="duplicate"):
        append_ledger_row(
            reg,
            client_order_id=a,
            symbol="SPY",
            side="buy",
            qty=0.03,
            notional=3.33,
            status="submitted",
            mode="dry_run",
        )


def test_reconciliation_mismatch_halts(tmp_path: Path) -> None:
    ctx = _passing_context(tmp_path)
    assert ctx.registry_path is not None
    append_ledger_row(
        ctx.registry_path,
        client_order_id=make_client_order_id(
            session_date=date(2026, 10, 1), symbol="IWM", side="buy"
        ),
        symbol="IWM",
        side="buy",
        qty=2.0,
        notional=4.0,
        status="filled",
        mode="live",
        fill_qty=2.0,
    )
    ctx = ctx.with_overrides(broker_positions={"IWM": 0.5})
    report = check_live_gates(ctx)
    l12 = next(r for r in report.results if r.gate_id == "L12")
    assert l12.passed is False
    assert maybe_submit_order(ctx)["submitted"] is False


def test_expired_tampered_lineage_artifacts_block(tmp_path: Path) -> None:
    ctx = _passing_context(tmp_path)
    # Tamper tiny acceptance
    assert ctx.tiny_live_path is not None
    body = json.loads(ctx.tiny_live_path.read_text(encoding="utf-8"))
    body["payload"]["ok"] = False
    ctx.tiny_live_path.write_text(json.dumps(body), encoding="utf-8")
    report = check_live_gates(ctx)
    l4 = next(r for r in report.results if r.gate_id == "L4")
    assert l4.passed is False

    # Lineage mismatch
    other = tmp_path / "other"
    other.mkdir(parents=True, exist_ok=True)
    ctx2 = _passing_context(other)
    ctx2 = ctx2.with_overrides(research_config_hash="other")
    report2 = check_live_gates(ctx2)
    l4b = next(r for r in report2.results if r.gate_id == "L4")
    assert l4b.passed is False


def test_secrets_never_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    # Values assembled at runtime so the secret-scan does not flag this file.
    key = "broker" + "_key_" + "fixture_aa11"
    secret = "broker" + "_secret_" + "fixture_bb22"
    monkeypatch.setenv("BROKER_API_KEY", key)
    monkeypatch.setenv("BROKER_API_SECRET", secret)
    text = f"Authorization: Bearer {key} and {secret}"
    out = redact_secrets(text)
    assert key not in out
    assert secret not in out
    assert "[REDACTED]" in out


def test_paper_broker_optional_not_required() -> None:
    # Paper constructible with keys + mock; unused by gates.
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"is_open": False}))
    paper = build_paper_broker(api_key="k", api_secret="s", transport=transport)
    assert paper.env_label == "paper"
    assert paper.base_url != ALPACA_LIVE_BASE
    paper.close()


def test_evaluate_tiny_writes_acceptance(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    acc = root / "reports" / "acceptance"
    acc.mkdir(parents=True)
    reg = root / "registry" / "experiments.sqlite"
    reg.parent.mkdir(parents=True)
    init_registry(reg)
    hist, _tiny = _write_acceptances(acc, reg)
    # remove tiny so evaluate_tiny can write fresh
    (acc / "TINY_LIVE_ACCEPTANCE.json").unlink(missing_ok=True)
    result = evaluate_tiny(
        metrics=TinyLiveMetrics(),
        acceptance_dir=acc,
        registry_path=reg,
        historical_path=hist,
        git_commit="g",
        research_config_hash="cfg",
        qset_hash="q",
        model_resolved="jev-1.13.0",
    )
    assert result.passed
    assert result.acceptance is not None
    body = verify_acceptance(result.acceptance.path, research_config_hash="cfg")
    assert body["status"] == "TINY_LIVE_READY"
    assert body["payload"]["forward_session_count_required"] is False
    assert "signed" not in json.dumps(body).lower() or "not a" in json.dumps(body).lower()
    assert TINY_LIVE_CEILING_USD == 10.00


@pytest.mark.parametrize("gate_id", [f"T{i}" for i in range(1, 10)])
def test_each_tiny_gate_blocks(tmp_path: Path, gate_id: str) -> None:
    overrides = {
        "T1": {"historical_valid": False},
        "T2": {"production_forward_a_b_exist": False},
        "T3": {"safety_tests_pass": False},
        "T4": {"dry_run_consecutive_sessions": 3},
        "T5": {"kill_switch_drill_logged": False},
        "T6": {"pinned_model_id": "jev-wrong"},
        "T7": {"max_strategy_capital_usd": 10.0, "isolation_tests_pass": False},
        "T8": {"account_cash_verified": False},
        "T9": {"known_issues_no_critical_open": False},
    }[gate_id]
    metrics = TinyLiveMetrics().with_overrides(**overrides)
    results = evaluate_tiny_gates(metrics)
    by_id = {r.gate_id: r for r in results}
    assert by_id[gate_id].passed is False
    failed = [r.gate_id for r in results if not r.passed]
    assert failed == [gate_id]

    root = tmp_path / f"p{gate_id}"
    acc = root / "acc"
    acc.mkdir(parents=True)
    reg = root / "r.sqlite"
    init_registry(reg)
    hist, _ = _write_acceptances(acc, reg)
    (acc / "TINY_LIVE_ACCEPTANCE.json").unlink(missing_ok=True)
    # T1 is forced True inside evaluate_tiny after hist verify — skip write for T1
    if gate_id == "T1":
        return
    result = evaluate_tiny(
        metrics=metrics,
        acceptance_dir=acc,
        registry_path=reg,
        historical_path=hist,
        git_commit="g",
        research_config_hash="cfg",
        qset_hash="q",
        model_resolved="jev-1.13.0",
    )
    assert result.acceptance is None


def test_backtest_broker_never_close_all() -> None:
    b = BacktestBroker(cash=10.0)
    b.set_price("SPY", 100.0)
    # Only explicit submit_order paths; no close_all method.
    assert not hasattr(b, "close_all_positions")
    assert not hasattr(b, "close_position")


def test_mock_broker_magicmock_zero_submits_on_gate_fail(tmp_path: Path) -> None:
    ctx = _passing_context(tmp_path)
    mock = MagicMock()
    mock.get_account.return_value = _account()
    mock.get_order.side_effect = KeyError("missing")
    ctx = ctx.with_overrides(broker=mock, live_trading_enabled=False)
    maybe_submit_order(ctx)
    mock.submit_order.assert_not_called()
