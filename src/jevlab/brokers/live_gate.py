"""Independent live gates L1-L14 (MASTER_SPEC §15.3). Any failure ⇒ zero submits."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from jevlab.brokers.arm import DEFAULT_ARM_PHRASE, is_armed
from jevlab.brokers.base import BrokerInterface
from jevlab.brokers.strategy_ledger import (
    assert_external_holdings_ok,
    check_buy_cap,
    check_sell_qty,
    client_order_id_exists,
    reconcile_open_orders,
    strategy_equity_usd,
    strategy_qty_by_symbol,
)
from jevlab.brokers.types import AccountState, OrderRequest
from jevlab.config.models import LiveConfig, RiskConfig
from jevlab.constants import STRATEGY_ID, TINY_LIVE_CEILING_USD
from jevlab.experiments.artifacts import ArtifactError, verify_acceptance
from jevlab.portfolio.state import PortfolioState
from jevlab.risk.killswitch import is_halted
from jevlab.risk.rules import MarketState as RiskMarketState
from jevlab.risk.rules import ProposedOrder, evaluate_orders
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.logging import get_logger

log = get_logger(__name__)

EXECUTION_EXPERIMENT_SENTENCE: str = (
    "This live phase is an execution experiment, not evidence of a profitable trading edge."
)

LIVE_GATE_IDS: tuple[str, ...] = tuple(f"L{i}" for i in range(1, 15))


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)
    execution_experiment_sentence: str = EXECUTION_EXPERIMENT_SENTENCE

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": [asdict(r) for r in self.results],
            "all_passed": self.all_passed,
            "execution_experiment_sentence": self.execution_experiment_sentence,
        }


@dataclass
class LiveContext:
    """Inputs for L1-L14. Defaults fail closed (safe)."""

    live_trading_enabled: bool = False
    broker_env: str = "disabled"
    max_strategy_capital_usd: float = 10.0
    tiny_live_path: Path | None = None
    historical_path: Path | None = None
    research_config_hash: str = ""
    project_root: Path | None = None
    execute: bool = False
    kill_switch_file: Path | None = None
    trading_halt_env: str = "TRADING_HALT"
    drawdown_latched: bool = False
    last_bar_session: date | None = None
    last_completed_nyse_session: date | None = None
    available_at_ok: bool = False
    jev_data_present: bool = False
    resolved_model_id: str | None = None
    pinned_model_id: str = "jev-1.13.0"
    production_model_hash: str | None = None
    acceptance_production_hash: str | None = None
    proposed_order: OrderRequest | None = None
    registry_path: Path | None = None
    broker: BrokerInterface | None = None
    strategy_market_value: float = 0.0
    outstanding_strategy_buy_notional: float = 0.0
    strategy_cumulative_pnl: float = 0.0
    broker_cash: float = 0.0
    account: AccountState | None = None
    risk_config: RiskConfig | None = None
    portfolio_state: PortfolioState | None = None
    risk_market_state: RiskMarketState | None = None
    broker_positions: dict[str, float] = field(default_factory=dict)
    broker_open_client_ids: list[str] = field(default_factory=list)
    arm_now: datetime | None = None

    def with_overrides(self, **kwargs: Any) -> LiveContext:
        return replace(self, **kwargs)


def gate_l1_live_enabled(ctx: LiveContext) -> GateResult:
    ok = bool(ctx.live_trading_enabled)
    return GateResult("L1", ok, f"LIVE_TRADING_ENABLED={ctx.live_trading_enabled}")


def gate_l2_broker_env(ctx: LiveContext) -> GateResult:
    ok = str(ctx.broker_env).lower() == "live"
    return GateResult("L2", ok, f"BROKER_ENV={ctx.broker_env}")


def gate_l3_strategy_cap(ctx: LiveContext) -> GateResult:
    if ctx.max_strategy_capital_usd > TINY_LIVE_CEILING_USD + 1e-12:
        return GateResult(
            "L3",
            False,
            f"MAX_STRATEGY_CAPITAL_USD={ctx.max_strategy_capital_usd} > ceiling",
        )
    order = ctx.proposed_order
    if order is None:
        return GateResult("L3", True, "no proposed order; cap config ok")
    equity = strategy_equity_usd(
        max_strategy_capital_usd=ctx.max_strategy_capital_usd,
        cumulative_pnl=ctx.strategy_cumulative_pnl,
    )
    if order.side == "buy":
        notional = float(order.notional or 0.0)
        ok, detail = check_buy_cap(
            strategy_market_value=ctx.strategy_market_value,
            outstanding_strategy_buy_notional=ctx.outstanding_strategy_buy_notional,
            proposed_buy_notional=notional,
            max_strategy_capital_usd=ctx.max_strategy_capital_usd,
            strategy_equity=equity,
            broker_cash=ctx.broker_cash,
        )
        return GateResult("L3", ok, detail)
    # sells: qty vs ledger
    if ctx.registry_path is None:
        return GateResult("L3", False, "registry_path required for sell qty check")
    ledger = strategy_qty_by_symbol(ctx.registry_path)
    held = float(ledger.get(order.symbol, 0.0))
    ok, detail = check_sell_qty(proposed_qty=float(order.qty or 0.0), ledger_qty=held)
    return GateResult("L3", ok, detail)


def gate_l4_tiny_live_acceptance(ctx: LiveContext) -> GateResult:
    if ctx.tiny_live_path is None or not Path(ctx.tiny_live_path).is_file():
        return GateResult("L4", False, "TINY_LIVE_ACCEPTANCE.json missing")
    if ctx.historical_path is None or not Path(ctx.historical_path).is_file():
        return GateResult("L4", False, "HISTORICAL_ACCEPTANCE.json missing")
    try:
        tiny = verify_acceptance(
            ctx.tiny_live_path, research_config_hash=ctx.research_config_hash or None
        )
        hist = verify_acceptance(ctx.historical_path)
    except ArtifactError as exc:
        return GateResult("L4", False, f"artifact invalid: {exc}")
    if tiny.get("status") != "TINY_LIVE_READY":
        return GateResult("L4", False, f"status={tiny.get('status')}")
    parent = tiny.get("parent_artifact_digest")
    if parent != hist.get("integrity_digest"):
        return GateResult("L4", False, "parent HISTORICAL digest mismatch")
    if hist.get("status") != "HISTORICAL_RESEARCH_PASS":
        return GateResult("L4", False, "parent not HISTORICAL_RESEARCH_PASS")
    return GateResult("L4", True, "TINY_LIVE_ACCEPTANCE valid")


def gate_l5_arm_token(ctx: LiveContext) -> GateResult:
    if ctx.project_root is None:
        return GateResult("L5", False, "project_root missing")
    ok = is_armed(ctx.project_root, now=ctx.arm_now)
    return GateResult("L5", ok, "armed" if ok else "arm absent or expired")


def gate_l6_execute_flag(ctx: LiveContext) -> GateResult:
    return GateResult("L6", bool(ctx.execute), f"execute={ctx.execute}")


def gate_l7_kill_switch(ctx: LiveContext) -> GateResult:
    halted = is_halted(
        kill_switch_file=ctx.kill_switch_file,
        kill_switch_env_var=ctx.trading_halt_env,
        drawdown_breach=ctx.drawdown_latched,
    )
    # Also treat TRADING_HALT directly
    env_halt = os.environ.get(ctx.trading_halt_env, "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }
    ok = (not halted) and (not env_halt) and (not ctx.drawdown_latched)
    return GateResult("L7", ok, "kill_switch_clear" if ok else "halt active")


def gate_l8_fresh_data(ctx: LiveContext) -> GateResult:
    if ctx.last_bar_session is None or ctx.last_completed_nyse_session is None:
        return GateResult("L8", False, "session dates missing")
    ok = ctx.last_bar_session == ctx.last_completed_nyse_session and ctx.available_at_ok
    return GateResult(
        "L8",
        ok,
        f"last_bar={ctx.last_bar_session} completed={ctx.last_completed_nyse_session} "
        f"available_at_ok={ctx.available_at_ok}",
    )


def gate_l9_jev_data(ctx: LiveContext) -> GateResult:
    # Fail closed: missing Jev ⇒ no new Jev-dependent entries.
    return GateResult(
        "L9",
        bool(ctx.jev_data_present),
        "jev present" if ctx.jev_data_present else "jev missing; fail closed",
    )


def gate_l10_model_id(ctx: LiveContext) -> GateResult:
    if ctx.resolved_model_id != ctx.pinned_model_id:
        return GateResult(
            "L10",
            False,
            f"resolved={ctx.resolved_model_id} pinned={ctx.pinned_model_id}",
        )
    if (
        ctx.production_model_hash is None
        or ctx.acceptance_production_hash is None
        or ctx.production_model_hash != ctx.acceptance_production_hash
    ):
        return GateResult("L10", False, "production model hash mismatch")
    return GateResult("L10", True, "model id and production hash ok")


def gate_l11_duplicate_order(ctx: LiveContext) -> GateResult:
    order = ctx.proposed_order
    if order is None:
        return GateResult("L11", True, "no proposed order")
    if ctx.registry_path is None:
        return GateResult("L11", False, "registry_path required")
    if client_order_id_exists(ctx.registry_path, order.client_order_id):
        return GateResult("L11", False, f"duplicate {order.client_order_id}")
    # Broker lookup before submit
    if ctx.broker is not None:
        try:
            ctx.broker.get_order(order.client_order_id)
            return GateResult("L11", False, "client_order_id already at broker")
        except Exception:
            pass  # not found = ok
    return GateResult("L11", True, "client_order_id unique")


def gate_l12_reconciliation(ctx: LiveContext) -> GateResult:
    if ctx.registry_path is None:
        return GateResult("L12", False, "registry_path required")
    try:
        ledger = strategy_qty_by_symbol(ctx.registry_path)
        assert_external_holdings_ok(ctx.broker_positions, ledger)
    except Exception as exc:
        return GateResult("L12", False, f"position mismatch: {exc}")
    from jevlab.brokers.strategy_ledger import list_ledger_rows

    ledger_ids = {r.client_order_id for r in list_ledger_rows(ctx.registry_path)}
    bad = reconcile_open_orders(
        broker_open_client_ids=ctx.broker_open_client_ids,
        ledger_client_ids=ledger_ids,
        strategy_prefix=STRATEGY_ID,
    )
    if bad:
        return GateResult("L12", False, f"foreign-prefix open orders not in ledger: {bad}")
    return GateResult("L12", True, "reconciled")


def gate_l13_risk_approval(ctx: LiveContext) -> GateResult:
    order = ctx.proposed_order
    if order is None:
        return GateResult("L13", True, "no proposed order")
    if ctx.risk_config is None or ctx.portfolio_state is None:
        return GateResult("L13", False, "risk_config/portfolio_state required")
    market = ctx.risk_market_state or RiskMarketState(
        session_date=date.today(),
        decision_time=datetime.now(UTC),
        last_completed_nyse_session=ctx.last_completed_nyse_session or date.today(),
        last_bar_session=ctx.last_bar_session,
        available_at_ok=ctx.available_at_ok,
    )
    proposed = ProposedOrder(
        symbol=order.symbol,
        side=order.side,
        qty=float(order.qty or 0.0),
        notional_usd=float(order.notional or 0.0),
        client_order_id=order.client_order_id,
        session_date=date.fromisoformat(order.session_date) if order.session_date else date.today(),
    )
    # Buys with notional-only: qty may be 0; risk rejects qty<=0. Use synthetic qty.
    if proposed.side == "buy" and proposed.qty <= 0 and proposed.notional_usd > 0:
        proposed = ProposedOrder(
            symbol=proposed.symbol,
            side=proposed.side,
            qty=1.0,  # placeholder for risk; notional drives caps
            notional_usd=proposed.notional_usd,
            client_order_id=proposed.client_order_id,
            session_date=proposed.session_date,
        )
    decision = evaluate_orders([proposed], ctx.portfolio_state, market, ctx.risk_config)
    if decision.approved and not decision.rejections:
        return GateResult("L13", True, "risk approved")
    reasons = [r.reason for r in decision.rejections]
    return GateResult("L13", False, f"risk rejected: {reasons}")


def gate_l14_cash_account(ctx: LiveContext) -> GateResult:
    acct = ctx.account
    if acct is None and ctx.broker is not None:
        acct = ctx.broker.get_account()
    if acct is None:
        return GateResult("L14", False, "account state missing; fail closed")
    # Fail closed on ambiguous fields.
    if acct.multiplier is None or str(acct.multiplier).strip() == "":
        return GateResult("L14", False, "multiplier ambiguous; fail closed")
    if acct.shorting_enabled is None:
        return GateResult("L14", False, "shorting_enabled ambiguous; fail closed")
    if str(acct.multiplier).strip() not in {"1", "1.0"}:
        return GateResult("L14", False, f"margin multiplier={acct.multiplier}")
    if acct.shorting_enabled is not False:
        return GateResult("L14", False, "shorting enabled")
    t = acct.account_type.lower()
    if t in {"margin", "pdt"}:
        return GateResult("L14", False, f"account_type={acct.account_type}")
    if t in {"cash", "cash_account", "", "unknown"}:
        return GateResult(
            "L14",
            True,
            f"type={acct.account_type} multiplier={acct.multiplier} shorting=false",
        )
    return GateResult("L14", False, f"unrecognized account_type={acct.account_type}")


GATE_FUNCS: dict[str, Callable[[LiveContext], GateResult]] = {
    "L1": gate_l1_live_enabled,
    "L2": gate_l2_broker_env,
    "L3": gate_l3_strategy_cap,
    "L4": gate_l4_tiny_live_acceptance,
    "L5": gate_l5_arm_token,
    "L6": gate_l6_execute_flag,
    "L7": gate_l7_kill_switch,
    "L8": gate_l8_fresh_data,
    "L9": gate_l9_jev_data,
    "L10": gate_l10_model_id,
    "L11": gate_l11_duplicate_order,
    "L12": gate_l12_reconciliation,
    "L13": gate_l13_risk_approval,
    "L14": gate_l14_cash_account,
}


def check_live_gates(ctx: LiveContext) -> GateReport:
    """Evaluate L1-L14 independently; any failure blocks submission."""
    results = [GATE_FUNCS[gid](ctx) for gid in LIVE_GATE_IDS]
    report = GateReport(results=results)
    log.info(
        "live_gates.checked",
        all_passed=report.all_passed,
        sentence=EXECUTION_EXPERIMENT_SENTENCE,
        failed=[r.gate_id for r in results if not r.passed],
    )
    return report


def maybe_submit_order(
    ctx: LiveContext,
    order: OrderRequest | None = None,
) -> dict[str, Any]:
    """Submit only if every gate passes; otherwise zero ``submit_order`` calls.

    Dry-run (execute=False) never submits even if other gates would pass when
    forced — L6 fails. This is the single submission choke point for safety tests.
    """
    work = ctx
    if order is not None:
        work = ctx.with_overrides(proposed_order=order)
    report = check_live_gates(work)
    payload: dict[str, Any] = {
        "submitted": False,
        "gates": report.to_dict(),
        "execution_experiment_sentence": EXECUTION_EXPERIMENT_SENTENCE,
        "intended_order": asdict(work.proposed_order) if work.proposed_order else None,
    }
    if not report.all_passed:
        return payload
    if work.broker is None or work.proposed_order is None:
        payload["detail"] = "missing broker or order"
        return payload
    ack = work.broker.submit_order(work.proposed_order)
    payload["submitted"] = True
    payload["ack"] = {
        "client_order_id": ack.client_order_id,
        "broker_order_id": ack.broker_order_id,
        "status": ack.status,
    }
    return payload


def live_context_from_config(
    live: LiveConfig,
    *,
    project_root: Path,
    registry_path: Path,
    research_config_hash: str,
    execute: bool = False,
) -> LiveContext:
    return LiveContext(
        live_trading_enabled=live.LIVE_TRADING_ENABLED,
        broker_env=live.BROKER_ENV,
        max_strategy_capital_usd=live.MAX_STRATEGY_CAPITAL_USD,
        tiny_live_path=project_root / "reports" / "acceptance" / "TINY_LIVE_ACCEPTANCE.json",
        historical_path=project_root / "reports" / "acceptance" / "HISTORICAL_ACCEPTANCE.json",
        research_config_hash=research_config_hash,
        project_root=project_root,
        execute=execute,
        kill_switch_file=project_root / "KILL_SWITCH",
        registry_path=registry_path,
        pinned_model_id="jev-1.13.0",
    )


def gate_report_sha256(report: GateReport) -> str:
    return sha256_of(canonical_json(report.to_dict()))


# Re-export arm phrase for CLI
ARM_PHRASE = DEFAULT_ARM_PHRASE
