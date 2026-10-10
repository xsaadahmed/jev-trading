"""Each §14 risk rule has a rejecting case (Phase 4)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jevlab.config.models import RiskConfig
from jevlab.portfolio.state import PortfolioState
from jevlab.risk.killswitch import is_halted
from jevlab.risk.rules import MarketState, ProposedOrder, evaluate_orders

DAY = date(2020, 6, 1)


def _cfg(**kwargs: object) -> RiskConfig:
    base = dict(
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
    base.update(kwargs)
    return RiskConfig.model_validate(base)


def _port(**kwargs: object) -> PortfolioState:
    base: dict[str, object] = dict(
        cash_usd=10_000.0,
        positions={},
        strategy_equity_usd=10_000.0,
        strategy_capital_usd=10_000.0,
        high_water_mark_usd=10_000.0,
        sod_equity_usd=10_000.0,
        market_value_usd=0.0,
        outstanding_buy_notional_usd=0.0,
        orders_today=0,
        kill_switch_active=False,
        existing_client_order_ids=set(),
    )
    base.update(kwargs)
    return PortfolioState(**base)  # type: ignore[arg-type]


def _mkt(**kwargs: object) -> MarketState:
    base: dict[str, object] = dict(
        session_date=DAY,
        decision_time=datetime(2020, 6, 1, 18, 30, tzinfo=ZoneInfo("America/New_York")),
        last_completed_nyse_session=DAY,
        last_bar_session=DAY,
        available_at_ok=True,
        missing_symbol_count=0,
    )
    base.update(kwargs)
    return MarketState(**base)  # type: ignore[arg-type]


def _buy(symbol: str = "SPY", notional: float = 1000.0, oid: str | None = None) -> ProposedOrder:
    return ProposedOrder(
        symbol=symbol,
        side="buy",
        qty=notional / 100.0,
        notional_usd=notional,
        client_order_id=oid or f"id-{symbol}-{notional}",
        session_date=DAY,
    )


def test_reject_max_position() -> None:
    # 34% of 10k = 3400; propose 4000
    out = evaluate_orders([_buy(notional=4000.0)], _port(), _mkt(), _cfg())
    assert len(out.approved) == 0
    assert out.rejections[0].reason == "max_position"


def test_reject_max_gross() -> None:
    port = _port(market_value_usd=9000.0, cash_usd=1000.0, strategy_equity_usd=10_000.0)
    out = evaluate_orders([_buy(notional=2000.0)], port, _mkt(), _cfg())
    assert out.rejections[0].reason == "max_gross_exposure"


def test_reject_max_orders_per_day() -> None:
    port = _port(orders_today=6)
    out = evaluate_orders([_buy()], port, _mkt(), _cfg())
    assert out.rejections[0].reason == "max_orders_per_day"


def test_reject_daily_loss_halt() -> None:
    # SOD 10k, now 9400 → 6% loss > 5%
    port = _port(strategy_equity_usd=9400.0, sod_equity_usd=10_000.0)
    out = evaluate_orders([_buy()], port, _mkt(), _cfg())
    assert out.rejections[0].reason == "daily_loss_halt"


def test_reject_drawdown_kill() -> None:
    # HWM 10k, equity 8000 → 20% DD
    port = _port(strategy_equity_usd=8000.0, high_water_mark_usd=10_000.0)
    out = evaluate_orders([_buy()], port, _mkt(), _cfg())
    assert out.rejections[0].reason == "kill_switch"


def test_reject_stale_data() -> None:
    mkt = _mkt(last_bar_session=date(2020, 5, 29), last_completed_nyse_session=DAY)
    out = evaluate_orders([_buy()], _port(), mkt, _cfg())
    assert out.rejections[0].reason == "stale_data"


def test_reject_kill_switch_file(tmp_path: Path) -> None:
    path = tmp_path / "KILL_SWITCH"
    path.write_text("1", encoding="utf-8")
    assert is_halted(kill_switch_file=path)
    out = evaluate_orders(
        [_buy()],
        _port(),
        _mkt(kill_switch_file=str(path)),
        _cfg(),
    )
    assert out.rejections[0].reason == "kill_switch"


def test_reject_duplicate_order() -> None:
    port = _port(existing_client_order_ids={"dup"})
    out = evaluate_orders([_buy(oid="dup")], port, _mkt(), _cfg())
    assert out.rejections[0].reason == "duplicate_order"


def test_approve_valid_buy() -> None:
    out = evaluate_orders([_buy(notional=2000.0)], _port(), _mkt(), _cfg())
    assert len(out.approved) == 1
    assert out.rejections == []
