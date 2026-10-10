"""LiveBroker / PaperBroker — Alpaca REST via httpx (MASTER_SPEC §15.2).

Hard-disabled by default: ``build_live_broker`` refuses unless
``LIVE_TRADING_ENABLED`` and ``BROKER_ENV=live``. Order submission is further
gated by ``live_gate.check_live_gates`` / ``maybe_submit_order``.

Never calls close-position / close-all endpoints. Sells use explicit qty only.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from jevlab.brokers.types import (
    AccountState,
    OrderAck,
    OrderRequest,
    OrderStatus,
    OrderStatusKind,
    Position,
    Side,
)
from jevlab.config.models import LiveConfig
from jevlab.utils.logging import get_logger, redact_secrets

log = get_logger(__name__)

ALPACA_LIVE_BASE: str = "https://api.alpaca.markets"
ALPACA_PAPER_BASE: str = "https://paper-api.alpaca.markets"
ORDER_PATH: str = "/v2/orders"
# Endpoints that must NEVER be called (isolation / fail-closed).
FORBIDDEN_PATH_FRAGMENTS: tuple[str, ...] = (
    "/v2/positions/",  # DELETE close position
    "/v2/positions",  # DELETE close all — exact match checked separately
)


class LiveBrokerDisabledError(RuntimeError):
    """Raised when constructing an order-capable LiveBroker under default config."""


class LiveBrokerError(RuntimeError):
    """Broker transport / response error (secrets redacted)."""


def _auth_headers(api_key: str, api_secret: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def assert_not_forbidden_path(method: str, path: str) -> None:
    """Refuse close-position / close-all style calls."""
    m = method.upper()
    p = path.split("?")[0]
    if m == "DELETE" and (p == "/v2/positions" or p.startswith("/v2/positions/")):
        raise LiveBrokerError("forbidden: close-position/close-all endpoints must never be called")
    if m == "DELETE" and "positions" in p:
        raise LiveBrokerError("forbidden: DELETE on positions is not allowed")


class LiveBroker:
    """Alpaca cash-account broker. Construction is hard-gated for live env."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        api_secret: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        env_label: Literal["live", "paper"] = "live",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.env_label = env_label
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=_auth_headers(api_key, api_secret),
            transport=transport,
            timeout=timeout,
        )
        self._request_log: list[tuple[str, str]] = []  # (method, path) for tests

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LiveBroker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def request_log(self) -> list[tuple[str, str]]:
        return list(self._request_log)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        assert_not_forbidden_path(method, path)
        self._request_log.append((method.upper(), path.split("?")[0]))
        try:
            resp = self._client.request(method, path, json=json_body, params=params)
        except httpx.HTTPError as exc:
            raise LiveBrokerError(redact_secrets(str(exc))) from exc
        if resp.status_code >= 400:
            raise LiveBrokerError(redact_secrets(f"broker HTTP {resp.status_code}: {resp.text}"))
        if not resp.content:
            return None
        return resp.json()

    def get_account(self) -> AccountState:
        raw = self._request("GET", "/v2/account")
        assert isinstance(raw, dict)
        # Alpaca cash accounts: account_type / multiplier / shorting fields.
        acct_type = str(raw.get("account_type") or raw.get("type") or "").lower()
        mult = raw.get("multiplier")
        mult_s = None if mult is None else str(mult)
        shorting = raw.get("shorting_enabled")
        if shorting is None and "shorting_enabled" not in raw:
            # Ambiguous if field missing entirely.
            shorting_b: bool | None = None
        else:
            shorting_b = bool(shorting)
        cash_v = raw.get("cash", 0.0)
        bp_v = raw.get("buying_power", cash_v if cash_v is not None else 0.0)
        eq_v = raw.get("equity", 0.0)
        return AccountState(
            cash=float(cash_v if cash_v is not None else 0.0),
            buying_power=float(bp_v if bp_v is not None else 0.0),
            equity=float(eq_v if eq_v is not None else 0.0),
            account_type=acct_type or "unknown",
            pattern_day_trader=bool(raw.get("pattern_day_trader", False)),
            multiplier=mult_s,
            shorting_enabled=shorting_b,
            status=str(raw.get("status", "")),
            raw=raw,
        )

    def get_positions(self) -> list[Position]:
        raw = self._request("GET", "/v2/positions")
        if not isinstance(raw, list):
            return []
        out: list[Position] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            out.append(
                Position(
                    symbol=str(item.get("symbol", "")),
                    qty=float(item.get("qty", 0.0)),
                    market_value=float(item.get("market_value", 0.0)),
                    avg_entry_price=float(item.get("avg_entry_price", 0.0)),
                    side=str(item.get("side", "long")),
                )
            )
        return out

    def list_open_orders(self, client_order_prefix: str | None = None) -> list[OrderStatus]:
        raw = self._request("GET", "/v2/orders", params={"status": "open", "limit": 500})
        if not isinstance(raw, list):
            return []
        out: list[OrderStatus] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("client_order_id") or "")
            if client_order_prefix and not cid.startswith(client_order_prefix):
                continue
            out.append(_parse_order_status(item))
        return out

    def submit_order(self, order: OrderRequest) -> OrderAck:
        if not self.asset_fractionable(order.symbol):
            raise LiveBrokerError(f"asset {order.symbol} is not fractionable")
        body: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
            "client_order_id": order.client_order_id,
        }
        if order.side == "buy":
            if order.notional is None or order.notional <= 0:
                raise LiveBrokerError("buy requires positive notional")
            body["notional"] = f"{float(order.notional):.2f}"
        else:
            if order.qty is None or order.qty <= 0:
                raise LiveBrokerError("sell requires positive qty")
            # Explicit qty only — never close-position.
            body["qty"] = str(float(order.qty))
        raw = self._request("POST", ORDER_PATH, json_body=body)
        assert isinstance(raw, dict)
        return OrderAck(
            client_order_id=str(raw.get("client_order_id") or order.client_order_id),
            broker_order_id=str(raw.get("id") or ""),
            status=_status_kind(str(raw.get("status", "new"))),
            submitted_at=datetime.now(UTC),
            raw=raw,
        )

    def get_order(self, client_order_id: str) -> OrderStatus:
        raw = self._request(
            "GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id}
        )
        # Alpaca uses /v2/orders:by_client_order_id — some deployments use query on /orders
        if not isinstance(raw, dict):
            raise LiveBrokerError("unexpected get_order response")
        return _parse_order_status(raw)

    def cancel_order(self, client_order_id: str) -> None:
        # Cancel by client order id via lookup then DELETE /v2/orders/{id} — not positions.
        st = self.get_order(client_order_id)
        self._request("DELETE", f"/v2/orders/{st.broker_order_id}")

    def is_market_open(self) -> bool:
        raw = self._request("GET", "/v2/clock")
        if not isinstance(raw, dict):
            return False
        return bool(raw.get("is_open"))

    def asset_fractionable(self, symbol: str) -> bool:
        raw = self._request("GET", f"/v2/assets/{symbol}")
        if not isinstance(raw, dict):
            return False
        return bool(raw.get("fractionable"))


def _status_kind(raw: str) -> OrderStatusKind:
    mapping: dict[str, OrderStatusKind] = {
        "new": "new",
        "accepted": "accepted",
        "pending_new": "pending_new",
        "filled": "filled",
        "partially_filled": "partially_filled",
        "canceled": "canceled",
        "cancelled": "canceled",
        "rejected": "rejected",
        "expired": "expired",
    }
    return mapping.get(raw.lower(), "new")


def _parse_order_status(item: dict[str, Any]) -> OrderStatus:
    side_raw = str(item.get("side", "buy")).lower()
    side: Side = "sell" if side_raw == "sell" else "buy"
    return OrderStatus(
        client_order_id=str(item.get("client_order_id") or ""),
        broker_order_id=str(item.get("id") or ""),
        symbol=str(item.get("symbol") or ""),
        side=side,
        status=_status_kind(str(item.get("status", "new"))),
        qty=float(item["qty"]) if item.get("qty") not in (None, "") else None,
        notional=float(item["notional"]) if item.get("notional") not in (None, "") else None,
        filled_qty=float(item.get("filled_qty") or 0.0),
        filled_avg_price=(
            float(item["filled_avg_price"])
            if item.get("filled_avg_price") not in (None, "")
            else None
        ),
        raw=item,
    )


def build_live_broker(
    live_config: LiveConfig,
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
    transport: httpx.BaseTransport | None = None,
    allow_unsafe_construct: bool = False,
) -> LiveBroker:
    """Construct order-capable LiveBroker only when live gates' env flags allow.

    Default config (LIVE_TRADING_ENABLED=false) raises ``LiveBrokerDisabledError``.
    Tests may pass ``allow_unsafe_construct=True`` with a mock transport.
    """
    if not allow_unsafe_construct:
        if not live_config.LIVE_TRADING_ENABLED:
            raise LiveBrokerDisabledError(
                "LIVE_TRADING_ENABLED is false; cannot construct LiveBroker"
            )
        if live_config.BROKER_ENV != "live":
            raise LiveBrokerDisabledError(
                f"BROKER_ENV={live_config.BROKER_ENV!r}; live broker requires 'live'"
            )
    key = api_key if api_key is not None else os.environ.get("BROKER_API_KEY", "")
    secret = api_secret if api_secret is not None else os.environ.get("BROKER_API_SECRET", "")
    if not key or not secret:
        raise LiveBrokerDisabledError("BROKER_API_KEY / BROKER_API_SECRET missing")
    return LiveBroker(
        base_url=ALPACA_LIVE_BASE,
        api_key=key,
        api_secret=secret,
        transport=transport,
        env_label="live",
    )


def build_paper_broker(
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> LiveBroker:
    """Optional PaperBroker (same code, paper base URL). Never required by gates."""
    key = api_key if api_key is not None else os.environ.get("BROKER_API_KEY", "")
    secret = api_secret if api_secret is not None else os.environ.get("BROKER_API_SECRET", "")
    if not key or not secret:
        raise LiveBrokerDisabledError("BROKER_API_KEY / BROKER_API_SECRET missing")
    return LiveBroker(
        base_url=ALPACA_PAPER_BASE,
        api_key=key,
        api_secret=secret,
        transport=transport,
        env_label="paper",
    )


def is_order_endpoint(method: str, path: str) -> bool:
    """True if the request would submit/cancel/mutate an order."""
    p = urlparse(path).path if "://" in path else path.split("?")[0]
    m = method.upper()
    if m == "POST" and p.rstrip("/").endswith("/v2/orders"):
        return True
    return bool(m == "DELETE" and "/v2/orders" in p)


# Alias for docs / imports
PaperBroker = build_paper_broker
