"""Mock-transport tests for Jev client (Phase 5)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from jevlab.config.models import JevConfig
from jevlab.features.columns import STATE_V1_SOURCE_COLUMNS
from jevlab.jev.client import AliasRefusedError, JevAPIError, JevClient, TokenBucketLimiter
from jevlab.jev.schemas import DOCUMENTED_RESPONSE_EXAMPLE
from jevlab.jev.state_builder import build_state_from_percentiles


def _cfg(**kwargs: Any) -> JevConfig:
    base = dict(
        route="systemone",
        endpoint="https://api.typesafe.ai/v1/systemone",
        model_id="jev-1.13.0",
        research_refuse_aliases=["jev-latest", "jev-preview"],
        rate_limit_per_minute=1000.0,
        max_retries=3,
        retry_statuses=[429, 500, 502, 503, 529],
        timeout_seconds=5.0,
        connect_timeout_seconds=2.0,
        bucket_edges=[0.05, 0.20, 0.40, 0.60, 0.80, 0.95],
        state_schema_version="state_v1",
        qset_version="qset_v1",
        qset_path="config/qset_v1.json",
        qset_hash=None,
        audit_qset_version="audit_qset_v1",
        audit_qset_path="config/audit_qset.json",
        audit_qset_hash=None,
        cache_dir="data/jev_cache",
    )
    base.update(kwargs)
    return JevConfig.model_validate(base)


def _state() -> dict[str, str]:
    pct = {c: 0.5 for c in STATE_V1_SOURCE_COLUMNS.values() if c != "down_streak"}
    return build_state_from_percentiles(pct, down_streak=1)


def _questions() -> dict[str, Any]:
    return {
        "q_noul": {
            "type": "noul",
            "instructions": "yes?",
            "criteria": {"true": "yes", "false": "no"},
        }
    }


class _SequencedHandler(httpx.MockTransport):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            if not self._responses:
                return httpx.Response(500, text="exhausted")
            return self._responses.pop(0)

        super().__init__(handler)


def test_alias_refusal() -> None:
    client = JevClient(config=_cfg(), api_key="test-key", validate_privacy=False)
    with pytest.raises(AliasRefusedError):
        client.build_request(_state(), _questions(), model="jev-latest")
    with pytest.raises(AliasRefusedError):
        client.build_request(_state(), _questions(), model="jev-preview")


def test_200_ok() -> None:
    body = dict(DOCUMENTED_RESPONSE_EXAMPLE)
    transport = _SequencedHandler([httpx.Response(200, json=body)])
    client = JevClient(
        config=_cfg(),
        api_key="test-key",
        transport=transport,
        limiter=TokenBucketLimiter(rate_per_minute=1000),
        validate_privacy=True,
    )
    # Use a minimal question set matching documented keys for parse path;
    # infer only needs valid JSON response.
    resp = client.infer(_state(), _questions())
    assert resp.model == "jev-1.13.0"
    assert transport.calls == 1


def test_401_no_retry() -> None:
    transport = _SequencedHandler([httpx.Response(401, text="nope")])
    client = JevClient(
        config=_cfg(max_retries=3),
        api_key="bad",
        transport=transport,
        limiter=TokenBucketLimiter(rate_per_minute=1000),
    )
    with pytest.raises(JevAPIError) as ei:
        client.infer(_state(), _questions())
    assert ei.value.status_code == 401
    assert transport.calls == 1


def test_422_no_retry() -> None:
    transport = _SequencedHandler([httpx.Response(422, text="bad schema")])
    client = JevClient(
        config=_cfg(),
        api_key="k",
        transport=transport,
        limiter=TokenBucketLimiter(rate_per_minute=1000),
    )
    with pytest.raises(JevAPIError) as ei:
        client.infer(_state(), _questions())
    assert ei.value.status_code == 422


def test_429_retries_then_ok() -> None:
    body = dict(DOCUMENTED_RESPONSE_EXAMPLE)
    transport = _SequencedHandler(
        [
            httpx.Response(429, text="slow", headers={"Retry-After": "0"}),
            httpx.Response(200, json=body),
        ]
    )
    client = JevClient(
        config=_cfg(max_retries=3),
        api_key="k",
        transport=transport,
        limiter=TokenBucketLimiter(rate_per_minute=1000),
    )
    resp = client.infer(_state(), _questions())
    assert resp.model == "jev-1.13.0"
    assert transport.calls == 2


def test_529_retries_exhausted() -> None:
    transport = _SequencedHandler(
        [
            httpx.Response(529, text="overload", headers={"Retry-After": "0"}),
            httpx.Response(529, text="overload", headers={"Retry-After": "0"}),
            httpx.Response(529, text="overload", headers={"Retry-After": "0"}),
            httpx.Response(529, text="overload", headers={"Retry-After": "0"}),
        ]
    )
    client = JevClient(
        config=_cfg(max_retries=2),
        api_key="k",
        transport=transport,
        limiter=TokenBucketLimiter(rate_per_minute=1000),
    )
    with pytest.raises(JevAPIError):
        client.infer(_state(), _questions())
    assert transport.calls == 3  # initial + 2 retries


def test_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    transport = httpx.MockTransport(handler)
    client = JevClient(
        config=_cfg(max_retries=1),
        api_key="k",
        transport=transport,
        limiter=TokenBucketLimiter(rate_per_minute=1000),
    )
    with pytest.raises(JevAPIError):
        client.infer(_state(), _questions())


def test_malformed_json() -> None:
    transport = _SequencedHandler([httpx.Response(200, text="{not-json")])
    client = JevClient(
        config=_cfg(),
        api_key="k",
        transport=transport,
        limiter=TokenBucketLimiter(rate_per_minute=1000),
    )
    with pytest.raises(JevAPIError):
        client.infer(_state(), _questions())


def test_missing_api_key() -> None:
    client = JevClient(
        config=_cfg(),
        api_key="",
        transport=_SequencedHandler([]),
        limiter=TokenBucketLimiter(rate_per_minute=1000),
    )
    with pytest.raises(JevAPIError):
        client.infer(_state(), _questions())
