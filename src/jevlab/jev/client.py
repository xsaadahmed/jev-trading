"""Official System One HTTP client (MASTER_SPEC §2; Phase 5).

Research mode refuses aliases (``jev-latest`` / ``jev-preview``) and requires the
pinned versioned model id from config. Logs never include secrets.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from jevlab.config.models import JevConfig
from jevlab.jev.privacy import PrivacyError, assert_state_private
from jevlab.jev.schemas import QuestionSpec, SystemOneRequest, SystemOneResponse
from jevlab.jev.vocabulary import CLOSED_VOCABULARY
from jevlab.utils.logging import get_logger, redact_secrets

log = get_logger(__name__)

DEFAULT_REFUSED_ALIASES: frozenset[str] = frozenset({"jev-latest", "jev-preview"})
API_KEY_ENV: str = "JEV_API_KEY"
AliasSet = frozenset[str] | set[str] | list[str] | tuple[str, ...]


class AliasRefusedError(ValueError):
    """Raised when research mode is asked to call an alias model id."""


class JevAPIError(RuntimeError):
    """Non-retryable or exhausted API failure."""

    def __init__(self, status_code: int | None, message: str, body: str | None = None) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(redact_secrets(f"Jev API error status={status_code}: {message}"))


class _RetryableHTTPError(Exception):
    def __init__(self, status_code: int, retry_after: float | None, body: str) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        self.body = body
        super().__init__(f"retryable HTTP {status_code}")


@dataclass
class TokenBucketLimiter:
    """Simple token-bucket limiter (requests per minute)."""

    rate_per_minute: float
    time_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], None] = time.sleep
    _tokens: float = field(init=False)
    _updated: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._tokens = float(self.rate_per_minute)
        self._updated = float(self.time_fn())

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = float(self.time_fn())
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(
                    self.rate_per_minute,
                    self._tokens + elapsed * (self.rate_per_minute / 60.0),
                )
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) * (60.0 / self.rate_per_minute)
            self.sleep_fn(max(wait, 0.01))


def assert_research_model_id(
    model_id: str,
    *,
    pinned_id: str,
    refused_aliases: AliasSet = DEFAULT_REFUSED_ALIASES,
) -> None:
    """Refuse aliases; require exact pinned versioned id."""
    if model_id in set(refused_aliases):
        raise AliasRefusedError(
            f"research mode refuses alias model id {model_id!r}; use pinned {pinned_id!r}"
        )
    if model_id != pinned_id:
        raise AliasRefusedError(
            f"research mode requires pinned model id {pinned_id!r}, got {model_id!r}"
        )


@dataclass
class JevClient:
    """httpx client for ``POST /v1/systemone``."""

    config: JevConfig
    api_key: str | None = None
    transport: httpx.BaseTransport | None = None
    limiter: TokenBucketLimiter | None = None
    vocab: frozenset[str] = CLOSED_VOCABULARY
    validate_privacy: bool = True

    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key if self.api_key is not None else os.environ.get(API_KEY_ENV, "")
        self.api_key = key
        if self.limiter is None:
            self.limiter = TokenBucketLimiter(
                rate_per_minute=float(self.config.rate_limit_per_minute)
            )
        timeout = httpx.Timeout(
            timeout=float(self.config.timeout_seconds),
            connect=float(self.config.connect_timeout_seconds),
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }
        self._client = httpx.Client(
            base_url="",
            headers=headers,
            timeout=timeout,
            transport=self.transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def build_request(
        self,
        state: Mapping[str, Any] | str | list[Any],
        questions: Mapping[str, QuestionSpec | Mapping[str, Any]],
        *,
        model: str | None = None,
    ) -> SystemOneRequest:
        model_id = model or self.config.model_id
        assert_research_model_id(
            model_id,
            pinned_id=self.config.model_id,
            refused_aliases=list(self.config.research_refuse_aliases),
        )
        if self.validate_privacy and isinstance(state, Mapping):
            assert_state_private(state, self.vocab)
        q_models: dict[str, QuestionSpec] = {}
        for qid, spec in questions.items():
            q_models[qid] = (
                spec if isinstance(spec, QuestionSpec) else QuestionSpec.model_validate(spec)
            )
        state_out: str | dict[str, Any] | list[Any]
        state_out = dict(state) if isinstance(state, Mapping) else state
        return SystemOneRequest(model=model_id, state=state_out, questions=q_models)

    def infer(
        self,
        state: Mapping[str, Any] | str | list[Any],
        questions: Mapping[str, QuestionSpec | Mapping[str, Any]],
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        """POST one System One request; retries 429/529/5xx honoring Retry-After."""
        if not self.api_key:
            raise JevAPIError(None, f"{API_KEY_ENV} is not set")
        req = self.build_request(state, questions, model=model)
        payload = req.model_dump(mode="json")
        assert self.limiter is not None
        self.limiter.acquire()
        log.info(
            "jev_request",
            route=self.config.route,
            endpoint=self.config.endpoint,
            model=req.model,
            n_questions=len(req.questions),
        )
        try:
            raw = self._post_with_retries(payload)
        except PrivacyError:
            raise
        except _RetryableHTTPError as exc:
            raise JevAPIError(exc.status_code, "retries exhausted", exc.body) from exc
        except httpx.TimeoutException as exc:
            raise JevAPIError(None, f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise JevAPIError(None, f"http error: {exc}") from exc

        try:
            parsed = SystemOneResponse.model_validate(raw)
        except Exception as exc:
            raise JevAPIError(200, f"malformed response: {exc}", str(raw)) from exc
        log.info(
            "jev_response",
            model_resolved=parsed.model,
            n_answers=len(parsed.answers),
        )
        return parsed

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        max_attempts = int(self.config.max_retries) + 1

        @retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(max=30.0),
            retry=retry_if_exception_type((_RetryableHTTPError, httpx.TimeoutException)),
            before_sleep=_before_sleep_retry_after,
        )
        def _once() -> dict[str, Any]:
            resp = self._client.post(self.config.endpoint, json=payload)
            if resp.status_code in set(self.config.retry_statuses) or (
                500 <= resp.status_code <= 599 and resp.status_code != 501
            ):
                retry_after = _parse_retry_after(resp)
                raise _RetryableHTTPError(resp.status_code, retry_after, resp.text)
            if resp.status_code == 401:
                raise JevAPIError(401, "unauthorized", resp.text)
            if resp.status_code == 422:
                raise JevAPIError(422, "unprocessable entity", resp.text)
            if resp.status_code >= 400:
                raise JevAPIError(resp.status_code, "client/server error", resp.text)
            try:
                data = resp.json()
            except Exception as exc:
                raise JevAPIError(resp.status_code, f"non-json body: {exc}", resp.text) from exc
            if not isinstance(data, dict):
                raise JevAPIError(resp.status_code, "response root must be object", resp.text)
            return data

        return _once()


def _parse_retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _before_sleep_retry_after(retry_state: Any) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _RetryableHTTPError) and exc.retry_after is not None:
        time.sleep(float(exc.retry_after))
    log.warning(
        "jev_retry",
        attempt=retry_state.attempt_number,
        error=redact_secrets(str(exc)) if exc else None,
    )
