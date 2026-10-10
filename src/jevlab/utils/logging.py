"""structlog setup with secret-redaction processor (MASTER_SPEC §19)."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import MutableMapping
from typing import Any, cast

import structlog

# Environment variable names whose values must never appear in logs.
SECRET_ENV_NAMES: tuple[str, ...] = (
    "JEV_API_KEY",
    "TIINGO_API_KEY",
    "FRED_API_KEY",
    "BROKER_API_KEY",
    "BROKER_API_SECRET",
)

_AUTHORIZATION_RE = re.compile(r"(?i)(authorization\s*[:=]\s*Bearer\s+)(\S+)")
_BEARER_RE = re.compile(r"(?i)(Bearer\s+)([A-Za-z0-9._\-+/=]{8,})")


def _secret_values() -> list[str]:
    values: list[str] = []
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            values.append(value)
    return values


def redact_secrets(text: str) -> str:
    """Redact Authorization headers and known secret env values from ``text``."""
    redacted = _AUTHORIZATION_RE.sub(r"\1[REDACTED]", text)
    redacted = _BEARER_RE.sub(r"\1[REDACTED]", redacted)
    for secret in _secret_values():
        if secret and secret in redacted:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    return value


def secret_redaction_processor(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """structlog processor that redacts secrets from all event fields."""
    for key in list(event_dict.keys()):
        event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def configure_logging(*, json_logs: bool = True, level: int = logging.INFO) -> None:
    """Configure structlog with JSON rendering and secret redaction."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        secret_redaction_processor,
    ]

    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            secret_redaction_processor,
            renderer,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger (call ``configure_logging`` once at startup)."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
