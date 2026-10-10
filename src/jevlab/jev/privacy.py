"""Semantic privacy validation for ``state_v1`` (MASTER_SPEC §7.1).

Not a blunt 'no digits' regex - six semantic checks listed in the Phase 5 prompt.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from jevlab.jev.vocabulary import SCHEMA_VALUE, STATE_KEYS

# Universe tickers (§21A) — always rejected if they appear as leaf strings.
UNIVERSE_TICKERS: frozenset[str] = frozenset(
    {
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
    }
)

# Uppercase tokens that may appear inside closed phrases (none today) or keys.
TICKER_ALLOWLIST: frozenset[str] = frozenset()

PROHIBITED_KEYS: frozenset[str] = frozenset(
    {
        "symbol",
        "ticker",
        "date",
        "timestamp",
        "price",
        "volume",
        "vix",
        "vix_level",
        "level",
        "rsi",
        "year",
        "asof",
        "as_of",
    }
)

_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
)

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR_RE = re.compile(r"\b(19[9]\d|20[0-9]\d|2100)\b")
_DATE_LIKE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}/\d{1,2}/\d{1,2}\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"[$€£¥]")
_PERCENT_RE = re.compile(r"\d+\s*%|\b\d+\.\d+\s*percent\b", re.IGNORECASE)
_FLOAT_RE = re.compile(r"(?<![a-zA-Z_])[-+]?\d+\.\d+(?![a-zA-Z_])")
_PRICE_LIKE_RE = re.compile(r"(?<![a-zA-Z_])\d{1,3}(,\d{3})+(\.\d+)?(?![a-zA-Z_])")
_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")


def validate_state_privacy(
    state: Mapping[str, Any],
    vocab: frozenset[str],
    *,
    universe_tickers: Iterable[str] = UNIVERSE_TICKERS,
    ticker_allowlist: Iterable[str] = TICKER_ALLOWLIST,
    expected_schema: str = SCHEMA_VALUE,
) -> list[str]:
    """Return a list of privacy violations (empty iff state is clean)."""
    violations: list[str] = []
    tickers = {t.upper() for t in universe_tickers}
    allow = {t.upper() for t in ticker_allowlist}

    if not isinstance(state, Mapping):
        return ["state_not_mapping"]

    # (6) schema/version fields must equal configured values
    schema_val = state.get("schema")
    if schema_val != expected_schema:
        violations.append(f"schema_mismatch:{schema_val!r}")

    # (5) prohibited metadata keys
    for key in state:
        if str(key).lower() in PROHIBITED_KEYS:
            violations.append(f"prohibited_key:{key}")

    # Required key set
    for key in STATE_KEYS:
        if key not in state:
            violations.append(f"missing_key:{key}")

    for key, value in state.items():
        key_s = str(key)
        if key_s not in vocab:
            violations.append(f"unknown_key:{key_s}")

        # (6) no numeric JSON types at all
        if isinstance(value, bool):
            # bool is a subclass of int — reject explicitly
            violations.append(f"numeric_json_type:{key_s}:bool")
            continue
        if isinstance(value, (int, float)):
            violations.append(f"numeric_json_type:{key_s}:{type(value).__name__}")
            continue
        if value is None:
            violations.append(f"null_value:{key_s}")
            continue
        if not isinstance(value, str):
            violations.append(f"non_string_leaf:{key_s}:{type(value).__name__}")
            continue

        # (1) closed vocabulary
        if value not in vocab:
            violations.append(f"leaf_not_in_vocab:{key_s}:{value!r}")

        low = value.lower()
        # (2) tickers / ticker-like tokens
        if value.upper() in tickers:
            violations.append(f"universe_ticker:{key_s}:{value}")
        for tok in _TICKER_TOKEN_RE.findall(value):
            if tok in tickers:
                violations.append(f"ticker_token:{key_s}:{tok}")
            elif tok not in allow and tok == value and len(tok) >= 2 and tok.isupper():
                violations.append(f"ticker_like:{key_s}:{tok}")

        # Also reject any substring that is exactly a universe ticker as a token.
        for t in tickers:
            if re.search(rf"\b{re.escape(t)}\b", value):
                violations.append(f"ticker_in_text:{key_s}:{t}")

        # (3) dates / years / months
        if _ISO_DATE_RE.search(value):
            violations.append(f"iso_date:{key_s}")
        if _DATE_LIKE_RE.search(value):
            violations.append(f"date_like:{key_s}")
        if _YEAR_RE.search(value):
            violations.append(f"year:{key_s}")
        for month in _MONTH_NAMES:
            if re.search(rf"\b{month}\b", low):
                violations.append(f"month_name:{key_s}:{month}")
                break

        # (4) currency, price-like, percentages, floats
        if _CURRENCY_RE.search(value):
            violations.append(f"currency:{key_s}")
        if _PERCENT_RE.search(value):
            violations.append(f"percentage:{key_s}")
        if _FLOAT_RE.search(value):
            violations.append(f"float_token:{key_s}")
        if _PRICE_LIKE_RE.search(value):
            violations.append(f"price_like:{key_s}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def assert_state_private(state: Mapping[str, Any], vocab: frozenset[str]) -> None:
    """Raise ``PrivacyError`` if any violation is found."""
    bad = validate_state_privacy(state, vocab)
    if bad:
        raise PrivacyError(bad)


class PrivacyError(ValueError):
    """State failed semantic privacy validation."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"state privacy violations: {violations}")
