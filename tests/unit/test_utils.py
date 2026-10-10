"""Phase 0 unit tests: hashing, digests, units, config, redaction, calendar."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from jevlab.config.loader import config_hash, load_all_configs, load_yaml_config
from jevlab.config.models import LiveConfig, UniverseConfig
from jevlab.constants import TINY_LIVE_CEILING_USD
from jevlab.utils.calendar import is_nyse_session, next_nyse_session, nyse_session_dates
from jevlab.utils.hashing import canonical_json, logical_table_digest, sha256_of
from jevlab.utils.logging import configure_logging, get_logger, redact_secrets
from jevlab.utils.registry import REGISTRY_TABLES, init_registry, list_registry_tables
from jevlab.utils.units import bps_to_decimal, decimal_to_bps


def test_canonical_json_stable_across_key_order_and_whitespace() -> None:
    a = {"z": 1, "a": {"y": 2.5, "b": True}, "m": [3, 1]}
    b = {"m": [3, 1], "a": {"b": True, "y": 2.5}, "z": 1}
    ca = canonical_json(a)
    cb = canonical_json(b)
    assert ca == cb
    assert " " not in ca
    # Equivalent JSON parsed with different whitespace still hashes equal via canonical form.
    messy = json.loads('{ "z" : 1, "a" : { "y": 2.5, "b": true }, "m": [3, 1] }')
    assert canonical_json(messy) == ca
    assert sha256_of(ca) == sha256_of(cb)


def test_canonical_json_float_rule_uses_float64_repr() -> None:
    text = canonical_json({"x": 0.1 + 0.2})
    assert text == '{"x":' + repr(float(0.1 + 0.2)) + "}"


def test_logical_table_digest_invariant_to_row_order_and_encoding(
    tmp_path: Path,
) -> None:
    df1 = pd.DataFrame(
        {
            "symbol": ["QQQ", "SPY"],
            "date": ["2020-01-02", "2020-01-01"],
            "value": [1.5, 2.25],
        }
    )
    df2 = df1.iloc[::-1].reset_index(drop=True)
    d1 = logical_table_digest(df1, primary_key=["symbol", "date"])
    d2 = logical_table_digest(df2, primary_key=["symbol", "date"])
    assert d1 == d2

    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    df1.to_parquet(path_a, index=False, compression="snappy")
    df2.to_parquet(path_b, index=False, compression="gzip")
    loaded_a = pd.read_parquet(path_a)
    loaded_b = pd.read_parquet(path_b)
    assert logical_table_digest(loaded_a, primary_key=["symbol", "date"]) == d1
    assert logical_table_digest(loaded_b, primary_key=["symbol", "date"]) == d1
    # File bytes may differ; logical digest must not.
    assert path_a.read_bytes() != path_b.read_bytes()


def test_logical_table_digest_sensitive_to_value_change() -> None:
    df = pd.DataFrame({"symbol": ["SPY"], "date": ["2020-01-01"], "value": [1.0]})
    base = logical_table_digest(df, primary_key=["symbol", "date"])
    changed = df.copy()
    changed.loc[0, "value"] = 1.0000001
    other = logical_table_digest(changed, primary_key=["symbol", "date"])
    assert base["content_sha256"] != other["content_sha256"]


def test_bps_decimal_roundtrip() -> None:
    assert bps_to_decimal(2.5) == pytest.approx(0.00025)
    assert decimal_to_bps(0.00025) == pytest.approx(2.5)
    assert bps_to_decimal(0.3) == pytest.approx(0.00003)
    # Normal round-trip cost example from §8: 2*2.5 + 0.3 = 5.3 bp = 0.00053
    roundtrip_cost_dec = 2 * bps_to_decimal(2.5) + bps_to_decimal(0.3)
    assert roundtrip_cost_dec == pytest.approx(0.00053)


def test_config_loads_and_hashes_stable() -> None:
    configs = load_all_configs()
    assert configs.live.LIVE_TRADING_ENABLED is False
    assert configs.live.MAX_STRATEGY_CAPITAL_USD == 10.00
    assert TINY_LIVE_CEILING_USD == 10.00
    h1 = configs.research_config_hash
    h2 = load_all_configs().research_config_hash
    assert h1 == h2
    assert len(h1) == 64
    assert config_hash(configs.universe) == configs.hashes["universe"]


def test_config_validation_failures(tmp_path: Path) -> None:
    bad_live = tmp_path / "live.yaml"
    bad_live.write_text(
        "LIVE_TRADING_ENABLED: false\n"
        "BROKER_ENV: disabled\n"
        "MAX_STRATEGY_CAPITAL_USD: 10.01\n"
        "arm_token_ttl_hours: 24\n"
        "historical_artifact_expiry_days: 180\n"
        "tiny_live_artifact_expiry_days: 90\n"
        "scale_artifact_expiry_days: 90\n"
        "arm_phrase: x\n"
        "require_execute_flag: true\n"
        "dry_run_default: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_yaml_config(bad_live, LiveConfig)

    with pytest.raises(ValidationError):
        UniverseConfig.model_validate(
            {
                "universe_version": "v1",
                "symbols": ["SPY", "QQQ"],
                "context_symbols": ["SPY"],
            }
        )


def test_secret_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "super-secret-jev-key-xyz")
    monkeypatch.setenv("TIINGO_API_KEY", "tiingo-secret-abc")
    text = "Authorization: Bearer super-secret-jev-key-xyz and also tiingo-secret-abc end"
    redacted = redact_secrets(text)
    assert "super-secret-jev-key-xyz" not in redacted
    assert "tiingo-secret-abc" not in redacted
    assert "[REDACTED]" in redacted

    configure_logging(json_logs=True)
    log = get_logger("test")
    # Ensure processor does not raise when logging a secret-bearing event.
    log.info("probe", authorization="Bearer super-secret-jev-key-xyz")


def test_nyse_calendar_holidays() -> None:
    # New Year's Day 2024 (Monday) and Independence Day 2024 are NYSE holidays.
    assert not is_nyse_session("2024-01-01")
    assert not is_nyse_session("2024-07-04")
    assert is_nyse_session("2024-01-02")
    assert is_nyse_session("2024-07-05")
    # Thanksgiving 2024 is Thursday Nov 28.
    assert not is_nyse_session("2024-11-28")
    assert next_nyse_session("2024-07-03").isoformat() == "2024-07-05"
    sessions = nyse_session_dates("2024-07-01", "2024-07-05")
    assert "2024-07-04" not in {d.isoformat() for d in sessions}
    assert any(d.isoformat() == "2024-07-05" for d in sessions)


def test_registry_init_creates_all_section_18_tables(tmp_path: Path) -> None:
    db = tmp_path / "experiments.sqlite"
    engine = init_registry(db)
    tables = set(list_registry_tables(engine))
    assert set(REGISTRY_TABLES) <= tables
    required = {
        "external_series",
        "forward_registry",
        "model_artifacts",
        "acceptance_artifacts",
        "strategy_ledger",
    }
    assert required <= tables


def test_tiny_live_ceiling_is_source_constant_not_config() -> None:
    assert TINY_LIVE_CEILING_USD == 10.00
    # Ceiling must not appear as a writable live.yaml field.
    fields = set(LiveConfig.model_fields.keys())
    assert "TINY_LIVE_CEILING_USD" not in fields
    assert "MAX_STRATEGY_CAPITAL_USD" in fields
