# Phase 0 Report — Scaffold, governance, utilities

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** `docs/MASTER_SPEC.md` §17, §18, §19, §20, §21 Phase 0, §21A, §22

---

## What was built

### Repository skeleton (§17)

- Installable `jevlab` package under `src/` with every subpackage from the architecture tree (`config`, `utils`, `data`, `features`, `targets`, `jev`, `models`, `cv`, `signals`, `portfolio`, `risk`, `backtest`, `evaluation`, `reporting`, `brokers`, `experiments`, `cli`), each with an `__init__.py` docstring stating responsibility.
- `pyproject.toml` (uv, Python ≥ 3.12, dependencies from §22), `uv.lock`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`, `Makefile` (`check`), `README.md`.
- Placeholder dirs: `config/`, `registry/`, `reports/acceptance/`, `scripts/`, `notebooks/`, `tests/{unit,integration,regression,safety}/`.
- Committed append-only header: `registry/trial_ledger.csv`.

### Config (§19)

- Pydantic v2 models for all YAML files: `data`, `features`, `jev`, `strategy`, `backtest`, `risk`, `live`, `universe`.
- `load_yaml_config` / `load_all_configs` / `config_hash` (canonical JSON → sha256); aggregate `research_config_hash`.
- `live.yaml`: `LIVE_TRADING_ENABLED: false`, `MAX_STRATEGY_CAPITAL_USD: 10.00`.
- `TINY_LIVE_CEILING_USD = 10.00` lives in `src/jevlab/constants.py` (not config); config values above the ceiling fail validation.
- `universe.yaml` pinned to the §21A 13-ETF list (validator enforces exact match).

### Utilities (§20 / Phase 0 tasks)

- `canonical_json`, `sha256_of`, `logical_table_digest` (sorted columns → normalized dtypes → sort by PK → canonical row serialization → sha256 + schema hash).
- Seeds, project paths, NYSE calendar helpers (`exchange_calendars`), `bps_to_decimal` / `decimal_to_bps`.
- structlog with secret-redaction processor (Authorization / Bearer + env secret values).
- SQLite registry init with **all** §18 tables, including `external_series`, `forward_registry`, `model_artifacts`, `acceptance_artifacts`, `strategy_ledger`.

### Docs / governance

- `docs/PREREGISTRATION.md` populated from §§3, 4, 8, 16.
- `docs/KNOWN_ISSUES.md` empty template with severity column.
- Pre-commit: ruff, mypy, secret-scan hook.
- Repo-wide secret-scan pytest.

### Explicitly not implemented (per Phase 0)

Data download, features, Jev client/state/qset, models, backtesting, live brokers.

---

## Decisions

1. **Logical digests** compare content, never parquet bytes; tests write snappy vs gzip encodings to prove invariance.
2. **Canonical floats** use `repr(float64(...))` as the fixed formatting rule.
3. **Feature/Jev columns in SQLite** are materialized as named DOUBLE columns matching §6 / §7 (parquet remains the planned bulk store in later phases).
4. **Labels PK** interpreted as `(symbol, date, horizon, label_version)` where the spec used `PRIMARY KEY(...)`.
5. **CLI** is a minimal Typer shell (`jev version` only); domain commands arrive in later phases.

---

## Deviations / ambiguities

| Item | Choice | Notes |
| ---- | ------ | ----- |
| `labels` PRIMARY KEY ellipsis in §18 | `(symbol, date, horizon, label_version)` | Conservative; revisit if Phase 3 needs a different key |
| `features` / `jev_features` wide columns in SQLite | Named columns now | Bulk analytics still intended via parquet + DuckDB later |
| Hatchling / uv dependency-groups | Used `[dependency-groups] dev` | Matches current uv; optional-dependencies also listed for extras |

No conflict with §21A invariants was introduced.

---

## How to run

```bash
uv sync --all-extras
cp .env.example .env   # leave empty or fill locally; never commit .env
make check             # ruff check + format --check + mypy + pytest
```

Individual steps:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/jevlab
uv run pytest -q
```

---

## Test results (acceptance)

| Check | Result |
| ----- | ------ |
| `ruff check` | pass |
| `ruff format --check` | pass |
| `mypy src/jevlab` | pass |
| `pytest -q` | 12 passed |
| Secrets in repo | none found by scan |

Covered invariants: hash stability (key order / whitespace); logical digest row-order and parquet-encoding invariance + value sensitivity; bps↔decimal; config validation failures (cap > ceiling, bad universe); secret redaction; NYSE holidays; registry table set; ceiling constant not in config.

---

## Known limitations

- No network/data code; configs are placeholders until Phase 1+.
- `qset_hash` / `audit_qset_hash` are `null` until Phase 5/6 freeze.
- structlog is configured by tests/callers; no global side effect at import.
- mypy overrides for future heavy deps (`lightgbm`, `plotly`, `duckdb`) are declared early; unused until those modules land.

---

## Stop

Phase 0 is complete. Do not proceed to Phase 1 until explicitly requested.
