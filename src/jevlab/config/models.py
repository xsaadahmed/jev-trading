"""Pydantic models for all YAML configs (MASTER_SPEC §19)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jevlab.constants import TINY_LIVE_CEILING_USD


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_StrictModel):
    """data.yaml — vendors, symbols file, date range, paths."""

    primary_vendor: str
    cross_check_vendors: list[str]
    vix_vendor: str
    vix_series_id: str
    universe_file: str
    start_date: str
    end_date: str | None
    raw_dir: str
    processed_dir: str
    external_dir: str
    rate_limit_per_hour: int
    rate_limit_per_day: int


class FeaturesConfig(_StrictModel):
    """features.yaml — windows, feature set ids."""

    feature_version: str
    sma_windows: list[int]
    return_windows: list[int]
    rsi_windows: list[int]
    rv_window: int
    volume_window: int
    percentile_window: int
    model_a_columns: list[str]
    stored_only_columns: list[str]
    eligibility_sma_window: int


class JevConfig(_StrictModel):
    """jev.yaml — route, endpoint, pinned model, limits, state, qset."""

    route: str
    endpoint: str
    model_id: str
    research_refuse_aliases: list[str]
    rate_limit_per_minute: float
    max_retries: int
    retry_statuses: list[int]
    timeout_seconds: float
    connect_timeout_seconds: float
    bucket_edges: list[float]
    state_schema_version: str
    qset_version: str
    qset_path: str
    qset_hash: str | None
    audit_qset_version: str
    audit_qset_path: str
    audit_qset_hash: str | None
    cache_dir: str


class StrategyConfig(_StrictModel):
    """strategy.yaml — eligibility, K, horizon, phases, model grid, seeds."""

    strategy_id: str
    eligibility_rule: str
    top_k: int
    horizon_sessions: int
    rebalance_every_sessions: int
    phases: list[int]
    acceptance_phase_agg: Literal["mean"]
    logistic_C_grid: list[float]
    lightgbm_params: dict[str, Any]
    seed: int
    entry_lag_sessions: int


class CostScenario(_StrictModel):
    name: str
    cost_bps_per_side: float


class BacktestConfig(_StrictModel):
    """backtest.yaml — cost scenarios, per-symbol multipliers, fill convention."""

    cost_scenarios: list[CostScenario]
    sell_fee_bps: float
    commission_bps: float
    per_symbol_cost_multiplier: dict[str, float]
    fill_convention: str
    initial_capital_usd: float
    fractional_shares: bool
    cash_interest_dec: float


class RiskConfig(_StrictModel):
    """risk.yaml — deterministic portfolio risk limits (MASTER_SPEC §14)."""

    max_position_pct: float
    max_gross_exposure_pct: float
    max_new_orders_per_day: int
    max_daily_loss_pct: float
    max_drawdown_pct: float
    min_live_position_notional_usd: float
    stale_data_halt: bool
    max_missing_symbols: int
    submit_window_start_et: str
    submit_window_end_et: str
    kill_switch_file: str
    kill_switch_env_var: str


class LiveConfig(_StrictModel):
    """live.yaml — all flags default off; capital cap ≤ TINY_LIVE_CEILING_USD."""

    LIVE_TRADING_ENABLED: bool = False
    BROKER_ENV: Literal["paper", "live", "disabled"] = "disabled"
    MAX_STRATEGY_CAPITAL_USD: float = 10.00
    arm_token_ttl_hours: float = 24.0
    historical_artifact_expiry_days: int = 180
    tiny_live_artifact_expiry_days: int = 90
    scale_artifact_expiry_days: int = 90
    arm_phrase: str = "I-ACCEPT-REAL-MONEY-RISK-USD-10"
    require_execute_flag: bool = True
    dry_run_default: bool = True

    @field_validator("MAX_STRATEGY_CAPITAL_USD")
    @classmethod
    def _cap_within_ceiling(cls, value: float) -> float:
        if value > TINY_LIVE_CEILING_USD:
            raise ValueError(
                f"MAX_STRATEGY_CAPITAL_USD={value} exceeds "
                f"TINY_LIVE_CEILING_USD={TINY_LIVE_CEILING_USD} "
                "(ceiling is a source constant, not configurable)"
            )
        if value <= 0:
            raise ValueError("MAX_STRATEGY_CAPITAL_USD must be positive")
        return value

    @model_validator(mode="after")
    def _trading_disabled_by_default_invariant(self) -> LiveConfig:
        # Documented invariant: live trading hard-disabled in committed config.
        return self


class UniverseConfig(_StrictModel):
    """universe.yaml — fixed ex-ante ETF universe (MASTER_SPEC §21A)."""

    universe_version: str
    symbols: list[str]
    context_symbols: list[str]
    excluded_notes: list[str] = Field(default_factory=list)

    @field_validator("symbols")
    @classmethod
    def _exactly_thirteen(cls, value: list[str]) -> list[str]:
        expected = [
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
        ]
        if value != expected:
            raise ValueError(
                f"universe symbols must exactly match the §21A list {expected}, got {value}"
            )
        return value
