"""SQLite experiment registry initialization (MASTER_SPEC §18)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# All tables defined in MASTER_SPEC §18.
REGISTRY_TABLES: tuple[str, ...] = (
    "market_bars",
    "downloads",
    "external_series",
    "features",
    "states",
    "jev_requests",
    "jev_inferences",
    "jev_features",
    "labels",
    "signals",
    "trades",
    "equity",
    "experiments",
    "metrics",
    "trial_ledger",
    "sealed_access_log",
    "live_orders",
    "strategy_ledger",
    "forward_registry",
    "forward_scores",
    "model_artifacts",
    "acceptance_artifacts",
)

_FEATURE_COLUMNS_SQL = """
    ret_1d DOUBLE, ret_3d DOUBLE, ret_5d DOUBLE, ret_20d DOUBLE,
    gap_1d DOUBLE,
    px_vs_sma20 DOUBLE, px_vs_sma50 DOUBLE, px_vs_sma200 DOUBLE,
    rsi_3 DOUBLE, rsi_14 DOUBLE,
    rv_20 DOUBLE, rv_pct_252 DOUBLE,
    vol_ratio_20 DOUBLE, dollar_vol_20 DOUBLE,
    spy_ret_5d DOUBLE, vix_lag1_pct_252 DOUBLE, spy_px_vs_sma200 DOUBLE,
    rel_ret_5d DOUBLE, xs_rank_ret_5d DOUBLE,
    down_streak DOUBLE,
    eligible INT
"""

_JEV_FEATURE_COLUMNS_SQL = """
    j_dislocation DOUBLE,
    j_downside_E DOUBLE,
    j_downside_p3 DOUBLE,
    j_regime DOUBLE,
    j_pullback_p DOUBLE,
    j_breakdown_p DOUBLE,
    j_sideways_p DOUBLE,
    j_setup_conf DOUBLE,
    j_order_disagree DOUBLE
"""

_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS market_bars (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        vendor TEXT NOT NULL,
        raw_open DOUBLE,
        raw_high DOUBLE,
        raw_low DOUBLE,
        raw_close DOUBLE,
        raw_volume BIGINT,
        adj_open DOUBLE,
        adj_high DOUBLE,
        adj_low DOUBLE,
        adj_close DOUBLE,
        adj_volume DOUBLE,
        div_cash DOUBLE,
        split_factor DOUBLE,
        download_id TEXT,
        available_at TIMESTAMP,
        PRIMARY KEY (symbol, date, vendor)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS downloads (
        download_id TEXT PRIMARY KEY,
        vendor TEXT,
        params_json TEXT,
        downloaded_at TIMESTAMP,
        file_sha256 TEXT,
        logical_digest TEXT,
        row_count INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS external_series (
        series_id TEXT NOT NULL,
        obs_date DATE NOT NULL,
        value DOUBLE,
        available_at TIMESTAMP,
        retrieved_at TIMESTAMP,
        download_id TEXT,
        PRIMARY KEY (series_id, obs_date)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS features (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        feature_version TEXT NOT NULL,
        decision_time TIMESTAMP,
        {_FEATURE_COLUMNS_SQL},
        PRIMARY KEY (symbol, date, feature_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS states (
        state_hash TEXT PRIMARY KEY,
        symbol TEXT,
        date DATE,
        state_schema_version TEXT,
        state_json TEXT,
        features_hash TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jev_requests (
        request_hash TEXT PRIMARY KEY,
        route TEXT,
        model_requested TEXT,
        state_hash TEXT,
        qset_version TEXT,
        purpose TEXT CHECK (purpose IN ('feature', 'audit', 'placebo', 'forward')),
        request_json_sha TEXT,
        created_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jev_inferences (
        inference_id TEXT PRIMARY KEY,
        request_hash TEXT NOT NULL,
        attempt INT NOT NULL,
        model_resolved TEXT,
        response_json_sha TEXT,
        shard_file TEXT,
        shard_offset INTEGER,
        input_tokens INT,
        output_tokens INT,
        latency_ms INT,
        http_status INT,
        created_at TIMESTAMP,
        UNIQUE (request_hash, attempt),
        FOREIGN KEY (request_hash) REFERENCES jev_requests (request_hash)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS jev_features (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        qset_version TEXT NOT NULL,
        state_schema_version TEXT NOT NULL,
        model_resolved TEXT NOT NULL,
        {_JEV_FEATURE_COLUMNS_SQL},
        inference_ids TEXT,
        PRIMARY KEY (symbol, date, qset_version, state_schema_version, model_resolved)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS labels (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        horizon INT NOT NULL,
        label_version TEXT NOT NULL,
        R DOUBLE,
        Rbar DOUBLE,
        X DOUBLE,
        y INT,
        roundtrip_cost_dec DOUBLE,
        y_cost INT,
        eligible INT,
        PRIMARY KEY (symbol, date, horizon, label_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        experiment_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        score DOUBLE,
        rank INT,
        selected INT,
        PRIMARY KEY (experiment_id, model_id, symbol, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        experiment_id TEXT,
        trade_id TEXT PRIMARY KEY,
        symbol TEXT,
        entry_date DATE,
        exit_date DATE,
        side TEXT,
        qty DOUBLE,
        entry_px DOUBLE,
        exit_px DOUBLE,
        cost_entry DOUBLE,
        cost_exit DOUBLE,
        pnl_gross DOUBLE,
        pnl_net DOUBLE,
        phase INT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS equity (
        experiment_id TEXT NOT NULL,
        date DATE NOT NULL,
        cash DOUBLE,
        positions_value DOUBLE,
        equity DOUBLE,
        PRIMARY KEY (experiment_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        created_at TIMESTAMP,
        git_commit TEXT,
        git_dirty INT,
        data_hash TEXT,
        feature_version TEXT,
        state_schema_version TEXT,
        qset_version TEXT,
        qset_hash TEXT,
        model_resolved TEXT,
        strategy_config_json TEXT,
        backtest_config_json TEXT,
        seed INT,
        date_start DATE,
        date_end DATE,
        costs_json TEXT,
        status TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        experiment_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        metric TEXT NOT NULL,
        value DOUBLE,
        ci_low DOUBLE,
        ci_high DOUBLE,
        PRIMARY KEY (experiment_id, model_id, scope, metric)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trial_ledger (
        trial_id TEXT PRIMARY KEY,
        created_at TIMESTAMP,
        qset_version TEXT,
        state_schema_version TEXT,
        feature_set TEXT,
        model_class TEXT,
        grid_json TEXT,
        universe_version TEXT,
        horizon INT,
        purpose TEXT CHECK (purpose IN ('confirmatory', 'exploratory', 'audit', 'placebo')),
        experiment_id TEXT,
        result_summary TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sealed_access_log (
        access_id TEXT PRIMARY KEY,
        at TIMESTAMP,
        user TEXT,
        experiment_hash TEXT,
        reason TEXT,
        partition TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS live_orders (
        order_id TEXT PRIMARY KEY,
        client_order_id TEXT UNIQUE,
        strategy_id TEXT,
        mode TEXT CHECK (mode IN ('dry_run', 'live')),
        cap_calc_json TEXT,
        resolved_model TEXT,
        model_artifact_hash TEXT,
        at TIMESTAMP,
        signal_json TEXT,
        feature_hash TEXT,
        inference_ids TEXT,
        portfolio_json TEXT,
        intended_order_json TEXT,
        risk_decision_json TEXT,
        request_json TEXT,
        response_json TEXT,
        fill_px DOUBLE,
        fill_qty DOUBLE,
        est_cost DOUBLE,
        realized_cost DOUBLE,
        reconcile_status TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_ledger (
        ledger_id TEXT PRIMARY KEY,
        client_order_id TEXT UNIQUE,
        strategy_id TEXT,
        symbol TEXT,
        side TEXT,
        qty DOUBLE,
        notional DOUBLE,
        status TEXT,
        fill_px DOUBLE,
        fill_qty DOUBLE,
        at TIMESTAMP,
        mode TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS forward_registry (
        registry_id TEXT PRIMARY KEY,
        deployment_timestamp TIMESTAMP,
        jev_model_id TEXT,
        qset_hash TEXT,
        feature_version TEXT,
        state_schema_version TEXT,
        git_commit TEXT,
        forward_start_date DATE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS forward_scores (
        session_date DATE NOT NULL,
        symbol TEXT NOT NULL,
        model_artifact_id TEXT NOT NULL,
        score DOUBLE,
        selected INT,
        jev_inference_ids TEXT,
        PRIMARY KEY (session_date, symbol, model_artifact_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_artifacts (
        model_artifact_id TEXT PRIMARY KEY,
        kind TEXT CHECK (kind IN ('sealed_test', 'production_forward')),
        model_id TEXT,
        columns_json TEXT,
        train_end DATE,
        hyperparams_json TEXT,
        content_digest TEXT,
        git_commit TEXT,
        created_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS acceptance_artifacts (
        artifact_id TEXT PRIMARY KEY,
        status TEXT CHECK (
            status IN (
                'HISTORICAL_RESEARCH_PASS',
                'TINY_LIVE_READY',
                'SCALE_READY'
            )
        ),
        path TEXT,
        integrity_digest TEXT,
        parent_digest TEXT,
        gate_report_sha256 TEXT,
        research_config_hash TEXT,
        git_commit TEXT,
        created_at TIMESTAMP,
        expires_at TIMESTAMP
    )
    """,
)


def init_registry(db_path: Path | str) -> Engine:
    """Create the SQLite registry database and all §18 tables if missing.

    Returns a SQLAlchemy engine connected to the database.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path.as_posix()}", future=True)
    with engine.begin() as conn:
        for ddl in _DDL_STATEMENTS:
            conn.execute(text(ddl))
    return engine


def list_registry_tables(engine: Engine) -> list[str]:
    """Return user table names present in the registry database."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ).fetchall()
    return [str(r[0]) for r in rows]
