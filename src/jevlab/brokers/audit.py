"""Append-only live_orders audit trail (MASTER_SPEC §15.5)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from jevlab.brokers.live_gate import EXECUTION_EXPERIMENT_SENTENCE
from jevlab.constants import STRATEGY_ID
from jevlab.utils.hashing import canonical_json
from jevlab.utils.logging import redact_secrets
from jevlab.utils.registry import init_registry


def append_live_order_audit(
    registry_path: Path,
    *,
    client_order_id: str,
    mode: str,
    payload: Mapping[str, Any],
    strategy_id: str = STRATEGY_ID,
) -> str:
    """Write one live_orders row; always includes the execution-experiment sentence."""
    engine = init_registry(registry_path)
    order_id = str(uuid.uuid4())
    body = dict(payload)
    body["execution_experiment_sentence"] = EXECUTION_EXPERIMENT_SENTENCE
    # Redact any accidental secrets in nested JSON.
    safe = json.loads(redact_secrets(canonical_json(body)))

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO live_orders (
                    order_id, client_order_id, strategy_id, mode, cap_calc_json,
                    resolved_model, model_artifact_hash, at, signal_json,
                    feature_hash, inference_ids, portfolio_json, intended_order_json,
                    risk_decision_json, request_json, response_json, fill_px,
                    fill_qty, est_cost, realized_cost, reconcile_status
                ) VALUES (
                    :order_id, :client_order_id, :strategy_id, :mode, :cap_calc_json,
                    :resolved_model, :model_artifact_hash, :at, :signal_json,
                    :feature_hash, :inference_ids, :portfolio_json, :intended_order_json,
                    :risk_decision_json, :request_json, :response_json, :fill_px,
                    :fill_qty, :est_cost, :realized_cost, :reconcile_status
                )
                """
            ),
            {
                "order_id": order_id,
                "client_order_id": client_order_id,
                "strategy_id": strategy_id,
                "mode": mode,
                "cap_calc_json": canonical_json(safe.get("cap_calc", {})),
                "resolved_model": str(safe.get("resolved_model", "")),
                "model_artifact_hash": str(safe.get("model_artifact_hash", "")),
                "at": datetime.now(UTC).isoformat(),
                "signal_json": canonical_json(safe.get("signal", {})),
                "feature_hash": str(safe.get("feature_hash", "")),
                "inference_ids": canonical_json(safe.get("inference_ids", [])),
                "portfolio_json": canonical_json(safe.get("portfolio", {})),
                "intended_order_json": canonical_json(safe.get("intended_order", {})),
                "risk_decision_json": canonical_json(safe.get("risk_decision", {})),
                "request_json": canonical_json(safe.get("request", {})),
                "response_json": canonical_json(safe.get("response", {})),
                "fill_px": safe.get("fill_px"),
                "fill_qty": safe.get("fill_qty"),
                "est_cost": safe.get("est_cost"),
                "realized_cost": safe.get("realized_cost"),
                "reconcile_status": str(safe.get("reconcile_status", "")),
            },
        )
    return order_id
