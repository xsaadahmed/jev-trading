"""Kill-switch detection: file, env, or drawdown breach (MASTER_SPEC §14)."""

from __future__ import annotations

import os
from pathlib import Path


def kill_switch_file_present(path: Path | str) -> bool:
    return Path(path).exists()


def kill_switch_env_set(var_name: str = "TRADING_HALT") -> bool:
    val = os.environ.get(var_name, "").strip()
    return val in {"1", "true", "TRUE", "yes", "YES"}


def is_halted(
    *,
    kill_switch_file: Path | str | None = None,
    kill_switch_env_var: str = "TRADING_HALT",
    drawdown_breach: bool = False,
) -> bool:
    """True if any halt condition is active (fail closed)."""
    if drawdown_breach:
        return True
    if kill_switch_file is not None and kill_switch_file_present(kill_switch_file):
        return True
    return bool(kill_switch_env_set(kill_switch_env_var))
