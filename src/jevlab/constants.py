"""Project-wide source constants that must not live in config files."""

# Hard ceiling for V1 strategy capital (USD). Config may set MAX_STRATEGY_CAPITAL_USD
# at or below this value; values above raise at startup. Never raise this without a
# new major version and explicit protocol change (MASTER_SPEC §15.1, §21A).
TINY_LIVE_CEILING_USD: float = 10.00

STRATEGY_ID: str = "jev-pullback-v1"
