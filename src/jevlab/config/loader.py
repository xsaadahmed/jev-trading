"""Load and hash YAML configs via pydantic models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from jevlab.config.models import (
    BacktestConfig,
    DataConfig,
    FeaturesConfig,
    JevConfig,
    LiveConfig,
    RiskConfig,
    StrategyConfig,
    UniverseConfig,
)
from jevlab.utils.hashing import canonical_json, sha256_of
from jevlab.utils.paths import ProjectPaths, get_project_root

CONFIG_MODEL_BY_NAME: dict[str, type[BaseModel]] = {
    "data": DataConfig,
    "features": FeaturesConfig,
    "jev": JevConfig,
    "strategy": StrategyConfig,
    "backtest": BacktestConfig,
    "risk": RiskConfig,
    "live": LiveConfig,
    "universe": UniverseConfig,
}


def load_yaml_config[T: BaseModel](path: Path | str, model: type[T]) -> T:
    """Load a YAML file and validate it against a pydantic model."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    return model.model_validate(raw)


def config_hash(model: BaseModel) -> str:
    """Return sha256 of the canonical JSON of a config model."""
    payload = model.model_dump(mode="json")
    return sha256_of(canonical_json(payload))


@dataclass(frozen=True, slots=True)
class ResearchConfigs:
    """All research configs plus an aggregate research_config_hash."""

    data: DataConfig
    features: FeaturesConfig
    jev: JevConfig
    strategy: StrategyConfig
    backtest: BacktestConfig
    risk: RiskConfig
    live: LiveConfig
    universe: UniverseConfig
    hashes: dict[str, str]
    research_config_hash: str


def load_all_configs(config_dir: Path | str | None = None) -> ResearchConfigs:
    """Load every §19 YAML config from ``config_dir`` (default: project config/)."""
    if config_dir is None:
        config_dir = ProjectPaths.from_root(get_project_root()).config
    else:
        config_dir = Path(config_dir)

    data = load_yaml_config(config_dir / "data.yaml", DataConfig)
    features = load_yaml_config(config_dir / "features.yaml", FeaturesConfig)
    jev = load_yaml_config(config_dir / "jev.yaml", JevConfig)
    strategy = load_yaml_config(config_dir / "strategy.yaml", StrategyConfig)
    backtest = load_yaml_config(config_dir / "backtest.yaml", BacktestConfig)
    risk = load_yaml_config(config_dir / "risk.yaml", RiskConfig)
    live = load_yaml_config(config_dir / "live.yaml", LiveConfig)
    universe = load_yaml_config(config_dir / "universe.yaml", UniverseConfig)

    hashes = {
        "data": config_hash(data),
        "features": config_hash(features),
        "jev": config_hash(jev),
        "strategy": config_hash(strategy),
        "backtest": config_hash(backtest),
        "risk": config_hash(risk),
        "live": config_hash(live),
        "universe": config_hash(universe),
    }
    research_config_hash = sha256_of(canonical_json(hashes))
    return ResearchConfigs(
        data=data,
        features=features,
        jev=jev,
        strategy=strategy,
        backtest=backtest,
        risk=risk,
        live=live,
        universe=universe,
        hashes=hashes,
        research_config_hash=research_config_hash,
    )
