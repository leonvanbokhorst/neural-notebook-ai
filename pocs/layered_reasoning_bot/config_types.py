"""
Shared configuration types for the layered reasoning bot.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set
import yaml


@dataclass
class ModelConfig:
    """Configuration for language models."""

    model_name: str
    temperature: float
    max_tokens: int
    tone_matching: bool = False


@dataclass
class AIRoleConfig:
    """Configuration for AI participant roles."""

    description: str
    model: ModelConfig
    capabilities: List[str]


@dataclass
class StrategyConfig:
    """Configuration for strategy formation."""

    decision_method: str
    confidence_threshold: float
    strategy_options: List[str]


@dataclass
class MemoryConfig:
    """Configuration for conversation memory."""

    conversation_history_length: int
    episodic_memory_enabled: bool
    memory_persistence: bool


@dataclass
class BotConfig:
    """Main configuration for the bot."""

    models: Dict[str, ModelConfig]
    ai_roles: Dict[str, AIRoleConfig]
    strategy: StrategyConfig
    memory: MemoryConfig
    features: Dict[str, bool]

    @classmethod
    def from_yaml(cls, config_path: str) -> "BotConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Load core model configs
        models = {
            "intent_recognition": ModelConfig(
                **config_dict["models"]["intent_recognition"]
            ),
            "response_generation": ModelConfig(
                **config_dict["models"]["response_generation"]
            ),
            "feature_extraction": ModelConfig(
                **config_dict["models"]["feature_extraction"]
            ),
        }

        # Load AI role configs
        ai_roles = {}
        for role_name, role_config in config_dict["ai_roles"].items():
            ai_roles[role_name] = AIRoleConfig(
                description=role_config["description"],
                model=ModelConfig(**role_config["model"]),
                capabilities=role_config["capabilities"],
            )

        strategy = StrategyConfig(**config_dict["models"]["strategy_formation"])
        memory = MemoryConfig(**config_dict["memory"])
        features = config_dict["features"]

        return cls(
            models=models,
            ai_roles=ai_roles,
            strategy=strategy,
            memory=memory,
            features=features,
        )
