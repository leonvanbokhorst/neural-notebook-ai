"""
LiteLLM POC for Neural Notebook AI
---------------------------------

A demonstration of using LiteLLM to interact with Ollama models.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from litellm import completion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ConfigurationParams:
    """Configuration parameters for the LiteLLM POC."""

    models: List[str]
    api_keys: Dict[str, str]
    prompt_template: str
    max_tokens: int

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigurationParams":
        return cls(
            models=config_dict["models"],
            api_keys=config_dict["api_keys"],
            prompt_template=config_dict["prompt_template"],
            max_tokens=config_dict["max_tokens"],
        )


class LiteLLMDemo:
    """Main implementation class for the LiteLLM POC."""

    def __init__(self, config: ConfigurationParams):
        self.config = config
        self._setup()

    def _setup(self) -> None:
        """Initialize any necessary resources."""
        logger.info("Setting up LiteLLM environment")
        self.models = self.config.models
        self.prompt = self.config.prompt_template

    def demonstrate_concept(self) -> Dict[str, Any]:
        """
        Demonstrate using multiple Ollama models through LiteLLM.

        Returns:
            Dict[str, Any]: Results from different models
        """
        logger.info("Running LiteLLM demonstration with Ollama models")
        results = {}

        for model in self.models:
            try:
                logger.info(f"Testing model: {model}")
                # Configure Ollama-specific parameters
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": self.prompt}],
                    max_tokens=self.config.max_tokens,
                )
                results[model] = {
                    "content": response.choices[0].message.content,
                    "usage": response.usage,
                }
                logger.info(f"Successfully got response from {model}")
            except Exception as e:
                logger.error(f"Error with model {model}: {str(e)}")
                results[model] = {"error": str(e)}

        return results

    def cleanup(self) -> None:
        """Cleanup any resources."""
        logger.info("Cleaning up resources")


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for the POC."""
    try:
        # Load configuration from YAML
        config_dict = load_config()
        config = ConfigurationParams.from_dict(config_dict)

        # Initialize and run demonstration
        demo = LiteLLMDemo(config)
        results = demo.demonstrate_concept()

        # Process results
        logger.info("Demonstration results:")
        for model, result in results.items():
            logger.info(f"\n{model}: {result}")

    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        raise
    finally:
        if "demo" in locals():
            demo.cleanup()


if __name__ == "__main__":
    main()
