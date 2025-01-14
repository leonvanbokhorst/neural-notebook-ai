"""
LiteLLM POC for Neural Notebook AI
---------------------------------

A demonstration of using LiteLLM to interact with Ollama models.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List

import litellm
import yaml
from litellm import completion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def usage_to_dict(usage: Any) -> Dict[str, int]:
    """Convert a Usage object to a dictionary."""
    if hasattr(usage, "__dict__"):
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    return {}


def format_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format result dictionary for JSON serialization."""
    if "error" in result:
        return result

    return {
        "content": result["content"],
        "usage": usage_to_dict(result["usage"]) if "usage" in result else {},
    }


def track_callback(
    kwargs: Dict[str, Any],  # kwargs to completion
    completion_response: Any,  # response from completion
    start_time: datetime,  # start time
    end_time: datetime,  # end time
) -> None:
    """Callback to track LLM usage and performance."""
    try:
        duration = (end_time - start_time).total_seconds()
        model = kwargs.get("model", "unknown")
        is_streaming = kwargs.get("stream", False)

        if usage := getattr(completion_response, "usage", None):
            logger.info(
                f"\nModel: {model} ({'streaming' if is_streaming else 'non-streaming'})"
                f"\n  Duration: {duration:.2f}s"
                f"\n  Prompt Tokens: {usage.prompt_tokens}"
                f"\n  Completion Tokens: {usage.completion_tokens}"
                f"\n  Total Tokens: {usage.total_tokens}"
            )

            # Get hidden params (if any)
            hidden_params = getattr(completion_response, "_hidden_params", {})
            if "response_cost" in hidden_params:
                logger.info(f"  Cost: ${hidden_params['response_cost']:.6f}")

    except Exception as e:
        logger.error(f"Error in tracking callback: {str(e)}")


@dataclass
class UsageStats:
    """Track usage statistics for LLM calls."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    request_count: int = 0

    def update_from_response(self, response: Any, model: str) -> None:
        """Update stats from a completion response."""
        if hasattr(response, "usage"):
            self.total_tokens += response.usage.total_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            self.request_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of usage statistics."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "requests": self.request_count,
            "tokens_per_request": (
                round(self.total_tokens / self.request_count, 2)
                if self.request_count > 0
                else 0
            ),
            "duration_seconds": round(duration, 2) if duration else None,
        }


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
        self.usage_stats = UsageStats()
        self._setup()

    def _setup(self) -> None:
        """Initialize any necessary resources."""
        logger.info("Setting up LiteLLM environment")
        self.models = self.config.models
        self.prompt = self.config.prompt_template

        # Set up callback for tracking
        litellm.success_callback = [track_callback]

    def demonstrate_streaming(self, model: str) -> Generator[str, None, None]:
        """
        Demonstrate streaming responses from a model.

        Args:
            model: The name of the model to use

        Yields:
            str: Chunks of the generated response
        """
        try:
            logger.info(f"Starting streaming with model: {model}")
            self.usage_stats.start_time = datetime.now()

            response = completion(
                model=model,
                messages=[{"role": "user", "content": self.prompt}],
                max_tokens=self.config.max_tokens,
                stream=True,  # Enable streaming
                stream_options={"include_usage": True},  # Include usage info in stream
            )

            full_response = None
            for chunk in response:
                if chunk and chunk.choices and chunk.choices[0].delta.content:
                    if not full_response:
                        full_response = chunk
                    elif hasattr(chunk, "usage") and hasattr(full_response, "usage"):
                        full_response.usage.completion_tokens += getattr(
                            chunk.usage, "completion_tokens", 0
                        )
                        full_response.usage.total_tokens += getattr(
                            chunk.usage, "total_tokens", 0
                        )
                    yield chunk.choices[0].delta.content

            self.usage_stats.end_time = datetime.now()
            if full_response:
                self.usage_stats.update_from_response(full_response, model)

        except Exception as e:
            logger.error(f"Error streaming from {model}: {str(e)}")
            yield f"Error: {str(e)}"

    def demonstrate_concept(self) -> Dict[str, Any]:
        """
        Demonstrate using multiple models through LiteLLM.

        Returns:
            Dict[str, Any]: Results from different models
        """
        logger.info("Running LiteLLM demonstration with models")
        results = {}
        self.usage_stats.start_time = datetime.now()

        for model in self.models:
            try:
                logger.info(f"Testing model: {model}")
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": self.prompt}],
                    max_tokens=self.config.max_tokens,
                )

                self.usage_stats.update_from_response(response, model)

                results[model] = {
                    "content": response.choices[0].message.content,
                    "usage": usage_to_dict(response.usage),
                }
                logger.info(f"Successfully got response from {model}")
            except Exception as e:
                logger.error(f"Error with model {model}: {str(e)}")
                results[model] = {"error": str(e)}

        self.usage_stats.end_time = datetime.now()
        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.get_summary()

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

        # Initialize demo
        demo = LiteLLMDemo(config)

        # Demonstrate streaming
        logger.info("\nTesting streaming responses:")
        for model in config.models:
            logger.info(f"\nStreaming from {model}:")
            for chunk in demo.demonstrate_streaming(model):
                print(chunk, end="", flush=True)
            print("\n")
            logger.info("Streaming Usage Statistics:")
            logger.info(json.dumps(demo.get_usage_stats(), indent=2))

        # Reset usage stats for non-streaming demo
        demo.usage_stats = UsageStats()

        # Demonstrate non-streaming
        results = demo.demonstrate_concept()
        logger.info("\nNon-streaming results:")
        for model, result in results.items():
            formatted_result = format_result(result)
            logger.info(f"\n{model}: {json.dumps(formatted_result, indent=2)}")

        # Display final usage stats
        logger.info("\nFinal Usage Statistics:")
        logger.info(json.dumps(demo.get_usage_stats(), indent=2))

    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        raise
    finally:
        if "demo" in locals():
            demo.cleanup()


if __name__ == "__main__":
    main()
