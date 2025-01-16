"""
Smolagents POC: Web Search and Research
----------------------------------------------

A simple agent system for web research using DuckDuckGo search and webpage visits.
"""

import logging
from pathlib import Path

import yaml
from smolagents import (TOOL_CALLING_SYSTEM_PROMPT, CodeAgent,
                        DuckDuckGoSearchTool, LiteLLMModel, ManagedAgent,
                        ToolCallingAgent)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_config("pocs/smolagents/config.yaml")

        # Initialize model
        model = LiteLLMModel(
            model_id=config["model"],
            temperature=config["temperature"],
            model_kwargs={"max_tokens": config["max_tokens"]},
        )

        # Create manager agent with demonstration config
        manager = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[],
            system_prompt=config["system_prompt"],
            additional_authorized_imports=config["additional_authorized_imports"],
        )

        # Run research task from demonstration config
        answer = manager.run(
            "Come up with a neural network takes a single number (0 to 9) and outputs that number in ascii art 5x5 matrix. Use pytorch. Design, train, validate and store the model."
        )

        # Save structured results to journey log
        journey_log = [
            f"# 🚀 RESEARCHERS ARE GO\n\n",
            str(answer),
        ]
        log_path = Path("journey_log.md")
        log_path.write_text("\n".join(journey_log))

        print("\n🔍 Research Results:")
        print("=" * 80)
        print(answer)
        print(f"\n✨ Journey log saved to {log_path}")

    except Exception as e:
        logging.error(f"Failed to run demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    main()
