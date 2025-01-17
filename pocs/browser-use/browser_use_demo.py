"""
Browser Automation POC for Neural Notebook AI
------------------------------------------

Demonstrates browser automation capabilities using Selenium and Playwright.
"""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from browser_use import Agent
from result_storage import BrowserResult, create_result_storage


async def try_with_models(
    task: str, models: List[Dict[str, Any]]
) -> tuple[Any, Dict[str, Any]]:
    """Try running the agent with different models until one succeeds."""
    last_error = None

    for model_config in models:
        try:
            agent = Agent(
                task=task,
                llm=ChatOpenAI(**model_config),
            )
            result = await agent.run(max_steps=10)
            return result, model_config

        except Exception as e:
            last_error = e
            print(f"\nFailed with model {model_config['model']}: {str(e)}")
            continue

    raise Exception(f"All models failed. Last error: {str(last_error)}")


def extract_content_from_agent_result(agent_result: Any) -> Dict[str, Any]:
    """Extract useful content from agent result."""
    content = {"steps": [], "final_result": None}

    # Handle string results
    if isinstance(agent_result, str):
        content["final_result"] = agent_result
        content["steps"].append(
            {"content": agent_result, "success": True, "error": None}
        )
        return content

    # Handle AgentHistoryList
    if hasattr(agent_result, "all_results"):
        for step in agent_result.all_results:
            step_content = None
            step_success = True
            step_error = None

            # Extract content based on different result types
            if hasattr(step, "extracted_content"):
                step_content = step.extracted_content
            elif hasattr(step, "result"):
                step_content = step.result
            elif hasattr(step, "output"):
                step_content = step.output
            elif isinstance(step, (str, dict)):
                step_content = step

            # Extract error information
            if hasattr(step, "error"):
                step_error = str(step.error) if step.error else None
                step_success = not bool(step.error)

            if step_content is not None:
                content["steps"].append(
                    {
                        "content": step_content,
                        "success": step_success,
                        "error": step_error,
                    }
                )

    # Set final result as the last successful step's content
    successful_steps = [step for step in content["steps"] if step["success"]]
    if successful_steps:
        content["final_result"] = successful_steps[-1]["content"]

    return content


async def main():
    # Ensure OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "Please set the OPENAI_API_KEY environment variable to use OpenAI's API"
        )

    # Initialize result storage
    storage = create_result_storage(Path("pocs/browser-use/results"))

    # Define models to try in order of preference
    models = [
        # gpt-4o-mini can use images - DO NOT CHANGE
        {"model": "gpt-4o-mini", "temperature": 0},
    ]

    task = "Find and return the latest news about the congressional hearings on Trump's nominees"

    try:
        # Try different models until one succeeds
        agent_result, used_model = await try_with_models(task, models)

        # Process the agent result
        content = extract_content_from_agent_result(agent_result)

        # Create and store the result
        result = BrowserResult.create(
            task=task,
            content=content,
            metadata={
                "model": used_model["model"],
                "temperature": used_model["temperature"],
                "max_steps": 10,
                "total_steps": len(content["steps"]),
                "successful_steps": len([s for s in content["steps"] if s["success"]]),
            },
        )
        result_path = storage.store_result(result)

        # Print the result and storage location
        if content["final_result"]:
            print("\nTask Result:", content["final_result"])
        print(f"\nResult stored in: {result_path}")

        # Analyze results
        all_results = storage.list_results()
        if all_results:
            stats = storage.analyze_results(all_results)
            print("\nResult Statistics:")
            print(f"Total Results: {stats['total_results']}")
            print(f"Success Rate: {stats['success_rate']:.1f}%")
            print(f"Average Content Length: {stats['avg_content_length']:.0f} chars")
            if stats["common_errors"]:
                print("\nCommon Errors:")
                for error, count in stats["common_errors"].items():
                    print(f"- {error}: {count} times")

    except Exception as e:
        # Store failed result with model information
        result = BrowserResult.create(
            task=task,
            content=None,
            success=False,
            error=str(e),
            metadata={"attempted_models": [m["model"] for m in models]},
        )
        storage.store_result(result)
        print(f"Error during execution: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
