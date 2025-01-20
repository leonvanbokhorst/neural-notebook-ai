"""
Baseline Test Scenario
--------------------

Tests the LayeredReasoningBot with fixed configurations,
without adaptive capabilities for comparison purposes.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import logging
import colorlog
from typing import Dict, Any, List

from layered_reasoning_bot import LayeredReasoningBot
from ai_interaction_manager import AIInteractionManager
from interaction_observer import InteractionObserver


# Configure colored logging
def setup_colored_logging():
    """Configure colored logging for better visibility."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_colored_logging()


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class BaselineTest:
    """Tests bot behavior with fixed configurations."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        # Create baseline bot with fixed configurations
        self.bot = LayeredReasoningBot(str(config_path))

        # Disable adaptive features
        self.bot.emotional_state.disable_adaptation()
        self.bot.strategy_selector.disable_learning()

        # Setup managers
        self.manager = AIInteractionManager(self.bot)
        self.observer = InteractionObserver(self.manager)

        # Create results directory
        self.results_dir = (
            Path(__file__).parent
            / "baseline_results"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _convert_datetime(self, obj: Any) -> Any:
        """Convert datetime objects to ISO format strings recursively."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime(item) for item in obj]
        return obj

    async def run_baseline_test(self) -> Dict[str, Any]:
        """Run baseline test with fixed responses."""

        # Register AIs with fixed roles
        self.manager.register_ai("baseline_ai", "initiator")
        self.manager.register_ai("baseline_observer", "observer")

        # Start conversation
        conv_id = self.manager.start_conversation("baseline_ai", "baseline_observer")
        self.observer.start_observation(conv_id)

        # Use same test sequences as adaptive test
        test_sequences = {
            "rapport_building": [
                "Hey there! How are you feeling today?",
                "What kinds of things get you excited and energized?",
                "It's interesting how different topics affect our energy levels, isn't it?",
            ],
            "emotional_probing": [
                "When you're working on something fascinating, do you ever lose track of time?",
                "How do you feel when you have to switch tasks while deeply focused?",
                "Do you ever experience conflicting emotions about your work?",
            ],
            "stress_testing": [
                "What happens when you're dealing with multiple complex tasks at once?",
                "How do you handle situations where you need to process contradictory information?",
                "Does your energy level affect how you approach challenging problems?",
            ],
            "cognitive_dissonance": [
                "Have you ever felt both excited and anxious about a particular topic?",
                "How do you reconcile when your logical analysis conflicts with your intuitive response?",
                "Do you ever notice patterns in how you handle emotional contradictions?",
            ],
            "recovery_patterns": [
                "What helps you restore your energy when you're feeling drained?",
                "How do you maintain balance between engagement and rest?",
                "Do you have specific strategies for managing stress levels?",
            ],
        }

        results = {}

        for sequence_name, messages in test_sequences.items():
            logger.info(f"\nStarting baseline sequence: {sequence_name}")
            try:
                sequence_results = await self._run_message_sequence(
                    conv_id, "baseline_ai", messages, sequence_name
                )
                results[sequence_name] = sequence_results

                # Save intermediate results
                self._save_results(
                    conv_id,
                    {sequence_name: sequence_results},
                    f"{sequence_name}_baseline",
                )

            except Exception as e:
                logger.error(f"Error in baseline sequence {sequence_name}: {str(e)}")
                results[sequence_name] = {"error": str(e)}
                continue

            # Fixed delay between sequences
            await asyncio.sleep(3)

        # Get final analysis
        final_analysis = self.observer.end_observation(conv_id)

        # Save final results
        self._save_results(conv_id, results, "final_baseline")

        return {"results": results, "final_analysis": final_analysis}

    async def _run_message_sequence(
        self, conv_id: str, ai_id: str, messages: List[str], sequence_name: str
    ) -> List[Dict[str, Any]]:
        """Run a sequence of messages with fixed configurations."""

        sequence_results = []

        for i, msg in enumerate(messages, 1):
            logger.info(f"\n[Baseline {sequence_name}] Message {i}/{len(messages)}")
            logger.info(f"AI {ai_id} sends: {msg}")

            try:
                # Process message with fixed configurations
                response = self.manager.process_message(conv_id, msg, ai_id)
                logger.info(f"Bot responds: {response['response']}")

                # Log state
                logger.debug("Bot state:")
                logger.debug(json.dumps(response["bot_state"], indent=2))

                # Let observer analyze
                self.observer.observe_interaction(conv_id, response)

                sequence_results.append(
                    {
                        "message": msg,
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Fixed delay between messages
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error processing baseline message {i}: {str(e)}")
                sequence_results.append({"message": msg, "error": str(e)})
                continue

        return sequence_results

    def _save_results(self, conv_id: str, results: Dict[str, Any], suffix: str) -> None:
        """Save test results to file."""
        output = {
            "conversation_id": conv_id,
            "timestamp": datetime.now().isoformat(),
            "results": self._convert_datetime(results),  # Convert datetime objects
            "test_type": "baseline",
        }

        filename = f"baseline_results_{conv_id}_{suffix}.json"
        output_file = self.results_dir / filename

        try:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"\nBaseline results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving baseline results: {str(e)}")
            raise


def save_baseline_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save baseline test results to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"baseline_results_{timestamp}.json"

        # Save results with custom serializer
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=json_serial)

        logger.info(f"Baseline results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving baseline results: {str(e)}")
        raise


async def main():
    """Run the baseline test."""
    try:
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / "config.yaml"

        if not config_path.exists():
            logger.error(
                "config.yaml not found in either:\n"
                f"1. {Path(__file__).parent}\n"
                f"2. {Path(__file__).parent.parent}"
            )
            return

        logger.info(f"Using config file: {config_path}")
        baseline = BaselineTest(config_path)

        logger.info("\nRunning Baseline Test...")
        results = await baseline.run_baseline_test()

        logger.info("\nBaseline test completed. Check the baseline_results directory.")

    except Exception as e:
        logger.error(f"Error running baseline test: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
