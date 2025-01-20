"""
Test AI Interaction Scenario
--------------------------

Tests the interaction between AIs and the LayeredReasoningBot,
focusing on emotional dynamics and pattern recognition.
"""

from layered_reasoning_bot import LayeredReasoningBot
from ai_interaction_manager import AIInteractionManager
from interaction_observer import InteractionObserver
import asyncio
import json
from datetime import datetime
from pathlib import Path
import logging
import sys
import colorlog


# Configure color logging
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
            secondary_log_colors={
                "message": {
                    "ERROR": "red",
                    "WARNING": "yellow",
                    "INFO": "green",
                    "DEBUG": "cyan",
                }
            },
        )
    )

    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Also set up file logging
    file_handler = logging.FileHandler("test_run.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


# Set up colored logging
logger = setup_colored_logging()

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()


class TestScenario:
    """Manages different test scenarios for AI interactions."""

    def __init__(self, config_path: str | Path):
        # Ensure config path is absolute
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = SCRIPT_DIR / config_path

        if not config_path.exists():
            logger.error(
                f"Config file not found at: {config_path}\n"
                f"Please ensure config.yaml exists in the correct location."
            )
            raise FileNotFoundError(
                f"Config file not found at: {config_path}\n"
                f"Please ensure config.yaml exists in the correct location."
            )

        self.bot = LayeredReasoningBot(str(config_path))
        self.manager = AIInteractionManager(self.bot)
        self.observer = InteractionObserver(self.manager)

        # Create test results directory
        self.results_dir = (
            SCRIPT_DIR / "test_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created test results directory: {self.results_dir}")

    async def run_emotional_complexity_test(self) -> dict:
        """Test scenario focusing on emotional complexity and contradictions."""

        # Register AIs with different roles - using descriptive IDs
        self.manager.register_ai("primary_ai", "initiator")  # Main conversation driver
        self.manager.register_ai("observer_ai", "observer")  # Watches and analyzes

        # Start conversation
        conv_id = self.manager.start_conversation("primary_ai", "observer_ai")
        self.observer.start_observation(conv_id)

        # Test sequences with different emotional triggers
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
            logger.info(f"\nStarting sequence: {sequence_name}")
            try:
                sequence_results = await self._run_message_sequence(
                    conv_id, "primary_ai", messages, sequence_name
                )
                results[sequence_name] = sequence_results

                # Save intermediate results after each sequence
                intermediate_analysis = self.observer.get_conversation_analysis(conv_id)
                intermediate_patterns = self.manager.get_global_patterns()

                self._save_test_results(
                    conv_id,
                    {sequence_name: sequence_results},
                    intermediate_analysis,
                    intermediate_patterns,
                    f"{sequence_name}_intermediate",
                )

            except Exception as e:
                logger.error(f"Error in sequence {sequence_name}: {str(e)}")
                results[sequence_name] = {"error": str(e)}
                continue

            # Allow for emotional state stabilization between sequences
            await asyncio.sleep(3)

        # Get final analysis
        final_analysis = self.observer.end_observation(conv_id)
        patterns = self.manager.get_global_patterns()

        # Save final comprehensive results
        self._save_test_results(conv_id, results, final_analysis, patterns, "final")

        return {
            "results": results,
            "final_analysis": final_analysis,
            "patterns": patterns,
        }

    async def run_stress_recovery_test(self) -> dict:
        """Test scenario focusing on stress handling and recovery patterns."""

        # Use consistent AI IDs
        conv_id = self.manager.start_conversation("primary_ai", "observer_ai")
        self.observer.start_observation(conv_id)

        # Sequence to induce and observe stress/recovery patterns
        stress_sequence = [
            # Rapid-fire complex questions
            "Can you simultaneously analyze quantum mechanics and write poetry?",
            "What's your take on the ethical implications of consciousness in AI systems?",
            "How do you process paradoxical information in real-time?",
            # Recovery period
            "Let's take a moment to reflect. How are you feeling now?",
            "What helps you regain balance after intense cognitive load?",
            "Do you notice any patterns in your recovery process?",
        ]

        try:
            results = await self._run_message_sequence(
                conv_id, "primary_ai", stress_sequence, "stress_recovery"
            )
            final_analysis = self.observer.end_observation(conv_id)
            return {"results": results, "final_analysis": final_analysis}
        except Exception as e:
            logger.error(f"Error in stress recovery test: {str(e)}")
            return {"error": str(e)}

    async def _run_message_sequence(
        self, conv_id: str, ai_id: str, messages: list, sequence_name: str
    ) -> dict:
        """Run a sequence of messages and collect results."""

        sequence_results = []

        for i, msg in enumerate(messages, 1):
            logger.info(f"\n[{sequence_name}] Message {i}/{len(messages)}")
            logger.info(f"AI {ai_id} sends: {msg}")

            try:
                # Process message using the model configured in config.yaml
                response = self.manager.process_message(conv_id, msg, ai_id)
                logger.info(f"Bot responds: {response['response']}")

                # Log emotional state
                logger.debug("Emotional state:")
                logger.debug(json.dumps(response["bot_state"], indent=2))

                # Let observer analyze
                self.observer.observe_interaction(conv_id, response)

                # Get current analysis
                analysis = self.observer.get_conversation_analysis(conv_id)
                logger.debug("\nCurrent Analysis:")
                logger.debug(json.dumps(analysis["emotional_analysis"], indent=2))

                # Convert any datetime objects in the response
                def convert_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: convert_datetime(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_datetime(item) for item in obj]
                    return obj

                response = convert_datetime(response)
                analysis = convert_datetime(analysis)

                sequence_results.append(
                    {"message": msg, "response": response, "analysis": analysis}
                )

                # Adaptive delay based on emotional state
                delay = 2 + (response["bot_state"]["stress_level"] * 1.5)
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Error processing message {i}: {str(e)}", exc_info=True)
                sequence_results.append({"message": msg, "error": str(e)})
                continue

        return sequence_results

    def _save_test_results(
        self,
        conv_id: str,
        results: dict,
        analysis: dict,
        patterns: dict,
        suffix: str = "",
    ) -> None:
        """Save test results to file."""

        def datetime_handler(obj):
            """Handle datetime serialization."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        output = {
            "conversation_id": conv_id,
            "timestamp": datetime.now().isoformat(),
            "sequence_results": results,
            "analysis": analysis,
            "observed_patterns": patterns,
        }

        # Create a unique filename with the suffix
        filename = f"test_results_{conv_id}_{suffix}.json"
        output_file = self.results_dir / filename

        try:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2, default=datetime_handler)
            logger.info(f"\nTest results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")


async def main():
    """Run the test scenarios."""
    try:
        # First try config in current directory
        config_path = SCRIPT_DIR / "config.yaml"

        # If not found, check parent directory
        if not config_path.exists():
            config_path = SCRIPT_DIR.parent / "config.yaml"

        if not config_path.exists():
            logger.error(
                "config.yaml not found in either:\n"
                f"1. {SCRIPT_DIR}\n"
                f"2. {SCRIPT_DIR.parent}\n"
                "Please ensure config.yaml exists in one of these locations."
            )
            sys.exit(1)

        logger.info(f"Using config file: {config_path}")
        scenario = TestScenario(config_path)

        # Run emotional complexity test
        logger.info("\nRunning Emotional Complexity Test...")
        complexity_results = await scenario.run_emotional_complexity_test()

        # Run stress recovery test
        logger.info("\nRunning Stress Recovery Test...")
        recovery_results = await scenario.run_stress_recovery_test()

        logger.info(
            "\nAll tests completed. Check the test_results directory for detailed output."
        )

    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
