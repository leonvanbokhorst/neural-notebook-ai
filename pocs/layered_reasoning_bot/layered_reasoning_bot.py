"""
Layered Reasoning Chatbot
------------------------

A chatbot implementation with layered reasoning capabilities for enhanced
understanding and response generation.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterator
from datetime import datetime
from pathlib import Path
import yaml
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential
from bayesian_strategy import BayesianStrategySelector
from config_types import ModelConfig, StrategyConfig, MemoryConfig, BotConfig

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()


class ConversationMemory:
    """Manages conversation history and episodic memory."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.episodic_memory: Dict[str, Any] = {}

    def add_interaction(
        self, user_input: str, bot_response: str, metadata: Dict[str, Any]
    ) -> None:
        """Add a new interaction to the conversation history."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "metadata": metadata,
        }
        self.history.append(interaction)

        if len(self.history) > self.config.conversation_history_length:
            self.history.pop(0)

    def get_context(self) -> str:
        """Get formatted conversation context for the model."""
        context = []
        for interaction in self.history[
            -3:
        ]:  # Last 3 interactions for immediate context
            context.extend(
                (
                    f"User: {interaction['user_input']}",
                    f"Assistant: {interaction['bot_response']}",
                )
            )
        return "\n".join(context)


class LayeredReasoningBot:
    """Main bot implementation with layered reasoning capabilities."""

    def __init__(self, config_path: str):
        self.config = BotConfig.from_yaml(config_path)
        self.memory = ConversationMemory(self.config.memory)
        self.strategy_selector = BayesianStrategySelector(
            self.config.strategy.strategy_options,
            self.config.models["feature_extraction"],
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_model(
        self, prompt: str, config: ModelConfig, stream: bool = False
    ) -> Any:
        """Make an API call to the language model using LiteLLM."""
        try:
            response = completion(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=stream,
            )

            return response if stream else response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LiteLLM: {str(e)}")
            raise

    def _stream_to_console(
        self, response_stream: Iterator[Any], prefix: str = "Bot"
    ) -> str:
        """Stream the response to console and return the full response."""
        full_response = []
        print(f"\n{prefix}: ", end="", flush=True)

        for chunk in response_stream:
            if hasattr(chunk.choices[0], "delta") and hasattr(
                chunk.choices[0].delta, "content"
            ):
                if content := chunk.choices[0].delta.content:
                    print(content, end="", flush=True)
                    full_response.append(content)

        print("\n")  # New line after response
        return "".join(full_response)

    def understand_intent(self, user_input: str) -> Dict[str, Any]:
        """Parse and understand user intent."""
        prompt = f"""Analyze this user message briefly and naturally:
        Message: {user_input}
        Context: {self.memory.get_context()}
        
        Give a quick analysis covering:
        1. Main intent (1-2 sentence)
        2. Emotional tone (1-2 words)
        3. Key topics (2-3 keywords)
        4. Personality (1-3 words)
        5. Hidden meaning/assumptions (1-2 sentence)
        6. Roleplay elements (if any):
           - Assumed identity/relationship
           - Shared context clues
           - Personal references
        
        Keep it short and human-like conversational. No formal headers or sections needed."""

        # Get streaming response for analysis
        response_stream = self._call_model(
            prompt, self.config.models["intent_recognition"], stream=True
        )

        analysis = self._stream_to_console(response_stream, prefix="Intent")
        return {"raw_analysis": analysis}

    def form_strategy(self, intent_analysis: Dict[str, Any]) -> str:
        """Decide on the response strategy using Bayesian inference."""
        strategy, probabilities = self.strategy_selector.select_strategy(
            intent_analysis["raw_analysis"]
        )

        # Log the strategy selection probabilities
        logger.error(f"Strategy probabilities: {probabilities}")

        return strategy

    def _estimate_response_success(self, response: str) -> float:
        """Estimate the success score of a response based on various factors."""
        # This is a simple heuristic - could be enhanced with more sophisticated metrics
        score = 0.5  # Base score

        # Length-based adjustment (penalize very short or very long responses)
        length = len(response.split())
        if 20 <= length <= 200:
            score += 0.2

        # Coherence indicators
        if "however" in response.lower() or "moreover" in response.lower():
            score += 0.1

        # Presence of examples or explanations
        if "for example" in response.lower() or "such as" in response.lower():
            score += 0.1

        # Normalize score to 0-1 range
        return min(1.0, score)

    def generate_response(
        self, user_input: str, intent_analysis: Dict[str, Any], strategy: str
    ) -> str:
        """Generate the final response using the chosen strategy."""
        prompt = f"""Generate a natural, conversational response using this strategy: {strategy}
        User Input: {user_input}
        Intent Analysis: {intent_analysis['raw_analysis']}
        Context: {self.memory.get_context()}
        
        Requirements:
        1. DO NOT mention or refer to the strategy in your response
        2. Keep the response natural and conversational
        3. Make it contextually appropriate
        4. Focus on engaging with the user's message
        5. If roleplay elements are present:
           - Stay in character consistently
           - Maintain the established relationship dynamic
           - Use appropriate personal references
           - Keep shared context in mind
        
        Just give the response directly, no meta-text or explanations."""

        # Get streaming response
        response_stream = self._call_model(
            prompt, self.config.models["response_generation"], stream=True
        )

        # Stream and collect the full response
        print("\n")
        full_response = self._stream_to_console(response_stream, prefix="Response")

        # Estimate response success and update strategy selector
        success_score = self._estimate_response_success(full_response)
        strategy_idx = self.config.strategy.strategy_options.index(strategy)
        self.strategy_selector.update_priors(strategy_idx, success_score)

        # Log strategy performance
        insights = self.strategy_selector.get_strategy_insights()
        logger.debug(f"Strategy insights: {insights}")

        return full_response

    def process_message(self, user_input: str) -> str:
        """Process a user message through all reasoning layers."""
        try:
            # Layer 1: Intent Recognition
            intent_analysis = self.understand_intent(user_input)
            logger.debug(f"Intent analysis: {intent_analysis}")

            # Layer 2: Strategy Formation
            strategy = self.form_strategy(intent_analysis)
            logger.info(f"Chosen strategy: {strategy}")

            # Layer 3: Response Generation
            response = self.generate_response(user_input, intent_analysis, strategy)

            # Update memory
            self.memory.add_interaction(
                user_input=user_input,
                bot_response=response,
                metadata={"intent_analysis": intent_analysis, "strategy": strategy},
            )

            return response

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error processing your message. Could you please try rephrasing it?"


def main():
    """Main entry point for the POC."""
    try:
        # Initialize bot with configuration using correct path
        config_path = SCRIPT_DIR / "config.yaml"
        bot = LayeredReasoningBot(str(config_path))

        # Interactive loop for testing
        print("Layered Reasoning Bot initialized. Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                break

            bot.process_message(user_input)  # Response is streamed directly

    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        raise


if __name__ == "__main__":
    main()