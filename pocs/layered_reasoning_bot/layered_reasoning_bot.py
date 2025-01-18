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
from typing import Any, Dict, List, Optional, Iterator, Tuple
from datetime import datetime
from pathlib import Path
import yaml
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential
from bayesian_strategy import BayesianStrategySelector
from config_types import ModelConfig, StrategyConfig, MemoryConfig, BotConfig
import json

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
        self.episodic_memory: Dict[str, Any] = {
            "greetings_used": set(),
            "topics_discussed": set(),
            "conversation_stage": "initial",  # initial, engaged, winding_down
            "interaction_patterns": {},
            "last_emotion_scores": {},
        }
        self.short_term_focus = {
            "recent_topics": [],  # Last 3 main topics
            "pending_questions": [],  # Questions waiting for answers
            "repeated_elements": {},  # Track repeated phrases/topics
            "active_context": set(),  # Currently active context elements
        }

    def _update_short_term_focus(
        self, user_input: str, bot_response: str, metadata: Dict[str, Any]
    ) -> None:
        """Update short-term memory focus."""
        # Track repeated elements
        for text in [user_input.lower(), bot_response.lower()]:
            for phrase in text.split():
                if phrase in self.short_term_focus["repeated_elements"]:
                    self.short_term_focus["repeated_elements"][phrase] += 1
                else:
                    self.short_term_focus["repeated_elements"][phrase] = 1

        # Update active context
        self.short_term_focus["active_context"].update(
            word.lower() for word in user_input.split()
        )

        # Maintain context window
        if len(self.short_term_focus["active_context"]) > 50:
            self.short_term_focus["active_context"] = set(
                list(self.short_term_focus["active_context"])[-50:]
            )

        # Track questions
        if "?" in bot_response:
            self.short_term_focus["pending_questions"].append(
                {
                    "question": bot_response,
                    "timestamp": datetime.now(),
                    "answered": False,
                }
            )
        elif self.short_term_focus["pending_questions"]:
            # Mark questions as answered if user responds
            self.short_term_focus["pending_questions"][-1]["answered"] = True

        # Update conversation stage
        if len(self.history) == 0:
            self.episodic_memory["conversation_stage"] = "initial"
        elif len(self.history) > 10:
            self.episodic_memory["conversation_stage"] = "engaged"

        # Track emotional progression
        if "evaluation_scores" in metadata:
            self.episodic_memory["last_emotion_scores"] = metadata[
                "evaluation_scores"
            ].get("emotional_alignment", {})

    def add_interaction(
        self, user_input: str, bot_response: str, metadata: Dict[str, Any]
    ) -> None:
        """Add a new interaction to the conversation history."""
        # Update short-term focus before adding new interaction
        self._update_short_term_focus(user_input, bot_response, metadata)

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "metadata": metadata,
            "context_state": {
                "active_topics": list(self.short_term_focus["active_context"]),
                "pending_questions": self.short_term_focus["pending_questions"],
                "conversation_stage": self.episodic_memory["conversation_stage"],
            },
        }
        self.history.append(interaction)

        # Update episodic memory
        if "greetings" in user_input.lower():
            self.episodic_memory["greetings_used"].add(bot_response)

        if len(self.history) > self.config.conversation_history_length:
            self.history.pop(0)

    def get_context(self) -> str:
        """Get formatted conversation context for the model."""
        context_elements = []

        # Add conversation stage context
        context_elements.append(
            f"Conversation Stage: {self.episodic_memory['conversation_stage']}"
        )

        # Add active context elements
        if self.short_term_focus["active_context"]:
            context_elements.append(
                f"Active Topics: {', '.join(self.short_term_focus['active_context'])}"
            )

        # Add pending questions
        unanswered_questions = [
            q["question"]
            for q in self.short_term_focus["pending_questions"]
            if not q["answered"]
        ]
        if unanswered_questions:
            context_elements.append(f"Pending Questions: {unanswered_questions[-1]}")

        # Add recent interactions
        for interaction in self.history[-3:]:
            context_elements.extend(
                [
                    f"User: {interaction['user_input']}",
                    f"Assistant: {interaction['bot_response']}",
                ]
            )

        # Add repetition warnings
        repeated = [
            word
            for word, count in self.short_term_focus["repeated_elements"].items()
            if count > 2
        ]
        if repeated:
            context_elements.append(
                f"Note: Already frequently used: {', '.join(repeated)}"
            )

        return "\n".join(context_elements)


class ResponseEvaluator:
    """Evaluates response quality across multiple dimensions with historical tracking."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

        # Dimension weights for final score
        self.weights = {
            "coherence": 0.25,
            "relevance": 0.25,
            "emotional_alignment": 0.2,
            "strategy_adherence": 0.15,
            "engagement": 0.15,
        }

        # Historical tracking of scores
        self.score_history: List[Dict[str, float]] = []
        self.moving_averages: Dict[str, float] = {dim: 0.5 for dim in self.weights}

        # Emotional analysis patterns
        self.emotion_patterns = {
            "joy": ["happy", "excited", "glad", "great", "wonderful", "fantastic"],
            "sadness": ["sad", "disappointed", "sorry", "upset", "unhappy"],
            "anger": ["angry", "frustrated", "annoyed", "mad", "irritated"],
            "fear": ["worried", "scared", "anxious", "nervous", "concerned"],
            "surprise": ["wow", "amazing", "unexpected", "incredible", "unbelievable"],
            "analytical": ["think", "consider", "analyze", "understand", "reason"],
            "confident": ["sure", "certain", "definitely", "absolutely", "clearly"],
            "tentative": ["maybe", "perhaps", "possibly", "might", "could"],
        }

    def _get_emotion_vector(self, text: str) -> Dict[str, float]:
        """Create an emotion vector for text using pattern matching and context."""
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_patterns}
        words = text.lower().split()

        # Score each emotion based on pattern matches
        for emotion, patterns in self.emotion_patterns.items():
            matches = sum(
                1
                for pattern in patterns
                if any(word.startswith(pattern) for word in words)
            )
            emotion_scores[emotion] = min(1.0, matches * 0.3)

        return emotion_scores

    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate response coherence and structure."""
        score = 0.5

        # Analyze sentence structure
        sentences = response.split(". ")
        if 2 <= len(sentences) <= 5:
            score += 0.2

        # Check for logical flow markers
        flow_markers = {
            "introduction": ["first", "to begin", "initially"],
            "continuation": ["additionally", "moreover", "furthermore"],
            "contrast": ["however", "although", "conversely"],
            "conclusion": ["therefore", "thus", "consequently"],
        }

        flow_score = 0.0
        for marker_type, markers in flow_markers.items():
            if any(marker in response.lower() for marker in markers):
                flow_score += 0.1

        score += min(0.3, flow_score)

        # Check for pronoun consistency
        pronouns = ["he", "she", "it", "they", "this", "that"]
        has_antecedents = all(
            response[:idx].lower().split()[-5:] != []  # Check previous context
            for idx in [
                i for i, word in enumerate(response.lower().split()) if word in pronouns
            ]
        )
        if has_antecedents:
            score += 0.1

        return min(1.0, score)

    def _evaluate_relevance(
        self, response: str, context: str, user_input: str
    ) -> float:
        """Evaluate response relevance using semantic and contextual analysis."""
        score = 0.5

        # Topic continuity
        user_topics = set(user_input.lower().split())
        response_topics = set(response.lower().split())
        context_topics = set(context.lower().split())

        # Direct topic overlap
        topic_overlap = len(user_topics.intersection(response_topics))
        score += min(0.2, topic_overlap * 0.05)

        # Context awareness
        context_overlap = len(context_topics.intersection(response_topics))
        score += min(0.2, context_overlap * 0.03)

        # Check for question-answer alignment
        if "?" in user_input:
            question_words = {"what", "why", "how", "when", "where", "who"}
            user_question_type = next(
                (word for word in user_input.lower().split() if word in question_words),
                None,
            )
            if user_question_type:
                # Check if response addresses the question type
                question_response_patterns = {
                    "what": ["is", "are", "was", "were"],
                    "why": ["because", "since", "as", "due to"],
                    "how": ["by", "through", "using", "steps"],
                    "when": ["at", "on", "in", "during"],
                    "where": ["at", "in", "near", "located"],
                    "who": ["person", "people", "they", "someone"],
                }
                if any(
                    pattern in response.lower()
                    for pattern in question_response_patterns.get(
                        user_question_type, []
                    )
                ):
                    score += 0.2

        return min(1.0, score)

    def _evaluate_emotional_alignment(
        self, response: str, intent_analysis: Dict[str, Any]
    ) -> float:
        """Evaluate emotional alignment using emotion vectors and context."""
        score = 0.5

        # Get emotion vectors
        intent_emotions = self._get_emotion_vector(intent_analysis["raw_analysis"])
        response_emotions = self._get_emotion_vector(response)

        # Calculate emotional alignment score
        emotion_similarity = sum(
            min(intent_emotions[e], response_emotions[e]) for e in intent_emotions
        ) / len(intent_emotions)
        score += emotion_similarity * 0.3

        # Check for appropriate emotion modulation
        if max(intent_emotions.values()) > 0.7:  # Strong emotion detected
            if (
                0.4 <= max(response_emotions.values()) <= 0.9
            ):  # Appropriate response intensity
                score += 0.2

        # Evaluate empathy markers in emotional contexts
        empathy_patterns = {
            "understanding": ["understand", "see why", "recognize"],
            "validation": ["valid", "natural to feel", "makes sense"],
            "support": ["here for you", "support", "help"],
        }

        if max(intent_emotions.values()) > 0.5:  # Emotional context
            empathy_score = sum(
                0.1
                for patterns in empathy_patterns.values()
                if any(pattern in response.lower() for pattern in patterns)
            )
            score += min(0.2, empathy_score)

        return min(1.0, score)

    def _evaluate_strategy_adherence(self, response: str, strategy: str) -> float:
        """Evaluate how well response follows chosen strategy."""
        score = 0.5

        strategy_patterns = {
            "factual_explanation": {
                "primary": ["because", "due to", "explains", "means"],
                "secondary": ["specifically", "research shows", "evidence", "studies"],
                "structure": ["first", "second", "finally", "in conclusion"],
            },
            "clarifying_question": {
                "primary": ["could you", "what do you", "how do you", "?"],
                "secondary": ["help me understand", "to clarify", "you mean"],
                "structure": ["earlier you mentioned", "about that", "regarding"],
            },
            "example_based": {
                "primary": ["for example", "like when", "similar to", "instance"],
                "secondary": ["imagine", "picture", "scenario", "case"],
                "structure": ["in this case", "another example", "similarly"],
            },
            "reframing": {
                "primary": ["another way", "perspective", "think about", "consider"],
                "secondary": ["alternatively", "different angle", "reframe", "shift"],
                "structure": ["instead of", "rather than", "looking at it"],
            },
        }

        if strategy in strategy_patterns:
            patterns = strategy_patterns[strategy]

            # Primary pattern matching
            primary_matches = sum(
                1 for pattern in patterns["primary"] if pattern in response.lower()
            )
            score += min(0.3, primary_matches * 0.15)

            # Secondary pattern matching
            secondary_matches = sum(
                1 for pattern in patterns["secondary"] if pattern in response.lower()
            )
            score += min(0.2, secondary_matches * 0.1)

            # Structure pattern matching
            structure_matches = sum(
                1 for pattern in patterns["structure"] if pattern in response.lower()
            )
            score += min(0.2, structure_matches * 0.1)

        return min(1.0, score)

    def _evaluate_engagement(self, response: str) -> float:
        """Evaluate how engaging and interactive the response is."""
        score = 0.5

        # Interactive elements
        interaction_patterns = {
            "questions": ["?", "what do you think", "how about"],
            "invitations": ["let's", "would you like", "try this"],
            "acknowledgments": ["you mentioned", "as you said", "your point about"],
            "personalizations": ["your", "you", "we", "our"],
        }

        for pattern_type, patterns in interaction_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in response.lower())
            score += min(0.15, matches * 0.05)

        # Dynamic language
        dynamic_phrases = [
            "imagine",
            "picture",
            "consider",
            "think about",
            "explore",
            "discover",
            "create",
            "build",
            "feel",
            "experience",
            "notice",
            "observe",
        ]

        dynamic_count = sum(
            1 for phrase in dynamic_phrases if phrase in response.lower()
        )
        score += min(0.2, dynamic_count * 0.1)

        return min(1.0, score)

    def evaluate_response(
        self,
        response: str,
        user_input: str,
        context: str,
        intent_analysis: Dict[str, Any],
        strategy: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate response quality across all dimensions and update history."""
        # Calculate current scores
        current_scores = {
            "coherence": self._evaluate_coherence(response),
            "relevance": self._evaluate_relevance(response, context, user_input),
            "emotional_alignment": self._evaluate_emotional_alignment(
                response, intent_analysis
            ),
            "strategy_adherence": self._evaluate_strategy_adherence(response, strategy),
            "engagement": self._evaluate_engagement(response),
        }

        # Update moving averages (with 0.1 learning rate)
        for dimension, score in current_scores.items():
            self.moving_averages[dimension] = (
                0.9 * self.moving_averages[dimension] + 0.1 * score
            )

        # Store historical data
        self.score_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "scores": current_scores.copy(),
                "moving_averages": self.moving_averages.copy(),
            }
        )

        # Calculate final score using weighted average
        final_score = sum(
            current_scores[dim] * self.weights[dim] for dim in current_scores
        )

        # Add trending information to the scores
        scores_with_trends = {
            dim: {"current": score, "trend": score - self.moving_averages[dim]}
            for dim, score in current_scores.items()
        }

        return final_score, scores_with_trends


class LayeredReasoningBot:
    """Main bot implementation with layered reasoning capabilities."""

    def __init__(self, config_path: str):
        self.config = BotConfig.from_yaml(config_path)
        self.memory = ConversationMemory(self.config.memory)
        self.strategy_selector = BayesianStrategySelector(
            self.config.strategy.strategy_options,
            self.config.models["feature_extraction"],
        )
        self.response_evaluator = ResponseEvaluator(
            self.config.models["response_generation"]
        )

        # Set up detailed logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Add file handler for detailed logs
        log_file = SCRIPT_DIR / "detailed_conversation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # set terminal handler
        terminal_handler = logging.StreamHandler()
        terminal_handler.setLevel(logging.ERROR)

        # Use a simpler formatter that doesn't require extra fields
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _log_step(self, step: str, details: Dict[str, Any]) -> None:
        """Log a processing step with detailed information."""
        self.logger.info(f"[{step}] {json.dumps(details, default=str)}")

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

    def _estimate_response_success(
        self,
        response: str,
        user_input: str,
        intent_analysis: Dict[str, Any],
        strategy: str,
    ) -> float:
        """Estimate the success score of a response using sophisticated metrics."""
        final_score, dimension_scores = self.response_evaluator.evaluate_response(
            response=response,
            user_input=user_input,
            context=self.memory.get_context(),
            intent_analysis=intent_analysis,
            strategy=strategy,
        )

        # Log detailed scores for analysis
        logger.debug(f"Response evaluation scores: {dimension_scores}")
        logger.debug(f"Final response score: {final_score}")

        return final_score

    def generate_response(
        self, user_input: str, intent_analysis: Dict[str, Any], strategy: str
    ) -> str:
        """Generate the final response using the chosen strategy."""
        # Get conversation stage and context
        conversation_stage = self.memory.episodic_memory["conversation_stage"]
        last_emotion_scores = self.memory.episodic_memory.get("last_emotion_scores", {})

        # Determine appropriate enthusiasm level
        enthusiasm_level = "moderate"
        if conversation_stage == "initial":
            enthusiasm_level = "high"
        elif conversation_stage == "engaged":
            # Adapt to user's emotional state
            if last_emotion_scores.get("current", 0.5) > 0.7:
                enthusiasm_level = "high"
            elif last_emotion_scores.get("current", 0.5) < 0.3:
                enthusiasm_level = "low"

        # Check for repetition warnings
        repeated_elements = self.memory.short_term_focus["repeated_elements"]
        repetition_warnings = [
            word for word, count in repeated_elements.items() if count > 2
        ]

        # Enhanced prompt with emotional calibration
        prompt = f"""Generate a natural, conversational response using this strategy: {strategy}
        User Input: {user_input}
        Intent Analysis: {intent_analysis['raw_analysis']}
        Context: {self.memory.get_context()}
        
        Requirements:
        1. DO NOT mention or refer to the strategy in your response
        2. Keep the response natural and conversational
        3. Make it contextually appropriate
        4. Focus on engaging with the user's message
        5. Maintain {enthusiasm_level} enthusiasm level
        6. Avoid repeating these elements: {', '.join(repetition_warnings) if repetition_warnings else 'None'}
        7. Match conversation stage: {conversation_stage}
        8. If roleplay elements are present:
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

        # Estimate response success using new evaluation system
        success_score = self._estimate_response_success(
            full_response, user_input, intent_analysis, strategy
        )

        # Update strategy selector with evaluated score
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
            self._log_step(
                "INTENT_RECOGNITION_START",
                {
                    "user_input": user_input,
                    "context_length": len(self.memory.get_context()),
                },
            )

            intent_analysis = self.understand_intent(user_input)

            self._log_step(
                "INTENT_RECOGNITION_COMPLETE",
                {
                    "analysis": intent_analysis,
                    "detected_emotions": self.response_evaluator._get_emotion_vector(
                        intent_analysis["raw_analysis"]
                    ),
                },
            )

            # Layer 2: Strategy Formation
            self._log_step(
                "STRATEGY_FORMATION_START",
                {
                    "current_priors": self.strategy_selector.priors.tolist(),
                    "intent_features": intent_analysis,
                },
            )

            strategy = self.form_strategy(intent_analysis)

            self._log_step(
                "STRATEGY_FORMATION_COMPLETE",
                {
                    "selected_strategy": strategy,
                    "strategy_probabilities": self.strategy_selector.get_strategy_insights(),
                },
            )

            # Layer 3: Response Generation
            self._log_step(
                "RESPONSE_GENERATION_START",
                {"strategy": strategy, "context_used": self.memory.get_context()},
            )

            response = self.generate_response(user_input, intent_analysis, strategy)

            # Evaluate response quality
            success_score, dimension_scores = self.response_evaluator.evaluate_response(
                response=response,
                user_input=user_input,
                context=self.memory.get_context(),
                intent_analysis=intent_analysis,
                strategy=strategy,
            )

            self._log_step(
                "RESPONSE_EVALUATION_COMPLETE",
                {
                    "response": response,
                    "success_score": success_score,
                    "dimension_scores": dimension_scores,
                    "historical_trends": {
                        dim: {
                            "current": scores["current"],
                            "trend": scores["trend"],
                            "moving_avg": self.response_evaluator.moving_averages[dim],
                        }
                        for dim, scores in dimension_scores.items()
                    },
                },
            )

            # Update memory
            self.memory.add_interaction(
                user_input=user_input,
                bot_response=response,
                metadata={
                    "intent_analysis": intent_analysis,
                    "strategy": strategy,
                    "evaluation_scores": dimension_scores,
                },
            )

            self._log_step(
                "INTERACTION_COMPLETE",
                {
                    "total_steps": 4,
                    "memory_size": len(self.memory.history),
                    "conversation_stats": {
                        "total_interactions": len(self.memory.history),
                        "avg_response_length": (
                            sum(
                                len(h["bot_response"].split())
                                for h in self.memory.history
                            )
                            / len(self.memory.history)
                            if self.memory.history
                            else 0
                        ),
                    },
                },
            )

            return response

        except Exception as e:
            self._log_step(
                "ERROR",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_stage": "Unknown",
                },
            )
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
