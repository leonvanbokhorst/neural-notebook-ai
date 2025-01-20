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
from emotional_state import EmotionalState
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
            "trust": ["believe", "trust", "rely", "depend", "faith"],
            "disgust": ["ugh", "gross", "disgusting", "awful", "terrible"],
            "anticipation": ["expect", "hope", "look forward", "await", "anticipate"],
        }

        # Emotional overspill tracking
        self.emotional_residue: Dict[str, float] = {
            emotion: 0.0 for emotion in self.emotion_patterns
        }
        self.residue_decay = 0.8  # Decay factor for emotional residue

        # Mood-based response bias tracking
        self.mood_bias = {
            "positivity": 0.0,  # -1.0 to 1.0
            "formality": 0.0,  # -1.0 to 1.0
            "engagement": 0.0,  # -1.0 to 1.0
        }
        self.mood_inertia = 0.7  # Resistance to mood changes

    def _get_emotion_vector(
        self, text: str, include_residue: bool = True
    ) -> Dict[str, float]:
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
            base_score = min(1.0, matches * 0.3)

            # Include emotional residue if enabled
            if include_residue:
                residue = self.emotional_residue.get(emotion, 0.0)
                emotion_scores[emotion] = min(1.0, base_score + (residue * 0.3))
            else:
                emotion_scores[emotion] = base_score

        return emotion_scores

    def _update_emotional_residue(self, emotion_scores: Dict[str, float]) -> None:
        """Update emotional residue based on current emotions."""
        # Decay existing residue
        for emotion in self.emotional_residue:
            self.emotional_residue[emotion] *= self.residue_decay

        # Add new emotional residue
        for emotion, score in emotion_scores.items():
            if score > 0.5:  # Only strong emotions create residue
                self.emotional_residue[emotion] = min(
                    1.0, self.emotional_residue.get(emotion, 0.0) + (score * 0.2)
                )

    def _update_mood_bias(self, emotion_scores: Dict[str, float]) -> None:
        """Update mood bias based on emotional state."""
        # Calculate new mood biases
        new_positivity = sum(
            emotion_scores.get(e, 0.0) for e in ["joy", "trust", "anticipation"]
        ) - sum(
            emotion_scores.get(e, 0.0) for e in ["sadness", "fear", "anger", "disgust"]
        )

        new_formality = (
            emotion_scores.get("analytical", 0.0) * 0.6
            + emotion_scores.get("confident", 0.0) * 0.4
            - emotion_scores.get("tentative", 0.0) * 0.3
        )

        new_engagement = (
            emotion_scores.get("anticipation", 0.0) * 0.4
            + emotion_scores.get("surprise", 0.0) * 0.3
            + emotion_scores.get("joy", 0.0) * 0.3
        )

        # Apply mood inertia
        self.mood_bias["positivity"] = self.mood_bias[
            "positivity"
        ] * self.mood_inertia + new_positivity * (1 - self.mood_inertia)
        self.mood_bias["formality"] = self.mood_bias[
            "formality"
        ] * self.mood_inertia + new_formality * (1 - self.mood_inertia)
        self.mood_bias["engagement"] = self.mood_bias[
            "engagement"
        ] * self.mood_inertia + new_engagement * (1 - self.mood_inertia)

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

        # Get emotion vectors with residue for intent
        intent_emotions = self._get_emotion_vector(
            intent_analysis["raw_analysis"], include_residue=True
        )

        # Get emotion vectors for response (without residue)
        response_emotions = self._get_emotion_vector(response, include_residue=False)

        # Update emotional residue and mood bias
        self._update_emotional_residue(response_emotions)
        self._update_mood_bias(response_emotions)

        # Calculate emotional alignment score with mood bias
        emotion_similarity = sum(
            min(intent_emotions[e], response_emotions[e]) for e in intent_emotions
        ) / len(intent_emotions)
        score += emotion_similarity * 0.3

        # Apply mood-based adjustments
        if self.mood_bias["positivity"] > 0.5:
            # In positive mood, reward positive emotional alignment more
            if emotion_similarity > 0.7:
                score += 0.1
        elif self.mood_bias["positivity"] < -0.5:
            # In negative mood, be more critical of emotional alignment
            if emotion_similarity < 0.3:
                score -= 0.1

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

        # Factor in emotional residue
        if any(residue > 0.3 for residue in self.emotional_residue.values()):
            # If strong emotional residue exists, expect some emotional carryover
            residue_alignment = any(
                response_emotions[e] > 0.3
                for e, residue in self.emotional_residue.items()
                if residue > 0.3
            )
            if residue_alignment:
                score += 0.1

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

        # Apply mood-based score adjustments
        if self.mood_bias["positivity"] < -0.5:  # Negative mood
            # Be more critical overall
            current_scores = {k: max(0.1, v * 0.9) for k, v in current_scores.items()}
        elif self.mood_bias["positivity"] > 0.5:  # Positive mood
            # Be more lenient
            current_scores = {k: min(1.0, v * 1.1) for k, v in current_scores.items()}

        # Update moving averages (with 0.1 learning rate)
        for dimension, score in current_scores.items():
            self.moving_averages[dimension] = (
                0.9 * self.moving_averages[dimension] + 0.1 * score
            )

        # Store historical data with mood information
        self.score_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "scores": current_scores.copy(),
                "moving_averages": self.moving_averages.copy(),
                "mood_bias": self.mood_bias.copy(),
                "emotional_residue": {
                    k: v for k, v in self.emotional_residue.items() if v > 0.1
                },
            }
        )

        # Calculate final score using weighted average
        final_score = sum(
            current_scores[dim] * self.weights[dim] for dim in current_scores
        )

        # Add trending information to the scores
        scores_with_trends = {
            dim: {
                "current": score,
                "trend": score - self.moving_averages[dim],
                "mood_influence": (
                    self.mood_bias["positivity"]
                    if dim == "emotional_alignment"
                    else 0.0
                ),
            }
            for dim, score in current_scores.items()
        }

        return final_score, scores_with_trends


class LayeredReasoningBot:
    """Main bot implementation with layered reasoning capabilities."""

    def __init__(self, config_path: str):
        # Load configuration
        self.config = BotConfig.from_yaml(config_path)

        # Initialize components
        self.memory = ConversationMemory(self.config.memory)
        self.emotional_state = EmotionalState()
        self.strategy_selector = BayesianStrategySelector(
            self.config.strategy.strategy_options,
            self.config.models["feature_extraction"],
        )
        self.response_evaluator = ResponseEvaluator(
            self.config.models["response_generation"]
        )

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

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
        # Get emotional influence
        emotional_influence = self.emotional_state.get_emotional_influence()

        # Add emotional context to prompt
        prompt = f"""Analyze this user message briefly and naturally:
        Message: {user_input}
        Context: {self.memory.get_context()}
        
        My current emotional state:
        - Dominant emotions: {', '.join(self.emotional_state.get_dominant_emotions())}
        - Energy level: {"High" if self.emotional_state.mood.energy_level > 0.7 else "Medium" if self.emotional_state.mood.energy_level > 0.4 else "Low"}
        - Stress level: {"High" if self.emotional_state.mood.stress_level > 0.7 else "Medium" if self.emotional_state.mood.stress_level > 0.4 else "Low"}
        
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
        
        Keep it short and human-like conversational. No formal headers or sections needed.
        Factor in my current emotional state when analyzing."""

        # Get streaming response for analysis
        response_stream = self._call_model(
            prompt, self.config.models["intent_recognition"], stream=True
        )

        analysis = self._stream_to_console(response_stream, prefix="Intent")

        # Update emotional state based on analysis
        emotion_triggers = self.response_evaluator._get_emotion_vector(analysis)
        self.emotional_state.update_state(emotion_triggers, context=user_input)

        return {"raw_analysis": analysis}

    def form_strategy(self, intent_analysis: Dict[str, Any]) -> str:
        """Decide on the response strategy using Bayesian inference and emotional state."""
        # Get gut feeling about the topic
        topics = intent_analysis.get("topics", [])
        gut_feelings = [
            (topic, self.emotional_state.get_gut_feeling(topic)) for topic in topics
        ]

        # Add emotional bias to strategy selection
        emotional_influence = self.emotional_state.get_emotional_influence()

        strategy, probabilities = self.strategy_selector.select_strategy(
            intent_analysis["raw_analysis"]
        )

        # Override strategy based on emotional state
        if self.emotional_state.mood.stress_level > 0.8:
            # When very stressed, bias towards reframing
            if random.random() < 0.4:  # 40% chance to override
                strategy = "reframing"
        elif self.emotional_state.mood.energy_level < 0.3:
            # When tired, bias towards simpler strategies
            if random.random() < 0.3:  # 30% chance to override
                strategy = "factual_explanation"

        # Log the strategy selection process
        self._log_step(
            "STRATEGY_FORMATION_COMPLETE",
            {
                "selected_strategy": strategy,
                "probabilities": (
                    probabilities.tolist()
                    if hasattr(probabilities, "tolist")
                    else probabilities
                ),
                "emotional_influence": emotional_influence,
                "gut_feelings": gut_feelings,
                "stress_level": self.emotional_state.mood.stress_level,
                "energy_level": self.emotional_state.mood.energy_level,
            },
        )

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
        self,
        user_input: str,
        intent_analysis: Dict[str, Any],
        strategy: str,
        model_config: Optional[ModelConfig] = None,
    ) -> str:
        """Generate the final response using the chosen strategy and emotional state."""
        # Get conversation stage and context
        conversation_stage = self.memory.episodic_memory["conversation_stage"]
        last_emotion_scores = self.memory.episodic_memory.get("last_emotion_scores", {})

        # Get current emotional state
        emotional_influence = self.emotional_state.get_emotional_influence()
        dominant_emotions = self.emotional_state.get_dominant_emotions()

        # Determine appropriate enthusiasm level with emotional influence
        base_enthusiasm = "moderate"
        if conversation_stage == "initial":
            base_enthusiasm = "high"
        elif conversation_stage == "engaged":
            if last_emotion_scores.get("current", 0.5) > 0.7:
                base_enthusiasm = "high"
            elif last_emotion_scores.get("current", 0.5) < 0.3:
                base_enthusiasm = "low"

        # Modify enthusiasm based on energy and mood
        if self.emotional_state.mood.energy_level < 0.4:
            base_enthusiasm = "low"  # Tired bot is less enthusiastic
        elif self.emotional_state.mood.stress_level > 0.7:
            base_enthusiasm = "high" if "anger" in dominant_emotions else "low"

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
        
        My Current State:
        - Emotional: {', '.join(dominant_emotions) if dominant_emotions else 'Neutral'}
        - Energy Level: {"Low" if self.emotional_state.mood.energy_level < 0.4 else "Normal"}
        - Stress Level: {"High" if self.emotional_state.mood.stress_level > 0.7 else "Normal"}
        
        Requirements:
        1. DO NOT mention or refer to the strategy in your response
        2. Keep the response natural and conversational
        3. Make it contextually appropriate
        4. Focus on engaging with the user's message
        5. Maintain {base_enthusiasm} enthusiasm level
        6. Avoid repeating these elements: {', '.join(repetition_warnings) if repetition_warnings else 'None'}
        7. Match conversation stage: {conversation_stage}
        8. If roleplay elements are present:
           - Stay in character consistently
           - Maintain the established relationship dynamic
           - Use appropriate personal references
           - Keep shared context in mind
        9. Factor in my current emotional state:
           - Let my stress level affect response tone
           - Show reduced patience if energy is low
           - Allow emotional overspill from previous interactions
           - Be more reactive if stress is high
        
        Just give the response directly, no meta-text or explanations."""

        # Get streaming response using provided model config or default
        response_stream = self._call_model(
            prompt,
            model_config or self.config.models["response_generation"],
            stream=True,
        )

        # Stream and collect the full response
        print("\n")
        full_response = self._stream_to_console(response_stream, prefix="Response")

        # Update emotional state based on the response
        response_emotions = self.response_evaluator._get_emotion_vector(full_response)
        self.emotional_state.update_state(response_emotions)

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

    def process_message(
        self, user_input: str, model_config: Optional[ModelConfig] = None
    ) -> str:
        """Process a user message through all reasoning layers.

        Args:
            user_input: The message from the user
            model_config: Optional model configuration to override default response generation model
        """
        try:
            # Check if bot needs a break
            if (
                self.emotional_state.mood.energy_level < 0.3
                or self.emotional_state.mood.stress_level > 0.8
            ):
                self.emotional_state.take_break()
                logger.info("Bot took a break to restore energy and reduce stress")

            # Layer 1: Intent Recognition
            self._log_step(
                "INTENT_RECOGNITION_START",
                {
                    "user_input": user_input,
                    "context_length": len(self.memory.get_context()),
                    "emotional_state": {
                        "dominant_emotions": self.emotional_state.get_dominant_emotions(),
                        "energy_level": self.emotional_state.mood.energy_level,
                        "stress_level": self.emotional_state.mood.stress_level,
                    },
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
                    "emotional_influence": self.emotional_state.get_emotional_influence(),
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
                {
                    "strategy": strategy,
                    "context_used": self.memory.get_context(),
                    "emotional_state": {
                        "dominant_emotions": self.emotional_state.get_dominant_emotions(),
                        "emotional_influence": self.emotional_state.get_emotional_influence(),
                    },
                    "model_config": (
                        model_config.model_name if model_config else "default"
                    ),
                },
            )

            # Use provided model config or default
            response = self.generate_response(
                user_input, intent_analysis, strategy, model_config=model_config
            )

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
                    "emotional_state": {
                        "dominant_emotions": self.emotional_state.get_dominant_emotions(),
                        "energy_level": self.emotional_state.mood.energy_level,
                        "stress_level": self.emotional_state.mood.stress_level,
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
                    "emotional_state": {
                        "dominant_emotions": self.emotional_state.get_dominant_emotions(),
                        "emotional_influence": self.emotional_state.get_emotional_influence(),
                    },
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
                    "emotional_state": {
                        "stress_level": self.emotional_state.mood.stress_level,
                        "energy_level": self.emotional_state.mood.energy_level,
                    },
                },
            )
            logger.error(f"Error processing message: {str(e)}")

            # Increase stress on error
            self.emotional_state.mood.stress_level = min(
                1.0, self.emotional_state.mood.stress_level + 0.2
            )

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
