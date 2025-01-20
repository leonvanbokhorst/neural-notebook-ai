"""
AI Interaction Manager
--------------------

Manages interactions between external AIs and the LayeredReasoningBot,
tracking conversations and emotional patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
import json
import logging
from pathlib import Path

from layered_reasoning_bot import LayeredReasoningBot
from emotional_state import EmotionalState, EmotionalLandmark
from config_types import BotConfig, ModelConfig, AIRoleConfig

logger = logging.getLogger(__name__)


@dataclass
class InteractionPattern:
    """Tracks recurring patterns in AI interactions."""

    pattern_type: str  # e.g., "emotional_trigger", "topic_shift", "strategy_change"
    frequency: int = 0
    last_observed: datetime = field(default_factory=datetime.now)
    context_examples: List[Dict[str, Any]] = field(default_factory=list)
    impact_score: float = 0.0  # -1.0 to 1.0


@dataclass
class AIParticipant:
    """Represents an AI participant in the conversation."""

    ai_id: str
    role: str  # "initiator" or "observer"
    role_config: AIRoleConfig
    interaction_count: int = 0
    emotional_impact: Dict[str, float] = field(default_factory=dict)
    observed_patterns: Dict[str, InteractionPattern] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)


class AIInteractionManager:
    """Manages and analyzes interactions between AIs and the bot."""

    def __init__(self, bot: LayeredReasoningBot, log_dir: Optional[Path] = None):
        self.bot = bot
        self.log_dir = log_dir or Path(__file__).parent / "interaction_logs"
        self.log_dir.mkdir(exist_ok=True)

        # Track AI participants
        self.participants: Dict[str, AIParticipant] = {}

        # Pattern recognition
        self.global_patterns: Dict[str, InteractionPattern] = {}
        self.emotional_triggers: Dict[str, List[str]] = defaultdict(list)
        self.strategy_effectiveness: Dict[str, List[float]] = defaultdict(list)

        # Conversation state
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_insights: List[Dict[str, Any]] = []

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure detailed logging for interactions."""
        interaction_log = self.log_dir / "ai_interactions.log"
        file_handler = logging.FileHandler(interaction_log)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def register_ai(self, ai_id: str, role: str) -> None:
        """Register a new AI participant."""
        if role not in self.bot.config.ai_roles:
            raise ValueError(
                f"Invalid role: {role}. Must be one of {list(self.bot.config.ai_roles.keys())}"
            )

        if ai_id not in self.participants:
            role_config = self.bot.config.ai_roles[role]
            self.participants[ai_id] = AIParticipant(
                ai_id=ai_id, role=role, role_config=role_config
            )
            logger.info(
                f"Registered new AI participant: {ai_id} as {role}\n"
                f"Using model: {role_config.model.model_name}\n"
                f"Capabilities: {', '.join(role_config.capabilities)}"
            )

    def start_conversation(
        self, initiator_id: str, observer_id: Optional[str] = None
    ) -> str:
        """Start a new conversation between AIs."""
        # Validate participant roles
        initiator = self.participants.get(initiator_id)
        if not initiator or initiator.role != "initiator":
            raise ValueError(
                f"Invalid initiator: {initiator_id}. Must be registered with 'initiator' role."
            )

        if observer_id:
            observer = self.participants.get(observer_id)
            if not observer or observer.role != "observer":
                raise ValueError(
                    f"Invalid observer: {observer_id}. Must be registered with 'observer' role."
                )

        conversation_id = (
            f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{initiator_id}"
        )

        self.active_conversations[conversation_id] = {
            "initiator_id": initiator_id,
            "observer_id": observer_id,
            "start_time": datetime.now(),
            "messages": [],
            "emotional_trajectory": [],
            "strategy_history": [],
            "patterns_observed": set(),
            "role_models": {
                initiator_id: initiator.role_config.model,
                observer_id: (
                    self.participants[observer_id].role_config.model
                    if observer_id
                    else None
                ),
            },
        }

        logger.info(
            f"Started new conversation: {conversation_id}\n"
            f"Initiator: {initiator_id} (Model: {initiator.role_config.model.model_name})\n"
            f"Observer: {observer_id or 'None'}"
        )
        return conversation_id

    def process_message(
        self, conversation_id: str, message: str, from_ai_id: str
    ) -> Dict[str, Any]:
        """Process a message in the conversation and return bot's response with analysis."""
        conv = self.active_conversations.get(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")

        participant = self.participants.get(from_ai_id)
        if not participant:
            raise ValueError(f"AI participant {from_ai_id} not found")

        # Get initial emotional state
        initial_state = self.bot.emotional_state.current_emotions.copy()

        # Process message through bot using role-specific model
        response = self.bot.process_message(
            message, model_config=participant.role_config.model
        )

        # Get final emotional state
        final_state = self.bot.emotional_state.current_emotions.copy()

        # Track emotional changes
        emotional_delta = {
            emotion: final_state[emotion] - initial_state[emotion]
            for emotion in initial_state
        }

        # Record interaction
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "from_ai": from_ai_id,
            "role": participant.role,
            "capabilities_used": participant.role_config.capabilities,
            "message": message,
            "response": response,
            "emotional_state": {
                "initial": initial_state,
                "final": final_state,
                "delta": emotional_delta,
            },
            "bot_state": {
                "energy_level": self.bot.emotional_state.mood.energy_level,
                "stress_level": self.bot.emotional_state.mood.stress_level,
                "dominant_emotions": self.bot.emotional_state.get_dominant_emotions(),
            },
        }

        conv["messages"].append(interaction_data)
        conv["emotional_trajectory"].append(emotional_delta)

        # Update participant stats
        participant.interaction_count += 1
        participant.conversation_history.append(interaction_data)

        # Analyze patterns
        self._analyze_interaction_patterns(conv, interaction_data)

        return {
            "response": response,
            "emotional_impact": emotional_delta,
            "emotional_state": interaction_data["emotional_state"],
            "bot_state": interaction_data["bot_state"],
            "patterns_detected": list(conv["patterns_observed"]),
            "role_capabilities": participant.role_config.capabilities,
        }

    def _analyze_interaction_patterns(
        self, conversation: Dict[str, Any], interaction: Dict[str, Any]
    ) -> None:
        """Analyze interaction for patterns and update tracking."""
        # Detect emotional triggers
        emotional_delta = interaction["emotional_state"]["delta"]
        significant_changes = [
            (emotion, delta)
            for emotion, delta in emotional_delta.items()
            if abs(delta) > 0.3
        ]

        for emotion, delta in significant_changes:
            pattern_id = f"emotional_trigger_{emotion}"
            if pattern_id not in self.global_patterns:
                self.global_patterns[pattern_id] = InteractionPattern(
                    pattern_type="emotional_trigger"
                )

            pattern = self.global_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_observed = datetime.now()
            pattern.context_examples.append(interaction)
            pattern.impact_score = (
                pattern.impact_score * (pattern.frequency - 1) + delta
            ) / pattern.frequency

            conversation["patterns_observed"].add(pattern_id)

        # Detect strategy effectiveness
        if len(conversation["messages"]) >= 2:
            prev_interaction = conversation["messages"][-2]
            current_state = interaction["bot_state"]
            prev_state = prev_interaction["bot_state"]

            # Check for significant state changes
            energy_delta = current_state["energy_level"] - prev_state["energy_level"]
            stress_delta = current_state["stress_level"] - prev_state["stress_level"]

            if abs(energy_delta) > 0.2 or abs(stress_delta) > 0.2:
                pattern_id = "significant_state_change"
                if pattern_id not in self.global_patterns:
                    self.global_patterns[pattern_id] = InteractionPattern(
                        pattern_type="state_change"
                    )

                pattern = self.global_patterns[pattern_id]
                pattern.frequency += 1
                pattern.context_examples.append(
                    {
                        "energy_delta": energy_delta,
                        "stress_delta": stress_delta,
                        "context": interaction,
                    }
                )

                conversation["patterns_observed"].add(pattern_id)

    def get_conversation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """Get detailed insights about a conversation."""
        conv = self.active_conversations.get(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Analyze emotional trajectory
        emotional_summary = defaultdict(list)
        for state in conv["emotional_trajectory"]:
            for emotion, value in state.items():
                emotional_summary[emotion].append(value)

        # Calculate emotional volatility
        emotional_volatility = {
            emotion: sum(abs(x) for x in values) / len(values)
            for emotion, values in emotional_summary.items()
        }

        # Get significant patterns
        significant_patterns = [
            pattern
            for pattern_id in conv["patterns_observed"]
            if (pattern := self.global_patterns.get(pattern_id))
            and pattern.frequency > 1
        ]

        return {
            "duration": (datetime.now() - conv["start_time"]).total_seconds(),
            "message_count": len(conv["messages"]),
            "emotional_volatility": emotional_volatility,
            "significant_patterns": [
                {
                    "type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "impact_score": pattern.impact_score,
                }
                for pattern in significant_patterns
            ],
            "bot_final_state": (
                conv["messages"][-1]["bot_state"] if conv["messages"] else None
            ),
        }

    def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """End a conversation and get final analysis."""
        insights = self.get_conversation_insights(conversation_id)

        # Save conversation log
        conv = self.active_conversations[conversation_id]
        log_path = self.log_dir / f"{conversation_id}.json"
        with open(log_path, "w") as f:
            json.dump({"conversation": conv, "insights": insights}, f, indent=2)

        # Remove from active conversations
        del self.active_conversations[conversation_id]

        return insights

    def get_global_patterns(self) -> Dict[str, Any]:
        """Get analysis of global interaction patterns."""
        return {
            pattern_id: {
                "type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "last_observed": pattern.last_observed.isoformat(),
                "impact_score": pattern.impact_score,
                "example_count": len(pattern.context_examples),
            }
            for pattern_id, pattern in self.global_patterns.items()
        }

    def get_participant_analysis(self, ai_id: str) -> Dict[str, Any]:
        """Get analysis of a specific AI participant's interactions."""
        participant = self.participants.get(ai_id)
        if not participant:
            raise ValueError(f"AI participant {ai_id} not found")

        return {
            "role": participant.role,
            "interaction_count": participant.interaction_count,
            "emotional_impact": participant.emotional_impact,
            "observed_patterns": {
                pattern_id: {
                    "frequency": pattern.frequency,
                    "impact_score": pattern.impact_score,
                }
                for pattern_id, pattern in participant.observed_patterns.items()
            },
            "conversation_history_length": len(participant.conversation_history),
        }
