"""
Interaction Observer
------------------

Observes and analyzes conversations between AIs and the LayeredReasoningBot,
focusing on emotional dynamics and emergent patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
import logging
from pathlib import Path
import json
import numpy as np

from emotional_state import EmotionalState, EmotionalLandmark
from ai_interaction_manager import AIInteractionManager, InteractionPattern

logger = logging.getLogger(__name__)


@dataclass
class ObservationNote:
    """Represents a significant observation during the conversation."""

    timestamp: str
    note_type: str  # e.g., "emotional_shift", "pattern_emergence", "unusual_behavior"
    description: str
    significance_score: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionalTrajectory:
    """Tracks the emotional journey through a conversation."""

    emotion_sequence: List[Dict[str, float]] = field(default_factory=list)
    volatility_scores: Dict[str, float] = field(default_factory=dict)
    key_transitions: List[Dict[str, Any]] = field(default_factory=list)
    dominant_emotions: List[str] = field(default_factory=list)


class InteractionObserver:
    """Observes and analyzes AI-bot interactions."""

    def __init__(
        self, interaction_manager: AIInteractionManager, log_dir: Optional[Path] = None
    ):
        self.manager = interaction_manager
        self.log_dir = log_dir or Path(__file__).parent / "observer_logs"
        self.log_dir.mkdir(exist_ok=True)

        # Observation storage
        self.observations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.observation_notes: Dict[str, List[ObservationNote]] = defaultdict(list)
        self.emotional_trajectories: Dict[str, EmotionalTrajectory] = {}

        # Pattern analysis
        self.recurring_patterns: Dict[str, List[InteractionPattern]] = defaultdict(list)
        self.unusual_behaviors: List[Dict[str, Any]] = []

        # Thresholds for analysis
        self.emotional_shift_threshold = 0.3
        self.pattern_significance_threshold = 0.4
        self.unusual_behavior_threshold = 0.7
        self.transition_threshold = 0.25
        self.significance_threshold = 0.6

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the observer."""
        observer_log = self.log_dir / "observer_notes.log"
        file_handler = logging.FileHandler(observer_log)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def start_observation(self, conversation_id: str) -> None:
        """Start observing a conversation."""
        self.emotional_trajectories[conversation_id] = EmotionalTrajectory()
        logger.info(f"Started observing conversation: {conversation_id}")

    def observe_interaction(
        self, conversation_id: str, interaction_data: Dict[str, Any]
    ) -> None:
        """Record and analyze a new interaction."""
        if conversation_id not in self.observations:
            self.observations[conversation_id] = []

        # Add timestamp if not present
        if "timestamp" not in interaction_data:
            interaction_data["timestamp"] = datetime.now()

        # Convert any datetime objects in the interaction data
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        processed_data = convert_datetime(interaction_data)

        self.observations[conversation_id].append(processed_data)

        # Create observation note
        note = ObservationNote(
            timestamp=datetime.now().isoformat(),
            note_type="interaction",
            description=f"Processed interaction with emotional impact: {processed_data.get('emotional_impact', 'N/A')}",
            significance_score=self._calculate_significance(processed_data),
        )

        self.observation_notes[conversation_id].append(note)

        # Log the observation
        logger.info(f"Observed interaction in conversation {conversation_id}:")
        logger.info(json.dumps(processed_data, indent=2))

    def _analyze_emotional_shift(
        self, conversation_id: str, trajectory: EmotionalTrajectory
    ) -> None:
        """Analyze significant emotional shifts in the conversation."""
        current = trajectory.emotion_sequence[-1]
        previous = trajectory.emotion_sequence[-2]

        # Calculate emotional shifts
        shifts = {emotion: current[emotion] - previous[emotion] for emotion in current}

        # Detect significant shifts
        significant_shifts = {
            emotion: shift
            for emotion, shift in shifts.items()
            if abs(shift) > self.emotional_shift_threshold
        }

        if significant_shifts:
            # Record the transition
            transition = {
                "timestamp": datetime.now(),
                "shifts": significant_shifts,
                "context": {"previous_state": previous, "current_state": current},
            }
            trajectory.key_transitions.append(transition)

            # Create observation note
            description = f"Significant emotional shift detected: " + ", ".join(
                f"{emotion}: {shift:+.2f}"
                for emotion, shift in significant_shifts.items()
            )

            self.observations[conversation_id].append(
                ObservationNote(
                    timestamp=datetime.now(),
                    note_type="emotional_shift",
                    description=description,
                    context={"transition": transition},
                    significance_score=max(
                        abs(shift) for shift in significant_shifts.values()
                    ),
                )
            )

    def _detect_unusual_behavior(
        self, conversation_id: str, interaction_data: Dict[str, Any]
    ) -> None:
        """Detect and analyze unusual bot behaviors."""
        bot_state = interaction_data["bot_state"]

        # Check for unusual energy/stress combinations
        if (bot_state["energy_level"] > 0.8 and bot_state["stress_level"] > 0.8) or (
            bot_state["energy_level"] < 0.2 and bot_state["stress_level"] < 0.2
        ):
            self.observations[conversation_id].append(
                ObservationNote(
                    timestamp=datetime.now(),
                    note_type="unusual_behavior",
                    description=f"Unusual energy/stress state: energy={bot_state['energy_level']:.2f}, stress={bot_state['stress_level']:.2f}",
                    context={"bot_state": bot_state},
                    significance_score=0.8,
                )
            )

        # Check for emotional contradictions
        dominant_emotions = set(bot_state["dominant_emotions"])
        contradictory_pairs = {
            frozenset(["joy", "sadness"]),
            frozenset(["trust", "disgust"]),
            frozenset(["fear", "anger"]),
            frozenset(["surprise", "anticipation"]),
        }

        for pair in contradictory_pairs:
            if len(dominant_emotions.intersection(pair)) > 1:
                self.observations[conversation_id].append(
                    ObservationNote(
                        timestamp=datetime.now(),
                        note_type="emotional_contradiction",
                        description=f"Contradictory emotions present: {', '.join(pair)}",
                        context={"dominant_emotions": list(dominant_emotions)},
                        significance_score=0.9,
                    )
                )

    def _analyze_patterns(
        self, conversation_id: str, interaction_data: Dict[str, Any]
    ) -> None:
        """Analyze emerging patterns in the conversation."""
        # Get conversation patterns from manager
        patterns = self.manager.get_global_patterns()

        # Look for recurring patterns
        for pattern_id, pattern_data in patterns.items():
            if (
                pattern_data["frequency"] > 2
                and pattern_data["impact_score"] > self.pattern_significance_threshold
            ):
                if pattern_id not in self.recurring_patterns[conversation_id]:
                    self.recurring_patterns[conversation_id].append(pattern_id)

                    self.observations[conversation_id].append(
                        ObservationNote(
                            timestamp=datetime.now(),
                            note_type="pattern_emergence",
                            description=f"Recurring pattern detected: {pattern_data['type']}",
                            context={"pattern_data": pattern_data},
                            significance_score=pattern_data["impact_score"],
                        )
                    )

    def _log_significant_observations(self, conversation_id: str) -> None:
        """Log significant observations for the conversation."""
        recent_observations = [
            obs
            for obs in self.observations[conversation_id]
            if (datetime.now() - obs.timestamp).total_seconds() < 300  # Last 5 minutes
        ]

        for obs in recent_observations:
            if obs.significance_score > 0.7:  # Only log highly significant observations
                logger.info(
                    f"[{conversation_id}] {obs.note_type}: {obs.description} "
                    f"(significance: {obs.significance_score:.2f})"
                )

    def get_conversation_analysis(self, conversation_id: str) -> Dict[str, Any]:
        """Get the current analysis for a conversation."""
        if conversation_id not in self.observations:
            return {}

        def convert_datetime(obj):
            """Convert datetime objects to ISO format strings."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        analysis = {
            "emotional_analysis": {
                "volatility": self._calculate_emotional_volatility(conversation_id),
                "key_transitions": self._identify_key_transitions(conversation_id),
                "dominant_patterns": self._get_dominant_patterns(conversation_id),
            },
            "interaction_metrics": {
                "total_interactions": len(self.observations[conversation_id]),
                "average_response_time": self._calculate_avg_response_time(
                    conversation_id
                ),
                "engagement_score": self._calculate_engagement_score(conversation_id),
            },
            "notable_moments": self._get_notable_moments(conversation_id),
            "timestamp": datetime.now().isoformat(),
        }

        # Convert any datetime objects in the analysis
        return convert_datetime(analysis)

    def _identify_key_transitions(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Identify significant emotional transitions in the conversation."""
        transitions = []
        observations = self.observations[conversation_id]

        for i in range(1, len(observations)):
            prev_state = observations[i - 1]["emotional_state"]["final"]
            curr_state = observations[i]["emotional_state"]["final"]

            # Calculate emotional shift
            shift = self._calculate_emotional_shift(prev_state, curr_state)

            if shift > self.transition_threshold:
                transitions.append(
                    {
                        "timestamp": observations[i]["timestamp"],
                        "shift_magnitude": shift,
                        "from_state": prev_state,
                        "to_state": curr_state,
                        "trigger_message": observations[i].get(
                            "message", "Unknown message"
                        ),
                    }
                )

        return transitions

    def _get_notable_moments(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get notable moments from the conversation."""
        notable_moments = []

        for note in self.observation_notes[conversation_id]:
            if note.significance_score > self.significance_threshold:
                notable_moments.append(
                    {
                        "timestamp": note.timestamp,
                        "type": note.note_type,
                        "description": note.description,
                        "significance_score": note.significance_score,
                    }
                )

        return notable_moments

    def end_observation(self, conversation_id: str) -> Dict[str, Any]:
        """End observation of a conversation and get final analysis."""
        analysis = self.get_conversation_analysis(conversation_id)

        # Save detailed observation log
        log_path = self.log_dir / f"observation_{conversation_id}.json"

        # Convert observations to serializable format
        observation_logs = []
        for obs in self.observation_notes[conversation_id]:
            observation_logs.append(
                {
                    "timestamp": obs.timestamp,
                    "type": obs.note_type,
                    "description": obs.description,
                    "significance_score": obs.significance_score,
                }
            )

        with open(log_path, "w") as f:
            json.dump(
                {
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis,
                    "observations": observation_logs,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved observation log to: {log_path}")

        # Cleanup
        if conversation_id in self.emotional_trajectories:
            del self.emotional_trajectories[conversation_id]
        if conversation_id in self.observations:
            del self.observations[conversation_id]
        if conversation_id in self.recurring_patterns:
            self.recurring_patterns[conversation_id].clear()
        if conversation_id in self.observation_notes:
            del self.observation_notes[conversation_id]

        return analysis

    def _calculate_significance(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate the significance score of an interaction."""
        significance = 0.0

        # Check emotional impact
        if "emotional_impact" in interaction_data:
            emotional_changes = interaction_data["emotional_impact"].values()
            significance += sum(abs(change) for change in emotional_changes) / len(
                emotional_changes
            )

        # Check stress and energy changes
        if "bot_state" in interaction_data:
            bot_state = interaction_data["bot_state"]
            significance += abs(bot_state.get("stress_level", 0)) * 0.3
            significance += abs(bot_state.get("energy_level", 0)) * 0.3

        # Normalize to 0-1 range
        return min(max(significance, 0.0), 1.0)

    def _calculate_emotional_volatility(self, conversation_id: str) -> Dict[str, float]:
        """Calculate emotional volatility scores for each emotion."""
        if (
            conversation_id not in self.observations
            or not self.observations[conversation_id]
        ):
            return {}

        # Extract emotional states from observations
        emotional_states = []
        for obs in self.observations[conversation_id]:
            if "emotional_state" in obs and "final" in obs["emotional_state"]:
                emotional_states.append(obs["emotional_state"]["final"])

        if not emotional_states:
            return {}

        # Calculate volatility (standard deviation) for each emotion
        volatility = {}
        for emotion in emotional_states[0].keys():
            values = [state.get(emotion, 0.0) for state in emotional_states]
            if values:
                # Calculate standard deviation, handle case with single value
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                    volatility[emotion] = variance**0.5
                else:
                    volatility[emotion] = 0.0

        return volatility

    def _calculate_emotional_shift(
        self, prev_state: Dict[str, float], curr_state: Dict[str, float]
    ) -> float:
        """Calculate the magnitude of emotional shift between two states."""
        if not prev_state or not curr_state:
            return 0.0

        try:
            # Calculate Euclidean distance between emotional states
            squared_diff_sum = 0.0
            common_emotions = set(prev_state.keys()) & set(curr_state.keys())

            if not common_emotions:
                return 0.0

            for emotion in common_emotions:
                prev_value = float(prev_state[emotion])
                curr_value = float(curr_state[emotion])
                diff = curr_value - prev_value
                squared_diff_sum += diff * diff

            return (squared_diff_sum**0.5) / len(common_emotions)
        except (TypeError, ValueError) as e:
            logger.warning(f"Error calculating emotional shift: {e}")
            return 0.0

    def _calculate_avg_response_time(self, conversation_id: str) -> float:
        """Calculate average response time in seconds."""
        if (
            conversation_id not in self.observations
            or len(self.observations[conversation_id]) < 2
        ):
            return 0.0

        response_times = []
        observations = self.observations[conversation_id]

        for i in range(1, len(observations)):
            curr_time = datetime.fromisoformat(observations[i]["timestamp"])
            prev_time = datetime.fromisoformat(observations[i - 1]["timestamp"])
            response_times.append((curr_time - prev_time).total_seconds())

        return sum(response_times) / len(response_times) if response_times else 0.0

    def _calculate_engagement_score(self, conversation_id: str) -> float:
        """Calculate engagement score based on response patterns."""
        if conversation_id not in self.observations:
            return 0.0

        observations = self.observations[conversation_id]
        if not observations:
            return 0.0

        # Factors that contribute to engagement
        factors = []

        # 1. Response consistency
        avg_time = self._calculate_avg_response_time(conversation_id)
        time_consistency = 1.0 / (1.0 + avg_time / 10) if avg_time > 0 else 0.0
        factors.append(time_consistency)

        # 2. Emotional responsiveness
        emotional_shifts = []
        for i in range(1, len(observations)):
            if (
                "emotional_state" in observations[i]
                and "emotional_state" in observations[i - 1]
                and "final" in observations[i]["emotional_state"]
                and "final" in observations[i - 1]["emotional_state"]
            ):

                prev_state = observations[i - 1]["emotional_state"]["final"]
                curr_state = observations[i]["emotional_state"]["final"]

                shift = self._calculate_emotional_shift(prev_state, curr_state)
                if shift > 0:  # Only count non-zero shifts
                    emotional_shifts.append(shift)

        emotional_responsiveness = (
            sum(emotional_shifts) / len(emotional_shifts) if emotional_shifts else 0.0
        )
        factors.append(emotional_responsiveness)

        # 3. Energy level maintenance
        energy_levels = [
            obs["bot_state"]["energy_level"]
            for obs in observations
            if "bot_state" in obs and "energy_level" in obs["bot_state"]
        ]
        avg_energy = sum(energy_levels) / len(energy_levels) if energy_levels else 0.0
        factors.append(avg_energy)

        # Calculate final score (weighted average)
        weights = [0.3, 0.4, 0.3]  # Adjust weights as needed
        engagement_score = sum(f * w for f, w in zip(factors, weights))

        return min(max(engagement_score, 0.0), 1.0)

    def _get_dominant_patterns(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get dominant patterns from the conversation."""
        if conversation_id not in self.observations:
            return []

        observations = self.observations[conversation_id]
        if not observations:
            return []

        # Track pattern frequencies and impacts
        pattern_stats = defaultdict(lambda: {"count": 0, "impact": 0.0, "examples": []})

        # Analyze patterns in emotional states and responses
        for obs in observations:
            # Check emotional patterns
            if "emotional_state" in obs:
                emotional_state = obs["emotional_state"].get("final", {})
                dominant_emotions = [
                    emotion
                    for emotion, value in emotional_state.items()
                    if value > 0.6  # Threshold for dominant emotion
                ]

                if dominant_emotions:
                    pattern_id = (
                        f"dominant_emotions_{'-'.join(sorted(dominant_emotions))}"
                    )
                    pattern_stats[pattern_id]["count"] += 1
                    pattern_stats[pattern_id]["impact"] += sum(
                        emotional_state[emotion] for emotion in dominant_emotions
                    ) / len(dominant_emotions)
                    pattern_stats[pattern_id]["examples"].append(
                        {
                            "timestamp": obs.get("timestamp", ""),
                            "emotions": dominant_emotions,
                            "values": {
                                e: emotional_state[e] for e in dominant_emotions
                            },
                        }
                    )

            # Check behavioral patterns
            if "bot_state" in obs:
                bot_state = obs["bot_state"]

                # High energy pattern
                if bot_state.get("energy_level", 0) > 0.8:
                    pattern_stats["high_energy"]["count"] += 1
                    pattern_stats["high_energy"]["impact"] += bot_state["energy_level"]
                    pattern_stats["high_energy"]["examples"].append(
                        {
                            "timestamp": obs.get("timestamp", ""),
                            "energy_level": bot_state["energy_level"],
                        }
                    )

                # High stress pattern
                if bot_state.get("stress_level", 0) > 0.7:
                    pattern_stats["high_stress"]["count"] += 1
                    pattern_stats["high_stress"]["impact"] += bot_state["stress_level"]
                    pattern_stats["high_stress"]["examples"].append(
                        {
                            "timestamp": obs.get("timestamp", ""),
                            "stress_level": bot_state["stress_level"],
                        }
                    )

        # Filter and format significant patterns
        dominant_patterns = []
        for pattern_id, stats in pattern_stats.items():
            if stats["count"] >= 2:  # Pattern must occur at least twice
                avg_impact = stats["impact"] / stats["count"]
                if avg_impact > 0.5:  # Pattern must have significant impact
                    dominant_patterns.append(
                        {
                            "pattern_id": pattern_id,
                            "frequency": stats["count"],
                            "average_impact": avg_impact,
                            "examples": stats["examples"][:3],  # Limit to 3 examples
                        }
                    )

        # Sort by impact and frequency
        return sorted(
            dominant_patterns,
            key=lambda x: (x["average_impact"], x["frequency"]),
            reverse=True,
        )
