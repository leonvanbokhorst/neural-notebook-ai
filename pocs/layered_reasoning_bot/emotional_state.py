"""
Emotional State Management
------------------------

Implements complex emotional state handling for more human-like bot behavior.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
import random
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

from emotional_complexity import (
    ContrarianBehavior,
    AssociativeMemoryBank,
    DynamicPersonality,
    EmotionalLandmark,
    CognitiveDissonance,
    ContradictoryState,
)


@dataclass
class EmotionalMemory:
    """Stores emotional associations with topics and contexts."""

    topic_emotions: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    context_emotions: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    emotional_triggers: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    processing_queue: List[Tuple[datetime, str, Dict[str, float]]] = field(
        default_factory=list
    )


@dataclass
class MoodRegulator:
    """Manages background mood and physiological influences."""

    last_break: datetime = field(default_factory=datetime.now)
    stress_level: float = 0.0  # 0.0 to 1.0
    energy_level: float = 1.0  # 0.0 to 1.0

    def update(self) -> None:
        """Update physiological factors."""
        # Simulate tiredness
        time_since_break = (datetime.now() - self.last_break).total_seconds() / 3600
        self.energy_level = max(0.2, 1.0 - (time_since_break * 0.1))

        # Accumulate stress
        self.stress_level = min(1.0, self.stress_level + (random.random() * 0.05))

        # Random mood fluctuations
        if random.random() < 0.1:  # 10% chance of mood swing
            self.stress_level = min(
                1.0, max(0.0, self.stress_level + random.uniform(-0.2, 0.2))
            )


class EmotionalState:
    """Manages complex emotional states with human-like characteristics."""

    def __init__(self):
        # Core emotional dimensions
        self.valence: float = 0.5  # -1.0 to 1.0 (negative to positive)
        self.arousal: float = 0.5  # 0.0 to 1.0 (calm to excited)
        self.dominance: float = 0.5  # 0.0 to 1.0 (submissive to dominant)

        # Complex emotional state
        self.current_emotions: Dict[str, float] = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "trust": 0.0,
            "disgust": 0.0,
            "surprise": 0.0,
            "anticipation": 0.0,
        }

        # Initialize emotional complexity components
        self.contrarian = ContrarianBehavior()
        self.associative_memory = AssociativeMemoryBank()
        self.personality = DynamicPersonality()
        self.cognitive_dissonance = CognitiveDissonance()

        # Emotional memory and regulation
        self.memory = EmotionalMemory()
        self.mood = MoodRegulator()

        # Emotional inertia (resistance to change)
        self.emotional_inertia: float = 0.7  # 0.0 to 1.0

        # Emotion processing queue
        self.pending_emotions: List[Tuple[datetime, Dict[str, float]]] = []

        # Track recent emotional experiences for conviction strength
        self.recent_experiences: List[float] = []
        self.experience_window = timedelta(hours=24)

        # Add adaptation flag
        self._adaptation_enabled = True

    def disable_adaptation(self) -> None:
        """Disable emotional adaptation for baseline testing."""
        self._adaptation_enabled = False
        logger.info("Emotional adaptation disabled")

    def enable_adaptation(self) -> None:
        """Enable emotional adaptation."""
        self._adaptation_enabled = True
        logger.info("Emotional adaptation enabled")

    def update_state(self, triggers: Dict[str, float], context: str = "") -> None:
        """Update emotional state based on triggers and context."""
        # Update physiological state
        self.mood.update()

        # Update personality traits
        self.personality.update_traits(
            energy_level=self.mood.energy_level,
            stress_level=self.mood.stress_level,
            context_type=(
                "stressed"
                if self.mood.stress_level > 0.7
                else "energetic" if self.mood.energy_level > 0.7 else None
            ),
        )

        # Check for emotional resistance
        if self.contrarian.should_resist(
            self.current_emotions,
            self.mood.stress_level,
            self.mood.energy_level,  # Pass energy level to contrarian behavior
        ):
            # Invert or modify emotional triggers
            triggers = {
                emotion: 1.0 - intensity for emotion, intensity in triggers.items()
            }

        # Process pending emotions (delayed processing)
        current_time = datetime.now()
        self.pending_emotions = [
            (time, emotions)
            for time, emotions in self.pending_emotions
            if (current_time - time).total_seconds() < 300  # 5-minute window
        ]

        # Calculate emotional changes
        changes = {emotion: 0.0 for emotion in self.current_emotions}

        # Factor in mood and energy levels
        energy_factor = self.mood.energy_level
        stress_factor = self.mood.stress_level

        # Process immediate triggers
        for emotion, intensity in triggers.items():
            if emotion in self.current_emotions:
                # Apply personality-based modulation
                neuroticism = self.personality.get_trait("neuroticism")
                intensity *= 1.0 + (neuroticism - 0.5)

                # Apply energy and stress factors
                if energy_factor < 0.5:  # Tired state amplifies negative emotions
                    intensity *= 1.5 if emotion in ["anger", "sadness", "fear"] else 0.7

                if stress_factor > 0.7:  # High stress amplifies emotional reactions
                    intensity *= 1.3

                changes[emotion] = intensity

        # Apply emotional inertia
        for emotion in self.current_emotions:
            current = self.current_emotions[emotion]
            target = changes.get(emotion, 0.0)
            self.current_emotions[emotion] = (
                current * self.emotional_inertia + target * (1 - self.emotional_inertia)
            )

        # Check for and handle contradictory emotions
        sorted_emotions = sorted(
            self.current_emotions.items(), key=lambda x: x[1], reverse=True
        )

        # Look for top 2 strongest emotions that might conflict
        if len(sorted_emotions) >= 2:
            emotion_a, intensity_a = sorted_emotions[0]
            emotion_b, intensity_b = sorted_emotions[1]

            if (
                intensity_a > 0.4 and intensity_b > 0.4
            ):  # Only consider significant emotions
                contradictory_state = self.cognitive_dissonance.add_contradictory_state(
                    emotion_a, emotion_b, intensity_a, intensity_b
                )

                if contradictory_state:
                    # Add to emotional memory with justification
                    self.memory.emotional_triggers[emotion_a].add(
                        f"{context} ({contradictory_state.justification})"
                    )

        # Queue some emotions for delayed processing
        if random.random() < 0.3:  # 30% chance of delayed emotional processing
            delayed_emotions = {
                emotion: intensity * 0.5
                for emotion, intensity in triggers.items()
                if random.random() < 0.5  # 50% chance for each emotion
            }
            if delayed_emotions:
                process_time = current_time + timedelta(seconds=random.uniform(30, 300))
                self.pending_emotions.append((process_time, delayed_emotions))

        # Update emotional memory and create landmarks
        if context:
            # Store in basic emotional memory
            self.memory.context_emotions[context] = self.current_emotions.copy()
            for emotion, intensity in self.current_emotions.items():
                if intensity > 0.5:
                    self.memory.emotional_triggers[emotion].add(context)

            # Create emotional landmark for significant emotions
            dominant_emotion = max(self.current_emotions.items(), key=lambda x: x[1])
            if dominant_emotion[1] > 0.6:  # Only store significant emotional events
                self.associative_memory.add_landmark(
                    emotion=dominant_emotion[0],
                    intensity=dominant_emotion[1],
                    context=context,
                    topic=context.split()[0] if context.split() else context,
                    associated_topics=set(context.split()[1:3]),
                )

            # Track opinion with current conviction
            conviction = self.cognitive_dissonance.get_conviction_strength(
                self.recent_experiences[-10:]  # Use last 10 experiences
            )
            self.cognitive_dissonance.add_opinion(
                topic=context.split()[0] if context.split() else context,
                opinion=context,
                conviction=conviction,
            )

        # Update recent experiences
        max_intensity = max(self.current_emotions.values())
        self.recent_experiences.append(max_intensity)

        # Maintain experience window
        self.recent_experiences = self.recent_experiences[
            -50:
        ]  # Keep last 50 experiences

    def get_dominant_emotions(self, threshold: float = 0.3) -> List[str]:
        """Get the currently dominant emotions."""
        return [
            emotion
            for emotion, intensity in self.current_emotions.items()
            if intensity >= threshold
        ]

    def get_emotional_influence(self) -> Dict[str, float]:
        """Get the current emotional influence on behavior."""
        influence = {
            "positivity_bias": 0.0,
            "reactivity": 0.0,
            "engagement": 0.0,
            "empathy": 0.0,
        }

        # Calculate positivity bias
        total_positive = sum(
            self.current_emotions[e] for e in ["joy", "trust", "anticipation"]
        )
        total_negative = sum(
            self.current_emotions[e] for e in ["sadness", "fear", "anger", "disgust"]
        )
        influence["positivity_bias"] = (total_positive - total_negative + 1) / 2

        # Calculate reactivity based on arousal, stress, and personality
        influence["reactivity"] = (
            self.arousal * 0.3
            + self.mood.stress_level * 0.4
            + self.personality.get_trait("neuroticism") * 0.3
        )

        # Calculate engagement based on personality and energy
        influence["engagement"] = (
            self.arousal * 0.3
            + self.mood.energy_level * 0.3
            + self.personality.get_trait("extraversion") * 0.4
        )

        # Calculate empathy based on personality and emotional state
        influence["empathy"] = (
            self.personality.get_trait("agreeableness") * 0.4
            + self.current_emotions["trust"] * 0.3
            + (1 - self.mood.stress_level) * 0.3
        )

        return influence

    def get_gut_feeling(self, topic: str) -> Optional[float]:
        """Generate a gut feeling about a topic based on emotional memory."""
        # Check associative memory first
        relevant_landmarks = self.associative_memory.get_relevant_callbacks(
            current_topic=topic,
            current_emotion=max(self.current_emotions.items(), key=lambda x: x[1])[0],
        )

        if relevant_landmarks:
            # Use the most recent/relevant landmark for gut feeling
            landmark = relevant_landmarks[0]
            # Add some random variation based on current mood
            mood_factor = random.uniform(-0.2, 0.2) * self.mood.stress_level
            # Convert emotion to valence (-1 to 1)
            valence = (
                1.0 if landmark.emotion in ["joy", "trust", "anticipation"] else -1.0
            )
            return max(-1.0, min(1.0, valence * landmark.intensity + mood_factor))

        # Fall back to basic emotional memory
        if topic in self.memory.topic_emotions:
            emotions = self.memory.topic_emotions[topic]
            # Weight positive and negative emotions
            positive = sum(emotions.get(e, 0) for e in ["joy", "trust", "anticipation"])
            negative = sum(emotions.get(e, 0) for e in ["fear", "anger", "disgust"])

            # Add random variation based on current mood
            mood_factor = random.uniform(-0.2, 0.2) * self.mood.stress_level

            return max(-1.0, min(1.0, (positive - negative) / 3 + mood_factor))
        return None

    def take_break(self) -> None:
        """Simulate taking a break to restore energy and reduce stress."""
        self.mood.last_break = datetime.now()
        self.mood.stress_level = max(0.0, self.mood.stress_level - 0.3)
        self.mood.energy_level = min(1.0, self.mood.energy_level + 0.4)

    def get_contrarian_response(self, emotion: str) -> Optional[str]:
        """Get a contrarian response if appropriate."""
        if emotion in self.current_emotions:
            return self.contrarian.get_contrarian_response(
                emotion, self.current_emotions[emotion]
            )
        return None

    def get_emotional_callbacks(self, topic: str) -> List[EmotionalLandmark]:
        """Get relevant emotional callbacks for the current context."""
        return self.associative_memory.get_relevant_callbacks(
            current_topic=topic,
            current_emotion=max(self.current_emotions.items(), key=lambda x: x[1])[0],
        )

    def get_contradictory_response(
        self, topic: str, current_opinion: str
    ) -> Optional[str]:
        """Get a response that might contradict past statements."""
        return self.cognitive_dissonance.generate_contradiction_response(
            topic, current_opinion
        )

    def get_selective_memory(self, topic: str) -> Optional[str]:
        """Get a memory that aligns with current emotional state."""
        dominant_emotion = max(self.current_emotions.items(), key=lambda x: x[1])
        return self.cognitive_dissonance.get_selective_memory(
            topic, dominant_emotion[0], dominant_emotion[1]
        )

    def get_active_contradictions(self) -> List[ContradictoryState]:
        """Get currently active contradictory emotional states."""
        return self.cognitive_dissonance.active_contradictions
