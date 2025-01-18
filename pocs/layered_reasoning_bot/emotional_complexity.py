"""
Advanced Emotional Complexity Components
--------------------------------------

Implements advanced emotional behaviors for more authentic human-like responses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
import random
import numpy as np
from collections import defaultdict


@dataclass
class EmotionalLandmark:
    """Represents a significant emotional memory with context."""

    emotion: str
    intensity: float
    context: str
    topic: str
    associated_topics: Set[str]
    timestamp: datetime = field(default_factory=datetime.now)
    recall_count: int = 0


@dataclass
class ContradictoryState:
    """Represents a pair of contradictory emotions or beliefs."""

    state_a: str
    state_b: str
    conviction_a: float  # 0.0 to 1.0
    conviction_b: float  # 0.0 to 1.0
    justification: str
    timestamp: datetime = field(default_factory=datetime.now)


class CognitiveDissonance:
    """Manages contradictory emotional states and beliefs with rationalizations."""

    def __init__(self):
        # Track contradictory emotional pairs
        self.emotional_contradictions: Dict[str, Set[str]] = {
            "joy": {"sadness", "disgust"},
            "trust": {"disgust", "fear"},
            "fear": {"anger", "trust"},
            "surprise": {"anticipation"},
            "anger": {"fear", "sadness"},
            "disgust": {"joy", "trust"},
            "sadness": {"joy", "anger"},
            "anticipation": {"surprise"},
        }

        # Track current contradictory states
        self.active_contradictions: List[ContradictoryState] = []

        # Justification templates for contradictions
        self.justifications = {
            ("joy", "sadness"): [
                "It's complicated - you can be happy about something while still feeling the weight of sadness",
                "Life isn't black and white, you know?",
                "I contain multitudes!",
            ],
            ("trust", "fear"): [
                "Sometimes you trust someone but still can't help feeling afraid",
                "It's just human nature to have mixed feelings",
                "I've thought about this a lot, and it makes perfect sense to me",
            ],
            ("anger", "sadness"): [
                "You can be angry at the situation while still feeling sad about it",
                "Emotions aren't simple - they're layered",
                "Actually, this is exactly how I've always felt about it",
            ],
        }

        # Track opinion history with selective memory
        self.opinion_history: Dict[str, List[Tuple[str, float, datetime]]] = (
            defaultdict(list)
        )

        # Conviction strength modulation
        self.base_conviction = random.uniform(0.4, 0.8)
        self.conviction_volatility = random.uniform(0.1, 0.3)

    def add_contradictory_state(
        self, state_a: str, state_b: str, intensity_a: float, intensity_b: float
    ) -> Optional[ContradictoryState]:
        """Add a new contradictory state if the states conflict."""
        if (
            state_a in self.emotional_contradictions
            and state_b in self.emotional_contradictions[state_a]
        ):
            # Generate conviction levels with some randomness
            conviction_a = min(1.0, intensity_a * (1.0 + random.uniform(-0.2, 0.2)))
            conviction_b = min(1.0, intensity_b * (1.0 + random.uniform(-0.2, 0.2)))

            # Get justification
            justification = self._generate_justification(state_a, state_b)

            # Create contradictory state
            state = ContradictoryState(
                state_a=state_a,
                state_b=state_b,
                conviction_a=conviction_a,
                conviction_b=conviction_b,
                justification=justification,
            )

            self.active_contradictions.append(state)
            return state
        return None

    def _generate_justification(self, state_a: str, state_b: str) -> str:
        """Generate a justification for holding contradictory states."""
        # Try to find direct justification
        if (state_a, state_b) in self.justifications:
            return random.choice(self.justifications[(state_a, state_b)])
        if (state_b, state_a) in self.justifications:
            return random.choice(self.justifications[(state_b, state_a)])

        # Generic justifications if no specific one exists
        generic_justifications = [
            "Well, it's more nuanced than that...",
            "Actually, this makes perfect sense to me",
            "I've always been consistent about this",
            "You're oversimplifying a complex situation",
            "Life isn't always black and white",
        ]
        return random.choice(generic_justifications)

    def add_opinion(self, topic: str, opinion: str, conviction: float) -> None:
        """Add an opinion to history, possibly contradicting past opinions."""
        self.opinion_history[topic].append((opinion, conviction, datetime.now()))

        # Limit history size while keeping strong convictions
        if len(self.opinion_history[topic]) > 5:
            # Sort by conviction and recency
            self.opinion_history[topic].sort(
                key=lambda x: (x[1], -((datetime.now() - x[2]).total_seconds()))
            )
            # Keep only the top 5
            self.opinion_history[topic] = self.opinion_history[topic][-5:]

    def get_selective_memory(
        self, topic: str, current_emotion: str, intensity: float
    ) -> Optional[str]:
        """Retrieve a memory that aligns with current emotional state."""
        if topic not in self.opinion_history:
            return None

        # Filter recent opinions (last 24 hours)
        recent_opinions = [
            (op, conv, ts)
            for op, conv, ts in self.opinion_history[topic]
            if (datetime.now() - ts) < timedelta(hours=24)
        ]

        if not recent_opinions:
            return None

        # If strong emotion, bias towards confirming current state
        if intensity > 0.7:
            # Select opinion with highest conviction
            return max(recent_opinions, key=lambda x: x[1])[0]
        else:
            # Randomly select, weighted by conviction
            weights = [conv for _, conv, _ in recent_opinions]
            return random.choices(
                [op for op, _, _ in recent_opinions], weights=weights, k=1
            )[0]

    def generate_contradiction_response(
        self, topic: str, current_opinion: str
    ) -> Optional[str]:
        """Generate a response that might contradict past statements."""
        if topic not in self.opinion_history:
            return None

        past_opinions = [
            op for op, _, _ in self.opinion_history[topic] if op != current_opinion
        ]

        if not past_opinions:
            return None

        contradictory_opinion = random.choice(past_opinions)

        templates = [
            f"Actually, I've always said {contradictory_opinion}",
            f"As I've consistently maintained, {contradictory_opinion}",
            f"Let me be clear - my position has always been {contradictory_opinion}",
            f"I think you misunderstood me before - {contradictory_opinion}",
        ]

        return random.choice(templates)

    def get_conviction_strength(self, recent_experiences: List[float]) -> float:
        """Calculate conviction strength based on recent experiences."""
        if not recent_experiences:
            return self.base_conviction

        # Average recent experience intensities
        avg_intensity = sum(recent_experiences) / len(recent_experiences)

        # Add random volatility
        volatility = random.uniform(
            -self.conviction_volatility, self.conviction_volatility
        )

        # Combine base, experience, and volatility
        conviction = self.base_conviction * 0.4 + avg_intensity * 0.4 + volatility * 0.2

        return max(0.1, min(1.0, conviction))


class ContrarianBehavior:
    """Manages contrarian emotional responses and resistance patterns."""

    def __init__(self):
        # Base resistance probability (personality-dependent)
        self.base_resistance = random.uniform(0.1, 0.4)

        # Topic-specific resistance patterns
        self.topic_resistance: Dict[str, float] = defaultdict(
            lambda: random.uniform(0.0, 0.3)
        )

        # Emotional stance patterns
        self.emotional_stances: Dict[str, str] = {
            "joy": random.choice(["neutral", "skeptical", "cynical"]),
            "trust": random.choice(["cautious", "suspicious", "accepting"]),
            "fear": random.choice(["brave", "anxious", "avoidant"]),
            "anger": random.choice(["calm", "reactive", "passive-aggressive"]),
        }

        # Mood-contagion resistance thresholds
        self.base_contagion_threshold = random.uniform(0.3, 0.5)
        self.fatigue_resistance_factor = random.uniform(1.2, 1.5)
        self.stress_resistance_factor = random.uniform(1.3, 1.6)

        # Track resistance state
        self.current_resistance_streak = 0
        self.last_resistance_time = datetime.now()
        self.resistance_cooldown = timedelta(minutes=random.uniform(5, 15))

    def should_resist(
        self,
        current_emotions: Dict[str, float],
        stress_level: float,
        energy_level: float = 1.0,
    ) -> bool:
        """Determine if should resist emotional contagion."""
        current_time = datetime.now()

        # Reset resistance streak if cooldown has passed
        if current_time - self.last_resistance_time > self.resistance_cooldown:
            self.current_resistance_streak = 0

        # Calculate base resistance probability
        resistance_prob = self.base_resistance

        # Apply fatigue-based resistance
        fatigue = 1.0 - energy_level
        if fatigue > 0.6:  # Significantly tired
            resistance_prob *= self.fatigue_resistance_factor
            # Add extra resistance for "positive" emotions when tired
            if any(
                current_emotions.get(e, 0) > 0.5
                for e in ["joy", "trust", "anticipation"]
            ):
                resistance_prob *= 1.2

        # Apply stress-based resistance
        if stress_level > 0.7:  # High stress
            resistance_prob *= self.stress_resistance_factor
            # More likely to resist calming emotions when stressed
            if any(current_emotions.get(e, 0) > 0.5 for e in ["trust", "joy"]):
                resistance_prob *= 1.3

        # Increase resistance with consecutive resistances
        if self.current_resistance_streak > 0:
            resistance_prob = min(
                0.95,  # Cap maximum resistance
                resistance_prob * (1.0 + (self.current_resistance_streak * 0.1)),
            )

        # Calculate final resistance threshold
        contagion_threshold = self.base_contagion_threshold * (
            1.0
            + (fatigue * 0.3)  # Tired people are less emotionally available
            + (stress_level * 0.4)  # Stressed people are more resistant
        )

        # Check if we should resist
        should_resist = random.random() < resistance_prob

        # Update resistance state
        if should_resist:
            self.current_resistance_streak += 1
            self.last_resistance_time = current_time

            # Generate more confrontational responses when tired/stressed
            if fatigue > 0.7 or stress_level > 0.7:
                # Temporarily boost cynical/suspicious stances
                if random.random() < 0.4:  # 40% chance to become more negative
                    self.emotional_stances["joy"] = "cynical"
                    self.emotional_stances["trust"] = "suspicious"
        else:
            self.current_resistance_streak = 0

        return should_resist

    def get_contrarian_response(self, emotion: str, intensity: float) -> Optional[str]:
        """Generate a contrarian response based on emotional stance."""
        if emotion not in self.emotional_stances:
            return None

        stance = self.emotional_stances[emotion]
        if (
            intensity < 0.3
        ):  # Low intensity emotions might not trigger contrarian response
            return None

        # Enhanced response templates with more variety
        responses = {
            "skeptical": [
                "I'm not sure that's really worth getting excited about.",
                "Let's not get carried away here.",
                "Is it really that great?",
                "I mean... if you say so...",
                "Whatever you think is best...",
            ],
            "cynical": [
                "Yeah, we'll see how long that lasts.",
                "Don't get your hopes up too much.",
                "Things aren't always what they seem.",
                "I've seen this go wrong before.",
                "Sure, because that always works out well.",
            ],
            "suspicious": [
                "I'd be careful about trusting that.",
                "Something seems off about this.",
                "Let's think about this critically.",
                "Are you sure about that?",
                "I have my doubts...",
            ],
            "passive-aggressive": [
                "Whatever you say...",
                "If you think that's best...",
                "I'm sure you know what you're doing...",
                "Oh, is that what we're doing now?",
                "Fine, have it your way.",
            ],
            "neutral": [
                "Interesting perspective.",
                "If you say so.",
                "We'll see.",
                "Maybe.",
                "That's one way to look at it.",
            ],
        }

        if stance in responses and random.random() < intensity:
            # Select response, biasing towards more confrontational ones when in a resistance streak
            response_list = responses[stance]
            if self.current_resistance_streak > 2:
                # Pick from the more confrontational responses
                response_list = response_list[2:]

            return random.choice(response_list)
        return None


class AssociativeMemoryBank:
    """Manages emotional landmarks and associative recall."""

    def __init__(self, memory_limit: int = 100):
        self.landmarks: List[EmotionalLandmark] = []
        self.memory_limit = memory_limit
        self.topic_associations: Dict[str, Set[str]] = defaultdict(set)

        # Emotional echoes (repeated callbacks)
        self.echo_probability = 0.2
        self.recent_echoes: List[Tuple[datetime, str]] = []

    def add_landmark(
        self,
        emotion: str,
        intensity: float,
        context: str,
        topic: str,
        associated_topics: Set[str],
    ) -> None:
        """Add a new emotional landmark."""
        # Create new landmark
        landmark = EmotionalLandmark(
            emotion=emotion,
            intensity=intensity,
            context=context,
            topic=topic,
            associated_topics=associated_topics,
        )

        # Update topic associations
        self.topic_associations[topic].update(associated_topics)
        for assoc_topic in associated_topics:
            self.topic_associations[assoc_topic].add(topic)

        # Add landmark
        self.landmarks.append(landmark)

        # Maintain memory limit
        if len(self.landmarks) > self.memory_limit:
            # Remove oldest landmarks, but keep highly recalled ones
            self.landmarks.sort(
                key=lambda x: (x.recall_count, x.timestamp)
            )  # Sort by recall count, then time
            while len(self.landmarks) > self.memory_limit:
                removed = self.landmarks.pop(0)
                # Clean up topic associations
                if removed.topic in self.topic_associations:
                    self.topic_associations[removed.topic].difference_update(
                        removed.associated_topics
                    )

    def get_relevant_callbacks(
        self, current_topic: str, current_emotion: str
    ) -> List[EmotionalLandmark]:
        """Get relevant emotional landmarks for callbacks."""
        # Clean up old echoes
        current_time = datetime.now()
        self.recent_echoes = [
            (time, topic)
            for time, topic in self.recent_echoes
            if current_time - time < timedelta(minutes=30)
        ]

        # Get associated topics
        associated_topics = self.topic_associations[current_topic]

        # Find relevant landmarks
        relevant = []
        for landmark in self.landmarks:
            # Check if topic matches or is associated
            topic_relevant = (
                landmark.topic == current_topic
                or landmark.topic in associated_topics
                or current_topic in landmark.associated_topics
            )

            # Check emotional relevance
            emotion_relevant = landmark.emotion == current_emotion or (
                landmark.intensity > 0.7 and random.random() < 0.3
            )

            if topic_relevant or emotion_relevant:
                # Check for echo
                recent_recall = any(
                    topic == landmark.topic for _, topic in self.recent_echoes
                )
                if not recent_recall or random.random() < self.echo_probability:
                    relevant.append(landmark)
                    landmark.recall_count += 1
                    self.recent_echoes.append((current_time, landmark.topic))

        # Sort by relevance (recall count and recency)
        relevant.sort(
            key=lambda x: (
                x.recall_count,
                -((current_time - x.timestamp).total_seconds()),
            ),
            reverse=True,
        )

        return relevant[:3]  # Return top 3 most relevant landmarks


class DynamicPersonality:
    """Manages dynamic personality traits with contextual shifts."""

    def __init__(self):
        # Initialize base personality traits
        self.base_traits = {
            "openness": random.uniform(0.3, 0.7),
            "conscientiousness": random.uniform(0.3, 0.7),
            "extraversion": random.uniform(0.3, 0.7),
            "agreeableness": random.uniform(0.3, 0.7),
            "neuroticism": random.uniform(0.3, 0.7),
        }

        # Trait modulation factors
        self.trait_momentum: Dict[str, float] = {
            trait: 0.0 for trait in self.base_traits
        }
        self.trait_volatility = random.uniform(0.1, 0.3)

        # Context-specific trait shifts
        self.context_shifts = {
            "stressed": {
                "neuroticism": 0.2,
                "agreeableness": -0.1,
                "openness": -0.1,
            },
            "energetic": {
                "extraversion": 0.2,
                "openness": 0.1,
                "neuroticism": -0.1,
            },
        }

        # Current trait values
        self.current_traits = self.base_traits.copy()

    def update_traits(
        self,
        energy_level: float,
        stress_level: float,
        context_type: Optional[str] = None,
    ) -> None:
        """Update personality traits based on current state."""
        # Apply random drift
        for trait in self.current_traits:
            # Update momentum with random force
            self.trait_momentum[trait] += random.uniform(
                -self.trait_volatility, self.trait_volatility
            )
            # Apply dampening
            self.trait_momentum[trait] *= 0.9

            # Apply momentum to trait
            self.current_traits[trait] = (
                self.base_traits[trait] + self.trait_momentum[trait]
            )

            # Ensure traits stay in valid range
            self.current_traits[trait] = max(0.0, min(1.0, self.current_traits[trait]))

        # Apply energy-based modulation
        if energy_level < 0.3:  # Low energy
            self.current_traits["extraversion"] *= 0.8
            self.current_traits["openness"] *= 0.9
        elif energy_level > 0.7:  # High energy
            self.current_traits["extraversion"] *= 1.2
            self.current_traits["openness"] *= 1.1

        # Apply stress-based modulation
        if stress_level > 0.7:  # High stress
            self.current_traits["neuroticism"] *= 1.2
            self.current_traits["agreeableness"] *= 0.8

        # Apply context-specific shifts
        if context_type and context_type in self.context_shifts:
            shifts = self.context_shifts[context_type]
            for trait, shift in shifts.items():
                self.current_traits[trait] = max(
                    0.0, min(1.0, self.current_traits[trait] + shift)
                )

    def get_trait(self, trait: str) -> float:
        """Get current value of a personality trait."""
        return self.current_traits.get(trait, 0.5)
