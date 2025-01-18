"""
Bayesian Strategy Formation Module
--------------------------------

Implements Bayesian updating for response strategy selection based on:
1. User intent analysis
2. Historical interaction success
3. Context-specific priors
"""

from dataclasses import dataclass
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from litellm import completion
from config_types import ModelConfig


@dataclass
class IntentFeatures:
    """Features extracted from intent analysis for strategy selection."""

    question_likelihood: float  # How likely is this a question (0-1)
    complexity: float  # How complex is the topic/request (0-1)
    emotional_charge: float  # How emotional is the content (0-1)
    clarity: float  # How clear is the intent (0-1)
    knowledge_requirement: float  # How much domain knowledge is needed (0-1)


class BayesianStrategySelector:
    """Bayesian model for selecting optimal response strategies."""

    def __init__(self, strategy_options: List[str], model_config: ModelConfig):
        self.strategies = strategy_options
        self.n_strategies = len(strategy_options)
        self.model_config = model_config

        # Initialize strategy priors (uniform distribution)
        self.priors = np.ones(self.n_strategies) / self.n_strategies

        # Strategy success likelihood matrix for different intent features
        # Shape: (n_strategies, n_features)
        self.likelihood_matrix = np.array(
            [
                # question, complexity, emotion, clarity, knowledge
                [
                    0.95,  # High for questions
                    0.15,  # Low for complexity
                    0.10,  # Very low for emotion
                    0.95,  # High for clarity
                    0.90,  # Very high for knowledge
                ],  # factual_explanation
                [
                    0.95,  # High for questions
                    0.70,  # High for complexity
                    0.30,  # Low for emotion
                    0.20,  # Low for clarity
                    0.40,  # Moderate for knowledge
                ],  # clarifying_question
                [
                    0.30,  # Low for questions
                    0.95,  # Very high for complexity
                    0.80,  # High for emotion
                    0.40,  # Moderate for clarity
                    0.20,  # Low for knowledge
                ],  # example_based
                [
                    0.15,  # Very low for questions
                    0.40,  # Moderate for complexity
                    0.95,  # Very high for emotion
                    0.15,  # Very low for clarity
                    0.30,  # Low for knowledge
                ],  # reframing
            ]
        )

        # Interaction history for strategy effectiveness
        self.history: List[Tuple[str, float]] = []

    def _call_llm_for_features(self, text: str) -> Dict[str, float]:
        """Use LLM to extract feature scores."""
        prompt = f"""Analyze this text and provide numerical scores (0.0 to 1.0) for these aspects:

Text: "{text}"

Score these dimensions:
1. Question Likelihood: How likely is this a question or seeking information? (0=statement, 1=clear question)
2. Complexity: How complex or technical is the topic? (0=simple, 1=very complex)
3. Emotional Charge: How emotional is the content? (0=neutral, 1=highly emotional)
4. Clarity: How clear and explicit is the meaning? (0=very vague, 1=crystal clear)
5. Knowledge Requirement: How much domain expertise is needed? (0=general knowledge, 1=expert knowledge)

Provide scores in this JSON format:
{{
    "question_likelihood": 0.0,
    "complexity": 0.0,
    "emotional_charge": 0.0,
    "clarity": 0.0,
    "knowledge_requirement": 0.0
}}

Only respond with the JSON, no other text."""

        try:
            response = completion(
                model=self.model_config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
            )

            # Extract and parse JSON from response
            scores = json.loads(response.choices[0].message.content)

            # Ensure all required fields are present
            required_fields = [
                "question_likelihood",
                "complexity",
                "emotional_charge",
                "clarity",
                "knowledge_requirement",
            ]
            for field in required_fields:
                if field not in scores:
                    scores[field] = 0.5  # Default to neutral if missing

            # Clip values to valid range
            for key in scores:
                scores[key] = max(0.1, min(1.0, float(scores[key])))

            return scores

        except Exception as e:
            # Fallback to neutral scores if LLM call fails
            return {
                "question_likelihood": 0.5,
                "complexity": 0.5,
                "emotional_charge": 0.5,
                "clarity": 0.5,
                "knowledge_requirement": 0.5,
            }

    def extract_intent_features(self, intent_analysis: str) -> IntentFeatures:
        """Extract numerical features from intent analysis text using LLM."""
        # Get scores from LLM
        scores = self._call_llm_for_features(intent_analysis)

        # Create features object
        features = IntentFeatures(
            question_likelihood=scores["question_likelihood"],
            complexity=scores["complexity"],
            emotional_charge=scores["emotional_charge"],
            clarity=scores["clarity"],
            knowledge_requirement=scores["knowledge_requirement"],
        )

        return features

    def compute_likelihood(self, features: IntentFeatures) -> np.ndarray:
        """Compute likelihood of success for each strategy given features."""
        feature_vector = np.array(
            [
                features.question_likelihood,
                features.complexity,
                features.emotional_charge,
                features.clarity,
                features.knowledge_requirement,
            ]
        )

        # Ensure feature vector is valid
        feature_vector = np.clip(feature_vector, 0.1, 1.0)  # Avoid zeros

        # Compute weighted dot product
        weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])  # Feature importance weights
        likelihoods = np.dot(self.likelihood_matrix, feature_vector * weights)

        # Ensure valid probabilities
        likelihoods = np.clip(likelihoods, 0.1, 1.0)  # Avoid zeros
        return likelihoods / np.sum(likelihoods)  # Now safe to normalize

    def update_priors(self, strategy_idx: int, success_score: float):
        """Update strategy priors based on observed success."""
        # Ensure valid success score
        success_score = max(0.1, min(1.0, success_score))

        # More aggressive update factors
        if success_score > 0.8:
            update_factor = 1.5  # Strong positive reinforcement
        elif success_score > 0.6:
            update_factor = 1.2  # Moderate positive reinforcement
        elif success_score < 0.4:
            update_factor = 0.7  # Strong negative reinforcement
        else:
            update_factor = 0.9  # Slight negative reinforcement

        # Update selected strategy
        self.priors[strategy_idx] *= update_factor

        # Decay other strategies more aggressively
        decay = 0.85  # Stronger decay for non-selected strategies
        for i in range(len(self.priors)):
            if i != strategy_idx:
                self.priors[i] *= decay

        # Ensure no zeros and normalize
        self.priors = np.clip(self.priors, 0.1, 1.0)
        self.priors /= np.sum(self.priors)

        # Store in history
        self.history.append((self.strategies[strategy_idx], success_score))

    def select_strategy(self, intent_analysis: str) -> Tuple[str, Dict[str, float]]:
        """Select the best strategy using Bayesian inference."""
        # Extract features from intent analysis
        features = self.extract_intent_features(intent_analysis)

        # Compute likelihoods
        likelihoods = self.compute_likelihood(features)

        # Compute posterior probabilities
        posteriors = self.priors * likelihoods
        posteriors /= np.sum(posteriors)

        # Select strategy with highest posterior probability
        selected_idx = np.argmax(posteriors)

        # Prepare probability distribution for logging
        strategy_probabilities = {
            strategy: float(prob) for strategy, prob in zip(self.strategies, posteriors)
        }

        return self.strategies[selected_idx], strategy_probabilities

    def get_strategy_insights(self) -> Dict[str, Any]:
        """Get insights about strategy performance."""
        if not self.history:
            return {"message": "No historical data available yet"}

        strategy_scores = {}
        for strategy in self.strategies:
            strategy_uses = [score for s, score in self.history if s == strategy]
            if strategy_uses:
                strategy_scores[strategy] = {
                    "uses": len(strategy_uses),
                    "avg_success": sum(strategy_uses) / len(strategy_uses),
                    "current_prior": float(
                        self.priors[self.strategies.index(strategy)]
                    ),
                }

        return {
            "strategy_performance": strategy_scores,
            "total_interactions": len(self.history),
        }
