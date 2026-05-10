from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class Amygdala:
    def __init__(
        self,
        emotional_weight: float = 0.35,
        novelty_weight: float = 0.25,
        social_weight: float = 0.15,
        goal_weight: float = 0.25,
    ):
        self.emotional_weight = emotional_weight
        self.novelty_weight = novelty_weight
        self.social_weight = social_weight
        self.goal_weight = goal_weight

    def assess_importance(
        self,
        emotional_valence: float = 0.0,
        novelty_score: float = 0.0,
        social_relevance: float = 0.0,
        goal_relevance: float = 0.0,
    ) -> float:
        emotional_component = abs(emotional_valence)
        novelty_component = min(1.0, novelty_score)
        social_component = min(1.0, social_relevance)
        goal_component = min(1.0, goal_relevance)

        importance = (
            self.emotional_weight * emotional_component
            + self.novelty_weight * novelty_component
            + self.social_weight * social_component
            + self.goal_weight * goal_component
        )
        return float(min(1.0, max(0.0, importance)))

    def modify_decay_rate(self, base_decay_rate: float, importance: float) -> float:
        importance_factor = 1.0 - 0.8 * importance
        modified_rate = base_decay_rate * importance_factor
        return float(max(0.001, modified_rate))

    def compute_emotional_boost(self, emotional_valence: float) -> float:
        return float(abs(emotional_valence) * 0.5)
