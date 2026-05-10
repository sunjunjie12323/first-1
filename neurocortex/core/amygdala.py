from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .memory_trace import EpisodicTrace

logger = logging.getLogger(__name__)


class Amygdala:
    """
    Amygdala-inspired importance modulation module.

    The amygdala assigns emotional significance to experiences,
    modulating how strongly they are encoded and how resistant
    they are to forgetting. This is a key mechanism behind
    the "flashbulb memory" effect in humans.

    Innovation: Unlike simple sentiment analysis, this module
    computes a multi-dimensional importance score that considers
    emotional intensity, personal relevance, novelty, and social
    significance - all factors known to influence amygdalar
    processing in the human brain.
    """

    def __init__(
        self,
        emotional_weight: float = 0.4,
        novelty_weight: float = 0.3,
        social_weight: float = 0.2,
        goal_relevance_weight: float = 0.1,
        arousal_threshold: float = 0.3,
    ):
        self.emotional_weight = emotional_weight
        self.novelty_weight = novelty_weight
        self.social_weight = social_weight
        self.goal_relevance_weight = goal_relevance_weight
        self.arousal_threshold = arousal_threshold

        self._current_goals: List[str] = []
        self._known_entities: Dict[str, int] = {}

    def assess_importance(
        self,
        content: str,
        emotional_valence: float = 0.0,
        emotional_intensity: float = 0.0,
        novelty_score: float = 0.0,
        social_relevance: float = 0.0,
        source: str = "unknown",
        current_goals: Optional[List[str]] = None,
    ) -> Tuple[float, float]:
        """
        Assess the importance and emotional tagging of a memory.

        Returns:
            (importance_score, emotional_valence) tuple

        The importance score combines:
        - Emotional intensity (flashbulb memory effect)
        - Novelty (new information is more important)
        - Social relevance (interactions with known people)
        - Goal relevance (alignment with current objectives)
        """
        emotional_component = min(1.0, abs(emotional_intensity))

        novelty_component = min(1.0, novelty_score)

        social_component = self._assess_social_relevance(content, source)

        goal_component = 0.0
        if current_goals:
            self._current_goals = current_goals
        if self._current_goals:
            goal_component = self._assess_goal_relevance(content)

        importance = (
            emotional_component * self.emotional_weight
            + novelty_component * self.novelty_weight
            + social_component * self.social_weight
            + goal_component * self.goal_relevance_weight
        )

        importance = min(1.0, importance)

        if emotional_intensity > self.arousal_threshold:
            importance = min(1.0, importance * (1.0 + 0.3 * emotional_intensity))

        return importance, emotional_valence

    def compute_decay_modifier(self, trace: EpisodicTrace) -> float:
        """
        Compute a decay rate modifier based on amygdalar importance.
        High-importance memories decay slower (emotionally significant
        events are better remembered).
        """
        base_modifier = 1.0 - trace.importance * 0.7

        if abs(trace.emotional_valence) > 0.5:
            emotional_protection = 1.0 - abs(trace.emotional_valence) * 0.3
            base_modifier *= emotional_protection

        if trace.reactivation_count > 0:
            reactivation_protection = 1.0 / (1.0 + 0.1 * trace.reactivation_count)
            base_modifier *= reactivation_protection

        return max(0.01, base_modifier)

    def update_entity_familiarity(self, entity: str) -> None:
        self._known_entities[entity] = self._known_entities.get(entity, 0) + 1

    def _assess_social_relevance(self, content: str, source: str) -> float:
        relevance = 0.0
        if source != "unknown" and source != "system":
            familiarity = self._known_entities.get(source, 0)
            relevance = min(1.0, 0.3 + 0.1 * familiarity)

        personal_indicators = [
            "我", "你", "我们", "我的", "你的", "喜欢", "讨厌",
            "开心", "难过", "生气", "害怕", "希望", "担心",
            "i", "you", "we", "my", "your", "like", "hate",
            "love", "fear", "hope", "worry",
        ]
        content_lower = content.lower()
        for indicator in personal_indicators:
            if indicator in content_lower:
                relevance = min(1.0, relevance + 0.15)
                break

        return relevance

    def _assess_goal_relevance(self, content: str) -> float:
        if not self._current_goals:
            return 0.0

        content_words = set(content.lower().split())
        max_relevance = 0.0
        for goal in self._current_goals:
            goal_words = set(goal.lower().split())
            if not goal_words:
                continue
            overlap = len(content_words & goal_words) / len(goal_words)
            max_relevance = max(max_relevance, overlap)

        return min(1.0, max_relevance)

    def get_status(self) -> Dict:
        return {
            "known_entities": len(self._known_entities),
            "active_goals": len(self._current_goals),
            "emotional_weight": self.emotional_weight,
            "novelty_weight": self.novelty_weight,
        }
