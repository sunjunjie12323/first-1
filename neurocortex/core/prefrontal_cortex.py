from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .memory_trace import WorkingMemoryItem

logger = logging.getLogger(__name__)


class PrefrontalCortex:
    """
    Prefrontal Cortex-inspired working memory and executive control module.

    Implements:
    1. Working memory: limited-capacity buffer for current context
    2. Attention gating: determines what enters working memory
    3. Goal maintenance: tracks current conversational goals
    4. Interference control: prevents irrelevant memories from
       overwhelming current processing

    The PFC acts as the executive controller that coordinates
    hippocampal retrieval and neocortical knowledge with the
    current task demands.
    """

    def __init__(
        self,
        working_memory_capacity: int = 7,
        attention_decay: float = 0.1,
        goal_persistence: float = 0.8,
    ):
        self.working_memory_capacity = working_memory_capacity
        self.attention_decay = attention_decay
        self.goal_persistence = goal_persistence

        self._working_memory: List[WorkingMemoryItem] = []
        self._current_goals: List[str] = []
        self._attention_focus: Optional[str] = None
        self._conversation_turn: int = 0

    def update_working_memory(
        self,
        content: str,
        embedding=None,
        source: str = "current",
        attention_weight: float = 1.0,
    ) -> WorkingMemoryItem:
        """
        Add an item to working memory with attention-weighted gating.
        If capacity is exceeded, the least-attended item is displaced.
        """
        item = WorkingMemoryItem(
            content=content,
            embedding=embedding,
            attention_weight=attention_weight,
            source=source,
        )

        self._working_memory.append(item)

        self._decay_attention()

        if len(self._working_memory) > self.working_memory_capacity:
            self._working_memory.sort(key=lambda x: x.attention_weight)
            displaced = self._working_memory.pop(0)
            logger.debug(f"PFC displaced item: {displaced.content[:50]}...")

        return item

    def focus_attention(self, focus: str) -> None:
        """
        Set the current attention focus. This modulates which
        memories are most relevant for retrieval.
        """
        self._attention_focus = focus

    def set_goals(self, goals: List[str]) -> None:
        """
        Set current conversational goals. Goals persist across
        turns but decay over time if not reinforced.
        """
        self._current_goals = goals

    def update_goals(self, new_goal: str) -> None:
        """
        Add or reinforce a goal. If the goal is similar to an
        existing one, it's reinforced rather than duplicated.
        """
        for i, existing in enumerate(self._current_goals):
            if self._goal_similarity(new_goal, existing) > 0.7:
                self._current_goals[i] = new_goal
                return
        self._current_goals.append(new_goal)

    def get_relevant_context(self, query: str) -> str:
        """
        Construct a context string from working memory items,
        prioritized by attention weight and goal relevance.
        """
        if not self._working_memory:
            return ""

        scored_items = []
        for item in self._working_memory:
            score = item.attention_weight
            if self._attention_focus:
                focus_overlap = self._compute_text_overlap(
                    item.content, self._attention_focus
                )
                score += focus_overlap * 0.5

            for goal in self._current_goals:
                goal_overlap = self._compute_text_overlap(item.content, goal)
                score += goal_overlap * self.goal_persistence

            scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        context_parts = []
        for item, _ in scored_items[:5]:
            context_parts.append(f"[{item.source}] {item.content}")

        return "\n".join(context_parts)

    def get_working_memory_contents(self) -> List[WorkingMemoryItem]:
        return sorted(
            self._working_memory, key=lambda x: x.attention_weight, reverse=True
        )

    def advance_turn(self) -> int:
        self._conversation_turn += 1
        return self._conversation_turn

    @property
    def current_turn(self) -> int:
        return self._conversation_turn

    @property
    def attention_focus(self) -> Optional[str]:
        return self._attention_focus

    @property
    def goals(self) -> List[str]:
        return self._current_goals.copy()

    def clear_working_memory(self) -> None:
        self._working_memory.clear()

    def get_status(self) -> Dict:
        return {
            "working_memory_items": len(self._working_memory),
            "capacity": self.working_memory_capacity,
            "current_turn": self._conversation_turn,
            "attention_focus": self._attention_focus,
            "active_goals": len(self._current_goals),
        }

    def _decay_attention(self) -> None:
        for item in self._working_memory:
            item.attention_weight *= (1.0 - self.attention_decay)

    def _goal_similarity(self, goal1: str, goal2: str) -> float:
        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def _compute_text_overlap(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
