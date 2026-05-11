from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    content: str
    embedding: Any = None
    attention_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrefrontalCortex:
    def __init__(self, capacity: int = 7, attention_decay: float = 0.1):
        self.capacity = capacity
        self.attention_decay = attention_decay
        self.working_memory: List[WorkingMemoryItem] = []
        self.current_goal: Optional[str] = None
        self.goal_embedding: Optional[Any] = None
        self.context_buffer: Dict[str, Any] = {}

    def add_to_working_memory(
        self,
        content: str,
        embedding: Any = None,
        attention_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        item = WorkingMemoryItem(
            content=content,
            embedding=embedding,
            attention_weight=attention_weight,
            metadata=metadata or {},
        )

        if self.current_goal and embedding is not None:
            goal_relevance = self._compute_goal_relevance(embedding)
            item.attention_weight *= goal_relevance

        self.working_memory.append(item)
        self._apply_attention_gating()

        if len(self.working_memory) > self.capacity:
            self.working_memory.sort(key=lambda x: x.attention_weight, reverse=True)
            self.working_memory = self.working_memory[: self.capacity]

        return True

    def set_goal(self, goal: str, goal_embedding: Any = None) -> None:
        self.current_goal = goal
        self.goal_embedding = goal_embedding
        self._reweight_by_goal()

    def clear_goal(self) -> None:
        self.current_goal = None
        self.goal_embedding = None

    def assemble_context(self) -> Dict[str, Any]:
        active_items = [item for item in self.working_memory if item.attention_weight > 0.1]
        active_items.sort(key=lambda x: x.attention_weight, reverse=True)

        return {
            "goal": self.current_goal,
            "active_items": [
                {
                    "content": item.content,
                    "attention_weight": item.attention_weight,
                    "metadata": item.metadata,
                }
                for item in active_items
            ],
            "working_memory_load": len(self.working_memory),
            "capacity": self.capacity,
            "context_buffer": dict(self.context_buffer),
        }

    def update_context_buffer(self, key: str, value: Any) -> None:
        self.context_buffer[key] = value

    def decay_attention(self) -> None:
        for item in self.working_memory:
            item.attention_weight *= (1.0 - self.attention_decay)
        self.working_memory = [item for item in self.working_memory if item.attention_weight > 0.05]

    def _apply_attention_gating(self) -> None:
        if not self.working_memory:
            return

        total_attention = sum(item.attention_weight for item in self.working_memory)
        if total_attention > 0:
            for item in self.working_memory:
                item.attention_weight /= total_attention

    def _compute_goal_relevance(self, embedding: Any) -> float:
        if self.goal_embedding is None:
            return 1.0
        import numpy as np

        norm_a = np.linalg.norm(embedding)
        norm_b = np.linalg.norm(self.goal_embedding)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.5
        similarity = float(np.dot(embedding, self.goal_embedding) / (norm_a * norm_b))
        return 0.5 + 0.5 * similarity

    def _reweight_by_goal(self) -> None:
        if not self.current_goal:
            return
        for item in self.working_memory:
            if item.embedding is not None:
                item.attention_weight *= self._compute_goal_relevance(item.embedding)
