"""
具身智能记忆系统
整合多种记忆类型的完整记忆架构
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from .vector_store import VectorStore


@dataclass
class WorkingMemoryItem:
    """工作记忆项"""
    content: str
    attention_level: float
    created_at: float


@dataclass
class EpisodicMemory:
    """情景记忆"""
    id: str
    event: str
    context: Dict
    emotions: List[str]
    timestamp: float
    outcome: str
    lessons: List[str]


@dataclass
class SemanticKnowledge:
    """语义知识"""
    concept: str
    definition: str
    relations: List[str]
    examples: List[str]
    confidence: float


class MemorySystem:
    """
    完整类脑记忆系统

    创新点：
    1. 三层记忆架构（工作/情景/语义）
    2. 情感增强记忆
    3. 自动记忆巩固
    4. 重要性驱动的记忆管理
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.working_memory: List[WorkingMemoryItem] = []
        self.episodic_memory: List[EpisodicMemory] = []
        self.semantic_knowledge: List[SemanticKnowledge] = []

        self.max_working_memory = 7
        self.consolidation_threshold = 50

    def encode_perception(self, perception_data: Any, context: str) -> str:
        """
        编码感知为记忆

        Args:
            perception_data: 感知数据
            context: 当前上下文

        Returns:
            记忆 ID
        """
        if hasattr(perception_data, '__dict__'):
            content = f"{context}: {str(perception_data.__dict__)}"
        else:
            content = f"{context}: {str(perception_data)}"

        metadata = {
            "type": "perception",
            "context": context,
            "timestamp": datetime.now().timestamp()
        }

        memory_id = self.vector_store.add(content, metadata, importance=0.8)

        self._update_working_memory(content, attention_level=1.0)

        return memory_id

    def encode_experience(self, experience: str, outcome: str,
                         emotions: List[str] = None) -> str:
        """
        编码经验为情景记忆

        Args:
            experience: 经验描述
            outcome: 结果
            emotions: 情感标签

        Returns:
            记忆 ID
        """
        memory_id = f"ep_{len(self.episodic_memory)}_{datetime.now().timestamp()}"

        emotion_boost = 0.0
        if emotions:
            emotion_boost = sum([
                0.2 for e in emotions
                if e in ['成功', '高兴', '兴奋', '惊讶', '失败', '恐惧', '焦虑']
            ])

        metadata = {
            "type": "experience",
            "outcome": outcome,
            "emotions": emotions or []
        }

        self.vector_store.add(
            experience,
            metadata,
            importance=0.6 + emotion_boost
        )

        episodic = EpisodicMemory(
            id=memory_id,
            event=experience,
            context={},
            emotions=emotions or [],
            timestamp=datetime.now().timestamp(),
            outcome=outcome,
            lessons=[]
        )
        self.episodic_memory.append(episodic)

        self._update_working_memory(
            f"经验: {experience[:50]}... 结果: {outcome}",
            attention_level=0.8
        )

        return memory_id

    def store_knowledge(self, concept: str, definition: str,
                      relations: List[str] = None, examples: List[str] = None):
        """存储语义知识"""
        knowledge = SemanticKnowledge(
            concept=concept,
            definition=definition,
            relations=relations or [],
            examples=examples or [],
            confidence=1.0
        )

        self.semantic_knowledge.append(knowledge)

        content = f"知识: {concept} - {definition}"
        self.vector_store.add(content, metadata={
            "type": "knowledge",
            "concept": concept
        }, importance=1.0)

    def recall(self, query: str, memory_type: str = "all",
              top_k: int = 5) -> List[Dict]:
        """
        记忆检索

        Args:
            query: 查询内容
            memory_type: 记忆类型
            top_k: 返回数量

        Returns:
            检索结果
        """
        results = self.vector_store.search(query, top_k=top_k)

        if memory_type == "experience":
            results = [r for r in results
                      if r["metadata"].get("type") == "experience"]
        elif memory_type == "knowledge":
            results = [r for r in results
                      if r["metadata"].get("type") == "knowledge"]
        elif memory_type == "perception":
            results = [r for r in results
                      if r["metadata"].get("type") == "perception"]

        return results[:top_k]

    def get_related_experiences(self, current_task: str,
                               limit: int = 5) -> List[str]:
        """获取相关经验"""
        memories = self.recall(current_task, memory_type="experience", top_k=limit)
        return [m["content"] for m in memories]

    def _update_working_memory(self, content: str, attention_level: float):
        """更新工作记忆"""
        item = WorkingMemoryItem(
            content=content,
            attention_level=attention_level,
            created_at=datetime.now().timestamp()
        )

        self.working_memory.append(item)

        if len(self.working_memory) > self.max_working_memory:
            self.working_memory.pop(0)

    def consolidate(self):
        """记忆巩固"""
        print("开始记忆巩固...")

        if len(self.working_memory) > self.consolidation_threshold:
            for item in self.working_memory[-20:]:
                self.vector_store.add(
                    item.content,
                    metadata={"type": "consolidated"},
                    importance=item.attention_level * 0.5
                )

        self.vector_store.consolidate()

        self.working_memory.clear()

        print(f"✓ 情景记忆: {len(self.episodic_memory)} 条")
        print(f"✓ 语义知识: {len(self.semantic_knowledge)} 条")

    def reflect_on_experiences(self) -> List[str]:
        """反思经验，提取教训"""
        if not self.episodic_memory:
            return []

        recent = self.episodic_memory[-10:]

        lessons = []
        for exp in recent:
            if exp.outcome == "成功":
                lessons.append(f"成功经验: {exp.event[:50]}")
            elif exp.outcome == "失败":
                lessons.append(f"失败教训: {exp.event[:50]}")

        return lessons[:5]

    def get_memory_summary(self) -> Dict:
        """获取记忆摘要"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "semantic_knowledge_size": len(self.semantic_knowledge),
            "recent_lessons": self.reflect_on_experiences()
        }

    def reset(self):
        """重置记忆系统"""
        self.vector_store = VectorStore()
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_knowledge.clear()
