"""
LongTermMemory - 长期记忆模块
模拟人脑长期记忆，持久化存储，容量无限
支持情景记忆和语义记忆
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json


@dataclass
class EpisodicMemory:
    """情景记忆"""
    id: str
    content: np.ndarray
    context: Dict[str, Any]
    timestamp: float
    strength: float = 1.0
    emotional_weight: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    associations: List[str] = field(default_factory=list)
    consolidation_level: int = 0


@dataclass
class SemanticMemory:
    """语义记忆"""
    id: str
    concept: str
    pattern: np.ndarray
    attributes: Dict[str, Any]
    strength: float = 1.0
    exemplars: List[str] = field(default_factory=list)


class LongTermMemory:
    """
    长期记忆模块
    
    特点：
    1. 持久化存储
    2. 容量无限
    3. 支持情景记忆和语义记忆
    4. 记忆重构
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        consolidation_threshold: float = 0.6,
        retrieval_threshold: float = 0.3
    ):
        self.embedding_dim = embedding_dim
        self.consolidation_threshold = consolidation_threshold
        self.retrieval_threshold = retrieval_threshold
        
        self._episodic_memories: Dict[str, EpisodicMemory] = {}
        self._semantic_memories: Dict[str, SemanticMemory] = {}
        self._concept_index: Dict[str, List[str]] = defaultdict(list)
        
        self._memory_counter = 0
        self._association_graph: Dict[str, List[str]] = defaultdict(list)
        
    def store_episodic(
        self, 
        content: np.ndarray,
        context: Optional[Dict] = None,
        emotional_weight: float = 0.5
    ) -> str:
        """
        存储情景记忆
        
        Args:
            content: 内容向量
            context: 上下文信息
            emotional_weight: 情感权重
            
        Returns:
            记忆ID
        """
        self._memory_counter += 1
        memory_id = f"epi_{self._memory_counter}"
        
        if context is None:
            context = {}
        
        memory = EpisodicMemory(
            id=memory_id,
            content=content.copy(),
            context=context,
            timestamp=time.time(),
            strength=1.0,
            emotional_weight=emotional_weight,
            last_accessed=time.time()
        )
        
        self._episodic_memories[memory_id] = memory
        
        self._auto_associate(memory_id, content)
        
        return memory_id
    
    def store_semantic(
        self, 
        concept: str,
        pattern: np.ndarray,
        attributes: Optional[Dict] = None
    ) -> str:
        """
        存储语义记忆
        
        Args:
            concept: 概念名称
            pattern: 模式向量
            attributes: 属性字典
            
        Returns:
            记忆ID
        """
        self._memory_counter += 1
        memory_id = f"sem_{self._memory_counter}"
        
        if attributes is None:
            attributes = {}
        
        memory = SemanticMemory(
            id=memory_id,
            concept=concept,
            pattern=pattern.copy(),
            attributes=attributes
        )
        
        self._semantic_memories[memory_id] = memory
        self._concept_index[concept].append(memory_id)
        
        return memory_id
    
    def retrieve_episodic(
        self, 
        query: np.ndarray,
        top_k: int = 5,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """
        检索情景记忆
        
        Args:
            query: 查询向量
            top_k: 返回数量
            time_range: 时间范围
            
        Returns:
            记忆列表
        """
        candidates = []
        
        for memory_id, memory in self._episodic_memories.items():
            if time_range:
                if not (time_range[0] <= memory.timestamp <= time_range[1]):
                    continue
            
            similarity = self._compute_similarity(query, memory.content)
            
            recency = 1.0 / (1.0 + time.time() - memory.last_accessed)
            strength_factor = memory.strength
            emotional_factor = 1.0 + memory.emotional_weight * 0.5
            consolidation_factor = 1.0 + 0.1 * memory.consolidation_level
            
            score = (
                similarity * 
                recency * 
                strength_factor * 
                emotional_factor * 
                consolidation_factor
            )
            
            if score > self.retrieval_threshold:
                candidates.append((memory_id, score, memory))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for memory_id, score, memory in candidates[:top_k]:
            memory.access_count += 1
            memory.last_accessed = time.time()
            
            results.append({
                'id': memory_id,
                'content': memory.content.copy(),
                'context': memory.context.copy(),
                'strength': memory.strength,
                'emotional_weight': memory.emotional_weight,
                'score': score,
                'timestamp': memory.timestamp
            })
        
        return results
    
    def retrieve_semantic(
        self, 
        query: Optional[np.ndarray] = None,
        concept: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        检索语义记忆
        
        Args:
            query: 查询向量
            concept: 概念名称
            top_k: 返回数量
            
        Returns:
            记忆列表
        """
        candidates = []
        
        if concept:
            memory_ids = self._concept_index.get(concept, [])
            for memory_id in memory_ids:
                if memory_id in self._semantic_memories:
                    memory = self._semantic_memories[memory_id]
                    candidates.append((memory_id, 1.0, memory))
        else:
            for memory_id, memory in self._semantic_memories.items():
                if query is not None:
                    similarity = self._compute_similarity(query, memory.pattern)
                else:
                    similarity = memory.strength
                
                if similarity > self.retrieval_threshold:
                    candidates.append((memory_id, similarity, memory))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for memory_id, score, memory in candidates[:top_k]:
            results.append({
                'id': memory_id,
                'concept': memory.concept,
                'pattern': memory.pattern.copy(),
                'attributes': memory.attributes.copy(),
                'strength': memory.strength,
                'score': score
            })
        
        return results
    
    def create_association(self, memory_id1: str, memory_id2: str, strength: float = 1.0):
        """
        创建记忆关联
        
        Args:
            memory_id1: 记忆ID1
            memory_id2: 记忆ID2
            strength: 关联强度
        """
        if memory_id1 in self._episodic_memories and memory_id2 in self._episodic_memories:
            if memory_id2 not in self._episodic_memories[memory_id1].associations:
                self._episodic_memories[memory_id1].associations.append(memory_id2)
            if memory_id1 not in self._episodic_memories[memory_id2].associations:
                self._episodic_memories[memory_id2].associations.append(memory_id1)
            
            self._association_graph[memory_id1].append(memory_id2)
            self._association_graph[memory_id2].append(memory_id1)
    
    def retrieve_associated(
        self, 
        memory_id: str, 
        depth: int = 2
    ) -> List[Dict]:
        """
        检索关联记忆
        
        Args:
            memory_id: 起始记忆ID
            depth: 搜索深度
            
        Returns:
            关联记忆列表
        """
        visited = set()
        queue = [(memory_id, 0)]
        results = []
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
            
            visited.add(current_id)
            
            if current_id in self._episodic_memories:
                memory = self._episodic_memories[current_id]
                results.append({
                    'id': current_id,
                    'content': memory.content.copy(),
                    'strength': memory.strength,
                    'depth': current_depth
                })
            
            for associated_id in self._association_graph.get(current_id, []):
                if associated_id not in visited:
                    queue.append((associated_id, current_depth + 1))
        
        return results[1:]
    
    def consolidate(self, memory_id: str) -> bool:
        """
        巩固记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否成功
        """
        if memory_id in self._episodic_memories:
            memory = self._episodic_memories[memory_id]
            memory.consolidation_level += 1
            memory.strength = min(2.0, memory.strength * 1.1)
            return True
        return False
    
    def strengthen(self, memory_id: str, factor: float = 1.1):
        """增强记忆"""
        if memory_id in self._episodic_memories:
            memory = self._episodic_memories[memory_id]
            memory.strength = min(2.0, memory.strength * factor)
        elif memory_id in self._semantic_memories:
            memory = self._semantic_memories[memory_id]
            memory.strength = min(2.0, memory.strength * factor)
    
    def forget(self, memory_id: str) -> bool:
        """遗忘记忆"""
        if memory_id in self._episodic_memories:
            del self._episodic_memories[memory_id]
            if memory_id in self._association_graph:
                for associated_id in self._association_graph[memory_id]:
                    if associated_id in self._association_graph:
                        self._association_graph[associated_id] = [
                            x for x in self._association_graph[associated_id] 
                            if x != memory_id
                        ]
                del self._association_graph[memory_id]
            return True
        elif memory_id in self._semantic_memories:
            memory = self._semantic_memories[memory_id]
            if memory.concept in self._concept_index:
                self._concept_index[memory.concept].remove(memory_id)
            del self._semantic_memories[memory_id]
            return True
        return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        episodic_strengths = [m.strength for m in self._episodic_memories.values()]
        semantic_strengths = [m.strength for m in self._semantic_memories.values()]
        
        return {
            'episodic_count': len(self._episodic_memories),
            'semantic_count': len(self._semantic_memories),
            'total_associations': sum(len(v) for v in self._association_graph.values()) // 2,
            'avg_episodic_strength': np.mean(episodic_strengths) if episodic_strengths else 0,
            'avg_semantic_strength': np.mean(semantic_strengths) if semantic_strengths else 0,
            'concept_count': len(self._concept_index)
        }
    
    def save(self, filepath: str):
        """保存到文件"""
        data = {
            'episodic': {
                k: {
                    'content': v.content.tolist(),
                    'context': v.context,
                    'timestamp': v.timestamp,
                    'strength': v.strength,
                    'emotional_weight': v.emotional_weight,
                    'associations': v.associations,
                    'consolidation_level': v.consolidation_level
                }
                for k, v in self._episodic_memories.items()
            },
            'semantic': {
                k: {
                    'concept': v.concept,
                    'pattern': v.pattern.tolist(),
                    'attributes': v.attributes,
                    'strength': v.strength
                }
                for k, v in self._semantic_memories.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for k, v in data['episodic'].items():
            self._episodic_memories[k] = EpisodicMemory(
                id=k,
                content=np.array(v['content']),
                context=v['context'],
                timestamp=v['timestamp'],
                strength=v['strength'],
                emotional_weight=v['emotional_weight'],
                associations=v['associations'],
                consolidation_level=v['consolidation_level']
            )
        
        for k, v in data['semantic'].items():
            self._semantic_memories[k] = SemanticMemory(
                id=k,
                concept=v['concept'],
                pattern=np.array(v['pattern']),
                attributes=v['attributes'],
                strength=v['strength']
            )
            self._concept_index[v['concept']].append(k)
    
    def _auto_associate(self, new_memory_id: str, content: np.ndarray):
        """自动关联相似记忆"""
        for memory_id, memory in self._episodic_memories.items():
            if memory_id == new_memory_id:
                continue
            
            similarity = self._compute_similarity(content, memory.content)
            
            if similarity > 0.7:
                self.create_association(new_memory_id, memory_id)
    
    def _compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算相似度"""
        min_len = min(len(v1), len(v2))
        v1, v2 = v1[:min_len], v2[:min_len]
        
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        
        return dot / norm
