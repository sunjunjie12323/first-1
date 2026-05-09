"""
ShortTermMemory - 短期记忆模块
模拟人脑短期记忆，持续时间较短，容量较大
作为工作记忆和长期记忆之间的缓冲
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class ShortTermMemoryItem:
    """短期记忆项"""
    id: str
    content: np.ndarray
    encoded_pattern: np.ndarray
    timestamp: float
    strength: float = 1.0
    emotional_weight: float = 0.5
    rehearsal_count: int = 0
    source: str = ""


class ShortTermMemory:
    """
    短期记忆模块
    
    特点：
    1. 持续时间约15-30秒
    2. 容量较大
    3. 支持复述强化
    4. 自动衰减
    """
    
    def __init__(
        self,
        capacity: int = 20,
        duration: float = 30.0,
        rehearsal_boost: float = 0.2,
        decay_rate: float = 0.05
    ):
        self.capacity = capacity
        self.duration = duration
        self.rehearsal_boost = rehearsal_boost
        self.decay_rate = decay_rate
        
        self._items: Dict[str, ShortTermMemoryItem] = {}
        self._queue = deque(maxlen=capacity)
        self._item_counter = 0
        
        self._rehearsal_buffer: List[str] = []
        
    def store(
        self, 
        content: np.ndarray,
        encoded_pattern: Optional[np.ndarray] = None,
        emotional_weight: float = 0.5,
        source: str = ""
    ) -> str:
        """
        存储短期记忆
        
        Args:
            content: 原始内容
            encoded_pattern: 编码模式
            emotional_weight: 情感权重
            source: 来源
            
        Returns:
            记忆ID
        """
        if len(self._items) >= self.capacity:
            self._evict_oldest()
        
        self._item_counter += 1
        item_id = f"stm_{self._item_counter}"
        
        if encoded_pattern is None:
            encoded_pattern = content.copy()
        
        item = ShortTermMemoryItem(
            id=item_id,
            content=content.copy(),
            encoded_pattern=encoded_pattern.copy(),
            timestamp=time.time(),
            strength=1.0,
            emotional_weight=emotional_weight,
            source=source
        )
        
        self._items[item_id] = item
        self._queue.append(item_id)
        
        return item_id
    
    def retrieve(self, item_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        检索短期记忆
        
        Args:
            item_id: 记忆ID
            
        Returns:
            (内容, 强度)或None
        """
        if item_id not in self._items:
            return None
        
        item = self._items[item_id]
        
        if time.time() - item.timestamp > self.duration:
            self._expire(item_id)
            return None
        
        return item.content.copy(), item.strength
    
    def retrieve_by_pattern(
        self, 
        pattern: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        按模式检索
        
        Args:
            pattern: 查询模式
            top_k: 返回数量
            
        Returns:
            (内容, 相似度, 强度)列表
        """
        self._cleanup_expired()
        
        similarities = []
        for item_id, item in self._items.items():
            similarity = self._compute_similarity(pattern, item.encoded_pattern)
            similarities.append((item_id, similarity, item))
        
        similarities.sort(key=lambda x: x[1] * x[2].strength, reverse=True)
        
        results = []
        for item_id, similarity, item in similarities[:top_k]:
            results.append((item.content.copy(), similarity, item.strength))
        
        return results
    
    def rehearse(self, item_id: str) -> bool:
        """
        复述强化
        
        Args:
            item_id: 记忆ID
            
        Returns:
            是否成功
        """
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        item.rehearsal_count += 1
        item.strength = min(2.0, item.strength + self.rehearsal_boost)
        item.timestamp = time.time()
        
        if item_id not in self._rehearsal_buffer:
            self._rehearsal_buffer.append(item_id)
        
        return True
    
    def batch_rehearse(self, item_ids: List[str]) -> int:
        """批量复述"""
        success_count = 0
        for item_id in item_ids:
            if self.rehearse(item_id):
                success_count += 1
        return success_count
    
    def should_consolidate(self, item_id: str) -> bool:
        """
        判断是否应该巩固到长期记忆
        
        Args:
            item_id: 记忆ID
            
        Returns:
            是否应该巩固
        """
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        
        consolidation_score = (
            item.strength * 0.4 +
            item.emotional_weight * 0.3 +
            min(1.0, item.rehearsal_count / 5) * 0.3
        )
        
        return consolidation_score > 0.6
    
    def get_consolidation_candidates(self) -> List[str]:
        """获取巩固候选"""
        candidates = []
        for item_id, item in self._items.items():
            if self.should_consolidate(item_id):
                candidates.append(item_id)
        return candidates
    
    def decay(self):
        """应用时间衰减"""
        current_time = time.time()
        to_remove = []
        
        for item_id, item in self._items.items():
            time_elapsed = current_time - item.timestamp
            
            if time_elapsed > self.duration:
                to_remove.append(item_id)
                continue
            
            decay_factor = np.exp(-self.decay_rate * time_elapsed)
            item.strength *= decay_factor
            
            if item.strength < 0.1:
                to_remove.append(item_id)
        
        for item_id in to_remove:
            self._expire(item_id)
    
    def remove(self, item_id: str) -> bool:
        """移除记忆"""
        if item_id in self._items:
            del self._items[item_id]
            if item_id in self._queue:
                self._queue.remove(item_id)
            if item_id in self._rehearsal_buffer:
                self._rehearsal_buffer.remove(item_id)
            return True
        return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self._items:
            return {
                'count': 0,
                'avg_strength': 0,
                'avg_rehearsal': 0
            }
        
        strengths = [item.strength for item in self._items.values()]
        rehearsals = [item.rehearsal_count for item in self._items.values()]
        
        return {
            'count': len(self._items),
            'capacity': self.capacity,
            'avg_strength': np.mean(strengths),
            'avg_rehearsal': np.mean(rehearsals),
            'rehearsal_buffer_size': len(self._rehearsal_buffer)
        }
    
    def _evict_oldest(self):
        """驱逐最旧的记忆"""
        while self._queue and len(self._items) >= self.capacity:
            oldest_id = self._queue[0]
            if oldest_id in self._items:
                del self._items[oldest_id]
            self._queue.popleft()
    
    def _expire(self, item_id: str):
        """过期记忆"""
        self.remove(item_id)
    
    def _cleanup_expired(self):
        """清理过期记忆"""
        current_time = time.time()
        to_remove = []
        
        for item_id, item in self._items.items():
            if current_time - item.timestamp > self.duration:
                to_remove.append(item_id)
        
        for item_id in to_remove:
            self._expire(item_id)
    
    def _compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算相似度"""
        min_len = min(len(v1), len(v2))
        v1, v2 = v1[:min_len], v2[:min_len]
        
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        
        return dot / norm
