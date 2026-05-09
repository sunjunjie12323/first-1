"""
WorkingMemory - 工作记忆模块
模拟人脑工作记忆，容量有限，快速存取
类似前额叶皮层的执行功能
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class WorkingMemoryItem:
    """工作记忆项"""
    id: str
    content: np.ndarray
    timestamp: float
    relevance: float = 1.0
    access_count: int = 0
    tag: str = ""


class WorkingMemory:
    """
    工作记忆模块
    
    特点：
    1. 容量有限（Miller's Law: 7±2）
    2. 快速存取
    3. 注意力调制
    4. 信息衰减
    """
    
    def __init__(
        self,
        capacity: int = 7,
        decay_rate: float = 0.1,
        attention_threshold: float = 0.3
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.attention_threshold = attention_threshold
        
        self._items: Dict[str, WorkingMemoryItem] = {}
        self._priority_queue: List[str] = []
        self._item_counter = 0
        
        self._attention_focus: Optional[np.ndarray] = None
        self._current_load = 0.0
        
    def store(
        self, 
        content: np.ndarray, 
        relevance: float = 1.0,
        tag: str = ""
    ) -> str:
        """
        存储信息到工作记忆
        
        Args:
            content: 内容向量
            relevance: 相关性
            tag: 标签
            
        Returns:
            项目ID
        """
        if len(self._items) >= self.capacity:
            self._evict_lowest_priority()
        
        self._item_counter += 1
        item_id = f"wm_{self._item_counter}"
        
        item = WorkingMemoryItem(
            id=item_id,
            content=content.copy(),
            timestamp=time.time(),
            relevance=relevance,
            tag=tag
        )
        
        self._items[item_id] = item
        self._update_priority(item_id, relevance)
        self._current_load = len(self._items) / self.capacity
        
        return item_id
    
    def retrieve(self, item_id: str) -> Optional[np.ndarray]:
        """
        检索工作记忆项
        
        Args:
            item_id: 项目ID
            
        Returns:
            内容向量或None
        """
        if item_id not in self._items:
            return None
        
        item = self._items[item_id]
        item.access_count += 1
        item.relevance = min(1.0, item.relevance + 0.1)
        
        self._update_priority(item_id, item.relevance)
        
        return item.content.copy()
    
    def retrieve_by_tag(self, tag: str) -> List[np.ndarray]:
        """按标签检索"""
        results = []
        for item in self._items.values():
            if item.tag == tag:
                item.access_count += 1
                results.append(item.content.copy())
        return results
    
    def retrieve_by_similarity(
        self, 
        query: np.ndarray, 
        top_k: int = 3
    ) -> List[Tuple[np.ndarray, float]]:
        """
        按相似性检索
        
        Args:
            query: 查询向量
            top_k: 返回数量
            
        Returns:
            (内容, 相似度)列表
        """
        similarities = []
        
        for item_id, item in self._items.items():
            similarity = self._cosine_similarity(query, item.content)
            similarities.append((item_id, similarity, item))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for item_id, similarity, item in similarities[:top_k]:
            item.access_count += 1
            results.append((item.content.copy(), similarity))
        
        return results
    
    def update_attention(self, focus: np.ndarray):
        """更新注意力焦点"""
        self._attention_focus = focus.copy()
        
        for item_id, item in self._items.items():
            if self._attention_focus is not None:
                similarity = self._cosine_similarity(
                    self._attention_focus, 
                    item.content
                )
                if similarity > self.attention_threshold:
                    item.relevance = min(1.0, item.relevance + 0.2)
                else:
                    item.relevance *= (1 - self.decay_rate)
    
    def decay(self):
        """应用时间衰减"""
        current_time = time.time()
        to_remove = []
        
        for item_id, item in self._items.items():
            time_elapsed = current_time - item.timestamp
            decay_factor = np.exp(-self.decay_rate * time_elapsed)
            item.relevance *= decay_factor
            
            if item.relevance < 0.1:
                to_remove.append(item_id)
        
        for item_id in to_remove:
            self.remove(item_id)
    
    def remove(self, item_id: str) -> bool:
        """移除项目"""
        if item_id in self._items:
            del self._items[item_id]
            if item_id in self._priority_queue:
                self._priority_queue.remove(item_id)
            self._current_load = len(self._items) / self.capacity
            return True
        return False
    
    def clear(self):
        """清空工作记忆"""
        self._items.clear()
        self._priority_queue.clear()
        self._current_load = 0.0
    
    def get_load(self) -> float:
        """获取当前负载"""
        return self._current_load
    
    def get_capacity_info(self) -> Dict:
        """获取容量信息"""
        return {
            'capacity': self.capacity,
            'current_items': len(self._items),
            'load': self._current_load,
            'available_slots': self.capacity - len(self._items)
        }
    
    def get_all_items(self) -> List[Dict]:
        """获取所有项目信息"""
        items = []
        for item in self._items.values():
            items.append({
                'id': item.id,
                'relevance': item.relevance,
                'access_count': item.access_count,
                'tag': item.tag,
                'timestamp': item.timestamp
            })
        return items
    
    def _evict_lowest_priority(self):
        """驱逐最低优先级项目"""
        if not self._priority_queue:
            return
        
        self._update_priorities()
        
        lowest_id = self._priority_queue[-1]
        self.remove(lowest_id)
    
    def _update_priority(self, item_id: str, relevance: float):
        """更新优先级"""
        if item_id in self._priority_queue:
            self._priority_queue.remove(item_id)
        
        inserted = False
        for i, existing_id in enumerate(self._priority_queue):
            if relevance > self._items[existing_id].relevance:
                self._priority_queue.insert(i, item_id)
                inserted = True
                break
        
        if not inserted:
            self._priority_queue.append(item_id)
    
    def _update_priorities(self):
        """重新排序优先级队列"""
        items_with_priority = [
            (item_id, self._items[item_id].relevance)
            for item_id in self._priority_queue
            if item_id in self._items
        ]
        items_with_priority.sort(key=lambda x: x[1], reverse=True)
        self._priority_queue = [item_id for item_id, _ in items_with_priority]
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        min_len = min(len(v1), len(v2))
        v1, v2 = v1[:min_len], v2[:min_len]
        
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        
        return dot / norm
