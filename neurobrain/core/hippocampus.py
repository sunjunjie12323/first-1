"""
Hippocampus - 海马体模块
负责记忆编码、短期记忆存储和记忆巩固的初始化
模拟人脑海马体的核心功能
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class MemoryTrace:
    """记忆痕迹"""
    id: str
    pattern: np.ndarray
    timestamp: float
    strength: float = 1.0
    emotional_weight: float = 0.5
    access_count: int = 0
    associations: List[str] = field(default_factory=list)
    context: Optional[np.ndarray] = None


class Hippocampus:
    """
    海马体模块
    
    功能：
    1. 记忆编码：将输入信息转换为神经编码
    2. 短期记忆存储：临时存储新信息
    3. 模式分离：区分相似的记忆
    4. 记忆巩固：将重要记忆传递给新皮层
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        memory_capacity: int = 7,
        consolidation_threshold: float = 0.7
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_capacity = memory_capacity
        self.consolidation_threshold = consolidation_threshold
        
        self._encoder_weights = self._init_encoder()
        self._decoder_weights = self._init_decoder()
        
        self._ca3_weights = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self._ca3_weights = (self._ca3_weights + self._ca3_weights.T) / 2
        np.fill_diagonal(self._ca3_weights, 0)
        
        self._memory_traces: Dict[str, MemoryTrace] = {}
        self._memory_queue = deque(maxlen=memory_capacity * 3)
        
        self._pattern_separation_scale = 0.1
        self._memory_counter = 0
        
        self._recency_buffer = deque(maxlen=100)
        
    def encode(
        self, 
        input_data: np.ndarray, 
        emotional_weight: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        编码输入信息
        
        Args:
            input_data: 输入数据
            emotional_weight: 情感权重
            
        Returns:
            编码结果和信息字典
        """
        if len(input_data) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[:len(input_data)] = input_data
            input_data = padded
        elif len(input_data) > self.input_dim:
            input_data = input_data[:self.input_dim]
        
        entorhinal_output = self._entorhinal_cortex(input_data)
        
        dentate_gyrus_output = self._pattern_separation(entorhinal_output)
        
        ca3_output = self._ca3_recurrence(dentate_gyrus_output)
        
        ca1_output = self._ca1_processing(ca3_output, entorhinal_output)
        
        memory_id = self._store_memory_trace(
            ca3_output,
            emotional_weight=emotional_weight,
            input_pattern=input_data
        )
        
        should_consolidate = emotional_weight > self.consolidation_threshold
        
        info = {
            'memory_id': memory_id,
            'encoding_strength': np.mean(np.abs(ca3_output)),
            'should_consolidate': should_consolidate,
            'pattern_separation': np.mean(np.abs(dentate_gyrus_output - entorhinal_output))
        }
        
        return ca1_output, info
    
    def recall(
        self, 
        query: np.ndarray, 
        top_k: int = 5,
        use_associations: bool = True
    ) -> List[Dict]:
        """
        记忆检索
        
        Args:
            query: 查询向量
            top_k: 返回数量
            use_associations: 是否使用联想
            
        Returns:
            匹配的记忆列表
        """
        if len(query) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[:len(query)] = query
            query = padded
        elif len(query) > self.input_dim:
            query = query[:self.input_dim]
        
        query_encoding = self._entorhinal_cortex(query)
        query_separated = self._pattern_separation(query_encoding)
        
        similarities = []
        for memory_id, trace in self._memory_traces.items():
            similarity = self._compute_similarity(query_separated, trace.pattern)
            
            recency_factor = 1.0 / (1.0 + time.time() - trace.timestamp)
            strength_factor = trace.strength
            emotional_factor = 1.0 + trace.emotional_weight
            access_factor = 1.0 + 0.1 * np.log1p(trace.access_count)
            
            combined_score = (
                similarity * 
                recency_factor * 
                strength_factor * 
                emotional_factor * 
                access_factor
            )
            
            similarities.append((memory_id, combined_score, trace))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for memory_id, score, trace in similarities[:top_k]:
            trace.access_count += 1
            
            result = {
                'id': memory_id,
                'pattern': trace.pattern.copy(),
                'strength': trace.strength,
                'emotional_weight': trace.emotional_weight,
                'timestamp': trace.timestamp,
                'access_count': trace.access_count,
                'score': score
            }
            
            if use_associations and trace.associations:
                result['associations'] = self._get_associated_memories(trace.associations)
            
            results.append(result)
        
        return results
    
    def update(
        self, 
        gradient: np.ndarray,
        dopamine_signal: float = 0.0
    ) -> np.ndarray:
        """
        更新权重（学习）
        
        Args:
            gradient: 梯度
            dopamine_signal: 多巴胺信号
            
        Returns:
            更新后的输出
        """
        learning_rate = 0.01 * (1.0 + dopamine_signal)
        
        if len(gradient) < self.hidden_dim:
            padded = np.zeros(self.hidden_dim)
            padded[:len(gradient)] = gradient
            gradient = padded
        
        self._encoder_weights['w2'] += learning_rate * np.outer(
            np.mean(self._last_hidden, axis=0) if hasattr(self, '_last_hidden') else np.zeros(self.hidden_dim),
            gradient[:self.hidden_dim]
        )
        
        self._update_memory_strengths(dopamine_signal)
        
        return gradient
    
    def prune_weak_memories(self, threshold: float = 0.1):
        """清理弱记忆"""
        to_remove = []
        for memory_id, trace in self._memory_traces.items():
            if trace.strength < threshold and trace.access_count < 2:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del self._memory_traces[memory_id]
            if memory_id in self._memory_queue:
                self._memory_queue.remove(memory_id)
    
    def create_association(self, memory_id1: str, memory_id2: str):
        """创建记忆关联"""
        if memory_id1 in self._memory_traces and memory_id2 in self._memory_traces:
            if memory_id2 not in self._memory_traces[memory_id1].associations:
                self._memory_traces[memory_id1].associations.append(memory_id2)
            if memory_id1 not in self._memory_traces[memory_id2].associations:
                self._memory_traces[memory_id2].associations.append(memory_id1)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self._memory_traces:
            return {
                'memory_count': 0,
                'avg_strength': 0,
                'avg_access_count': 0
            }
        
        strengths = [t.strength for t in self._memory_traces.values()]
        access_counts = [t.access_count for t in self._memory_traces.values()]
        
        return {
            'memory_count': len(self._memory_traces),
            'avg_strength': np.mean(strengths),
            'avg_access_count': np.mean(access_counts),
            'queue_size': len(self._memory_queue)
        }
    
    def get_state(self) -> Dict:
        """获取状态"""
        return {
            'encoder_weights': self._encoder_weights,
            'decoder_weights': self._decoder_weights,
            'ca3_weights': self._ca3_weights,
            'memory_traces': {
                k: {
                    'pattern': v.pattern,
                    'timestamp': v.timestamp,
                    'strength': v.strength,
                    'emotional_weight': v.emotional_weight,
                    'access_count': v.access_count,
                    'associations': v.associations
                }
                for k, v in self._memory_traces.items()
            }
        }
    
    def set_state(self, state: Dict):
        """设置状态"""
        self._encoder_weights = state['encoder_weights']
        self._decoder_weights = state['decoder_weights']
        self._ca3_weights = state['ca3_weights']
        
        self._memory_traces = {}
        for k, v in state['memory_traces'].items():
            self._memory_traces[k] = MemoryTrace(
                id=k,
                pattern=v['pattern'],
                timestamp=v['timestamp'],
                strength=v['strength'],
                emotional_weight=v['emotional_weight'],
                access_count=v['access_count'],
                associations=v['associations']
            )
    
    def _init_encoder(self) -> Dict[str, np.ndarray]:
        """初始化编码器权重"""
        return {
            'w1': np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim),
            'b1': np.zeros(self.hidden_dim),
            'w2': np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim),
            'b2': np.zeros(self.hidden_dim)
        }
    
    def _init_decoder(self) -> Dict[str, np.ndarray]:
        """初始化解码器权重"""
        return {
            'w1': np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim),
            'b1': np.zeros(self.hidden_dim),
            'w2': np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.hidden_dim),
            'b2': np.zeros(self.input_dim)
        }
    
    def _entorhinal_cortex(self, input_data: np.ndarray) -> np.ndarray:
        """内嗅皮层处理"""
        hidden = np.maximum(0, np.dot(input_data, self._encoder_weights['w1']) + self._encoder_weights['b1'])
        output = np.dot(hidden, self._encoder_weights['w2']) + self._encoder_weights['b2']
        return np.tanh(output)
    
    def _pattern_separation(self, input_pattern: np.ndarray) -> np.ndarray:
        """模式分离（齿状回）"""
        noise = np.random.randn(*input_pattern.shape) * self._pattern_separation_scale
        separated = input_pattern + noise
        
        threshold = np.percentile(np.abs(separated), 70)
        separated[np.abs(separated) < threshold] = 0
        
        return separated / (np.linalg.norm(separated) + 1e-8)
    
    def _ca3_recurrence(self, input_pattern: np.ndarray) -> np.ndarray:
        """CA3区循环处理"""
        output = input_pattern.copy()
        
        for _ in range(3):
            recurrent = np.dot(output, self._ca3_weights)
            output = np.tanh(output + 0.5 * recurrent)
        
        return output
    
    def _ca1_processing(
        self, 
        ca3_output: np.ndarray, 
        entorhinal_input: np.ndarray
    ) -> np.ndarray:
        """CA1区处理"""
        combined = 0.7 * ca3_output + 0.3 * entorhinal_input
        
        hidden = np.maximum(0, np.dot(combined, self._decoder_weights['w1']) + self._decoder_weights['b1'])
        output = np.dot(hidden, self._decoder_weights['w2']) + self._decoder_weights['b2']
        
        return output
    
    def _store_memory_trace(
        self, 
        pattern: np.ndarray, 
        emotional_weight: float,
        input_pattern: np.ndarray
    ) -> str:
        """存储记忆痕迹"""
        self._memory_counter += 1
        memory_id = f"mem_{self._memory_counter}_{int(time.time() * 1000)}"
        
        trace = MemoryTrace(
            id=memory_id,
            pattern=pattern.copy(),
            timestamp=time.time(),
            strength=1.0,
            emotional_weight=emotional_weight,
            context=input_pattern.copy()
        )
        
        self._memory_traces[memory_id] = trace
        self._memory_queue.append(memory_id)
        self._recency_buffer.append(memory_id)
        
        self._update_ca3_weights(pattern)
        
        if len(self._memory_queue) > self.memory_capacity * 3:
            oldest_id = self._memory_queue[0]
            if oldest_id in self._memory_traces:
                self._memory_traces[oldest_id].strength *= 0.9
        
        return memory_id
    
    def _update_ca3_weights(self, pattern: np.ndarray):
        """更新CA3权重（Hebbian学习）"""
        outer = np.outer(pattern, pattern)
        self._ca3_weights += 0.01 * outer
        np.fill_diagonal(self._ca3_weights, 0)
        
        max_weight = np.max(np.abs(self._ca3_weights))
        if max_weight > 1.0:
            self._ca3_weights /= max_weight
    
    def _compute_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """计算模式相似度"""
        min_len = min(len(pattern1), len(pattern2))
        p1, p2 = pattern1[:min_len], pattern2[:min_len]
        
        dot_product = np.dot(p1, p2)
        norm_product = np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8
        
        return dot_product / norm_product
    
    def _update_memory_strengths(self, dopamine_signal: float):
        """根据多巴胺信号更新记忆强度"""
        for trace in self._memory_traces.values():
            if dopamine_signal > 0:
                trace.strength = min(2.0, trace.strength * (1.0 + 0.1 * dopamine_signal))
            else:
                trace.strength = max(0.1, trace.strength * (1.0 + 0.05 * dopamine_signal))
    
    def _get_associated_memories(self, association_ids: List[str]) -> List[Dict]:
        """获取关联记忆"""
        associated = []
        for assoc_id in association_ids:
            if assoc_id in self._memory_traces:
                trace = self._memory_traces[assoc_id]
                associated.append({
                    'id': assoc_id,
                    'strength': trace.strength,
                    'emotional_weight': trace.emotional_weight
                })
        return associated
