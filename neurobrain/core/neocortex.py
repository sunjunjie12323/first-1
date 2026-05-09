"""
Neocortex - 新皮层模块
负责长期记忆存储、高级认知功能和模式识别
模拟人脑新皮层的层次化处理结构
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CorticalColumn:
    """皮层柱"""
    weights: np.ndarray
    bias: np.ndarray
    activation: np.ndarray
    plasticity_trace: np.ndarray
    

@dataclass 
class LongTermMemory:
    """长期记忆"""
    id: str
    pattern: np.ndarray
    category: str
    strength: float = 1.0
    emotional_weight: float = 0.5
    creation_time: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    consolidation_level: int = 0


class Neocortex:
    """
    新皮层模块
    
    功能：
    1. 长期记忆存储：持久化存储重要记忆
    2. 层次化处理：多层抽象和特征提取
    3. 模式识别：识别和分类输入模式
    4. 预测生成：基于经验预测未来
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 10,
        plasticity_threshold: float = 0.5
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.plasticity_threshold = plasticity_threshold
        
        self._layers = self._init_layers()
        self._prediction_weights = self._init_prediction_weights()
        
        self._long_term_memories: Dict[str, LongTermMemory] = {}
        self._category_index: Dict[str, List[str]] = defaultdict(list)
        self._memory_counter = 0
        
        self._synaptic_traces: Dict[str, np.ndarray] = {}
        self._global_inhibition = 0.1
        
        self._prediction_context = np.zeros(output_dim)
        
    def process(
        self, 
        input_data: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            输出和预测信息
        """
        if len(input_data) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[:len(input_data)] = input_data
            input_data = padded
        elif len(input_data) > self.input_dim:
            input_data = input_data[:self.input_dim]
        
        current_activation = input_data
        layer_activations = [current_activation]
        
        for i, layer in enumerate(self._layers):
            current_activation = self._process_layer(layer, current_activation, i)
            layer_activations.append(current_activation)
        
        prediction = self._generate_prediction(current_activation, context)
        
        category = self._classify_pattern(current_activation)
        
        info = {
            'layer_activations': [a.copy() for a in layer_activations],
            'prediction': prediction.copy(),
            'category': category,
            'activation_strength': np.mean(np.abs(current_activation))
        }
        
        return current_activation, info
    
    def store(
        self, 
        pattern: np.ndarray,
        emotional_weight: float = 0.5,
        category: Optional[str] = None
    ) -> bool:
        """
        存储长期记忆
        
        Args:
            pattern: 记忆模式
            emotional_weight: 情感权重
            category: 类别标签
            
        Returns:
            是否成功存储
        """
        self._memory_counter += 1
        memory_id = f"ltm_{self._memory_counter}"
        
        if category is None:
            category = self._classify_pattern(pattern)
        
        memory = LongTermMemory(
            id=memory_id,
            pattern=pattern.copy(),
            category=category,
            strength=1.0,
            emotional_weight=emotional_weight,
            creation_time=self._memory_counter,
            last_accessed=self._memory_counter
        )
        
        self._long_term_memories[memory_id] = memory
        self._category_index[category].append(memory_id)
        
        self._update_synaptic_traces(memory_id, pattern)
        
        return True
    
    def retrieve(
        self, 
        query: np.ndarray,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        检索长期记忆
        
        Args:
            query: 查询向量
            top_k: 返回数量
            category_filter: 类别过滤
            
        Returns:
            匹配的记忆列表
        """
        candidates = []
        
        if category_filter:
            memory_ids = self._category_index.get(category_filter, [])
        else:
            memory_ids = list(self._long_term_memories.keys())
        
        for memory_id in memory_ids:
            memory = self._long_term_memories[memory_id]
            
            similarity = self._compute_similarity(query, memory.pattern)
            
            recency_factor = 1.0 / (1.0 + self._memory_counter - memory.last_accessed)
            strength_factor = memory.strength
            emotional_factor = 1.0 + memory.emotional_weight * 0.5
            consolidation_factor = 1.0 + 0.1 * memory.consolidation_level
            
            score = similarity * recency_factor * strength_factor * emotional_factor * consolidation_factor
            
            candidates.append({
                'id': memory_id,
                'pattern': memory.pattern,
                'category': memory.category,
                'strength': memory.strength,
                'emotional_weight': memory.emotional_weight,
                'score': score
            })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        for candidate in candidates[:top_k]:
            memory = self._long_term_memories[candidate['id']]
            memory.last_accessed = self._memory_counter
            memory.access_count += 1
        
        return candidates[:top_k]
    
    def backpropagate(self, error: np.ndarray) -> np.ndarray:
        """
        反向传播误差
        
        Args:
            error: 误差信号
            
        Returns:
            输入层梯度
        """
        current_error = error
        if len(current_error) < self.hidden_dims[-1]:
            padded = np.zeros(self.hidden_dims[-1])
            padded[:len(current_error)] = current_error
            current_error = padded
        
        layer_errors = [current_error]
        
        for i in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[i]
            
            gradient = current_error * self._relu_derivative(layer.activation)
            
            if i > 0:
                prev_activation = self._layers[i-1].activation
            else:
                prev_activation = np.zeros(self.input_dim)
            
            min_len = min(layer.weights.shape[1], len(gradient))
            weight_update = np.outer(prev_activation[:layer.weights.shape[0]], gradient[:min_len])
            layer.weights[:, :min_len] += 0.01 * weight_update[:, :min_len] if weight_update.shape[1] >= min_len else weight_update
            layer.bias[:min_len] += 0.01 * gradient[:min_len]
            
            if i > 0:
                min_len = min(layer.weights.shape[0], len(current_error))
                current_error = np.dot(layer.weights.T[:min_len, :min_len], current_error[:min_len])
                layer_errors.insert(0, current_error)
        
        return layer_errors[0] if layer_errors else np.zeros(self.input_dim)
    
    def strengthen_memory(self, pattern: np.ndarray, factor: float = 1.1):
        """增强相关记忆"""
        for memory in self._long_term_memories.values():
            similarity = self._compute_similarity(pattern, memory.pattern)
            if similarity > 0.7:
                memory.strength = min(2.0, memory.strength * factor)
                memory.consolidation_level += 1
    
    def apply_synaptic_plasticity(self) -> int:
        """应用突触可塑性规则"""
        changes = 0
        
        for layer in self._layers:
            active_neurons = np.abs(layer.activation) > self.plasticity_threshold
            
            layer.plasticity_trace[active_neurons] += 0.1
            layer.plasticity_trace[~active_neurons] *= 0.95
            
            potentiated = layer.plasticity_trace > 1.0
            if np.any(potentiated):
                layer.weights[:, potentiated] *= 1.05
                layer.plasticity_trace[potentiated] = 0.5
                changes += np.sum(potentiated)
            
            depressed = layer.plasticity_trace < 0.1
            if np.any(depressed):
                layer.weights[:, depressed] *= 0.95
                changes += np.sum(depressed)
        
        return changes
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self._long_term_memories:
            return {
                'memory_count': 0,
                'categories': 0,
                'avg_strength': 0
            }
        
        strengths = [m.strength for m in self._long_term_memories.values()]
        
        return {
            'memory_count': len(self._long_term_memories),
            'categories': len(self._category_index),
            'avg_strength': np.mean(strengths),
            'layer_count': len(self._layers)
        }
    
    def get_state(self) -> Dict:
        """获取状态"""
        return {
            'layers': [
                {
                    'weights': layer.weights.copy(),
                    'bias': layer.bias.copy(),
                    'plasticity_trace': layer.plasticity_trace.copy()
                }
                for layer in self._layers
            ],
            'prediction_weights': {k: v.copy() for k, v in self._prediction_weights.items()},
            'long_term_memories': {
                k: {
                    'pattern': v.pattern.copy(),
                    'category': v.category,
                    'strength': v.strength,
                    'emotional_weight': v.emotional_weight,
                    'consolidation_level': v.consolidation_level
                }
                for k, v in self._long_term_memories.items()
            }
        }
    
    def set_state(self, state: Dict):
        """设置状态"""
        for i, layer_state in enumerate(state['layers']):
            self._layers[i].weights = layer_state['weights']
            self._layers[i].bias = layer_state['bias']
            self._layers[i].plasticity_trace = layer_state['plasticity_trace']
        
        self._prediction_weights = state['prediction_weights']
        
        self._long_term_memories = {}
        for k, v in state['long_term_memories'].items():
            self._long_term_memories[k] = LongTermMemory(
                id=k,
                pattern=v['pattern'],
                category=v['category'],
                strength=v['strength'],
                emotional_weight=v['emotional_weight'],
                consolidation_level=v['consolidation_level']
            )
    
    def _init_layers(self) -> List[CorticalColumn]:
        """初始化皮层柱"""
        layers = []
        dims = [self.input_dim] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            weights = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            bias = np.zeros(dims[i+1])
            activation = np.zeros(dims[i+1])
            plasticity_trace = np.zeros(dims[i+1])
            
            layers.append(CorticalColumn(
                weights=weights,
                bias=bias,
                activation=activation,
                plasticity_trace=plasticity_trace
            ))
        
        return layers
    
    def _init_prediction_weights(self) -> Dict[str, np.ndarray]:
        """初始化预测权重"""
        last_dim = self.hidden_dims[-1] if self.hidden_dims else self.input_dim
        return {
            'w': np.random.randn(last_dim, self.output_dim) * 0.1,
            'b': np.zeros(self.output_dim)
        }
    
    def _process_layer(
        self, 
        layer: CorticalColumn, 
        input_data: np.ndarray,
        layer_idx: int
    ) -> np.ndarray:
        """处理单层"""
        min_len = min(len(input_data), layer.weights.shape[0])
        
        linear = np.dot(input_data[:min_len], layer.weights[:min_len, :]) + layer.bias
        
        if layer_idx < len(self._layers) - 1:
            activation = self._relu(linear)
        else:
            activation = np.tanh(linear)
        
        activation = activation * (1 - self._global_inhibition)
        
        layer.activation = activation
        
        return activation
    
    def _generate_prediction(
        self, 
        current_state: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """生成预测"""
        min_len = min(len(current_state), self._prediction_weights['w'].shape[0])
        
        prediction = np.dot(
            current_state[:min_len], 
            self._prediction_weights['w'][:min_len, :]
        ) + self._prediction_weights['b']
        
        if context is not None and len(context) == len(prediction):
            prediction = 0.7 * prediction + 0.3 * context
        
        return prediction
    
    def _classify_pattern(self, pattern: np.ndarray) -> str:
        """分类模式"""
        if len(self._long_term_memories) == 0:
            return "category_0"
        
        best_category = "category_0"
        best_similarity = 0.0
        
        for category, memory_ids in self._category_index.items():
            if not memory_ids:
                continue
            
            for memory_id in memory_ids[:5]:
                memory = self._long_term_memories[memory_id]
                similarity = self._compute_similarity(pattern, memory.pattern)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category
        
        if best_similarity < 0.5:
            return f"category_{len(self._category_index)}"
        
        return best_category
    
    def _update_synaptic_traces(self, memory_id: str, pattern: np.ndarray):
        """更新突触痕迹"""
        self._synaptic_traces[memory_id] = pattern.copy()
    
    def _compute_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """计算相似度"""
        min_len = min(len(pattern1), len(pattern2))
        p1, p2 = pattern1[:min_len], pattern2[:min_len]
        
        dot = np.dot(p1, p2)
        norm = np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8
        
        return dot / norm
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (x > 0).astype(float)
