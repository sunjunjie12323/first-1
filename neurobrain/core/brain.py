"""
Brain - 类脑主控制器
整合海马体、新皮层、杏仁核等核心组件
实现类似人脑的信息处理和记忆管理
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

from .hippocampus import Hippocampus
from .neocortex import Neocortex
from .amygdala import Amygdala


class BrainState(Enum):
    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    REM_SLEEP = "rem_sleep"
    FOCUSED = "focused"
    RELAXED = "relaxed"


@dataclass
class BrainConfig:
    input_dim: int = 784
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    output_dim: int = 10
    working_memory_capacity: int = 7
    short_term_memory_duration: float = 30.0
    long_term_memory_threshold: float = 0.7
    learning_rate: float = 0.01
    consolidation_rate: float = 0.1
    emotional_weight: float = 0.3
    plasticity_threshold: float = 0.5


class Brain:
    """
    类脑主控制器
    
    模拟人脑的主要功能区域：
    - 海马体：负责记忆编码和巩固
    - 新皮层：负责长期记忆存储和高级认知
    - 杏仁核：负责情感标记和记忆增强
    """
    
    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()
        
        self.hippocampus = Hippocampus(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dims[0],
            memory_capacity=self.config.working_memory_capacity,
            consolidation_threshold=self.config.long_term_memory_threshold
        )
        
        self.neocortex = Neocortex(
            input_dim=self.config.hidden_dims[0],
            hidden_dims=self.config.hidden_dims[1:],
            output_dim=self.config.output_dim,
            plasticity_threshold=self.config.plasticity_threshold
        )
        
        self.amygdala = Amygdala(
            emotional_weight=self.config.emotional_weight
        )
        
        self.state = BrainState.AWAKE
        self.attention_level = 1.0
        self.global_time = 0.0
        
        self._experience_buffer = deque(maxlen=1000)
        self._consolidation_queue = deque()
        self._memory_index: Dict[str, Any] = {}
        
        self._attention_weights = np.ones(self.config.input_dim)
        self._global_context = np.zeros(self.config.hidden_dims[0])
        
    def process(self, input_data: np.ndarray, context: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        处理输入信息，模拟人脑的信息处理流程
        
        Args:
            input_data: 输入数据
            context: 可选的上下文信息
            
        Returns:
            输出结果和处理信息
        """
        self.global_time += 1
        
        attention_input = self._apply_attention(input_data)
        
        emotional_response = self.amygdala.process(attention_input)
        
        hippocampal_output, memory_info = self.hippocampus.encode(
            attention_input, 
            emotional_weight=emotional_response['weight']
        )
        
        cortical_output, prediction = self.neocortex.process(
            hippocampal_output,
            context=self._global_context
        )
        
        self._update_global_context(hippocampal_output)
        
        if memory_info.get('should_consolidate', False):
            self._consolidation_queue.append({
                'data': hippocampal_output,
                'emotional_weight': emotional_response['weight'],
                'timestamp': self.global_time
            })
        
        experience = {
            'input': input_data.copy(),
            'attention_input': attention_input.copy(),
            'hippocampal_output': hippocampal_output.copy(),
            'cortical_output': cortical_output.copy(),
            'emotional_response': emotional_response,
            'prediction': prediction,
            'timestamp': self.global_time
        }
        self._experience_buffer.append(experience)
        
        info = {
            'emotional_response': emotional_response,
            'memory_info': memory_info,
            'prediction': prediction,
            'attention_level': self.attention_level,
            'brain_state': self.state.value
        }
        
        return cortical_output, info
    
    def recall(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        记忆检索，模拟人脑的回忆过程
        
        Args:
            query: 查询向量
            top_k: 返回的记忆数量
            
        Returns:
            相关记忆列表
        """
        attention_query = self._apply_attention(query)
        
        hippocampal_memories = self.hippocampus.recall(attention_query, top_k=top_k)
        
        cortical_memories = self.neocortex.retrieve(attention_query, top_k=top_k)
        
        all_memories = []
        
        for mem in hippocampal_memories:
            mem['source'] = 'hippocampus'
            mem['recency_weight'] = 1.0 / (1.0 + self.global_time - mem.get('timestamp', 0))
            all_memories.append(mem)
            
        for mem in cortical_memories:
            mem['source'] = 'neocortex'
            mem['recency_weight'] = 0.8
            all_memories.append(mem)
        
        all_memories.sort(key=lambda x: x.get('strength', 0) * x.get('recency_weight', 1), reverse=True)
        
        return all_memories[:top_k]
    
    def consolidate(self, sleep_cycles: int = 3) -> Dict:
        """
        记忆巩固，模拟睡眠时的记忆整合过程
        
        Args:
            sleep_cycles: 睡眠周期数
            
        Returns:
            巩固统计信息
        """
        original_state = self.state
        self.state = BrainState.DEEP_SLEEP
        
        stats = {
            'memories_consolidated': 0,
            'hippocampus_to_neocortex': 0,
            'memory_strength_updated': 0,
            'synaptic_changes': 0
        }
        
        for cycle in range(sleep_cycles):
            self.state = BrainState.DEEP_SLEEP if cycle % 2 == 0 else BrainState.REM_SLEEP
            
            while self._consolidation_queue:
                memory_data = self._consolidation_queue.popleft()
                
                success = self.neocortex.store(
                    memory_data['data'],
                    emotional_weight=memory_data['emotional_weight']
                )
                
                if success:
                    stats['memories_consolidated'] += 1
                    stats['hippocampus_to_neocortex'] += 1
            
            replay_stats = self._sleep_replay()
            stats['memory_strength_updated'] += replay_stats['updated']
            stats['synaptic_changes'] += replay_stats['changes']
        
        self.hippocampus.prune_weak_memories()
        
        self.state = original_state
        
        return stats
    
    def learn(self, input_data: np.ndarray, target: np.ndarray, 
              reward: float = 0.0) -> Dict:
        """
        学习过程，结合强化学习和突触可塑性
        
        Args:
            input_data: 输入数据
            target: 目标输出
            reward: 奖励信号
            
        Returns:
            学习统计信息
        """
        output, info = self.process(input_data)
        
        min_len = min(len(target), len(output))
        error = np.zeros_like(output)
        error[:min_len] = target[:min_len] - output[:min_len]
        loss = np.mean(error ** 2)
        
        dopamine_signal = self._compute_dopamine(reward, loss)
        
        self.amygdala.update_emotional_state(reward)
        
        cortical_grad = self.neocortex.backpropagate(error)
        
        hippocampal_grad = self.hippocampus.update(
            cortical_grad,
            dopamine_signal=dopamine_signal
        )
        
        self._update_attention(input_data, error)
        
        return {
            'loss': loss,
            'output': output,
            'dopamine': dopamine_signal,
            'emotional_state': info['emotional_response']
        }
    
    def set_attention(self, attention_mask: np.ndarray):
        """设置注意力权重"""
        self._attention_weights = attention_mask / (np.sum(attention_mask) + 1e-8)
        
    def set_state(self, state: BrainState):
        """设置大脑状态"""
        self.state = state
        self.attention_level = 1.0 if state == BrainState.FOCUSED else 0.5
        
    def get_memory_stats(self) -> Dict:
        """获取记忆系统统计信息"""
        return {
            'hippocampus': self.hippocampus.get_stats(),
            'neocortex': self.neocortex.get_stats(),
            'amygdala': self.amygdala.get_stats(),
            'experience_buffer_size': len(self._experience_buffer),
            'consolidation_queue_size': len(self._consolidation_queue),
            'global_time': self.global_time
        }
    
    def _apply_attention(self, input_data: np.ndarray) -> np.ndarray:
        """应用注意力机制"""
        min_len = min(len(self._attention_weights), len(input_data))
        return input_data[:min_len] * self._attention_weights[:min_len] * self.attention_level
    
    def _update_global_context(self, hippocampal_output: np.ndarray):
        """更新全局上下文"""
        alpha = 0.1
        min_len = min(len(self._global_context), len(hippocampal_output))
        self._global_context[:min_len] = (
            (1 - alpha) * self._global_context[:min_len] + 
            alpha * hippocampal_output[:min_len]
        )
    
    def _compute_dopamine(self, reward: float, loss: float) -> float:
        """计算多巴胺信号（奖励预测误差）"""
        prediction_error = reward - loss
        return np.tanh(prediction_error)
    
    def _update_attention(self, input_data: np.ndarray, error: np.ndarray):
        """根据错误更新注意力"""
        error_influence = np.abs(error).mean()
        input_importance = np.abs(input_data)
        
        attention_update = input_importance * error_influence
        min_len = min(len(self._attention_weights), len(attention_update))
        self._attention_weights[:min_len] = (
            0.9 * self._attention_weights[:min_len] + 
            0.1 * attention_update[:min_len]
        )
        self._attention_weights = self._attention_weights / (np.sum(self._attention_weights) + 1e-8)
    
    def _sleep_replay(self) -> Dict:
        """睡眠时的记忆重放"""
        stats = {'updated': 0, 'changes': 0}
        
        if len(self._experience_buffer) < 10:
            return stats
        
        replay_indices = np.random.choice(
            len(self._experience_buffer),
            size=min(50, len(self._experience_buffer)),
            replace=False
        )
        
        for idx in replay_indices:
            experience = self._experience_buffer[idx]
            
            emotional_weight = experience['emotional_response'].get('weight', 0.5)
            if emotional_weight > 0.6:
                self.neocortex.strengthen_memory(
                    experience['hippocampal_output'],
                    factor=1.0 + emotional_weight * 0.1
                )
                stats['updated'] += 1
        
        stats['changes'] = self.neocortex.apply_synaptic_plasticity()
        
        return stats
    
    def save_state(self, filepath: str):
        """保存大脑状态"""
        import pickle
        state = {
            'config': self.config,
            'hippocampus': self.hippocampus.get_state(),
            'neocortex': self.neocortex.get_state(),
            'amygdala': self.amygdala.get_state(),
            'attention_weights': self._attention_weights,
            'global_context': self._global_context,
            'global_time': self.global_time
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """加载大脑状态"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.hippocampus.set_state(state['hippocampus'])
        self.neocortex.set_state(state['neocortex'])
        self.amygdala.set_state(state['amygdala'])
        self._attention_weights = state['attention_weights']
        self._global_context = state['global_context']
        self.global_time = state['global_time']
