"""
Synapse - 突触模型
模拟生物突触的可塑性和传递特性
实现STDP、LTP/LTD等学习规则
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class SynapseType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


class PlasticityRule(Enum):
    STDP = "stdp"
    HEBBIAN = "hebbian"
    ANTI_HEBBIAN = "anti_hebbian"
    HOMEOSTATIC = "homeostatic"


@dataclass
class SynapseState:
    """突触状态"""
    weight: float = 0.5
    efficacy: float = 1.0
    last_presynaptic_spike: float = 0.0
    last_postsynaptic_spike: float = 0.0
    eligibility_trace: float = 0.0
    calcium_trace: float = 0.0


class SynapticPlasticity:
    """
    突触可塑性管理器
    实现多种学习规则
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.1,
        a_minus: float = 0.12,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        
    def compute_stdp(
        self, 
        delta_t: float,
        current_weight: float
    ) -> float:
        """
        计算STDP权重变化
        
        Args:
            delta_t: t_post - t_pre
            current_weight: 当前权重
            
        Returns:
            权重变化
        """
        if delta_t > 0:
            delta_w = self.a_plus * np.exp(-delta_t / self.tau_plus)
        else:
            delta_w = -self.a_minus * np.exp(delta_t / self.tau_minus)
        
        return delta_w * self.learning_rate
    
    def compute_hebbian(
        self, 
        pre_activity: float,
        post_activity: float,
        current_weight: float
    ) -> float:
        """
        计算Hebbian权重变化
        
        Args:
            pre_activity: 突触前活动
            post_activity: 突触后活动
            current_weight: 当前权重
            
        Returns:
            权重变化
        """
        delta_w = pre_activity * post_activity
        
        delta_w -= 0.01 * current_weight ** 2
        
        return delta_w * self.learning_rate
    
    def compute_homeostatic(
        self, 
        current_weight: float,
        target_weight: float = 0.5
    ) -> float:
        """
        计算稳态权重变化
        
        Args:
            current_weight: 当前权重
            target_weight: 目标权重
            
        Returns:
            权重变化
        """
        return (target_weight - current_weight) * self.learning_rate * 0.1
    
    def clip_weight(self, weight: float) -> float:
        """限制权重范围"""
        return np.clip(weight, self.w_min, self.w_max)


class Synapse:
    """
    突触模型
    
    实现突触传递和可塑性
    """
    
    def __init__(
        self,
        synapse_id: str,
        pre_neuron_id: str,
        post_neuron_id: str,
        initial_weight: float = 0.5,
        delay: float = 1.0,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        plasticity_rule: PlasticityRule = PlasticityRule.STDP
    ):
        self.synapse_id = synapse_id
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.delay = delay
        self.synapse_type = synapse_type
        self.plasticity_rule = plasticity_rule
        
        self.state = SynapseState(weight=initial_weight)
        self.plasticity = SynapticPlasticity()
        
        self._pending_spikes: List[Tuple[float, float]] = []
        self._update_history: List[Tuple[float, float]] = []
        
    def transmit(self, spike_time: float) -> Optional[float]:
        """
        传递信号
        
        Args:
            spike_time: 突触前发放时间
            
        Returns:
            传递的电流或None
        """
        self.state.last_presynaptic_spike = spike_time
        
        arrival_time = spike_time + self.delay
        current = self.state.weight * self.state.efficacy
        
        if self.synapse_type == SynapseType.INHIBITORY:
            current = -current
        
        self._pending_spikes.append((arrival_time, current))
        
        return current
    
    def receive(self, current_time: float) -> Optional[float]:
        """
        接收传递的信号
        
        Args:
            current_time: 当前时间
            
        Returns:
            到达的电流或None
        """
        total_current = 0.0
        remaining_spikes = []
        
        for arrival_time, current in self._pending_spikes:
            if arrival_time <= current_time:
                total_current += current
            else:
                remaining_spikes.append((arrival_time, current))
        
        self._pending_spikes = remaining_spikes
        
        return total_current if total_current != 0 else None
    
    def update_plasticity(
        self, 
        pre_spike: bool = False,
        post_spike: bool = False,
        pre_activity: float = 0.0,
        post_activity: float = 0.0,
        dopamine: float = 0.0
    ):
        """
        更新突触可塑性
        
        Args:
            pre_spike: 突触前是否发放
            post_spike: 突触后是否发放
            pre_activity: 突触前活动水平
            post_activity: 突触后活动水平
            dopamine: 多巴胺信号
        """
        current_time = time.time()
        
        if post_spike:
            self.state.last_postsynaptic_spike = current_time
        
        delta_w = 0.0
        
        if self.plasticity_rule == PlasticityRule.STDP:
            if pre_spike and post_spike:
                delta_t = self.state.last_postsynaptic_spike - self.state.last_presynaptic_spike
                delta_w = self.plasticity.compute_stdp(delta_t, self.state.weight)
        
        elif self.plasticity_rule == PlasticityRule.HEBBIAN:
            delta_w = self.plasticity.compute_hebbian(
                pre_activity, post_activity, self.state.weight
            )
        
        elif self.plasticity_rule == PlasticityRule.ANTI_HEBBIAN:
            delta_w = -self.plasticity.compute_hebbian(
                pre_activity, post_activity, self.state.weight
            )
        
        elif self.plasticity_rule == PlasticityRule.HOMEOSTATIC:
            delta_w = self.plasticity.compute_homeostatic(self.state.weight)
        
        delta_w *= (1.0 + dopamine * 0.5)
        
        self.state.weight = self.plasticity.clip_weight(self.state.weight + delta_w)
        
        self._update_history.append((current_time, delta_w))
        
        if len(self._update_history) > 1000:
            self._update_history = self._update_history[-500:]
    
    def get_weight(self) -> float:
        """获取当前权重"""
        return self.state.weight
    
    def set_weight(self, weight: float):
        """设置权重"""
        self.state.weight = self.plasticity.clip_weight(weight)
    
    def depress(self, factor: float = 0.9):
        """短时程抑制"""
        self.state.efficacy *= factor
    
    def facilitate(self, factor: float = 1.1):
        """短时程增强"""
        self.state.efficacy = min(2.0, self.state.efficacy * factor)
    
    def get_state_dict(self) -> Dict:
        """获取状态字典"""
        return {
            'synapse_id': self.synapse_id,
            'pre_neuron': self.pre_neuron_id,
            'post_neuron': self.post_neuron_id,
            'type': self.synapse_type.value,
            'weight': self.state.weight,
            'efficacy': self.state.efficacy,
            'plasticity_rule': self.plasticity_rule.value
        }
    
    def reset(self):
        """重置突触"""
        self.state = SynapseState(weight=self.state.weight)
        self._pending_spikes.clear()
