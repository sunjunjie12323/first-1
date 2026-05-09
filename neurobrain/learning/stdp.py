"""
STDP - 脉冲时间依赖可塑性
Spike-Timing-Dependent Plasticity
基于突触前后脉冲时间差的学习规则
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class STDPVariant(Enum):
    STANDARD = "standard"
    SYMMETRIC = "symmetric"
    ANTI_HEBBIAN = "anti_hebbian"
    TRIPLET = "triplet"
    DOPAMINE_MODULATED = "dopamine_modulated"


@dataclass
class STDPConfig:
    """STDP配置"""
    learning_rate: float = 0.01
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.1
    a_minus: float = 0.12
    w_min: float = 0.0
    w_max: float = 1.0
    variant: STDPVariant = STDPVariant.STANDARD
    eligibility_decay: float = 0.95


@dataclass
class SpikePair:
    """脉冲对"""
    pre_time: float
    post_time: float
    delta_t: float


class STDP:
    """
    脉冲时间依赖可塑性
    
    实现多种STDP变体：
    - 标准STDP
    - 对称STDP
    - 反赫布STDP
    - 三脉冲STDP
    - 多巴胺调制STDP
    """
    
    def __init__(self, config: Optional[STDPConfig] = None):
        self.config = config or STDPConfig()
        
        self._eligibility_trace = 0.0
        self._pre_trace = 0.0
        self._post_trace = 0.0
        
        self._recent_pre_spikes: deque = deque(maxlen=100)
        self._recent_post_spikes: deque = deque(maxlen=100)
        
        self._spike_pairs: List[SpikePair] = []
        self._weight_updates: List[Tuple[float, float]] = []
        
        self._dopamine_signal = 0.0
        self._update_count = 0
        
    def process_pre_spike(
        self, 
        spike_time: float,
        post_spike_times: Optional[List[float]] = None
    ) -> float:
        """
        处理突触前脉冲
        
        Args:
            spike_time: 脉冲时间
            post_spike_times: 突触后脉冲时间列表
            
        Returns:
            权重变化
        """
        self._recent_pre_spikes.append(spike_time)
        self._pre_trace = 1.0
        
        delta_w = 0.0
        
        if post_spike_times:
            for post_time in post_spike_times:
                delta_t = post_time - spike_time
                
                if delta_t < 0:
                    self._spike_pairs.append(SpikePair(
                        pre_time=spike_time,
                        post_time=post_time,
                        delta_t=delta_t
                    ))
                    
                    delta_w += self._compute_ltd(delta_t)
        
        return delta_w
    
    def process_post_spike(
        self, 
        spike_time: float,
        pre_spike_times: Optional[List[float]] = None
    ) -> float:
        """
        处理突触后脉冲
        
        Args:
            spike_time: 脉冲时间
            pre_spike_times: 突触前脉冲时间列表
            
        Returns:
            权重变化
        """
        self._recent_post_spikes.append(spike_time)
        self._post_trace = 1.0
        
        delta_w = 0.0
        
        if pre_spike_times:
            for pre_time in pre_spike_times:
                delta_t = spike_time - pre_time
                
                if delta_t > 0:
                    self._spike_pairs.append(SpikePair(
                        pre_time=pre_time,
                        post_time=spike_time,
                        delta_t=delta_t
                    ))
                    
                    delta_w += self._compute_ltp(delta_t)
        
        return delta_w
    
    def compute_weight_update(
        self,
        current_weight: float,
        pre_spiked: bool = False,
        post_spiked: bool = False,
        dopamine: float = 0.0
    ) -> float:
        """
        计算权重更新
        
        Args:
            current_weight: 当前权重
            pre_spiked: 突触前是否发放
            post_spiked: 突触后是否发放
            dopamine: 多巴胺信号
            
        Returns:
            权重变化
        """
        self._update_count += 1
        current_time = time.time()
        
        self._pre_trace *= self.config.eligibility_decay
        self._post_trace *= self.config.eligibility_decay
        
        if pre_spiked:
            self._pre_trace = 1.0
        if post_spiked:
            self._post_trace = 1.0
        
        delta_w = 0.0
        
        if self.config.variant == STDPVariant.STANDARD:
            if pre_spiked and self._post_trace > 0:
                delta_w = -self.config.a_minus * self._post_trace
            if post_spiked and self._pre_trace > 0:
                delta_w += self.config.a_plus * self._pre_trace
        
        elif self.config.variant == STDPVariant.SYMMETRIC:
            if pre_spiked or post_spiked:
                delta_w = self.config.learning_rate * (self._pre_trace + self._post_trace)
        
        elif self.config.variant == STDPVariant.ANTI_HEBBIAN:
            if pre_spiked and self._post_trace > 0:
                delta_w = self.config.a_plus * self._post_trace
            if post_spiked and self._pre_trace > 0:
                delta_w -= self.config.a_minus * self._pre_trace
        
        elif self.config.variant == STDPVariant.TRIPLET:
            delta_w = self._compute_triplet_update(pre_spiked, post_spiked)
        
        elif self.config.variant == STDPVariant.DOPAMINE_MODULATED:
            self._dopamine_signal = dopamine
            self._eligibility_trace = self._pre_trace * self._post_trace
            delta_w = self.config.learning_rate * self._eligibility_trace * dopamine
        
        delta_w *= self.config.learning_rate
        
        self._weight_updates.append((current_time, delta_w))
        if len(self._weight_updates) > 1000:
            self._weight_updates = self._weight_updates[-500:]
        
        return delta_w
    
    def apply_update(
        self, 
        current_weight: float,
        delta_w: float
    ) -> float:
        """
        应用权重更新
        
        Args:
            current_weight: 当前权重
            delta_w: 权重变化
            
        Returns:
            新权重
        """
        new_weight = current_weight + delta_w
        return np.clip(new_weight, self.config.w_min, self.config.w_max)
    
    def get_eligibility_trace(self) -> float:
        """获取资格迹"""
        return self._eligibility_trace
    
    def get_recent_activity(self) -> Dict:
        """获取最近活动"""
        return {
            'pre_trace': self._pre_trace,
            'post_trace': self._post_trace,
            'eligibility': self._eligibility_trace,
            'recent_pre_spikes': len(self._recent_pre_spikes),
            'recent_post_spikes': len(self._recent_post_spikes)
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'update_count': self._update_count,
            'variant': self.config.variant.value,
            'learning_rate': self.config.learning_rate,
            'tau_plus': self.config.tau_plus,
            'tau_minus': self.config.tau_minus,
            'spike_pairs_recorded': len(self._spike_pairs)
        }
    
    def reset(self):
        """重置状态"""
        self._eligibility_trace = 0.0
        self._pre_trace = 0.0
        self._post_trace = 0.0
        self._recent_pre_spikes.clear()
        self._recent_post_spikes.clear()
        self._spike_pairs.clear()
        self._weight_updates.clear()
        self._dopamine_signal = 0.0
        self._update_count = 0
    
    def _compute_ltp(self, delta_t: float) -> float:
        """计算长时程增强"""
        return self.config.a_plus * np.exp(-delta_t / self.config.tau_plus)
    
    def _compute_ltd(self, delta_t: float) -> float:
        """计算长时程抑制"""
        return -self.config.a_minus * np.exp(delta_t / self.config.tau_minus)
    
    def _compute_triplet_update(
        self, 
        pre_spiked: bool,
        post_spiked: bool
    ) -> float:
        """计算三脉冲STDP更新"""
        delta_w = 0.0
        
        if len(self._recent_pre_spikes) >= 2 and post_spiked:
            t1 = self._recent_pre_spikes[-1]
            t2 = self._recent_pre_spikes[-2]
            r1 = np.exp(-(time.time() - t1) / self.config.tau_plus)
            r2 = np.exp(-(time.time() - t2) / self.config.tau_plus)
            delta_w += self.config.a_plus * r1 * r2
        
        if len(self._recent_post_spikes) >= 2 and pre_spiked:
            t1 = self._recent_post_spikes[-1]
            t2 = self._recent_post_spikes[-2]
            o1 = np.exp(-(time.time() - t1) / self.config.tau_minus)
            o2 = np.exp(-(time.time() - t2) / self.config.tau_minus)
            delta_w -= self.config.a_minus * o1 * o2
        
        return delta_w
