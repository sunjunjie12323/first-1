"""
Neuron - 类脑神经元模型
模拟生物神经元的电生理特性
支持多种神经元类型和发放模式
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class NeuronType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


@dataclass
class NeuronState:
    """神经元状态"""
    membrane_potential: float = -70.0
    refractory_time: float = 0.0
    calcium_concentration: float = 0.0
    adaptation_current: float = 0.0
    last_spike_time: float = 0.0


class Neuron:
    """
    类脑神经元
    
    实现LIF (Leaky Integrate-and-Fire) 模型
    支持自适应和多种发放模式
    """
    
    def __init__(
        self,
        neuron_id: str,
        neuron_type: NeuronType = NeuronType.EXCITATORY,
        resting_potential: float = -70.0,
        threshold: float = -55.0,
        reset_potential: float = -75.0,
        membrane_resistance: float = 10.0,
        membrane_capacitance: float = 1.0,
        refractory_period: float = 2.0,
        adaptation_strength: float = 0.1
    ):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.resting_potential = resting_potential
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.membrane_resistance = membrane_resistance
        self.membrane_capacitance = membrane_capacitance
        self.refractory_period = refractory_period
        self.adaptation_strength = adaptation_strength
        
        self.state = NeuronState()
        self._spike_history: List[float] = []
        self._input_current = 0.0
        self._tau_m = membrane_resistance * membrane_capacitance
        
    def inject_current(self, current: float):
        """注入电流"""
        self._input_current += current
    
    def update(self, dt: float = 1.0) -> bool:
        """
        更新神经元状态
        
        Args:
            dt: 时间步长
            
        Returns:
            是否发放
        """
        current_time = time.time()
        
        if self.state.refractory_time > 0:
            self.state.refractory_time -= dt
            self._input_current = 0.0
            return False
        
        leak_current = (self.state.membrane_potential - self.resting_potential) / self.membrane_resistance
        
        adaptation = self.adaptation_strength * self.state.adaptation_current
        
        dV = (self._input_current - leak_current - adaptation) * dt / self._tau_m
        self.state.membrane_potential += dV
        
        self.state.adaptation_current *= 0.9
        
        self.state.calcium_concentration *= 0.95
        
        spiked = False
        if self.state.membrane_potential >= self.threshold:
            spiked = True
            self._spike_history.append(current_time)
            self.state.membrane_potential = self.reset_potential
            self.state.refractory_time = self.refractory_period
            self.state.last_spike_time = current_time
            self.state.calcium_concentration += 1.0
            self.state.adaptation_current += 1.0
        
        self._input_current = 0.0
        
        return spiked
    
    def get_firing_rate(self, window: float = 1000.0) -> float:
        """获取发放率"""
        current_time = time.time()
        recent_spikes = [t for t in self._spike_history if current_time - t < window / 1000.0]
        return len(recent_spikes) / (window / 1000.0)
    
    def get_state_dict(self) -> Dict:
        """获取状态字典"""
        return {
            'neuron_id': self.neuron_id,
            'type': self.neuron_type.value,
            'membrane_potential': self.state.membrane_potential,
            'threshold': self.threshold,
            'refractory_time': self.state.refractory_time,
            'calcium': self.state.calcium_concentration,
            'firing_rate': self.get_firing_rate(),
            'spike_count': len(self._spike_history)
        }
    
    def reset(self):
        """重置神经元"""
        self.state = NeuronState()
        self._input_current = 0.0


class NeuronCluster:
    """
    神经元集群
    模拟皮层柱或核团
    """
    
    def __init__(
        self,
        cluster_id: str,
        size: int,
        neuron_type: NeuronType = NeuronType.EXCITATORY,
        connectivity: float = 0.1
    ):
        self.cluster_id = cluster_id
        self.size = size
        self.neuron_type = neuron_type
        self.connectivity = connectivity
        
        self.neurons = [
            Neuron(
                neuron_id=f"{cluster_id}_n{i}",
                neuron_type=neuron_type
            )
            for i in range(size)
        ]
        
        self._internal_weights = self._init_internal_connections()
        self._external_inputs: Dict[str, np.ndarray] = {}
        
    def process(self, external_input: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        处理输入并产生输出
        
        Args:
            external_input: 外部输入
            dt: 时间步长
            
        Returns:
            输出发放模式
        """
        if len(external_input) != self.size:
            if len(external_input) < self.size:
                padded = np.zeros(self.size)
                padded[:len(external_input)] = external_input
                external_input = padded
            else:
                external_input = external_input[:self.size]
        
        for i, neuron in enumerate(self.neurons):
            neuron.inject_current(external_input[i])
        
        spikes = np.zeros(self.size)
        for i, neuron in enumerate(self.neurons):
            if neuron.update(dt):
                spikes[i] = 1.0
        
        recurrent_input = np.dot(spikes, self._internal_weights)
        for i, neuron in enumerate(self.neurons):
            if recurrent_input[i] != 0:
                neuron.inject_current(recurrent_input[i] * 0.1)
        
        return spikes
    
    def get_activity(self) -> np.ndarray:
        """获取当前活动"""
        return np.array([
            neuron.state.membrane_potential 
            for neuron in self.neurons
        ])
    
    def get_firing_rates(self) -> np.ndarray:
        """获取所有神经元的发放率"""
        return np.array([
            neuron.get_firing_rate() 
            for neuron in self.neurons
        ])
    
    def get_mean_firing_rate(self) -> float:
        """获取平均发放率"""
        return np.mean(self.get_firing_rates())
    
    def reset(self):
        """重置所有神经元"""
        for neuron in self.neurons:
            neuron.reset()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'cluster_id': self.cluster_id,
            'size': self.size,
            'type': self.neuron_type.value,
            'mean_firing_rate': self.get_mean_firing_rate(),
            'mean_membrane_potential': np.mean(self.get_activity()),
            'total_spikes': sum(len(n._spike_history) for n in self.neurons)
        }
    
    def _init_internal_connections(self) -> np.ndarray:
        """初始化内部连接"""
        weights = np.random.randn(self.size, self.size) * 0.1
        
        if self.neuron_type == NeuronType.INHIBITORY:
            weights = -np.abs(weights)
        else:
            np.fill_diagonal(weights, 0)
        
        mask = np.random.random((self.size, self.size)) < self.connectivity
        weights = weights * mask
        
        return weights
