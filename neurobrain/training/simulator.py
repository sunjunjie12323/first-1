"""
BrainSimulator - 大脑仿真器
提供仿真训练环境，支持反复试错学习
实现具身智能的基本训练框架
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.brain import Brain, BrainConfig, BrainState
from core.hippocampus import Hippocampus
from core.neocortex import Neocortex
from core.amygdala import Amygdala


class SimulationMode(Enum):
    TRAINING = "training"
    TESTING = "testing"
    INFERENCE = "inference"
    SLEEP = "sleep"


@dataclass
class SimulationConfig:
    """仿真配置"""
    max_episodes: int = 1000
    max_steps_per_episode: int = 100
    consolidation_interval: int = 100
    sleep_cycles: int = 3
    reward_threshold: float = 0.8
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995


@dataclass
class EpisodeResult:
    """回合结果"""
    episode: int
    total_reward: float
    steps: int
    memories_formed: int
    consolidations: int
    avg_loss: float


class BrainSimulator:
    """
    大脑仿真器
    
    提供完整的仿真训练环境：
    1. 环境交互
    2. 奖励计算
    3. 记忆巩固
    4. 睡眠重放
    5. 性能评估
    """
    
    def __init__(
        self,
        brain: Brain,
        config: Optional[SimulationConfig] = None
    ):
        self.brain = brain
        self.config = config or SimulationConfig()
        
        self.mode = SimulationMode.TRAINING
        self.current_episode = 0
        self.current_step = 0
        
        self._episode_history: List[EpisodeResult] = []
        self._reward_history = deque(maxlen=1000)
        self._loss_history = deque(maxlen=1000)
        
        self._exploration_rate = self.config.exploration_rate
        
        self._environment: Optional[Any] = None
        self._reward_function: Optional[Callable] = None
        
    def set_environment(
        self, 
        environment: Any,
        reward_function: Optional[Callable] = None
    ):
        """
        设置仿真环境
        
        Args:
            environment: 环境对象
            reward_function: 奖励函数
        """
        self._environment = environment
        self._reward_function = reward_function
    
    def run_episode(
        self,
        input_generator: Optional[Callable] = None,
        target_generator: Optional[Callable] = None
    ) -> EpisodeResult:
        """
        运行单个回合
        
        Args:
            input_generator: 输入生成器
            target_generator: 目标生成器
            
        Returns:
            回合结果
        """
        self.current_episode += 1
        self.current_step = 0
        
        total_reward = 0.0
        total_loss = 0.0
        memories_formed = 0
        consolidations = 0
        
        initial_stats = self.brain.get_memory_stats()
        
        for step in range(self.config.max_steps_per_episode):
            self.current_step = step
            
            if input_generator:
                input_data = input_generator()
            else:
                input_data = np.random.randn(self.brain.config.input_dim)
            
            if target_generator:
                target = target_generator()
            else:
                target = np.random.randn(self.brain.config.output_dim)
            
            if self.mode == SimulationMode.TRAINING:
                if np.random.random() < self._exploration_rate:
                    output, info = self.brain.process(input_data)
                    reward = self._compute_exploration_reward(output, target)
                else:
                    learn_result = self.brain.learn(input_data, target, reward=0.0)
                    output = learn_result['output']
                    reward = -learn_result['loss']
                    total_loss += learn_result['loss']
            else:
                output, info = self.brain.process(input_data)
                reward = self._compute_reward(output, target)
            
            total_reward += reward
            
            if self._should_consolidate(step):
                consolidation_stats = self.brain.consolidate(
                    sleep_cycles=self.config.sleep_cycles
                )
                consolidations += consolidation_stats['memories_consolidated']
        
        final_stats = self.brain.get_memory_stats()
        memories_formed = (
            final_stats['hippocampus']['memory_count'] - 
            initial_stats['hippocampus']['memory_count']
        )
        
        avg_loss = total_loss / max(1, self.config.max_steps_per_episode)
        
        result = EpisodeResult(
            episode=self.current_episode,
            total_reward=total_reward,
            steps=self.config.max_steps_per_episode,
            memories_formed=memories_formed,
            consolidations=consolidations,
            avg_loss=avg_loss
        )
        
        self._episode_history.append(result)
        self._reward_history.append(total_reward)
        self._loss_history.append(avg_loss)
        
        self._exploration_rate *= self.config.exploration_decay
        
        return result
    
    def run_training(
        self,
        input_generator: Optional[Callable] = None,
        target_generator: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ) -> List[EpisodeResult]:
        """
        运行完整训练
        
        Args:
            input_generator: 输入生成器
            target_generator: 目标生成器
            callback: 回调函数
            
        Returns:
            训练结果列表
        """
        self.mode = SimulationMode.TRAINING
        results = []
        
        for episode in range(self.config.max_episodes):
            result = self.run_episode(input_generator, target_generator)
            results.append(result)
            
            if callback:
                callback(result)
            
            if self._should_sleep(episode):
                self._run_sleep_phase()
            
            if self._check_convergence():
                print(f"Converged at episode {episode}")
                break
        
        return results
    
    def evaluate(
        self,
        num_episodes: int = 10,
        input_generator: Optional[Callable] = None,
        target_generator: Optional[Callable] = None
    ) -> Dict:
        """
        评估性能
        
        Args:
            num_episodes: 评估回合数
            input_generator: 输入生成器
            target_generator: 目标生成器
            
        Returns:
            评估结果
        """
        self.mode = SimulationMode.TESTING
        original_exploration = self._exploration_rate
        self._exploration_rate = 0.0
        
        rewards = []
        for _ in range(num_episodes):
            result = self.run_episode(input_generator, target_generator)
            rewards.append(result.total_reward)
        
        self._exploration_rate = original_exploration
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'memory_stats': self.brain.get_memory_stats()
        }
    
    def run_sleep(self, cycles: int = 3) -> Dict:
        """
        运行睡眠阶段
        
        Args:
            cycles: 睡眠周期数
            
        Returns:
            巩固统计
        """
        self.mode = SimulationMode.SLEEP
        return self.brain.consolidate(sleep_cycles=cycles)
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        if not self._episode_history:
            return {
                'episodes': 0,
                'avg_reward': 0,
                'avg_loss': 0
            }
        
        recent_rewards = list(self._reward_history)[-100:]
        recent_losses = list(self._loss_history)[-100:]
        
        return {
            'episodes': len(self._episode_history),
            'current_episode': self.current_episode,
            'avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_loss': np.mean(recent_losses) if recent_losses else 0,
            'exploration_rate': self._exploration_rate,
            'total_memories': self.brain.get_memory_stats()['hippocampus']['memory_count']
        }
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        self.brain.save_state(filepath)
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        self.brain.load_state(filepath)
    
    def _compute_reward(
        self, 
        output: np.ndarray, 
        target: np.ndarray
    ) -> float:
        """计算奖励"""
        if self._reward_function:
            return self._reward_function(output, target)
        
        min_len = min(len(output), len(target))
        error = np.mean((output[:min_len] - target[:min_len]) ** 2)
        return -error
    
    def _compute_exploration_reward(
        self, 
        output: np.ndarray, 
        target: np.ndarray
    ) -> float:
        """计算探索奖励"""
        min_len = min(len(output), len(target))
        similarity = np.corrcoef(output[:min_len], target[:min_len])[0, 1]
        if np.isnan(similarity):
            similarity = 0.0
        return similarity * 0.1
    
    def _should_consolidate(self, step: int) -> bool:
        """是否应该巩固"""
        return step > 0 and step % self.config.consolidation_interval == 0
    
    def _should_sleep(self, episode: int) -> bool:
        """是否应该睡眠"""
        return episode > 0 and episode % (self.config.consolidation_interval * 10) == 0
    
    def _run_sleep_phase(self):
        """运行睡眠阶段"""
        original_state = self.brain.state
        self.brain.set_state(BrainState.DEEP_SLEEP)
        
        self.run_sleep(cycles=self.config.sleep_cycles)
        
        self.brain.set_state(original_state)
    
    def _check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self._reward_history) < 100:
            return False
        
        recent_rewards = list(self._reward_history)[-100:]
        avg_reward = np.mean(recent_rewards)
        
        return avg_reward > self.config.reward_threshold
