"""
ReinforcementTrainer - 强化学习训练器
实现基于奖励的学习，支持多种强化学习算法
模拟多巴胺驱动的学习过程
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.brain import Brain, BrainState
from neurons.synapse import SynapticPlasticity


class RLAlgorithm(Enum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    ACTOR_CRITIC = "actor_critic"
    POLICY_GRADIENT = "policy_gradient"
    DOPAMINE_MODULATED = "dopamine_modulated"


@dataclass
class TrainerConfig:
    """训练器配置"""
    algorithm: RLAlgorithm = RLAlgorithm.DOPAMINE_MODULATED
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100
    dopamine_baseline: float = 0.5
    dopamine_decay: float = 0.95


@dataclass
class Experience:
    """经验元组"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    dopamine: float = 0.0


class ReinforcementTrainer:
    """
    强化学习训练器
    
    实现多巴胺调制的强化学习：
    1. 经验回放
    2. 目标网络
    3. 多巴胺信号
    4. 探索-利用平衡
    """
    
    def __init__(
        self,
        brain: Brain,
        config: Optional[TrainerConfig] = None
    ):
        self.brain = brain
        self.config = config or TrainerConfig()
        
        self._replay_buffer: deque = deque(maxlen=self.config.memory_size)
        self._dopamine_level = self.config.dopamine_baseline
        
        self._step_count = 0
        self._episode_count = 0
        self._total_reward = 0.0
        
        self._q_values = np.zeros(100)
        self._policy_weights = np.random.randn(100, 10) * 0.1
        
        self._loss_history = deque(maxlen=1000)
        self._reward_history = deque(maxlen=1000)
        
        self._target_weights = None
        
    def select_action(
        self, 
        state: np.ndarray,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            available_actions: 可用动作列表
            
        Returns:
            选择的动作
        """
        if available_actions is None:
            available_actions = list(range(10))
        
        if np.random.random() < self.config.exploration_rate:
            return np.random.choice(available_actions)
        
        output, info = self.brain.process(state)
        
        q_values = self._compute_q_values(output)
        
        valid_q = [(a, q_values[a]) for a in available_actions if a < len(q_values)]
        if valid_q:
            best_action = max(valid_q, key=lambda x: x[1])[0]
        else:
            best_action = available_actions[0]
        
        return best_action
    
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict:
        """
        学习更新
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            
        Returns:
            学习信息
        """
        self._step_count += 1
        
        dopamine = self._compute_dopamine(reward)
        
        experience = Experience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            dopamine=dopamine
        )
        self._replay_buffer.append(experience)
        
        self._update_dopamine(reward)
        
        if self.config.algorithm == RLAlgorithm.Q_LEARNING:
            loss = self._q_learning_update(experience)
        elif self.config.algorithm == RLAlgorithm.SARSA:
            loss = self._sarsa_update(experience)
        elif self.config.algorithm == RLAlgorithm.ACTOR_CRITIC:
            loss = self._actor_critic_update(experience)
        elif self.config.algorithm == RLAlgorithm.POLICY_GRADIENT:
            loss = self._policy_gradient_update(experience)
        else:
            loss = self._dopamine_modulated_update(experience)
        
        self._loss_history.append(loss)
        self._reward_history.append(reward)
        self._total_reward += reward
        
        if self._step_count % self.config.target_update_freq == 0:
            self._update_target()
        
        self.config.exploration_rate = max(
            self.config.min_exploration,
            self.config.exploration_rate * self.config.exploration_decay
        )
        
        return {
            'loss': loss,
            'dopamine': dopamine,
            'exploration_rate': self.config.exploration_rate,
            'buffer_size': len(self._replay_buffer)
        }
    
    def batch_learn(self, batch_size: Optional[int] = None) -> Dict:
        """
        批量学习
        
        Args:
            batch_size: 批量大小
            
        Returns:
            学习信息
        """
        batch_size = batch_size or self.config.batch_size
        
        if len(self._replay_buffer) < batch_size:
            return {'loss': 0.0, 'batch_size': 0}
        
        indices = np.random.choice(len(self._replay_buffer), batch_size, replace=False)
        batch = [self._replay_buffer[i] for i in indices]
        
        total_loss = 0.0
        
        for exp in batch:
            if self.config.algorithm == RLAlgorithm.DOPAMINE_MODULATED:
                loss = self._dopamine_modulated_update(exp)
            else:
                loss = self._q_learning_update(exp)
            total_loss += loss
        
        avg_loss = total_loss / batch_size
        
        self._loss_history.append(avg_loss)
        
        return {
            'loss': avg_loss,
            'batch_size': batch_size
        }
    
    def end_episode(self):
        """结束回合"""
        self._episode_count += 1
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        recent_rewards = list(self._reward_history)[-100:]
        recent_losses = list(self._loss_history)[-100:]
        
        return {
            'steps': self._step_count,
            'episodes': self._episode_count,
            'total_reward': self._total_reward,
            'avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_loss': np.mean(recent_losses) if recent_losses else 0,
            'exploration_rate': self.config.exploration_rate,
            'dopamine_level': self._dopamine_level,
            'buffer_size': len(self._replay_buffer)
        }
    
    def save(self, filepath: str):
        """保存训练器状态"""
        import pickle
        state = {
            'config': self.config,
            'step_count': self._step_count,
            'episode_count': self._episode_count,
            'total_reward': self._total_reward,
            'dopamine_level': self._dopamine_level,
            'q_values': self._q_values,
            'policy_weights': self._policy_weights
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """加载训练器状态"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self._step_count = state['step_count']
        self._episode_count = state['episode_count']
        self._total_reward = state['total_reward']
        self._dopamine_level = state['dopamine_level']
        self._q_values = state['q_values']
        self._policy_weights = state['policy_weights']
    
    def _compute_q_values(self, state_encoding: np.ndarray) -> np.ndarray:
        """计算Q值"""
        min_len = min(len(state_encoding), len(self._q_values))
        self._q_values[:min_len] = state_encoding[:min_len]
        return self._q_values
    
    def _compute_dopamine(self, reward: float) -> float:
        """计算多巴胺信号"""
        prediction_error = reward - self._dopamine_level
        return np.tanh(prediction_error)
    
    def _update_dopamine(self, reward: float):
        """更新多巴胺水平"""
        self._dopamine_level = (
            self.config.dopamine_decay * self._dopamine_level +
            (1 - self.config.dopamine_decay) * reward
        )
    
    def _q_learning_update(self, experience: Experience) -> float:
        """Q学习更新"""
        current_q = self._q_values[experience.action] if experience.action < len(self._q_values) else 0
        
        next_output, _ = self.brain.process(experience.next_state)
        next_q = self._compute_q_values(next_output)
        max_next_q = np.max(next_q)
        
        target = experience.reward + (1 - float(experience.done)) * self.config.discount_factor * max_next_q
        
        td_error = target - current_q
        loss = td_error ** 2
        
        if experience.action < len(self._q_values):
            self._q_values[experience.action] += self.config.learning_rate * td_error
        
        self.brain.learn(
            experience.state,
            np.array([target] * self.brain.config.output_dim),
            reward=experience.reward
        )
        
        return loss
    
    def _sarsa_update(self, experience: Experience) -> float:
        """SARSA更新"""
        current_q = self._q_values[experience.action] if experience.action < len(self._q_values) else 0
        
        next_action = self.select_action(experience.next_state)
        next_q = self._q_values[next_action] if next_action < len(self._q_values) else 0
        
        target = experience.reward + (1 - float(experience.done)) * self.config.discount_factor * next_q
        
        td_error = target - current_q
        loss = td_error ** 2
        
        if experience.action < len(self._q_values):
            self._q_values[experience.action] += self.config.learning_rate * td_error
        
        return loss
    
    def _actor_critic_update(self, experience: Experience) -> float:
        """Actor-Critic更新"""
        output, _ = self.brain.process(experience.state)
        
        value = np.mean(output)
        
        next_output, _ = self.brain.process(experience.next_state)
        next_value = np.mean(next_output)
        
        td_error = (
            experience.reward +
            (1 - float(experience.done)) * self.config.discount_factor * next_value -
            value
        )
        
        loss = td_error ** 2
        
        self.brain.learn(
            experience.state,
            output + self.config.learning_rate * td_error,
            reward=experience.reward
        )
        
        return loss
    
    def _policy_gradient_update(self, experience: Experience) -> float:
        """策略梯度更新"""
        output, _ = self.brain.process(experience.state)
        
        if experience.action < self._policy_weights.shape[1]:
            min_len = min(len(output), self._policy_weights.shape[0])
            gradient = (
                self.config.learning_rate *
                experience.reward *
                output[:min_len]
            )
            self._policy_weights[:min_len, experience.action] += gradient
        
        loss = -experience.reward * np.log(np.abs(output.mean()) + 1e-8)
        
        return loss
    
    def _dopamine_modulated_update(self, experience: Experience) -> float:
        """多巴胺调制更新"""
        dopamine = experience.dopamine
        
        current_q = self._q_values[experience.action] if experience.action < len(self._q_values) else 0
        
        next_output, _ = self.brain.process(experience.next_state)
        next_q = self._compute_q_values(next_output)
        max_next_q = np.max(next_q)
        
        target = experience.reward + (1 - float(experience.done)) * self.config.discount_factor * max_next_q
        
        td_error = target - current_q
        loss = td_error ** 2
        
        modulated_lr = self.config.learning_rate * (1 + dopamine)
        
        if experience.action < len(self._q_values):
            self._q_values[experience.action] += modulated_lr * td_error
        
        self.brain.learn(
            experience.state,
            np.array([target] * self.brain.config.output_dim),
            reward=dopamine
        )
        
        return loss
    
    def _update_target(self):
        """更新目标网络"""
        pass
