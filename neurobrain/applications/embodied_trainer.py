"""
具身智能训练器
使用NeuroBrain框架训练机器人在仿真环境中完成任务
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurobrain import Brain, BrainConfig, BrainState
from neurobrain.training.simulator import BrainSimulator, SimulationConfig
from neurobrain.training.reinforcement import ReinforcementTrainer, TrainerConfig, RLAlgorithm


@dataclass
class TrainingMetrics:
    """训练指标"""
    episode: int
    total_reward: float
    success_rate: float
    avg_loss: float
    exploration_rate: float
    memory_count: int
    dopamine_level: float


class EmbodiedTrainer:
    """
    具身智能训练器

    功能：
    1. 环境交互
    2. 神经网络决策
    3. 记忆增强学习
    4. 性能评估
    5. 模型保存/加载
    """

    def __init__(
        self,
        environment,
        brain: Optional[Brain] = None,
        brain_config: Optional[BrainConfig] = None,
        train_config: Optional[TrainerConfig] = None
    ):
        self.env = environment

        if brain is None:
            brain_config = brain_config or BrainConfig(
                input_dim=20,
                hidden_dims=[128, 64, 32],
                output_dim=2,
                working_memory_capacity=7,
                emotional_weight=0.3
            )
            brain = Brain(brain_config)

        self.brain = brain

        train_config = train_config or TrainerConfig(
            algorithm=RLAlgorithm.DOPAMINE_MODULATED,
            learning_rate=0.001,
            exploration_rate=0.3,
            exploration_decay=0.995,
            min_exploration=0.01,
            batch_size=32,
            memory_size=10000
        )
        self.trainer = ReinforcementTrainer(self.brain, train_config)

        self.episode_count = 0
        self.total_steps = 0

        self._reward_history = deque(maxlen=100)
        self._success_history = deque(maxlen=100)
        self._loss_history = deque(maxlen=100)

        self._best_reward = float('-inf')
        self._patience = 50
        self._no_improvement = 0

    def train(
        self,
        num_episodes: int = 1000,
        callback: Optional[Callable] = None,
        save_interval: int = 100,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict:
        """
        训练智能体

        Args:
            num_episodes: 训练回合数
            callback: 每回合回调函数
            save_interval: 保存间隔
            checkpoint_dir: 检查点目录

        Returns:
            训练历史
        """
        print("\n" + "=" * 60)
        print("开始具身智能训练")
        print("=" * 60)
        print(f"环境: {type(self.env).__name__}")
        print(f"训练回合: {num_episodes}")
        print(f"初始探索率: {self.trainer.config.exploration_rate}")
        print("=" * 60 + "\n")

        os.makedirs(checkpoint_dir, exist_ok=True)

        episode_rewards = []
        episode_losses = []
        success_count = 0

        for episode in range(num_episodes):
            episode_reward, episode_loss, success = self._run_episode()

            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss)
            self._reward_history.append(episode_reward)
            self._loss_history.append(episode_loss)

            if success:
                success_count += 1
                self._success_history.append(1.0)
            else:
                self._success_history.append(0.0)

            success_rate = success_count / (episode + 1)
            avg_reward = np.mean(list(self._reward_history))
            avg_loss = np.mean(list(self._loss_history))

            metrics = TrainingMetrics(
                episode=episode + 1,
                total_reward=episode_reward,
                success_rate=success_rate,
                avg_loss=avg_loss,
                exploration_rate=self.trainer.config.exploration_rate,
                memory_count=self.brain.hippocampus.get_stats()['memory_count'],
                dopamine_level=self.trainer.get_training_stats()['dopamine_level']
            )

            if callback:
                callback(metrics)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | "
                      f"Success: {success_rate:.1%} | "
                      f"Explore: {self.trainer.config.exploration_rate:.2%} | "
                      f"Memory: {metrics.memory_count}")

            if episode_reward > self._best_reward:
                self._best_reward = episode_reward
                self._no_improvement = 0
                self._save_checkpoint(os.path.join(checkpoint_dir, "best_model.pkl"))
            else:
                self._no_improvement += 1

            if save_interval > 0 and (episode + 1) % save_interval == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_{episode + 1}.pkl")
                )

            if self._no_improvement >= self._patience:
                print(f"\n早停: 连续 {self._patience} 个回合没有改进")
                break

        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)

        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'best_reward': self._best_reward,
            'final_success_rate': success_count / num_episodes,
            'total_episodes': episode + 1
        }

    def _run_episode(self) -> Tuple[float, float, bool]:
        """运行单个回合"""
        obs = self.env.reset()
        self.brain.set_state(BrainState.FOCUSED)

        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        success = False

        done = False
        while not done:
            action = self._select_action(obs)

            next_obs, reward, done, info = self.env.step(action)

            learn_result = self.trainer.learn(
                obs, int(action[0] > 0), reward, next_obs, done
            )

            total_reward += reward
            total_loss += learn_result.get('loss', 0.0)
            steps += 1

            if info.get('success', False):
                success = True

            if done:
                break

            obs = next_obs

        self.trainer.end_episode()

        if steps > 0:
            total_loss /= steps

        self.episode_count += 1
        self.total_steps += steps

        return total_reward, total_loss, success

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        """选择动作"""
        action_output, _ = self.brain.process(obs)

        linear_action = np.tanh(action_output[0]) if len(action_output) > 0 else 0.0
        angular_action = np.tanh(action_output[1]) if len(action_output) > 1 else 0.0

        if np.random.random() < self.trainer.config.exploration_rate:
            linear_action = np.random.uniform(-1, 1)
            angular_action = np.random.uniform(-1, 1)

        return np.array([linear_action, angular_action])

    def evaluate(self, num_episodes: int = 20) -> Dict:
        """
        评估训练好的智能体

        Args:
            num_episodes: 评估回合数

        Returns:
            评估结果
        """
        print("\n" + "=" * 60)
        print("开始评估")
        print("=" * 60)

        original_exploration = self.trainer.config.exploration_rate
        self.trainer.config.exploration_rate = 0.0

        rewards = []
        successes = []
        steps_list = []

        for episode in range(num_episodes):
            obs = self.env.reset()

            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                action = self._select_action(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            successes.append(1.0 if info.get('success', False) else 0.0)
            steps_list.append(steps)

            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {total_reward:7.2f}, "
                  f"Steps = {steps}, "
                  f"Success = {info.get('success', False)}")

        self.trainer.config.exploration_rate = original_exploration

        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'success_rate': np.mean(successes),
            'mean_steps': np.mean(steps_list),
            'all_rewards': rewards,
            'all_successes': successes
        }

        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"成功率: {results['success_rate']:.1%}")
        print(f"平均步数: {results['mean_steps']:.1f}")
        print("=" * 60 + "\n")

        return results

    def _save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'brain_config': self.brain.config,
            'brain_state': self.brain.hippocampus.get_state(),
            'neocortex_state': self.brain.neocortex.get_state(),
            'amygdala_state': self.brain.amygdala.get_state(),
            'trainer_config': self.trainer.config,
            'episode_count': self.episode_count,
            'best_reward': self._best_reward
        }

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"保存检查点: {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        import pickle
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.brain.hippocampus.set_state(checkpoint['brain_state'])
        self.brain.neocortex.set_state(checkpoint['neocortex_state'])
        self.brain.amygdala.set_state(checkpoint['amygdala_state'])
        self.trainer.config = checkpoint['trainer_config']
        self.episode_count = checkpoint['episode_count']
        self._best_reward = checkpoint['best_reward']

        print(f"加载检查点: {filepath}")

    def get_training_history(self) -> Dict:
        """获取训练历史"""
        return {
            'reward_history': list(self._reward_history),
            'success_history': list(self._success_history),
            'loss_history': list(self._loss_history),
            'best_reward': self._best_reward,
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps
        }
