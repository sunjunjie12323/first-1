"""
具身智能训练器
整合强化学习训练流程
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from ..embodied.robot_agent import RobotAgent


@dataclass
class TrainingConfig:
    """训练配置"""
    num_episodes: int = 100
    max_steps_per_episode: int = 100
    target_success_rate: float = 0.8
    save_frequency: int = 10
    eval_frequency: int = 20
    memory_consolidation_interval: int = 50


@dataclass
class TrainingMetrics:
    """训练指标"""
    episode: int
    total_reward: float
    steps: int
    success: bool
    exploration_rate: float
    timestamp: float


class EmbodiedTrainer:
    """
    具身智能训练器

    功能：
    1. 管理训练流程
    2. 记录训练指标
    3. 自动记忆巩固
    4. 训练效果评估
    """

    def __init__(self, agent: RobotAgent, config: Optional[TrainingConfig] = None):
        self.agent = agent
        self.config = config or TrainingConfig()

        self.metrics_history: List[TrainingMetrics] = []
        self.best_reward = float('-inf')
        self.best_episode = 0

        self.training_start_time = None
        self.training_end_time = None

    def train(self, callback: Optional[Callable] = None) -> Dict:
        """
        训练智能体

        Args:
            callback: 训练过程中的回调函数

        Returns:
            训练结果
        """
        print("=" * 70)
        print("具身智能训练开始")
        print("=" * 70)
        print(f"配置:")
        print(f"  - 训练回合: {self.config.num_episodes}")
        print(f"  - 每回合最大步数: {self.config.max_steps_per_episode}")
        print(f"  - 目标成功率: {self.config.target_success_rate * 100:.0f}%")
        print("=" * 70)

        self.training_start_time = datetime.now()

        success_history = []
        reward_history = []

        for episode in range(self.config.num_episodes):
            episode_start = datetime.now()

            result = self.agent.run_episode(
                max_steps=self.config.max_steps_per_episode,
                verbose=False
            )

            metric = TrainingMetrics(
                episode=episode,
                total_reward=result["total_reward"],
                steps=result["steps"],
                success=result["success"],
                exploration_rate=result["exploration_rate"],
                timestamp=datetime.now().timestamp()
            )

            self.metrics_history.append(metric)

            success_history.append(1 if result["success"] else 0)
            reward_history.append(result["total_reward"])

            if len(success_history) > 20:
                success_history.pop(0)
            if len(reward_history) > 20:
                reward_history.pop(0)

            recent_success_rate = np.mean(success_history)
            recent_avg_reward = np.mean(reward_history[-10:])

            if result["total_reward"] > self.best_reward:
                self.best_reward = result["total_reward"]
                self.best_episode = episode

            if episode % self.config.save_frequency == 0:
                self._save_checkpoint(episode)

            if episode % self.config.eval_frequency == 0:
                self._print_eval(episode, recent_success_rate, recent_avg_reward)

            if callback:
                callback(episode, result)

            if recent_success_rate >= self.config.target_success_rate:
                print(f"\n✓ 达到目标成功率 {self.config.target_success_rate * 100:.0f}%！")
                print(f"✓ 训练提前结束于第 {episode} 个回合")
                break

            if episode % self.config.memory_consolidation_interval == 0 and episode > 0:
                self.agent.memory.consolidate()

        self.training_end_time = datetime.now()

        return self._generate_training_report()

    def _print_eval(self, episode: int, success_rate: float, avg_reward: float):
        """打印评估信息"""
        print(f"回合 {episode:3d} | "
              f"成功率: {success_rate*100:5.1f}% | "
              f"平均奖励: {avg_reward:7.2f} | "
              f"最佳奖励: {self.best_reward:7.2f} (回合 {self.best_episode})")

    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        checkpoint = {
            "episode": episode,
            "metrics": [
                {
                    "episode": m.episode,
                    "total_reward": m.total_reward,
                    "success": m.success
                }
                for m in self.metrics_history[-100:]
            ],
            "agent_stats": self.agent.get_stats()
        }
        print(f"  [检查点保存] 回合 {episode}")

    def _generate_training_report(self) -> Dict:
        """生成训练报告"""
        training_duration = None
        if self.training_start_time and self.training_end_time:
            training_duration = (
                self.training_end_time - self.training_start_time
            ).total_seconds()

        success_count = sum(1 for m in self.metrics_history if m.success)
        total_episodes = len(self.metrics_history)

        return {
            "status": "completed",
            "total_episodes": total_episodes,
            "successful_episodes": success_count,
            "overall_success_rate": success_count / max(1, total_episodes) if total_episodes > 0 else 0,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "training_duration_seconds": training_duration,
            "final_exploration_rate": self.metrics_history[-1].exploration_rate if self.metrics_history else 1.0,
            "metrics_summary": {
                "avg_reward": np.mean([m.total_reward for m in self.metrics_history]),
                "avg_steps": np.mean([m.steps for m in self.metrics_history]),
                "max_reward": max([m.total_reward for m in self.metrics_history]) if self.metrics_history else 0,
                "min_reward": min([m.total_reward for m in self.metrics_history]) if self.metrics_history else 0
            }
        }

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        评估训练后的智能体

        Args:
            num_episodes: 评估回合数

        Returns:
            评估结果
        """
        print(f"\n评估智能体 ({num_episodes} 个回合)...")

        eval_results = []
        for i in range(num_episodes):
            result = self.agent.run_episode(
                max_steps=self.config.max_steps_per_episode,
                verbose=False
            )
            eval_results.append(result)

        success_count = sum(1 for r in eval_results if r["success"])
        avg_reward = np.mean([r["total_reward"] for r in eval_results])

        print(f"\n评估结果:")
        print(f"  成功率: {success_count / num_episodes * 100:.1f}%")
        print(f"  平均奖励: {avg_reward:.2f}")

        return {
            "num_episodes": num_episodes,
            "success_rate": success_count / num_episodes,
            "avg_reward": avg_reward,
            "results": eval_results
        }

    def plot_training_curve(self):
        """绘制训练曲线（文本版）"""
        if len(self.metrics_history) < 10:
            print("数据不足，无法绘制训练曲线")
            return

        print("\n训练曲线 (最近 50 回合):")
        print("-" * 60)

        recent = self.metrics_history[-50:]

        rewards = [m.total_reward for m in recent]
        max_reward = max(rewards)
        min_reward = min(rewards)

        for i, metric in enumerate(recent[::5]):
            bar_length = int((metric.total_reward - min_reward) /
                           (max_reward - min_reward + 1e-8) * 40)

            success_marker = "✓" if metric.success else " "
            print(f"{metric.episode:3d} [{success_marker}] "
                  f"{'█' * bar_length}{'░' * (40 - bar_length)} "
                  f"{metric.total_reward:7.2f}")

        print("-" * 60)

    def get_learning_progress(self) -> Dict:
        """获取学习进度"""
        if len(self.metrics_history) < 20:
            return {"status": "insufficient_data"}

        first_half = self.metrics_history[:len(self.metrics_history)//2]
        second_half = self.metrics_history[len(self.metrics_history)//2:]

        first_success_rate = sum(1 for m in first_half if m.success) / len(first_half)
        second_success_rate = sum(1 for m in second_half if m.success) / len(second_half)

        first_avg_reward = np.mean([m.total_reward for m in first_half])
        second_avg_reward = np.mean([m.total_reward for m in second_half])

        improvement = second_success_rate - first_success_rate

        if improvement > 0.1:
            status = "learning_well"
            message = "学习效果良好"
        elif improvement > 0:
            status = "learning_slowly"
            message = "学习速度较慢"
        else:
            status = "not_learning"
            message = "未观察到学习进步"

        return {
            "status": status,
            "message": message,
            "first_half_success_rate": first_success_rate,
            "second_half_success_rate": second_success_rate,
            "improvement": improvement,
            "first_half_avg_reward": first_avg_reward,
            "second_half_avg_reward": second_avg_reward
        }
