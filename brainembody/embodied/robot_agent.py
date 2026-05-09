"""
具身智能机器人智能体
整合大脑、记忆和感知
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..core.brain_core import BrainCore, Perception
from ..memory.memory_system import MemorySystem
from .simulator import EmbodiedSimulator


@dataclass
class Experience:
    """经验记录"""
    state: Dict
    action: str
    reward: float
    next_state: Dict
    done: bool
    lesson: str


class RobotAgent:
    """
    具身智能机器人智能体

    整合：
    1. BrainCore - 大脑推理
    2. MemorySystem - 记忆系统
    3. EmbodiedSimulator - 仿真环境
    """

    def __init__(self, name: str = "Robot"):
        self.name = name

        self.brain = BrainCore()
        self.memory = MemorySystem()
        self.simulator = EmbodiedSimulator()

        self.experience_buffer: List[Experience] = []
        self.total_steps = 0
        self.successful_episodes = 0

        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration = 0.1

    def perceive(self) -> Perception:
        """感知环境"""
        observation = self.simulator.get_observation()

        perception = Perception(
            visual=np.array(observation.get("lidar", [])),
            proprioception=np.array(observation.get("position", [])),
            touch=None,
            audio=None
        )

        analysis = self.brain.perceive(perception)

        self.memory.encode_perception(
            observation,
            context=f"步骤 {self.total_steps}"
        )

        return perception

    def think(self, task: str) -> str:
        """思考"""
        context = {
            "环境状态": self.simulator.render_text(),
            "记忆摘要": str(self.memory.get_memory_summary()),
            "探索率": f"{self.exploration_rate:.2f}"
        }

        related_experiences = self.memory.get_related_experiences(task, limit=3)

        if related_experiences:
            context["相关经验"] = "\n".join(related_experiences)

        thought = self.brain.think(task, context)

        return thought

    def decide_action(self) -> tuple:
        """决定动作"""
        if np.random.random() < self.exploration_rate:
            action_type = np.random.choice(["move_forward", "turn", "explore"])
            params = {}

            if action_type == "move_forward":
                params["distance"] = np.random.uniform(0.5, 2.0)
            elif action_type == "turn":
                params["angle"] = np.random.uniform(-0.5, 0.5)

            return action_type, params, "exploration"
        else:
            observation = self.simulator.get_observation()

            action = self.brain.plan_action(
                goal="到达绿色目标位置",
                current_state=observation
            )

            return action.action_type, action.parameters, "planned"

    def act(self, action_type: str, params: Dict) -> tuple:
        """
        执行动作

        Returns:
            (下一观测, 奖励, 是否结束, 动作类型)
        """
        observation, reward, done = self.simulator.step(action_type, params)

        self.total_steps += 1

        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

        return observation, reward, done, action_type

    def remember_experience(self, state: Dict, action: str,
                          reward: float, next_state: Dict, done: bool):
        """记录经验"""
        lesson = self._extract_lesson(action, reward, done)

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            lesson=lesson
        )

        self.experience_buffer.append(experience)

        outcome = "成功" if done and reward > 0 else "失败"
        self.memory.encode_experience(
            experience=f"执行动作 {action}，奖励 {reward:.2f}",
            outcome=outcome,
            emotions=["成功"] if reward > 0 else ["失败"]
        )

        if done and reward > 0:
            self.successful_episodes += 1

        if len(self.experience_buffer) > 100:
            self.experience_buffer.pop(0)

    def _extract_lesson(self, action: str, reward: float, done: bool) -> str:
        """提取教训"""
        if done and reward > 0:
            return "成功到达目标！继续保持。"
        elif done and reward < 0:
            return "撞到障碍物，需要避开。"
        elif reward > 0:
            return "正在接近目标。"
        else:
            return "继续探索。"

    def reflect(self):
        """反思"""
        if len(self.experience_buffer) < 10:
            return

        recent = self.experience_buffer[-10:]

        experience_text = "\n".join([
            f"动作: {e.action}, 奖励: {e.reward:.2f}, 教训: {e.lesson}"
            for e in recent
        ])

        reflection = self.brain.reflect(experience_text)

        print(f"\n🤔 反思结果:\n{reflection}")

    def run_episode(self, max_steps: int = 100, verbose: bool = True) -> Dict:
        """
        运行一个回合

        Args:
            max_steps: 最大步数
            verbose: 是否输出详细信息

        Returns:
            回合统计
        """
        state = self.simulator.reset()
        total_reward = 0
        actions_taken = []

        for step in range(max_steps):
            self.perceive()

            action_type, params, decision_type = self.decide_action()

            next_state, reward, done, action = self.act(action_type, params)

            self.remember_experience(
                state, action, reward, next_state, done
            )

            total_reward += reward
            actions_taken.append(action)

            if verbose and step % 20 == 0:
                print(f"步骤 {step}: 动作={action}, 奖励={reward:.2f}, 累计奖励={total_reward:.2f}")

            if done:
                if verbose:
                    print(f"\n✓ 回合完成！总奖励: {total_reward:.2f}, 步数: {step}")
                break

            state = next_state

        if verbose and step >= max_steps - 1:
            print(f"\n⚠ 超时！总奖励: {total_reward:.2f}, 步数: {step}")

        if len(self.experience_buffer) >= 50:
            self.reflect()

        return {
            "total_reward": total_reward,
            "steps": step,
            "actions": actions_taken,
            "success": done and reward > 0,
            "exploration_rate": self.exploration_rate
        }

    def get_stats(self) -> Dict:
        """获取统计"""
        success_rate = 0.0
        if self.total_steps > 0:
            success_rate = self.successful_episodes / max(1, self.total_steps // 100)

        return {
            "name": self.name,
            "total_steps": self.total_steps,
            "successful_episodes": self.successful_episodes,
            "success_rate": success_rate,
            "current_exploration_rate": self.exploration_rate,
            "memory_stats": self.memory.get_memory_summary(),
            "brain_stats": self.brain.get_state_summary()
        }

    def train(self, num_episodes: int = 100, target_success_rate: float = 0.8):
        """
        训练智能体

        Args:
            num_episodes: 训练回合数
            target_success_rate: 目标成功率
        """
        print(f"\n开始训练 {num_episodes} 个回合...")
        print(f"目标成功率: {target_success_rate * 100:.0f}%")
        print("=" * 60)

        success_history = []

        for episode in range(num_episodes):
            result = self.run_episode(max_steps=100, verbose=False)

            success_history.append(1 if result["success"] else 0)

            if len(success_history) > 20:
                success_history.pop(0)

            recent_success_rate = np.mean(success_history)

            if episode % 10 == 0:
                print(f"回合 {episode}: "
                      f"奖励={result['total_reward']:.2f}, "
                      f"成功率={recent_success_rate*100:.0f}%, "
                      f"探索率={result['exploration_rate']:.2f}")

            if recent_success_rate >= target_success_rate:
                print(f"\n✓ 达到目标成功率！训练完成。")
                break

        print(f"\n训练完成！")
        print(f"总回合: {episode + 1}")
        print(f"成功率: {recent_success_rate*100:.0f}%")

    def reset(self):
        """重置智能体"""
        self.brain.reset()
        self.memory.reset()
        self.experience_buffer.clear()
        self.total_steps = 0
        self.successful_episodes = 0
        self.exploration_rate = 1.0
