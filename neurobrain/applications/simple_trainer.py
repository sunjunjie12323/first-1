"""
简化版类脑具身智能训练器 - 修复版
专注于验证核心学习能力
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from collections import deque
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleBrain:
    """简化但功能完整的大脑模型"""

    def __init__(self, input_dim: int = 20, hidden_dims: List[int] = [128, 64, 32], output_dim: int = 2):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = []
        self.biases = []
        self.activations = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
            self.activations.append(np.zeros(dims[i]))

        self.learning_rate = 0.001
        self.memory_traces = []
        self.emotional_weight = 0.3
        self.dopamine_level = 0.5

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        x = x[:self.input_dim]

        self.activations = [x]
        current = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            linear = np.dot(current, w) + b

            if i < len(self.weights) - 1:
                current = np.maximum(0, linear)
            else:
                current = np.tanh(linear)

            self.activations.append(current)
            current = current

        return current

    def backward(self, target: np.ndarray, reward: float) -> float:
        """反向传播"""
        output = self.activations[-1]

        if len(target) > len(output):
            target = target[:len(output)]
        elif len(target) < len(output):
            target = np.pad(target, (0, len(output) - len(target)))

        error = target - output
        loss = np.mean(error ** 2)

        self.dopamine_level = 0.9 * self.dopamine_level + 0.1 * reward
        lr = self.learning_rate * (1 + self.dopamine_level)

        deltas = []

        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                delta = error * (1 - output ** 2)
            else:
                act = self.activations[i + 1]
                delta = deltas[-1]
                delta = np.dot(delta, self.weights[i + 1].T)
                delta = delta * (act > 0).astype(float)

            deltas.append(delta)

            w_grad = np.outer(self.activations[i], delta)
            self.weights[i] += lr * w_grad

            self.biases[i] += lr * delta

        deltas.reverse()
        return loss

    def store_memory(self, observation: np.ndarray, reward: float):
        """存储记忆"""
        if reward > 0.5 or abs(reward) > 0.1:
            self.memory_traces.append({
                'observation': observation.copy(),
                'reward': reward,
                'timestamp': time.time()
            })

            if len(self.memory_traces) > 1000:
                self.memory_traces = self.memory_traces[-500:]

    def consolidate(self) -> int:
        """记忆巩固"""
        consolidated = 0

        important_memories = [m for m in self.memory_traces if m['reward'] > 0.5]

        for mem in important_memories[-10:]:
            for _ in range(5):
                obs = mem['observation']
                output = self.forward(obs)
                self.backward(output + 0.1, mem['reward'])
                consolidated += 1

        return consolidated

    def get_memory_count(self) -> int:
        return len(self.memory_traces)


class SimpleEnvironment:
    """简化但功能完整的环境"""

    def __init__(self, task: str = "navigation"):
        self.task = task
        self.world_size = (10.0, 10.0)
        self.max_steps = 200

        self.position = np.array([0.0, 0.0])
        self.target = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.obstacles = []

        self.step_count = 0
        self._initialize()

    def _initialize(self):
        self.position = np.array([0.5, 0.5])
        self.target = np.array([
            np.random.uniform(7, 9),
            np.random.uniform(7, 9)
        ])
        self.obstacles = [
            np.array([3.0, 5.0]),
            np.array([5.0, 3.0]),
            np.array([7.0, 7.0])
        ]
        self.orientation = 0.0
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self._initialize()
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1

        linear = np.clip(action[0], -1.0, 1.0) * 0.3
        angular = np.clip(action[1], -1.0, 1.0) * 0.5

        self.orientation += angular
        self.orientation = self.orientation % (2 * np.pi)

        dx = linear * np.cos(self.orientation)
        dy = linear * np.sin(self.orientation)

        new_pos = self.position + np.array([dx, dy])
        new_pos = np.clip(new_pos, 0, np.array(self.world_size))

        collision = False
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs) < 1.0:
                collision = True
                break

        if not collision:
            self.position = new_pos

        reward = 0.0
        dist = np.linalg.norm(self.position - self.target)
        reward = -dist * 0.1

        if dist < 0.5:
            reward = 100.0
            done = True
        elif collision:
            reward = -10.0
            done = True
        else:
            done = self.step_count >= self.max_steps

        return self._get_observation(), reward, done, {'distance': dist, 'collision': collision}

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(20)

        obs[0:2] = self.position / np.array(self.world_size)
        obs[2] = self.orientation / (2 * np.pi)

        for i in range(8):
            angle = self.orientation + i * np.pi / 4
            direction = np.array([np.cos(angle), np.sin(angle)])

            dist = 5.0
            for d in np.linspace(0.1, 5.0, 20):
                test_pos = self.position + direction * d
                for obs_pos in self.obstacles:
                    if np.linalg.norm(test_pos - obs_pos) < 0.5:
                        dist = d
                        break
                if dist < d:
                    break

            obs[3 + i] = 1.0 - dist / 5.0

        obs[11:13] = self.target / np.array(self.world_size)
        obs[13] = dist / 10.0

        return obs


def train_agent(num_episodes: int = 200) -> Tuple[List[float], List[float]]:
    """训练智能体"""
    print("\n" + "=" * 70)
    print("类脑具身智能训练系统")
    print("=" * 70)
    print(f"训练回合: {num_episodes}")
    print("=" * 70 + "\n")

    env = SimpleEnvironment("navigation")
    brain = SimpleBrain(input_dim=20, hidden_dims=[128, 64, 32], output_dim=2)

    exploration_rate = 0.5
    exploration_decay = 0.99
    min_exploration = 0.05

    episode_rewards = []
    success_count = 0

    for episode in range(num_episodes):
        obs = env.reset()

        total_reward = 0.0
        done = False

        while not done:
            if np.random.random() < exploration_rate:
                action = np.random.uniform(-1, 1, 2)
            else:
                action = brain.forward(obs)
                action = np.tanh(action)

            next_obs, reward, done, info = env.step(action)

            brain.store_memory(obs, reward)

            target = brain.forward(next_obs)
            target_action = action + 0.1 * (next_obs[11:13] - obs[11:13])
            target_action = np.clip(target_action, -1, 1)

            loss = brain.backward(target_action, reward)

            total_reward += reward
            obs = next_obs

        brain.consolidate()

        exploration_rate = max(min_exploration, exploration_rate * exploration_decay)

        episode_rewards.append(total_reward)

        if info.get('distance', 100) < 0.5:
            success_count += 1

        if (episode + 1) % 20 == 0:
            recent_rewards = episode_rewards[-20:]
            recent_success = sum(1 for r in episode_rewards[-20:] if r > 50)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Avg(20): {np.mean(recent_rewards):8.2f} | "
                  f"Success: {recent_success}/20 | "
                  f"Explore: {exploration_rate:.2%} | "
                  f"Memory: {brain.get_memory_count()}")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

    final_success = sum(1 for r in episode_rewards[-50:] if r > 50)
    print(f"最后50回合成功率: {final_success}/50 ({final_success/50:.1%})")
    print(f"最后50回合平均奖励: {np.mean(episode_rewards[-50:]):.2f}")

    return episode_rewards, episode_rewards


def evaluate_agent(num_episodes: int = 20) -> Dict:
    """评估智能体"""
    print("\n" + "=" * 70)
    print("评估智能体")
    print("=" * 70 + "\n")

    env = SimpleEnvironment("navigation")
    brain = SimpleBrain(input_dim=20, hidden_dims=[128, 64, 32], output_dim=2)

    for _ in range(200):
        obs = env.reset()
        for _ in range(100):
            action = brain.forward(obs)
            obs, reward, done, _ = env.step(np.tanh(action))
            brain.backward(action, reward)
            if done:
                break

    rewards = []
    successes = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = brain.forward(obs)
            obs, reward, done, info = env.step(np.tanh(action))
            total_reward += reward

        rewards.append(total_reward)
        successes.append(1.0 if info.get('distance', 100) < 0.5 else 0.0)

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {total_reward:8.2f}, "
              f"Success = {successes[-1] > 0.5}")

    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': np.mean(successes),
        'rewards': rewards
    }

    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"成功率: {results['success_rate']:.1%}")
    print("=" * 70 + "\n")

    return results


def run_verification():
    """运行验证测试"""
    print("\n" + "=" * 70)
    print("NeuroBrain 简化版验证测试")
    print("=" * 70)

    print("\n[测试 1] 环境功能测试")
    env = SimpleEnvironment()
    obs = env.reset()
    print(f"✓ 环境重置成功，观测维度: {len(obs)}")

    action = np.random.uniform(-1, 1, 2)
    obs, reward, done, info = env.step(action)
    print(f"✓ 执行动作成功，奖励: {reward:.4f}")

    print("\n[测试 2] 大脑功能测试")
    brain = SimpleBrain(input_dim=20, output_dim=2)
    output = brain.forward(obs)
    print(f"✓ 前向传播成功，输出维度: {len(output)}")

    loss = brain.backward(np.random.randn(2), 0.5)
    print(f"✓ 反向传播成功，损失: {loss:.4f}")

    print("\n[测试 3] 记忆功能测试")
    for i in range(10):
        brain.store_memory(np.random.randn(20), np.random.random())
    print(f"✓ 存储10条记忆，当前记忆数: {brain.get_memory_count()}")

    consolidated = brain.consolidate()
    print(f"✓ 记忆巩固完成，巩固操作数: {consolidated}")

    print("\n[测试 4] 完整训练测试 (50回合)")
    rewards, _ = train_agent(num_episodes=50)

    early_avg = np.mean(rewards[:10])
    late_avg = np.mean(rewards[-10:])

    print(f"\n学习效果:")
    print(f"  前10回合平均奖励: {early_avg:.2f}")
    print(f"  后10回合平均奖励: {late_avg:.2f}")

    if late_avg > early_avg:
        print(f"  ✓ 学习效果验证通过! 奖励提升了 {late_avg - early_avg:.2f}")
    else:
        print(f"  ⚠ 奖励变化: {late_avg - early_avg:.2f}")

    print("\n" + "=" * 70)
    print("验证测试完成!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_verification()
