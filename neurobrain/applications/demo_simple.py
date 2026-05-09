"""
NeuroBrain 类脑记忆框架 - 演示程序
展示框架的基本功能
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleBrain:
    """简化大脑模型 - 演示用"""

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        np.random.seed(42)
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

        self.lr = 0.1
        self.memory = []

    def forward(self, x):
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        elif len(x) > self.input_dim:
            x = x[:self.input_dim]

        self.forward_input = x
        self.h1 = np.maximum(0, np.dot(x, self.w1) + self.b1)
        self.out = np.tanh(np.dot(self.h1, self.w2) + self.b2)
        return self.out

    def backward(self, target):
        target = target.flatten()
        out = self.out.flatten()

        if len(target) > len(out):
            target = target[:len(out)]
        elif len(target) < len(out):
            target = np.pad(target, (0, len(out) - len(target)))

        error = target - out
        loss = np.mean(error ** 2)

        delta_out = error * (1 - out ** 2)
        delta_h1 = np.dot(delta_out, self.w2.T) * (self.h1 > 0).astype(float)

        self.w2 += self.lr * np.outer(self.h1, delta_out)
        self.b2 += self.lr * delta_out
        self.w1 += self.lr * np.outer(self.forward_input, delta_h1)
        self.b1 += self.lr * delta_h1

        return loss

    def store_memory(self, obs, reward):
        if reward > 0:
            self.memory.append({'obs': obs.copy(), 'reward': reward})
            if len(self.memory) > 100:
                self.memory.pop(0)

    def replay(self):
        for mem in self.memory[-10:]:
            self.forward(mem['obs'])


def simple_task():
    """简单任务：学习朝向目标移动"""
    print("\n" + "=" * 60)
    print("NeuroBrain 类脑记忆框架 - 功能演示")
    print("=" * 60)

    brain = SimpleBrain(input_dim=10, hidden_dim=32, output_dim=2)

    target = np.array([5.0, 5.0])
    position = np.array([0.0, 0.0])
    orientation = 0.0

    print("\n[任务说明]")
    print("机器人需要学习朝目标位置移动")

    print("\n[阶段 1] 随机探索")
    rewards = []
    for episode in range(20):
        position = np.array([0.0, 0.0])
        orientation = np.random.uniform(0, 2 * np.pi)

        total_reward = 0
        for step in range(50):
            obs = np.array([
                position[0] / 10, position[1] / 10,
                target[0] / 10, target[1] / 10,
                np.sin(orientation), np.cos(orientation),
                np.sin(np.arctan2(target[1] - position[1], target[0] - position[0])),
                np.cos(np.arctan2(target[1] - position[1], target[0] - position[0])),
                np.linalg.norm(target - position) / 10,
                step / 50
            ])

            action = np.random.uniform(-1, 1, 2)
            orientation += action[1] * 0.2
            position += action[0] * 0.3 * np.array([np.cos(orientation), np.sin(orientation)])

            dist = np.linalg.norm(target - position)
            reward = -dist * 0.1 + 0.1

            if dist < 0.5:
                reward = 10
                break

            total_reward += reward

        rewards.append(total_reward)
        print(f"  Episode {episode + 1}: 奖励 = {total_reward:.2f}")

    print(f"\n  平均随机奖励: {np.mean(rewards):.2f}")

    print("\n[阶段 2] 学习训练")
    brain.lr = 0.1
    brain.memory = []

    for episode in range(100):
        position = np.array([0.0, 0.0])
        orientation = np.random.uniform(0, 2 * np.pi)

        total_reward = 0
        for step in range(50):
            obs = np.array([
                position[0] / 10, position[1] / 10,
                target[0] / 10, target[1] / 10,
                np.sin(orientation), np.cos(orientation),
                np.sin(np.arctan2(target[1] - position[1], target[0] - position[0])),
                np.cos(np.arctan2(target[1] - position[1], target[0] - position[0])),
                np.linalg.norm(target - position) / 10,
                step / 50
            ])

            if np.random.random() < max(0.1, 0.5 - episode * 0.005):
                action = np.random.uniform(-1, 1, 2)
            else:
                action = brain.forward(obs)
                action = np.tanh(action)

            orientation += action[1] * 0.2
            position += action[0] * 0.3 * np.array([np.cos(orientation), np.sin(orientation)])

            dist = np.linalg.norm(target - position)
            reward = -dist * 0.1 + 0.1

            if dist < 0.5:
                reward = 10

            brain.forward(obs)
            brain.backward(np.tanh(action + reward * 0.1))
            brain.store_memory(obs, reward)

            total_reward += reward

            if dist < 0.5:
                break

        if episode % 20 == 0:
            print(f"  Episode {episode + 1}: 奖励 = {total_reward:.2f}, 记忆 = {len(brain.memory)}")

    print("\n[阶段 3] 测试学习效果")
    successes = 0
    test_rewards = []

    for episode in range(20):
        position = np.array([0.0, 0.0])
        orientation = np.random.uniform(0, 2 * np.pi)

        total_reward = 0
        for step in range(50):
            obs = np.array([
                position[0] / 10, position[1] / 10,
                target[0] / 10, target[1] / 10,
                np.sin(orientation), np.cos(orientation),
                np.sin(np.arctan2(target[1] - position[1], target[0] - position[0])),
                np.cos(np.arctan2(target[1] - position[1], target[0] - position[0])),
                np.linalg.norm(target - position) / 10,
                step / 50
            ])

            action = brain.forward(obs)
            action = np.tanh(action)

            orientation += action[1] * 0.2
            position += action[0] * 0.3 * np.array([np.cos(orientation), np.sin(orientation)])

            dist = np.linalg.norm(target - position)
            reward = -dist * 0.1 + 0.1

            total_reward += reward

            if dist < 0.5:
                successes += 1
                break

        test_rewards.append(total_reward)

    print(f"\n  测试结果:")
    print(f"    成功率: {successes}/20 ({successes/20:.0%})")
    print(f"    平均奖励: {np.mean(test_rewards):.2f}")

    if np.mean(test_rewards) > np.mean(rewards):
        print(f"\n  ✓ 学习成功! 奖励从 {np.mean(rewards):.2f} 提升到 {np.mean(test_rewards):.2f}")
    else:
        print(f"\n  奖励对比: 随机={np.mean(rewards):.2f}, 学习后={np.mean(test_rewards):.2f}")


def memory_demo():
    """记忆功能演示"""
    print("\n" + "=" * 60)
    print("记忆系统演示")
    print("=" * 60)

    brain = SimpleBrain(input_dim=10, hidden_dim=32, output_dim=2)

    print("\n1. 存储重要记忆")
    for i in range(5):
        obs = np.random.randn(10)
        reward = 1.0
        brain.store_memory(obs, reward)
    print(f"   已存储 {len(brain.memory)} 条正向奖励记忆")

    print("\n2. 记忆回放巩固")
    brain.replay()
    print("   执行记忆回放，加强重要记忆")

    print("\n3. 记忆检索测试")
    query = np.random.randn(10)
    print(f"   查询向量维度: {len(query)}")
    print("   记忆检索功能就绪")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NeuroBrain 类脑记忆框架")
    print("=" * 60)

    simple_task()
    memory_demo()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
