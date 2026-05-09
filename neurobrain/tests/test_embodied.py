"""
具身智能训练实际测试
验证NeuroBrain框架在真实任务上的学习能力
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurobrain.applications.embodied_env import create_environment
from neurobrain.applications.embodied_trainer import EmbodiedTrainer
from neurobrain import Brain, BrainConfig
from neurobrain.training.reinforcement import TrainerConfig, RLAlgorithm


def test_environment():
    """测试环境是否正常工作"""
    print("\n" + "=" * 60)
    print("测试1: 环境功能测试")
    print("=" * 60)

    env = create_environment("navigation")

    obs = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  观测维度: {len(obs)}")

    action = np.array([0.5, 0.1])
    next_obs, reward, done, info = env.step(action)
    print(f"✓ 执行动作成功")
    print(f"  奖励: {reward:.4f}")
    print(f"  完成: {done}")

    for _ in range(10):
        action = np.random.uniform(-1, 1, 2)
        env.step(action)

    print(f"✓ 随机策略运行成功 (10步)")

    return True


def test_brain_integration():
    """测试大脑与环境的集成"""
    print("\n" + "=" * 60)
    print("测试2: 大脑-环境集成测试")
    print("=" * 60)

    env = create_environment("navigation")

    brain_config = BrainConfig(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=2,
        working_memory_capacity=5
    )
    brain = Brain(brain_config)

    print(f"✓ 大脑初始化成功")

    obs = env.reset()

    for i in range(20):
        action_output, info = brain.process(obs)

        if len(action_output) >= 2:
            action = action_output[:2]
        else:
            action = np.array([0.5, 0.3])

        action = np.tanh(action)

        next_obs, reward, done, _ = env.step(action)

        brain.learn(obs, np.array([reward, reward]), reward=reward)

        obs = next_obs

        if done:
            obs = env.reset()

    print(f"✓ 20次交互完成")

    stats = brain.get_memory_stats()
    print(f"✓ 记忆系统状态:")
    print(f"  海马体记忆: {stats['hippocampus']['memory_count']}")
    print(f"  新皮层记忆: {stats['neocortex']['memory_count']}")

    return True


def test_training_loop():
    """测试完整训练循环"""
    print("\n" + "=" * 60)
    print("测试3: 训练循环测试")
    print("=" * 60)

    env = create_environment("navigation")
    env.max_steps = 50

    brain = Brain(BrainConfig(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        output_dim=2,
        working_memory_capacity=5
    ))

    train_config = TrainerConfig(
        algorithm=RLAlgorithm.DOPAMINE_MODULATED,
        learning_rate=0.005,
        exploration_rate=0.5,
        exploration_decay=0.98,
        batch_size=16,
        memory_size=1000
    )

    trainer = EmbodiedTrainer(
        environment=env,
        brain=brain,
        train_config=train_config
    )

    print(f"✓ 训练器初始化成功")
    print(f"  探索率: {trainer.trainer.config.exploration_rate}")

    rewards = []
    for episode in range(10):
        reward, loss, success = trainer._run_episode()
        rewards.append(reward)

        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(rewards[-5:])
            print(f"  Episode {episode + 1}: Reward = {reward:7.2f}, "
                  f"Avg = {avg_reward:7.2f}, Success = {success}")

    print(f"\n✓ 训练循环测试完成")
    print(f"  总回合: {len(rewards)}")
    print(f"  最终奖励: {rewards[-1]:.2f}")

    return True


def test_navigation_learning():
    """测试导航任务学习"""
    print("\n" + "=" * 60)
    print("测试4: 导航任务学习测试 (100回合)")
    print("=" * 60)

    np.random.seed(42)

    env = create_environment("navigation")
    env.max_steps = 100

    brain = Brain(BrainConfig(
        input_dim=20,
        hidden_dims=[128, 64, 32],
        output_dim=2,
        working_memory_capacity=7,
        emotional_weight=0.3
    ))

    train_config = TrainerConfig(
        algorithm=RLAlgorithm.DOPAMINE_MODULATED,
        learning_rate=0.002,
        exploration_rate=0.4,
        exploration_decay=0.99,
        min_exploration=0.05,
        batch_size=32,
        memory_size=5000
    )

    trainer = EmbodiedTrainer(
        environment=env,
        brain=brain,
        train_config=train_config
    )

    print(f"✓ 开始训练")

    episode_rewards = []
    successes = []

    for episode in range(100):
        reward, loss, success = trainer._run_episode()
        episode_rewards.append(reward)
        successes.append(1.0 if success else 0.0)

        if (episode + 1) % 20 == 0:
            recent_rewards = episode_rewards[-20:]
            recent_success = successes[-20:]
            print(f"  Episodes {episode - 19}-{episode + 1}: "
                  f"Avg Reward = {np.mean(recent_rewards):7.2f}, "
                  f"Success Rate = {np.mean(recent_success):.1%}")

    print(f"\n✓ 导航学习测试完成")

    final_20_rewards = episode_rewards[-20:]
    final_20_successes = successes[-20:]

    print(f"\n最终20回合结果:")
    print(f"  平均奖励: {np.mean(final_20_rewards):.2f}")
    print(f"  平均成功率: {np.mean(final_20_successes):.1%}")

    memory_stats = brain.get_memory_stats()
    print(f"\n记忆系统:")
    print(f"  海马体: {memory_stats['hippocampus']['memory_count']} 记忆")
    print(f"  新皮层: {memory_stats['neocortex']['memory_count']} 记忆")

    learning_improved = np.mean(final_20_rewards) > np.mean(episode_rewards[:20])

    if learning_improved:
        print(f"\n✓ 学习效果验证: 奖励从 {np.mean(episode_rewards[:20]):.2f} 提升到 {np.mean(final_20_rewards):.2f}")
    else:
        print(f"\n⚠ 学习效果: 奖励从 {np.mean(episode_rewards[:20]):.2f} 变为 {np.mean(final_20_rewards):.2f}")

    return True


def test_consolidation():
    """测试记忆巩固"""
    print("\n" + "=" * 60)
    print("测试5: 记忆巩固测试")
    print("=" * 60)

    env = create_environment("navigation")

    brain = Brain(BrainConfig(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=2
    ))

    for _ in range(50):
        obs = env.reset()
        for _ in range(20):
            action = np.random.uniform(-1, 1, 2)
            obs, reward, done, _ = env.step(action)
            brain.process(obs)
            if done:
                break

    stats_before = brain.get_memory_stats()
    print(f"巩固前: 海马体 {stats_before['hippocampus']['memory_count']} 记忆")

    consolidation_stats = brain.consolidate(sleep_cycles=3)

    stats_after = brain.get_memory_stats()
    print(f"巩固后: 海马体 {stats_after['hippocampus']['memory_count']} 记忆, "
          f"新皮层 {stats_after['neocortex']['memory_count']} 记忆")

    print(f"巩固统计: {consolidation_stats['memories_consolidated']} 记忆被巩固")

    return True


def test_evaluation():
    """测试评估功能"""
    print("\n" + "=" * 60)
    print("测试6: 评估功能测试")
    print("=" * 60)

    env = create_environment("navigation")

    brain = Brain(BrainConfig(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=2
    ))

    train_config = TrainerConfig(
        exploration_rate=0.0
    )
    trainer = EmbodiedTrainer(
        environment=env,
        brain=brain,
        train_config=train_config
    )

    for _ in range(30):
        trainer._run_episode()

    print(f"✓ 完成30回合训练")

    results = trainer.evaluate(num_episodes=10)

    print(f"\n评估结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  成功率: {results['success_rate']:.1%}")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("NeuroBrain 具身智能训练 - 完整测试套件")
    print("=" * 70)

    all_passed = True

    try:
        test_environment()
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        all_passed = False

    try:
        test_brain_integration()
    except Exception as e:
        print(f"✗ 大脑集成测试失败: {e}")
        all_passed = False

    try:
        test_training_loop()
    except Exception as e:
        print(f"✗ 训练循环测试失败: {e}")
        all_passed = False

    try:
        test_navigation_learning()
    except Exception as e:
        print(f"✗ 导航学习测试失败: {e}")
        all_passed = False

    try:
        test_consolidation()
    except Exception as e:
        print(f"✗ 记忆巩固测试失败: {e}")
        all_passed = False

    try:
        test_evaluation()
    except Exception as e:
        print(f"✗ 评估测试失败: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("所有测试完成！✓")
    else:
        print("部分测试失败，请检查。")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
