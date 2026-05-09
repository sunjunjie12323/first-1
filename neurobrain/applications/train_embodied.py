"""
具身智能训练主程序
运行完整的训练和评估流程
"""

import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurobrain.applications.embodied_env import create_environment, EmbodiedEnvironment, TaskType
from neurobrain.applications.embodied_trainer import EmbodiedTrainer
from neurobrain import Brain, BrainConfig
from neurobrain.training.reinforcement import TrainerConfig, RLAlgorithm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='具身智能训练')

    parser.add_argument('--task', type=str, default='navigation',
                        choices=['navigation', 'maze', 'manipulation'],
                        help='任务类型')
    parser.add_argument('--episodes', type=int, default=500,
                        help='训练回合数')
    parser.add_argument('--eval_episodes', type=int, default=20,
                        help='评估回合数')
    parser.add_argument('--world_size', type=float, nargs=2, default=[10.0, 10.0],
                        help='世界大小')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点目录')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='加载检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')

    return parser.parse_args()


def create_brain(input_dim: int = 20, output_dim: int = 2) -> Brain:
    """创建大脑"""
    config = BrainConfig(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=output_dim,
        working_memory_capacity=7,
        short_term_memory_duration=30.0,
        long_term_memory_threshold=0.6,
        emotional_weight=0.3,
        plasticity_threshold=0.5
    )
    return Brain(config)


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "=" * 70)
    print("NeuroBrain 具身智能训练系统")
    print("=" * 70)
    print(f"任务: {args.task}")
    print(f"训练回合: {args.episodes}")
    print(f"世界大小: {args.world_size}")
    print("=" * 70 + "\n")

    env = create_environment(args.task)
    env.world_size = tuple(args.world_size)

    brain = create_brain(input_dim=20, output_dim=2)

    train_config = TrainerConfig(
        algorithm=RLAlgorithm.DOPAMINE_MODULATED,
        learning_rate=0.001,
        exploration_rate=0.3,
        exploration_decay=0.995,
        min_exploration=0.01,
        batch_size=32,
        memory_size=10000,
        dopamine_baseline=0.5
    )

    trainer = EmbodiedTrainer(
        environment=env,
        brain=brain,
        train_config=train_config
    )

    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    if args.eval_only:
        print("评估模式")
        results = trainer.evaluate(num_episodes=args.eval_episodes)
    else:
        print("训练模式")
        history = trainer.train(
            num_episodes=args.episodes,
            save_interval=100,
            checkpoint_dir=args.checkpoint_dir
        )

        print("\n进行最终评估...")
        results = trainer.evaluate(num_episodes=args.eval_episodes)

        print("\n保存训练历史...")
        history_file = os.path.join(args.checkpoint_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump({
                'reward_history': history['episode_rewards'][-100:],
                'best_reward': history['best_reward'],
                'final_success_rate': history['final_success_rate']
            }, f)
        print(f"历史已保存到: {history_file}")

    print("\n训练流程完成!")
    return results


if __name__ == "__main__":
    import json
    main()
