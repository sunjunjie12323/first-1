"""
BrainEmbody 完整演示
展示具身智能类脑系统的完整功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainembody import RobotAgent, EmbodiedTrainer, TrainingConfig
from brainembody.core.llm_adapter import LLMConfig


def demo_single_episode():
    """演示1：单回合运行"""
    print("=" * 70)
    print("演示1：单回合具身智能任务")
    print("=" * 70)

    agent = RobotAgent(name="DemoRobot")

    print("\n任务：控制机器人在仿真环境中导航到目标位置")
    print(f"初始探索率: {agent.exploration_rate:.2f}")
    print(f"环境信息:\n{agent.simulator.render_text()}")

    input("\n按 Enter 开始演示...")

    result = agent.run_episode(max_steps=100, verbose=True)

    print(f"\n回合统计:")
    print(f"  总奖励: {result['total_reward']:.2f}")
    print(f"  步数: {result['steps']}")
    print(f"  成功: {'是' if result['success'] else '否'}")
    print(f"  探索率: {result['exploration_rate']:.2f}")

    stats = agent.get_stats()
    print(f"\n智能体统计:")
    print(f"  总步数: {stats['total_steps']}")
    print(f"  成功回合: {stats['successful_episodes']}")
    print(f"  记忆大小: {stats['memory_stats']['vector_store']['total_memories']}")


def demo_memory_system():
    """演示2：记忆系统"""
    print("\n" + "=" * 70)
    print("演示2：类脑记忆系统")
    print("=" * 70)

    from brainembody.memory.memory_system import MemorySystem

    memory = MemorySystem()

    print("\n存储记忆...")

    memory.encode_experience(
        "在仿真环境中成功绕过障碍物",
        outcome="成功",
        emotions=["成功", "高兴"]
    )

    memory.encode_experience(
        "尝试直接穿过障碍物导致碰撞",
        outcome="失败",
        emotions=["失败", "失望"]
    )

    memory.store_knowledge(
        concept="导航",
        definition="从一个位置移动到另一个位置的过程",
        relations=["定位", "路径规划", "避障"],
        examples=["室内导航", "室外导航", "机器人导航"]
    )

    print("\n检索记忆...")
    print(f"\n查询 '移动':")
    results = memory.recall("移动", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['content'][:50]}... (相关性: {r['score']:.3f})")

    print(f"\n查询 '导航知识':")
    results = memory.recall("导航", memory_type="knowledge", top_k=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['content']}")

    print(f"\n记忆摘要:")
    summary = memory.get_memory_summary()
    print(f"  向量存储: {summary['vector_store']['total_memories']} 条")
    print(f"  情景记忆: {summary['episodic_memory_size']} 条")
    print(f"  语义知识: {summary['semantic_knowledge_size']} 条")

    lessons = memory.reflect_on_experiences()
    if lessons:
        print(f"\n从经验中提取的教训:")
        for lesson in lessons:
            print(f"  - {lesson}")


def demo_llm_integration():
    """演示3：LLM集成"""
    print("\n" + "=" * 70)
    print("演示3：DeepSeek LLM 集成")
    print("=" * 70)

    from brainembody.core.brain_core import BrainCore

    config = LLMConfig()
    brain = BrainCore(config)

    print(f"\nLLM 模式: {'模拟' if brain.llm.mock_mode else '真实 API'}")

    print("\n测试思考能力...")
    thought = brain.think(
        task="分析当前环境并制定导航策略",
        context={"环境": "有障碍物的2D网格世界", "目标": "到达指定位置"}
    )
    print(f"\n思考结果:\n{thought[:200]}...")

    print("\n测试动作规划...")
    action = brain.plan_action(
        goal="安全高效地到达目标位置",
        current_state={
            "位置": [45.0, 55.0],
            "目标": [80.0, 20.0],
            "障碍物": 5,
            "距离": 50.0
        }
    )
    print(f"\n规划的动作:")
    print(f"  类型: {action.action_type}")
    print(f"  参数: {action.parameters}")
    print(f"  理由: {action.reasoning}")
    print(f"  置信度: {action.confidence:.2f}")

    print("\n测试反思能力...")
    reflection = brain.reflect(
        "这次导航尝试中，我成功地绕过了大部分障碍物，"
        "但在接近目标时犹豫了太久。下次应该更果断。"
    )
    print(f"\n反思结果:\n{reflection[:200]}...")


def demo_training():
    """演示4：快速训练"""
    print("\n" + "=" * 70)
    print("演示4：具身智能训练")
    print("=" * 70)

    agent = RobotAgent(name="TrainingBot")

    config = TrainingConfig(
        num_episodes=50,
        max_steps_per_episode=100,
        target_success_rate=0.6,
        save_frequency=10,
        eval_frequency=10
    )

    trainer = EmbodiedTrainer(agent, config)

    print("\n开始快速训练 (50 回合)...")
    print("这将训练机器人学习导航任务\n")

    result = trainer.train()

    print("\n" + "=" * 70)
    print("训练报告")
    print("=" * 70)
    print(f"状态: {result['status']}")
    print(f"总回合: {result['total_episodes']}")
    print(f"成功率: {result['overall_success_rate']*100:.1f}%")
    print(f"最佳奖励: {result['best_reward']:.2f} (回合 {result['best_episode']})")

    if result['training_duration_seconds']:
        print(f"训练时长: {result['training_duration_seconds']:.1f} 秒")

    print(f"\n指标摘要:")
    print(f"  平均奖励: {result['metrics_summary']['avg_reward']:.2f}")
    print(f"  平均步数: {result['metrics_summary']['avg_steps']:.1f}")
    print(f"  最高奖励: {result['metrics_summary']['max_reward']:.2f}")
    print(f"  最低奖励: {result['metrics_summary']['min_reward']:.2f}")

    trainer.plot_training_curve()

    print("\n评估训练后的智能体...")
    eval_result = trainer.evaluate(num_episodes=5)
    print(f"\n评估成功率: {eval_result['success_rate']*100:.1f}%")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("BrainEmbody - 具身智能类脑系统")
    print("结合 DeepSeek LLM + 创新记忆架构")
    print("=" * 70)

    demos = [
        ("演示1：单回合运行", demo_single_episode),
        ("演示2：记忆系统", demo_memory_system),
        ("演示3：LLM 集成", demo_llm_integration),
        ("演示4：快速训练", demo_training),
    ]

    print("\n可用的演示:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\n输入数字选择演示 (1-4)，或输入 'all' 运行全部: ", end="")

    choice = input().strip().lower()

    if choice == 'all':
        for name, demo_func in demos:
            try:
                demo_func()
            except Exception as e:
                print(f"\n演示出错: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= 4:
        demos[int(choice) - 1][1]()
    else:
        print("运行所有演示...")
        for name, demo_func in demos:
            try:
                demo_func()
            except Exception as e:
                print(f"\n演示出错: {e}")


if __name__ == "__main__":
    main()
