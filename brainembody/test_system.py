"""
快速测试脚本
验证 BrainEmbody 系统核心功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainembody import RobotAgent
from brainembody.memory.memory_system import MemorySystem
from brainembody.core.llm_adapter import LLMAdapter


def test_memory_system():
    """测试记忆系统"""
    print("\n【测试1】记忆系统")
    print("-" * 40)

    memory = MemorySystem()

    memory.encode_experience(
        "机器人成功导航到目标",
        outcome="成功",
        emotions=["成功", "高兴"]
    )

    results = memory.recall("导航", top_k=5)
    print(f"✓ 存储和检索记忆: {len(results)} 条")

    memory.store_knowledge(
        concept="具身智能",
        definition="智能体通过身体与环境交互的能力"
    )
    print(f"✓ 存储知识: {len(memory.semantic_knowledge)} 条")


def test_llm_adapter():
    """测试 LLM 适配器"""
    print("\n【测试2】LLM 适配器")
    print("-" * 40)

    adapter = LLMAdapter()

    print(f"模式: {'模拟' if adapter.mock_mode else 'API'}")

    response = adapter.chat("你好，请介绍自己")
    print(f"✓ LLM 响应: {response[:50]}...")

    stats = adapter.get_stats()
    print(f"✓ 统计信息: {stats}")


def test_simulator():
    """测试仿真环境"""
    print("\n【测试3】仿真环境")
    print("-" * 40)

    from brainembody.embodied.simulator import EmbodiedSimulator

    sim = EmbodiedSimulator()

    observation = sim.get_observation()
    print(f"✓ 环境观测: 位置={observation['position']}")

    obs, reward, done = sim.step("move_forward", {"distance": 1.0})
    print(f"✓ 执行动作: 奖励={reward:.2f}, 完成={done}")

    obs, reward, done = sim.step("turn", {"angle": 0.5})
    print(f"✓ 转向动作: 奖励={reward:.2f}, 完成={done}")


def test_robot_agent():
    """测试机器人智能体"""
    print("\n【测试4】机器人智能体")
    print("-" * 40)

    agent = RobotAgent(name="TestRobot")

    result = agent.run_episode(max_steps=10, verbose=False)
    print(f"✓ 运行回合: 奖励={result['total_reward']:.2f}, 步数={result['steps']}")

    stats = agent.get_stats()
    print(f"✓ 智能体统计: 总步数={stats['total_steps']}")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("BrainEmbody 系统测试")
    print("=" * 60)

    try:
        test_memory_system()
        test_llm_adapter()
        test_simulator()
        test_robot_agent()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
