"""
NeuroBrain 类脑记忆框架使用示例
演示如何使用框架进行记忆编码、检索和学习
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurobrain import Brain, BrainConfig
from neurobrain.core.brain import BrainState
from neurobrain.training.simulator import BrainSimulator, SimulationConfig
from neurobrain.training.reinforcement import ReinforcementTrainer, TrainerConfig, RLAlgorithm


def basic_memory_example():
    """基本记忆功能示例"""
    print("=" * 60)
    print("基本记忆功能示例")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=100,
        hidden_dims=[64, 32],
        output_dim=10,
        working_memory_capacity=7
    )
    brain = Brain(config)
    
    print("\n1. 编码和存储记忆")
    for i in range(5):
        input_data = np.random.randn(100)
        output, info = brain.process(input_data)
        print(f"   记忆 {i+1}: 情感权重={info['emotional_response']['weight']:.3f}")
    
    print("\n2. 记忆检索")
    query = np.random.randn(100)
    memories = brain.recall(query, top_k=3)
    print(f"   检索到 {len(memories)} 条记忆")
    for i, mem in enumerate(memories):
        print(f"   - 记忆 {i+1}: 强度={mem.get('strength', 0):.3f}")
    
    print("\n3. 记忆统计")
    stats = brain.get_memory_stats()
    print(f"   海马体记忆数: {stats['hippocampus']['memory_count']}")
    print(f"   新皮层记忆数: {stats['neocortex']['memory_count']}")


def learning_example():
    """学习功能示例"""
    print("\n" + "=" * 60)
    print("学习功能示例")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=50,
        hidden_dims=[32, 16],
        output_dim=5
    )
    brain = Brain(config)
    
    print("\n1. 训练阶段")
    losses = []
    for epoch in range(20):
        input_data = np.random.randn(50)
        target = np.random.randn(5)
        
        result = brain.learn(input_data, target, reward=np.random.random())
        losses.append(result['loss'])
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}: Loss={result['loss']:.4f}, "
                  f"Dopamine={result['dopamine']:.3f}")
    
    print("\n2. 记忆巩固")
    consolidation_stats = brain.consolidate(sleep_cycles=2)
    print(f"   巩固的记忆数: {consolidation_stats['memories_consolidated']}")
    print(f"   突触变化: {consolidation_stats['synaptic_changes']}")


def emotional_memory_example():
    """情感记忆示例"""
    print("\n" + "=" * 60)
    print("情感记忆示例")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=50,
        hidden_dims=[32, 16],
        output_dim=5,
        emotional_weight=0.4
    )
    brain = Brain(config)
    
    print("\n1. 存储带情感标记的记忆")
    
    positive_inputs = [np.random.randn(50) * 0.5 + 0.5 for _ in range(3)]
    negative_inputs = [np.random.randn(50) * 0.5 - 0.5 for _ in range(3)]
    
    for i, inp in enumerate(positive_inputs):
        output, info = brain.process(inp)
        brain.amygdala.update_emotional_state(reward=0.8)
        print(f"   正面记忆 {i+1}: 情感={info['emotional_response']['dominant_emotion']}")
    
    for i, inp in enumerate(negative_inputs):
        output, info = brain.process(inp)
        brain.amygdala.update_emotional_state(reward=-0.5)
        print(f"   负面记忆 {i+1}: 情感={info['emotional_response']['dominant_emotion']}")
    
    print("\n2. 情感状态")
    amygdala_stats = brain.amygdala.get_stats()
    print(f"   当前效价: {amygdala_stats['current_valence']:.3f}")
    print(f"   当前唤醒度: {amygdala_stats['current_arousal']:.3f}")
    print(f"   主导情感: {amygdala_stats['dominant_emotion']}")


def simulation_training_example():
    """仿真训练示例"""
    print("\n" + "=" * 60)
    print("仿真训练示例")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=50,
        hidden_dims=[32, 16],
        output_dim=5
    )
    brain = Brain(config)
    
    sim_config = SimulationConfig(
        max_episodes=10,
        max_steps_per_episode=20,
        consolidation_interval=10
    )
    simulator = BrainSimulator(brain, sim_config)
    
    def input_generator():
        return np.random.randn(50)
    
    def target_generator():
        return np.random.randn(5)
    
    print("\n1. 运行仿真训练")
    results = simulator.run_training(
        input_generator=input_generator,
        target_generator=target_generator
    )
    
    print("\n2. 训练结果")
    for i, result in enumerate(results[:5]):
        print(f"   Episode {result.episode}: "
              f"Reward={result.total_reward:.3f}, "
              f"Memories={result.memories_formed}")
    
    print("\n3. 训练统计")
    stats = simulator.get_training_stats()
    print(f"   总回合数: {stats['episodes']}")
    print(f"   平均奖励: {stats['avg_reward']:.3f}")
    print(f"   探索率: {stats['exploration_rate']:.3f}")


def reinforcement_learning_example():
    """强化学习示例"""
    print("\n" + "=" * 60)
    print("强化学习示例")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=50,
        hidden_dims=[32, 16],
        output_dim=5
    )
    brain = Brain(config)
    
    trainer_config = TrainerConfig(
        algorithm=RLAlgorithm.DOPAMINE_MODULATED,
        learning_rate=0.01,
        exploration_rate=0.2
    )
    trainer = ReinforcementTrainer(brain, trainer_config)
    
    print("\n1. 强化学习训练")
    for episode in range(10):
        state = np.random.randn(50)
        action = trainer.select_action(state)
        next_state = np.random.randn(50)
        reward = np.random.random()
        
        result = trainer.learn(state, action, reward, next_state, done=False)
        
        if (episode + 1) % 3 == 0:
            print(f"   Episode {episode+1}: "
                  f"Action={action}, Reward={reward:.3f}, "
                  f"Dopamine={result['dopamine']:.3f}")
    
    print("\n2. 训练统计")
    stats = trainer.get_training_stats()
    print(f"   总步数: {stats['steps']}")
    print(f"   总奖励: {stats['total_reward']:.3f}")
    print(f"   多巴胺水平: {stats['dopamine_level']:.3f}")


def memory_consolidation_example():
    """记忆巩固示例"""
    print("\n" + "=" * 60)
    print("记忆巩固示例")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=50,
        hidden_dims=[32, 16],
        output_dim=5
    )
    brain = Brain(config)
    
    print("\n1. 存储大量记忆")
    for i in range(20):
        input_data = np.random.randn(50)
        emotional_weight = np.random.random()
        output, info = brain.process(input_data)
    
    stats_before = brain.get_memory_stats()
    print(f"   巩固前 - 海马体: {stats_before['hippocampus']['memory_count']}, "
          f"新皮层: {stats_before['neocortex']['memory_count']}")
    
    print("\n2. 执行记忆巩固（模拟睡眠）")
    brain.set_state(BrainState.DEEP_SLEEP)
    consolidation_stats = brain.consolidate(sleep_cycles=3)
    brain.set_state(BrainState.AWAKE)
    
    print(f"   巩固的记忆数: {consolidation_stats['memories_consolidated']}")
    print(f"   海马体到新皮层迁移: {consolidation_stats['hippocampus_to_neocortex']}")
    
    stats_after = brain.get_memory_stats()
    print(f"   巩固后 - 海马体: {stats_after['hippocampus']['memory_count']}, "
          f"新皮层: {stats_after['neocortex']['memory_count']}")


def save_load_example():
    """保存和加载示例"""
    print("\n" + "=" * 60)
    print("保存和加载示例")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=50,
        hidden_dims=[32, 16],
        output_dim=5
    )
    brain = Brain(config)
    
    print("\n1. 训练并存储记忆")
    for i in range(10):
        input_data = np.random.randn(50)
        brain.process(input_data)
    
    stats_before = brain.get_memory_stats()
    print(f"   记忆数: {stats_before['hippocampus']['memory_count']}")
    
    print("\n2. 保存状态")
    brain.save_state("brain_state.pkl")
    print("   已保存到 brain_state.pkl")
    
    print("\n3. 加载状态到新大脑")
    new_brain = Brain(config)
    new_brain.load_state("brain_state.pkl")
    
    stats_after = new_brain.get_memory_stats()
    print(f"   加载后记忆数: {stats_after['hippocampus']['memory_count']}")
    
    import os
    if os.path.exists("brain_state.pkl"):
        os.remove("brain_state.pkl")
        print("\n4. 清理临时文件")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("NeuroBrain 类脑记忆框架 - 使用示例")
    print("=" * 60)
    
    basic_memory_example()
    learning_example()
    emotional_memory_example()
    memory_consolidation_example()
    simulation_training_example()
    reinforcement_learning_example()
    save_load_example()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
