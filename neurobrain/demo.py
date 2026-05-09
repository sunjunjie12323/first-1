"""
NeuroBrain 类脑记忆框架

仿生人脑的记忆系统，支持长期记忆、深度学习和具身智能训练

核心特性：
- 海马体-新皮层记忆系统
- 工作记忆、短期记忆、长期记忆
- 情感增强记忆（杏仁核）
- 突触可塑性和Hebbian学习
- STDP脉冲时间依赖可塑性
- 记忆巩固和睡眠重放
- 强化学习训练框架
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neurobrain import (
    Brain, BrainConfig,
    Hippocampus,
    Neocortex,
    Amygdala,
    WorkingMemory,
    ShortTermMemory,
    LongTermMemory,
    Neuron, NeuronCluster, NeuronType,
    Synapse, SynapseType, PlasticityRule,
    HebbianLearning, HebbianVariant,
    STDP, STDPVariant,
    BrainSimulator,
    ReinforcementTrainer
)


def demo():
    """快速演示"""
    print("=" * 60)
    print("NeuroBrain 类脑记忆框架演示")
    print("=" * 60)
    
    config = BrainConfig(
        input_dim=100,
        hidden_dims=[64, 32],
        output_dim=10,
        working_memory_capacity=7
    )
    
    brain = Brain(config)
    print("\n[1] 大脑初始化完成")
    print(f"    输入维度: {config.input_dim}")
    print(f"    隐藏层: {config.hidden_dims}")
    print(f"    输出维度: {config.output_dim}")
    
    print("\n[2] 记忆编码...")
    for i in range(5):
        input_data = np.random.randn(100)
        output, info = brain.process(input_data)
        print(f"    记忆 {i+1}: 情感={info['emotional_response']['dominant_emotion']}")
    
    print("\n[3] 记忆检索...")
    query = np.random.randn(100)
    memories = brain.recall(query, top_k=3)
    print(f"    检索到 {len(memories)} 条相关记忆")
    
    print("\n[4] 学习训练...")
    for epoch in range(5):
        input_data = np.random.randn(100)
        target = np.random.randn(10)
        result = brain.learn(input_data, target, reward=np.random.random())
        print(f"    Epoch {epoch+1}: Loss={result['loss']:.4f}")
    
    print("\n[5] 记忆巩固...")
    stats = brain.consolidate(sleep_cycles=2)
    print(f"    巩固记忆数: {stats['memories_consolidated']}")
    
    print("\n[6] 系统状态...")
    mem_stats = brain.get_memory_stats()
    print(f"    海马体记忆: {mem_stats['hippocampus']['memory_count']}")
    print(f"    新皮层记忆: {mem_stats['neocortex']['memory_count']}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    demo()
