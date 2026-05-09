"""
NeuroBrain 类脑记忆框架测试
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    STDP, STDPVariant
)


def test_brain_initialization():
    """测试大脑初始化"""
    config = BrainConfig(
        input_dim=100,
        hidden_dims=[64, 32],
        output_dim=10
    )
    brain = Brain(config)
    
    assert brain.config.input_dim == 100
    assert brain.hippocampus is not None
    assert brain.neocortex is not None
    assert brain.amygdala is not None
    print("✓ 大脑初始化测试通过")


def test_memory_processing():
    """测试记忆处理"""
    config = BrainConfig(input_dim=50, hidden_dims=[32, 16], output_dim=5)
    brain = Brain(config)
    
    input_data = np.random.randn(50)
    output, info = brain.process(input_data)
    
    assert output is not None
    assert 'emotional_response' in info
    assert 'memory_info' in info
    print("✓ 记忆处理测试通过")


def test_memory_recall():
    """测试记忆检索"""
    config = BrainConfig(input_dim=50, hidden_dims=[32, 16], output_dim=5)
    brain = Brain(config)
    
    for _ in range(5):
        input_data = np.random.randn(50)
        brain.process(input_data)
    
    query = np.random.randn(50)
    memories = brain.recall(query, top_k=3)
    
    assert len(memories) > 0
    print("✓ 记忆检索测试通过")


def test_learning():
    """测试学习功能"""
    config = BrainConfig(input_dim=50, hidden_dims=[32, 16], output_dim=5)
    brain = Brain(config)
    
    input_data = np.random.randn(50)
    target = np.random.randn(5)
    
    result = brain.learn(input_data, target, reward=1.0)
    
    assert 'loss' in result
    assert 'dopamine' in result
    print("✓ 学习功能测试通过")


def test_consolidation():
    """测试记忆巩固"""
    config = BrainConfig(input_dim=50, hidden_dims=[32, 16], output_dim=5)
    brain = Brain(config)
    
    for _ in range(10):
        input_data = np.random.randn(50)
        brain.process(input_data)
    
    stats = brain.consolidate(sleep_cycles=2)
    
    assert 'memories_consolidated' in stats
    print("✓ 记忆巩固测试通过")


def test_hippocampus():
    """测试海马体"""
    hippocampus = Hippocampus(input_dim=50, hidden_dim=32)
    
    input_data = np.random.randn(50)
    output, info = hippocampus.encode(input_data, emotional_weight=0.7)
    
    assert output is not None
    assert 'memory_id' in info
    
    memories = hippocampus.recall(input_data, top_k=3)
    assert isinstance(memories, list)
    print("✓ 海马体测试通过")


def test_neocortex():
    """测试新皮层"""
    neocortex = Neocortex(input_dim=32, hidden_dims=[16, 8], output_dim=5)
    
    input_data = np.random.randn(32)
    output, info = neocortex.process(input_data)
    
    assert output is not None
    assert 'prediction' in info
    
    success = neocortex.store(input_data, emotional_weight=0.5)
    assert success
    print("✓ 新皮层测试通过")


def test_amygdala():
    """测试杏仁核"""
    amygdala = Amygdala(emotional_weight=0.3)
    
    input_data = np.random.randn(50)
    response = amygdala.process(input_data)
    
    assert 'valence' in response
    assert 'arousal' in response
    assert 'dominant_emotion' in response
    assert 'weight' in response
    print("✓ 杏仁核测试通过")


def test_working_memory():
    """测试工作记忆"""
    wm = WorkingMemory(capacity=7)
    
    item_id = wm.store(np.random.randn(50), relevance=0.8)
    assert item_id is not None
    
    content = wm.retrieve(item_id)
    assert content is not None
    
    load = wm.get_load()
    assert 0 <= load <= 1
    print("✓ 工作记忆测试通过")


def test_short_term_memory():
    """测试短期记忆"""
    stm = ShortTermMemory(capacity=20, duration=30.0)
    
    item_id = stm.store(np.random.randn(50))
    assert item_id is not None
    
    result = stm.retrieve(item_id)
    assert result is not None
    
    stm.rehearse(item_id)
    print("✓ 短期记忆测试通过")


def test_long_term_memory():
    """测试长期记忆"""
    ltm = LongTermMemory(embedding_dim=50)
    
    epi_id = ltm.store_episodic(np.random.randn(50), context={'type': 'test'})
    assert epi_id is not None
    
    sem_id = ltm.store_semantic('concept1', np.random.randn(50))
    assert sem_id is not None
    
    memories = ltm.retrieve_episodic(np.random.randn(50), top_k=3)
    assert isinstance(memories, list)
    print("✓ 长期记忆测试通过")


def test_neuron():
    """测试神经元"""
    neuron = Neuron("test_neuron", NeuronType.EXCITATORY)
    
    neuron.inject_current(1.0)
    spiked = neuron.update()
    
    assert isinstance(spiked, bool)
    
    firing_rate = neuron.get_firing_rate()
    assert firing_rate >= 0
    print("✓ 神经元测试通过")


def test_neuron_cluster():
    """测试神经元集群"""
    cluster = NeuronCluster("test_cluster", 10, NeuronType.EXCITATORY)
    
    input_data = np.random.randn(10)
    spikes = cluster.process(input_data)
    
    assert len(spikes) == 10
    assert all(s in [0, 1] for s in spikes)
    print("✓ 神经元集群测试通过")


def test_synapse():
    """测试突触"""
    synapse = Synapse(
        "test_synapse",
        "pre_1",
        "post_1",
        plasticity_rule=PlasticityRule.STDP
    )
    
    current = synapse.transmit(time.time())
    assert current is not None
    
    synapse.update_plasticity(pre_spike=True, post_spike=True)
    weight = synapse.get_weight()
    assert 0 <= weight <= 1
    print("✓ 突触测试通过")


def test_hebbian_learning():
    """测试赫布学习"""
    hebbian = HebbianLearning()
    
    weights = np.random.randn(5, 10) * 0.1
    pre = np.random.randn(10)
    post = np.random.randn(5)
    
    delta_w = hebbian.compute_weight_update(weights, pre, post)
    assert delta_w.shape == weights.shape
    print("✓ 赫布学习测试通过")


def test_stdp():
    """测试STDP"""
    stdp = STDP()
    
    delta_w = stdp.compute_weight_update(
        current_weight=0.5,
        pre_spiked=True,
        post_spiked=True,
        dopamine=0.5
    )
    
    assert isinstance(delta_w, float)
    print("✓ STDP测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("NeuroBrain 测试套件")
    print("=" * 50 + "\n")
    
    test_brain_initialization()
    test_memory_processing()
    test_memory_recall()
    test_learning()
    test_consolidation()
    test_hippocampus()
    test_neocortex()
    test_amygdala()
    test_working_memory()
    test_short_term_memory()
    test_long_term_memory()
    test_neuron()
    test_neuron_cluster()
    test_synapse()
    test_hebbian_learning()
    test_stdp()
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_all_tests()
