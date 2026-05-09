"""
NeuroBrain - 类脑记忆框架
仿生人脑的记忆系统，支持长期记忆、深度学习和具身智能训练
"""

from .core.brain import Brain, BrainConfig, BrainState
from .core.hippocampus import Hippocampus
from .core.neocortex import Neocortex
from .core.amygdala import Amygdala, EmotionType
from .memory.working_memory import WorkingMemory
from .memory.short_term_memory import ShortTermMemory
from .memory.long_term_memory import LongTermMemory
from .memory.dialog_memory import DialogMemory
from .memory.thinking_memory import ThinkingMemory
from .neurons.synapse import Synapse, SynapticPlasticity, SynapseType, PlasticityRule
from .neurons.neuron import Neuron, NeuronCluster, NeuronType
from .learning.hebbian import HebbianLearning, HebbianVariant
from .learning.stdp import STDP, STDPVariant
from .training.simulator import BrainSimulator, SimulationConfig
from .training.reinforcement import ReinforcementTrainer, TrainerConfig, RLAlgorithm

__version__ = "1.1.0"
__all__ = [
    "Brain",
    "BrainConfig",
    "BrainState",
    "Hippocampus",
    "Neocortex",
    "Amygdala",
    "EmotionType",
    "WorkingMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "DialogMemory",
    "ThinkingMemory",
    "Synapse",
    "SynapticPlasticity",
    "SynapseType",
    "PlasticityRule",
    "Neuron",
    "NeuronCluster",
    "NeuronType",
    "HebbianLearning",
    "HebbianVariant",
    "STDP",
    "STDPVariant",
    "BrainSimulator",
    "SimulationConfig",
    "ReinforcementTrainer",
    "TrainerConfig",
    "RLAlgorithm",
]
