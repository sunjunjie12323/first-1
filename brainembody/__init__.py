"""
BrainEmbody - 具身智能类脑系统
结合 DeepSeek LLM 和创新记忆架构 PHMEG 的具身智能框架
"""

from .core.brain_core import BrainCore
from .core.llm_adapter import LLMAdapter
from .memory.memory_system import MemorySystem
from .memory.vector_store import VectorStore
from .memory.phmeg import PHMEGMemory, EmotionalState, TaskTrajectory
from .embodied.simulator import EmbodiedSimulator
from .embodied.robot_agent import RobotAgent
from .training.trainer import EmbodiedTrainer

__version__ = "2.0.0"
__all__ = [
    "BrainCore",
    "LLMAdapter",
    "MemorySystem",
    "VectorStore",
    "PHMEGMemory",
    "EmotionalState",
    "TaskTrajectory",
    "EmbodiedSimulator",
    "RobotAgent",
    "EmbodiedTrainer",
]
