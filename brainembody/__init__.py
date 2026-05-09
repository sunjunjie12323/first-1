"""
BrainEmbody - 具身智能类脑系统
结合 DeepSeek LLM 和创新记忆架构的具身智能框架
"""

from .core.brain_core import BrainCore
from .core.llm_adapter import LLMAdapter
from .memory.memory_system import MemorySystem
from .memory.vector_store import VectorStore
from .embodied.simulator import EmbodiedSimulator
from .embodied.robot_agent import RobotAgent
from .training.trainer import EmbodiedTrainer

__version__ = "1.0.0"
__all__ = [
    "BrainCore",
    "LLMAdapter",
    "MemorySystem",
    "VectorStore",
    "EmbodiedSimulator",
    "RobotAgent",
    "EmbodiedTrainer",
]
