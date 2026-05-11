from __future__ import annotations

import os
from typing import Optional


class Config:
    LLM_BASE_URL: str = os.environ.get("NEURO_LLM_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.environ.get("NEURO_LLM_MODEL", "llama3.2")
    EMBEDDING_MODEL: str = os.environ.get("NEURO_EMBEDDING_MODEL", "nomic-embed-text")
    API_TYPE: str = os.environ.get("NEURO_API_TYPE", "ollama")
    API_KEY: Optional[str] = os.environ.get("NEURO_API_KEY", None)
    LLM_TIMEOUT: float = float(os.environ.get("NEURO_LLM_TIMEOUT", "60.0"))

    EPSILON: float = float(os.environ.get("NEURO_EPSILON", "0.3"))
    WORKING_MEMORY_CAPACITY: int = int(os.environ.get("NEURO_WORKING_MEMORY_CAPACITY", "7"))
    MAX_TRACES: int = int(os.environ.get("NEURO_MAX_TRACES", "10000"))
    DECAY_THRESHOLD: float = float(os.environ.get("NEURO_DECAY_THRESHOLD", "0.01"))

    CONSOLIDATION_THRESHOLD: float = float(os.environ.get("NEURO_CONSOLIDATION_THRESHOLD", "0.3"))
    MAX_CONSOLIDATION_PER_CYCLE: int = int(os.environ.get("NEURO_MAX_CONSOLIDATION_PER_CYCLE", "50"))

    SPREAD_DEPTH: int = int(os.environ.get("NEURO_SPREAD_DEPTH", "2"))
    ACTIVATION_THRESHOLD: float = float(os.environ.get("NEURO_ACTIVATION_THRESHOLD", "0.15"))
    ALPHA_FULL: float = float(os.environ.get("NEURO_ALPHA_FULL", "0.7"))
    ALPHA_GIST: float = float(os.environ.get("NEURO_ALPHA_GIST", "0.3"))

    MERGE_SIMILARITY_THRESHOLD: float = float(os.environ.get("NEURO_MERGE_SIMILARITY_THRESHOLD", "0.75"))

    EMOTIONAL_WEIGHT: float = float(os.environ.get("NEURO_EMOTIONAL_WEIGHT", "0.35"))
    NOVELTY_WEIGHT: float = float(os.environ.get("NEURO_NOVELTY_WEIGHT", "0.25"))
    SOCIAL_WEIGHT: float = float(os.environ.get("NEURO_SOCIAL_WEIGHT", "0.15"))
    GOAL_WEIGHT: float = float(os.environ.get("NEURO_GOAL_WEIGHT", "0.25"))

    SERVER_HOST: str = os.environ.get("NEURO_SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.environ.get("NEURO_SERVER_PORT", "8000"))

    STATE_DIR: str = os.environ.get("NEURO_STATE_DIR", "/tmp/neurocortex_state")

    LOG_LEVEL: str = os.environ.get("NEURO_LOG_LEVEL", "INFO")
