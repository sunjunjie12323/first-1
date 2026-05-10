from __future__ import annotations

import os
from typing import Optional


class Config:
    """
    NeuroCortex configuration.

    All settings can be overridden via environment variables
    with the NEURO_ prefix.
    """

    def __init__(self):
        self.llm_base_url = os.getenv("NEURO_LLM_BASE_URL", "http://localhost:11434")
        self.llm_model = os.getenv("NEURO_LLM_MODEL", "qwen2.5:7b")
        self.llm_api_type = os.getenv("NEURO_LLM_API_TYPE", "ollama")
        self.llm_embedding_model = os.getenv("NEURO_LLM_EMBEDDING_MODEL", "")

        self.embedding_dim = int(os.getenv("NEURO_EMBEDDING_DIM", "768"))

        self.consolidation_interval = int(os.getenv("NEURO_CONSOLIDATION_INTERVAL", "10"))

        self.data_dir = os.getenv("NEURO_DATA_DIR", "./neurocortex_data")

        self.system_identity = os.getenv(
            "NEURO_SYSTEM_IDENTITY",
            "你是一个具有类脑记忆系统的智能体，部署在机器人上。"
            "你像人脑一样记忆和回忆——记住对话的过程和情感，"
            "而不是死记硬背答案。",
        )

        self.server_host = os.getenv("NEURO_SERVER_HOST", "0.0.0.0")
        self.server_port = int(os.getenv("NEURO_SERVER_PORT", "8900"))

        self.log_level = os.getenv("NEURO_LOG_LEVEL", "INFO")

        self.hippocampus_max_traces = int(os.getenv("NEURO_HIPPOCAMPUS_MAX_TRACES", "10000"))
        self.hippocampus_pattern_separation = float(
            os.getenv("NEURO_HIPPOCAMPUS_PATTERN_SEPARATION", "0.3")
        )
        self.hippocampus_ca3_recurrent = float(
            os.getenv("NEURO_HIPPOCAMPUS_CA3_RECURRENT", "0.6")
        )

        self.neocortex_max_schemas = int(os.getenv("NEURO_NEOCORTEX_MAX_SCHEMAS", "5000"))
        self.neocortex_merge_threshold = float(
            os.getenv("NEURO_NEOCORTEX_MERGE_THRESHOLD", "0.75")
        )

        self.prefrontal_capacity = int(os.getenv("NEURO_PREFRONTAL_CAPACITY", "7"))

        self.amygdala_emotional_weight = float(
            os.getenv("NEURO_AMYGDALA_EMOTIONAL_WEIGHT", "0.4")
        )
        self.amygdala_novelty_weight = float(
            os.getenv("NEURO_AMYGDALA_NOVELTY_WEIGHT", "0.3")
        )

        self.recall_cue_threshold = float(os.getenv("NEURO_RECALL_CUE_THRESHOLD", "0.15"))
        self.recall_temperature = float(os.getenv("NEURO_RECALL_TEMPERATURE", "0.7"))
        self.recall_distortion_sensitivity = float(
            os.getenv("NEURO_RECALL_DISTORTION_SENSITIVITY", "0.3")
        )

        self.consolidation_batch_size = int(os.getenv("NEURO_CONSOLIDATION_BATCH", "5"))
        self.consolidation_temperature = float(
            os.getenv("NEURO_CONSOLIDATION_TEMPERATURE", "0.3")
        )

    def to_dict(self) -> dict:
        return {
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_api_type": self.llm_api_type,
            "embedding_dim": self.embedding_dim,
            "consolidation_interval": self.consolidation_interval,
            "data_dir": self.data_dir,
            "server_host": self.server_host,
            "server_port": self.server_port,
            "log_level": self.log_level,
        }
