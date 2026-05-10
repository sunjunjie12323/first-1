from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .amygdala import Amygdala
from .basal_forebrain import BasalForebrain
from .consolidation import ConsolidationEngine
from .hippocampus import Hippocampus
from .llm_engine import LLMEngine
from .memory_trace import (
    ContextTag,
    EpisodicTrace,
    MemoryPhase,
    NeuromodulatoryState,
    ReconstructedMemory,
    SemanticSchema,
)
from .neocortex import Neocortex
from .prefrontal_cortex import PrefrontalCortex
from .reconstructive_recall import ReconstructiveRecall

logger = logging.getLogger(__name__)


class BrainSystem:
    """
    The whole-brain orchestrator for NeuroCortex.

    Coordinates all brain-region modules to implement the complete
    perception → encoding → recall → response → consolidation cycle.

    Information flow (mirroring the human brain):
    ┌─────────────┐
    │   INPUT      │ (user message / robot sensor)
    └──────┬──────┘
           │
    ┌──────▼──────┐     ┌──────────────┐
    │  Basal       │────▶│   Amygdala   │
    │  Forebrain   │     │ (importance) │
    │ (novelty)    │     └──────┬───────┘
    └──────┬──────┘            │
           │                   │
    ┌──────▼───────────────────▼──────┐
    │        HIPPOCAMPUS              │
    │   (episodic encoding +          │
    │    pattern completion)          │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │   RECONSTRUCTIVE RECALL         │  ◀── CORE INNOVATION
    │   (fragment assembly +          │
    │    LLM reconstruction)         │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │   PREFRONTAL CORTEX             │
    │   (working memory +             │
    │    attention + goal tracking)   │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │        LLM ENGINE               │
    │   (reasoning + generation)      │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │        OUTPUT                   │
    │   (response to user/robot)      │
    └─────────────────────────────────┘

    During idle periods (sleep):
    ┌──────────────┐     ┌──────────────┐
    │  HIPPOCAMPUS │────▶│  NEOCORTEX   │
    │  (replay)    │     │ (consolidate)│
    └──────────────┘     └──────────────┘
    """

    def __init__(
        self,
        llm_base_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5:7b",
        llm_api_type: str = "ollama",
        embedding_dim: int = 768,
        consolidation_interval: int = 10,
        data_dir: str = "./neurocortex_data",
        system_identity: str = "",
    ):
        self.data_dir = data_dir
        self.system_identity = system_identity or (
            "你是一个具有类脑记忆系统的智能体，部署在机器人上。"
            "你像人脑一样记忆和回忆——记住对话的过程和情感，"
            "而不是死记硬背答案。"
        )

        self.llm_engine = LLMEngine(
            base_url=llm_base_url,
            model=llm_model,
            api_type=llm_api_type,
        )

        self.hippocampus = Hippocampus(embedding_dim=embedding_dim)
        self.neocortex = Neocortex(embedding_dim=embedding_dim)
        self.prefrontal = PrefrontalCortex()
        self.amygdala = Amygdala()
        self.basal_forebrain = BasalForebrain()
        self.reconstructive_recall = ReconstructiveRecall()
        self.consolidation_engine = ConsolidationEngine()

        self.consolidation_interval = consolidation_interval
        self._interaction_count = 0
        self._last_consolidation = 0

        os.makedirs(data_dir, exist_ok=True)

    async def process_input(
        self,
        user_message: str,
        source: str = "user",
        context: Optional[Dict[str, str]] = None,
        emotional_feedback: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Process a user input through the complete brain pipeline.

        This is the main entry point for the system. It implements:
        1. Perception → novelty detection
        2. Encoding → hippocampal + amygdalar processing
        3. Recall → reconstructive memory retrieval
        4. Response → LLM generation with memory context
        5. Learning → feedback-driven neuromodulation
        """
        self._interaction_count += 1
        self.prefrontal.advance_turn()

        logger.info(f"Processing input #{self._interaction_count}: {user_message[:50]}...")

        # === PHASE 1: Perception & Novelty Detection ===
        embedding = await self._get_embedding(user_message)

        existing_embeddings = self._get_existing_embeddings()
        novelty = self.basal_forebrain.compute_novelty(embedding, existing_embeddings)

        # === PHASE 2: Importance Assessment (Amygdala) ===
        importance, emotional_valence = self.amygdala.assess_importance(
            content=user_message,
            emotional_intensity=abs(emotional_feedback),
            novelty_score=novelty,
            source=source,
            current_goals=self.prefrontal.goals,
        )

        # === PHASE 3: Hippocampal Encoding ===
        context_tag = ContextTag(
            interlocutor=source,
            **(context or {}),
        )

        encoding_gate = self.basal_forebrain.encoding_gate
        effective_importance = importance * encoding_gate

        trace = self.hippocampus.encode(
            content=user_message,
            embedding=embedding,
            context=context_tag,
            importance=effective_importance,
            emotional_valence=emotional_valence,
            source=source,
            novelty_score=novelty,
            reward_score=self.basal_forebrain.state.reward_signal,
        )

        # === PHASE 4: Reconstructive Recall ===
        reconstructed_memory = await self.reconstructive_recall.recall(
            query=user_message,
            query_embedding=embedding,
            hippocampus=self.hippocampus,
            neocortex=self.neocortex,
            llm_engine=self.llm_engine,
            emotional_valence=emotional_valence,
            current_context=self.prefrontal.get_relevant_context(user_message),
            neuromodulatory_state=self.basal_forebrain.state,
        )

        # === PHASE 5: Working Memory Update ===
        self.prefrontal.update_working_memory(
            content=user_message,
            source=source,
            attention_weight=effective_importance,
        )
        self.prefrontal.focus_attention(user_message)

        # === PHASE 6: LLM Response Generation ===
        working_memory_context = self.prefrontal.get_relevant_context(user_message)

        response = await self.llm_engine.generate_with_memory_context(
            user_message=user_message,
            reconstructed_memory=reconstructed_memory.reconstructed_narrative,
            working_memory_context=working_memory_context,
            system_prompt=self.system_identity,
        )

        # === PHASE 7: Encode Response ===
        response_embedding = await self._get_embedding(response)
        response_importance = effective_importance * 0.7

        self.hippocampus.encode(
            content=f"[回应] {response}",
            embedding=response_embedding,
            context=context_tag,
            importance=response_importance,
            emotional_valence=emotional_valence * 0.5,
            source="self",
            novelty_score=novelty * 0.3,
        )

        # === PHASE 8: Neuromodulatory Update ===
        self.basal_forebrain.compute_reward(emotional_feedback)
        self.basal_forebrain.update_social_signal(source != "system")
        self.basal_forebrain.decay_to_baseline()

        # === PHASE 9: Check Consolidation Need ===
        if self._should_consolidate():
            asyncio.create_task(self._run_consolidation())

        return {
            "response": response,
            "memory_trace_id": trace.trace_id,
            "reconstruction_id": reconstructed_memory.reconstruction_id,
            "importance": effective_importance,
            "novelty": novelty,
            "emotional_valence": emotional_valence,
            "memory_confidence": reconstructed_memory.confidence,
            "distortion_score": reconstructed_memory.distortion_score,
            "neuromodulatory_state": self.basal_forebrain.state.to_dict(),
        }

    async def recall_memory(
        self,
        query: str,
        source: str = "user",
    ) -> Dict[str, Any]:
        """
        Explicitly recall a memory without generating a response.
        Useful for testing memory properties.
        """
        embedding = await self._get_embedding(query)

        reconstructed = await self.reconstructive_recall.recall(
            query=query,
            query_embedding=embedding,
            hippocampus=self.hippocampus,
            neocortex=self.neocortex,
            llm_engine=self.llm_engine,
            current_context=self.prefrontal.get_relevant_context(query),
        )

        return {
            "query": query,
            "reconstructed_narrative": reconstructed.reconstructed_narrative,
            "confidence": reconstructed.confidence,
            "distortion_score": reconstructed.distortion_score,
            "source_traces": reconstructed.source_traces,
            "source_schemas": reconstructed.source_schemas,
        }

    async def force_consolidation(self) -> Dict[str, Any]:
        """Manually trigger a consolidation round."""
        return await self._run_consolidation()

    def set_goals(self, goals: List[str]) -> None:
        for goal in goals:
            self.prefrontal.update_goals(goal)

    async def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all brain regions."""
        llm_health = await self.llm_engine.check_health()

        return {
            "interaction_count": self._interaction_count,
            "llm": llm_health,
            "hippocampus": self.hippocampus.get_status(),
            "neocortex": self.neocortex.get_status(),
            "prefrontal_cortex": self.prefrontal.get_status(),
            "amygdala": self.amygdala.get_status(),
            "basal_forebrain": self.basal_forebrain.get_status(),
            "consolidation_rounds": self.consolidation_engine.consolidation_count,
        }

    async def save_state(self) -> None:
        """Save the complete brain state to disk."""
        state = {
            "interaction_count": self._interaction_count,
            "last_consolidation": self._last_consolidation,
            "hippocampus_traces": [
                self._serialize_trace(t)
                for t in self.hippocampus.get_active_traces()
            ],
            "neocortex_schemas": [
                self._serialize_schema(s)
                for s in self.neocortex.get_all_schemas()
            ],
            "prefrontal_state": self.prefrontal.get_status(),
            "saved_at": datetime.now().isoformat(),
        }

        path = os.path.join(self.data_dir, "brain_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        logger.info(f"Brain state saved to {path}")

    async def load_state(self) -> None:
        """Load brain state from disk."""
        path = os.path.join(self.data_dir, "brain_state.json")
        if not os.path.exists(path):
            logger.info("No saved state found, starting fresh")
            return

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self._interaction_count = state.get("interaction_count", 0)
        self._last_consolidation = state.get("last_consolidation", 0)

        logger.info(
            f"Brain state loaded: {self._interaction_count} interactions, "
            f"{len(state.get('hippocampus_traces', []))} traces, "
            f"{len(state.get('neocortex_schemas', []))} schemas"
        )

    async def shutdown(self) -> None:
        """Gracefully shut down the brain system."""
        await self.save_state()
        await self.llm_engine.close()
        logger.info("Brain system shut down gracefully")

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with fallback to hash-based embedding."""
        raw_embedding = await self.llm_engine.get_embedding(text)

        if raw_embedding and len(raw_embedding) > 0:
            embedding = np.array(raw_embedding, dtype=np.float32)
            if embedding.shape[0] != self.hippocampus.embedding_dim:
                if embedding.shape[0] > self.hippocampus.embedding_dim:
                    embedding = embedding[: self.hippocampus.embedding_dim]
                else:
                    padded = np.zeros(self.hippocampus.embedding_dim, dtype=np.float32)
                    padded[: embedding.shape[0]] = embedding
                    embedding = padded
            return embedding

        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Fallback: generate a deterministic pseudo-embedding from text hash."""
        import hashlib

        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        embedding = np.frombuffer(hash_bytes, dtype=np.float32).copy()

        while embedding.shape[0] < self.hippocampus.embedding_dim:
            hash_bytes = hashlib.sha256(hash_bytes).digest()
            more = np.frombuffer(hash_bytes, dtype=np.float32).copy()
            embedding = np.concatenate([embedding, more])

        embedding = embedding[: self.hippocampus.embedding_dim]
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding

    def _get_existing_embeddings(self) -> Optional[np.ndarray]:
        """Get the matrix of existing embeddings for novelty computation."""
        traces = self.hippocampus.get_active_traces()
        if not traces:
            return None
        embeddings = [t.embedding for t in traces if t.embedding is not None]
        if not embeddings:
            return None
        return np.stack(embeddings)

    def _should_consolidate(self) -> bool:
        return (
            self._interaction_count - self._last_consolidation
            >= self.consolidation_interval
        )

    async def _run_consolidation(self) -> Dict[str, Any]:
        """Run a consolidation round."""
        self._last_consolidation = self._interaction_count
        result = await self.consolidation_engine.consolidate(
            hippocampus=self.hippocampus,
            neocortex=self.neocortex,
            llm_engine=self.llm_engine,
            amygdala=self.amygdala,
        )
        return result

    def _serialize_trace(self, trace: EpisodicTrace) -> Dict:
        d = trace.to_dict()
        if trace.embedding is not None:
            d["embedding_shape"] = list(trace.embedding.shape)
        return d

    def _serialize_schema(self, schema: SemanticSchema) -> Dict:
        d = schema.to_dict()
        if schema.embedding is not None:
            d["embedding_shape"] = list(schema.embedding.shape)
        return d
