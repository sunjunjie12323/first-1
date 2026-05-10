from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from neurocortex.core.amygdala import Amygdala
from neurocortex.core.basal_forebrain import BasalForebrain
from neurocortex.core.consolidation import Consolidation
from neurocortex.core.hippocampus import Hippocampus
from neurocortex.core.llm_engine import LLMEngine
from neurocortex.core.memory_trace import (
    ContextTag,
    EpisodicTrace,
    MemoryPhase,
    NeuromodulatoryState,
    ReconstructedMemory,
    SemanticSchema,
)
from neurocortex.core.neocortex import Neocortex
from neurocortex.core.prefrontal import PrefrontalCortex
from neurocortex.core.reconstructive_recall import ReconstructiveRecall
from neurocortex.core.theory import (
    BarcodeCapacityTheorem,
    ReconstructiveDistortionBound,
    SchacterSinsMapping,
    SeparationCompletionDuality,
)

logger = logging.getLogger(__name__)


class BrainSystem:
    def __init__(
        self,
        llm_base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        api_type: str = "ollama",
        api_key: Optional[str] = None,
        barcode_dim: int = 256,
        barcode_sparsity: int = 16,
        lambda_param: float = 0.5,
        working_memory_capacity: int = 7,
    ):
        self.llm_engine = LLMEngine(
            base_url=llm_base_url,
            model=llm_model,
            embedding_model=embedding_model,
            api_type=api_type,
            api_key=api_key,
        )
        self.hippocampus = Hippocampus(
            barcode_dim=barcode_dim,
            barcode_sparsity=barcode_sparsity,
            content_dim=128,
            lambda_param=lambda_param,
            use_projection=True,
        )
        self.neocortex = Neocortex()
        self.prefrontal = PrefrontalCortex(capacity=working_memory_capacity)
        self.amygdala = Amygdala()
        self.basal_forebrain = BasalForebrain()
        self.reconstructive_recall = ReconstructiveRecall(
            hippocampus=self.hippocampus,
            neocortex=self.neocortex,
            llm_engine=self.llm_engine,
        )
        self.consolidation = Consolidation(
            hippocampus=self.hippocampus,
            neocortex=self.neocortex,
            amygdala=self.amygdala,
            llm_engine=self.llm_engine,
        )

    async def process_input(
        self,
        content: str,
        context: Optional[ContextTag] = None,
        source: str = "user",
        emotional_valence: float = 0.0,
        social_relevance: float = 0.0,
        goal_relevance: float = 0.0,
    ) -> Dict[str, Any]:
        embedding = await self.llm_engine.get_embedding(content)

        existing_matrix = self._get_existing_embeddings()
        novelty_score = self._compute_novelty(embedding, existing_matrix)

        self.basal_forebrain.compute_novelty(novelty_score)

        importance = self.amygdala.assess_importance(
            emotional_valence=emotional_valence,
            novelty_score=novelty_score,
            social_relevance=social_relevance,
            goal_relevance=goal_relevance,
        )

        encoding_gate = self.basal_forebrain.get_encoding_gate()

        trace = self.hippocampus.encode(
            content=content,
            embedding=embedding,
            context=context or ContextTag(),
            importance=importance,
            emotional_valence=emotional_valence,
            novelty_score=novelty_score,
            source=source,
            encoding_gate=encoding_gate,
        )

        if trace is not None:
            modified_decay = self.amygdala.modify_decay_rate(trace.decay_rate, importance)
            trace.decay_rate = modified_decay

        self.prefrontal.add_to_working_memory(
            content=content,
            embedding=embedding,
            attention_weight=importance,
            metadata={"trace_id": trace.trace_id if trace else None, "importance": importance},
        )

        recall_result = await self.reconstructive_recall.recall(content, embedding, top_k=3)

        memory_context = recall_result.reconstructed_narrative
        response = await self.llm_engine.generate_with_memory_context(
            prompt=content,
            memory_context=memory_context,
            max_tokens=512,
        )

        reward_signal = self._estimate_reward(response, importance)
        self.basal_forebrain.compute_reward(reward_signal)

        self.basal_forebrain.homeostatic_decay()
        self.prefrontal.decay_attention()

        return {
            "response": response,
            "trace_id": trace.trace_id if trace else None,
            "importance": importance,
            "novelty_score": novelty_score,
            "encoding_gate": encoding_gate,
            "recall_confidence": recall_result.confidence,
            "distortion_score": recall_result.distortion_score,
            "neuromodulatory_state": self.basal_forebrain.get_state().to_dict(),
        }

    async def recall_memory(self, query: str) -> ReconstructedMemory:
        query_embedding = await self.llm_engine.get_embedding(query)
        result = await self.reconstructive_recall.recall(query, query_embedding)
        return result

    async def force_consolidation(self) -> Dict[str, int]:
        consolidation_gate = self.basal_forebrain.get_consolidation_gate()
        return await self.consolidation.consolidate(consolidation_gate)

    def get_memory_status(self) -> Dict[str, Any]:
        traces = list(self.hippocampus.traces.values())
        schemas = list(self.neocortex.schemas.values())

        phase_counts = {}
        for phase in MemoryPhase:
            phase_counts[phase.value] = sum(1 for t in traces if t.phase == phase)

        avg_strength = float(np.mean([t.memory_strength for t in traces])) if traces else 0.0
        avg_importance = float(np.mean([t.importance for t in traces])) if traces else 0.0

        return {
            "total_traces": len(traces),
            "total_schemas": len(schemas),
            "phase_distribution": phase_counts,
            "avg_memory_strength": avg_strength,
            "avg_importance": avg_importance,
            "working_memory_load": len(self.prefrontal.working_memory),
            "working_memory_capacity": self.prefrontal.capacity,
            "current_goal": self.prefrontal.current_goal,
            "neuromodulatory_state": self.basal_forebrain.get_state().to_dict(),
            "barcode_dim": self.hippocampus.bam.barcode_dim,
            "barcode_sparsity": self.hippocampus.bam.barcode_sparsity,
            "lambda_param": self.hippocampus.bam.lambda_param,
        }

    def analyze_distortion(self, query_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        traces = list(self.hippocampus.traces.values())
        if not traces:
            return {"distortion_score": 1.0, "schacter_sins": {}}

        activations = []
        consolidation_levels = []
        if query_embedding is not None:
            results = self.hippocampus.retrieve_by_cue(query_embedding, top_k=5)
            for trace, sim in results:
                activations.append(sim)
                consolidation_levels.append(trace.consolidation_level)

        if not activations:
            activations = [t.memory_strength for t in traces[:5]]
            consolidation_levels = [t.consolidation_level for t in traces[:5]]

        distortion = SchacterSinsMapping.compute_distortion(
            activations=activations,
            consolidation_levels=consolidation_levels,
            n_spread_traces=max(0, len(traces) - 5),
            n_schemas=len(self.neocortex.schemas),
        )

        avg_trace = traces[0]
        if len(traces) > 1:
            avg_emotion = float(np.mean([t.emotional_valence for t in traces]))
            avg_importance = float(np.mean([t.importance for t in traces]))
            avg_decay = float(np.mean([t.decay_rate for t in traces]))
        else:
            avg_emotion = avg_trace.emotional_valence
            avg_importance = avg_trace.importance
            avg_decay = avg_trace.decay_rate

        from datetime import datetime, timezone as tz
        avg_age = float(np.mean([
            (datetime.now(tz.utc) - t.timestamp).total_seconds() / 3600.0 for t in traces
        ]))

        sins = SchacterSinsMapping.compute_schacter_sins(
            trace_age_hours=avg_age,
            decay_rate=avg_decay,
            encoding_gate=self.basal_forebrain.get_encoding_gate(),
            activation=float(np.mean(activations)) if activations else 0.0,
            detail_level="full",
            n_spread=max(0, len(traces) - 1),
            n_schemas=len(self.neocortex.schemas),
            emotional_valence=avg_emotion,
            importance=avg_importance,
        )

        return {"distortion_score": distortion, "schacter_sins": sins}

    async def save_state(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        traces_data = {}
        for tid, trace in self.hippocampus.traces.items():
            d = trace.to_dict()
            d["embedding"] = trace.embedding.tolist() if trace.embedding.size > 0 else []
            d["barcode"] = trace.barcode.tolist() if trace.barcode.size > 0 else []
            traces_data[tid] = d

        schemas_data = {}
        for sid, schema in self.neocortex.schemas.items():
            d = schema.to_dict()
            d["embedding"] = schema.embedding.tolist() if schema.embedding.size > 0 else []
            schemas_data[sid] = d

        state = {
            "traces": traces_data,
            "schemas": schemas_data,
            "associations": {
                tid: dict(neighbors)
                for tid, neighbors in self.hippocampus.association_graph.items()
            },
            "neuromodulatory_state": self.basal_forebrain.get_state().to_dict(),
            "barcode_dim": self.hippocampus.bam.barcode_dim,
            "barcode_sparsity": self.hippocampus.bam.barcode_sparsity,
            "lambda_param": self.hippocampus.bam.lambda_param,
        }

        with open(os.path.join(path, "brain_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Brain state saved to {path}")

    async def load_state(self, path: str) -> None:
        state_file = os.path.join(path, "brain_state.json")
        if not os.path.exists(state_file):
            logger.warning(f"State file not found: {state_file}")
            return

        with open(state_file, "r") as f:
            state = json.load(f)

        self.hippocampus.traces = {}
        for tid, d in state.get("traces", {}).items():
            emb_list = d.pop("embedding", [])
            embedding = np.array(emb_list, dtype=np.float32) if emb_list else np.array([], dtype=np.float32)
            bc_list = d.pop("barcode", [])
            barcode = np.array(bc_list, dtype=np.float32) if bc_list else np.array([], dtype=np.float32)
            d.pop("embedding_shape", None)
            d.pop("barcode_shape", None)
            d.pop("memory_strength", None)
            context_data = d.pop("context", {})
            context = ContextTag(**context_data)
            phase_str = d.pop("phase", "episodic")
            phase = MemoryPhase(phase_str)
            ts_str = d.pop("timestamp", None)
            timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
            lr_str = d.pop("last_reactivation", None)
            last_reactivation = datetime.fromisoformat(lr_str) if lr_str else None

            trace = EpisodicTrace(
                trace_id=d.get("trace_id", tid),
                timestamp=timestamp,
                content=d.get("content", ""),
                embedding=embedding,
                barcode=barcode,
                context=context,
                importance=d.get("importance", 0.5),
                emotional_valence=d.get("emotional_valence", 0.0),
                consolidation_level=d.get("consolidation_level", 0.0),
                reactivation_count=d.get("reactivation_count", 0),
                last_reactivation=last_reactivation,
                decay_rate=d.get("decay_rate", 0.1),
                associations=d.get("associations", []),
                source=d.get("source", ""),
                novelty_score=d.get("novelty_score", 0.0),
                phase=phase,
            )
            self.hippocampus.traces[tid] = trace

        self.neocortex.schemas = {}
        for sid, d in state.get("schemas", {}).items():
            emb_list = d.pop("embedding", [])
            embedding = np.array(emb_list, dtype=np.float32) if emb_list else np.array([], dtype=np.float32)
            d.pop("embedding_shape", None)
            d.pop("maturity", None)
            schema = SemanticSchema(
                schema_id=d.get("schema_id", sid),
                gist=d.get("gist", ""),
                embedding=embedding,
                source_traces=d.get("source_traces", []),
                confidence=d.get("confidence", 0.5),
                key_entities=d.get("key_entities", []),
                associations=d.get("associations", []),
                reinforcement_count=d.get("reinforcement_count", 0),
            )
            self.neocortex.schemas[sid] = schema

        self.hippocampus.association_graph = {}
        for tid, neighbors in state.get("associations", {}).items():
            self.hippocampus.association_graph[tid] = {n: float(w) for n, w in neighbors.items()}

        neuro_data = state.get("neuromodulatory_state", {})
        self.basal_forebrain.state = NeuromodulatoryState(
            acetylcholine=neuro_data.get("acetylcholine", 0.5),
            dopamine=neuro_data.get("dopamine", 0.5),
            serotonin=neuro_data.get("serotonin", 0.5),
            norepinephrine=neuro_data.get("norepinephrine", 0.5),
        )

        bc_dim = state.get("barcode_dim", 256)
        bc_sparsity = state.get("barcode_sparsity", 16)
        lam = state.get("lambda_param", 0.5)
        self.hippocampus.bam.barcode_dim = bc_dim
        self.hippocampus.bam.barcode_sparsity = bc_sparsity
        self.hippocampus.bam.lambda_param = lam

        logger.info(f"Brain state loaded from {path}")

    def _get_existing_embeddings(self) -> Optional[np.ndarray]:
        embeddings = [t.embedding for t in self.hippocampus.traces.values() if t.embedding.size > 0]
        if not embeddings:
            return None
        return np.stack(embeddings)

    @staticmethod
    def _compute_novelty(embedding: np.ndarray, existing_matrix: Optional[np.ndarray]) -> float:
        if existing_matrix is None or len(existing_matrix) == 0:
            return 1.0
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return 0.5
        normalized = embedding / norm
        mat_norms = np.linalg.norm(existing_matrix, axis=1, keepdims=True)
        mat_norms = np.maximum(mat_norms, 1e-8)
        sims = (existing_matrix / mat_norms) @ normalized
        max_sim = float(np.max(sims))
        return float(max(0.0, 1.0 - max_sim))

    @staticmethod
    def _estimate_reward(response: str, importance: float) -> float:
        length_factor = min(1.0, len(response) / 200.0)
        return 0.5 * length_factor + 0.5 * importance
