from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurocortex.core.hippocampus import Hippocampus
from neurocortex.core.llm_engine import LLMEngine
from neurocortex.core.memory_trace import EpisodicTrace, ReconstructedMemory, SemanticSchema
from neurocortex.core.neocortex import Neocortex
from neurocortex.core.theory import (
    ReconstructiveDistortionBound,
    SchacterSinsMapping,
    SeparationCompletionDuality,
)

logger = logging.getLogger(__name__)


class ReconstructiveRecall:
    def __init__(
        self,
        hippocampus: Hippocampus,
        neocortex: Neocortex,
        llm_engine: LLMEngine,
        spread_depth: int = 2,
        activation_threshold: float = 0.15,
        alpha_full: float = 0.7,
        alpha_gist: float = 0.3,
    ):
        self.hippocampus = hippocampus
        self.neocortex = neocortex
        self.llm_engine = llm_engine
        self.spread_depth = spread_depth
        self.activation_threshold = activation_threshold
        self.alpha_full = alpha_full
        self.alpha_gist = alpha_gist

    async def recall(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        lambda_param: Optional[float] = None,
    ) -> ReconstructedMemory:
        cue_results = self.hippocampus.retrieve_by_cue(
            query_embedding, top_k=top_k, min_similarity=0.2, lambda_param=lambda_param
        )

        if not cue_results:
            return ReconstructedMemory(
                query=query,
                reconstructed_narrative="",
                confidence=0.0,
                distortion_score=1.0,
            )

        seed_ids = [trace.trace_id for trace, _ in cue_results]
        activations = self.hippocampus.spread_activation(
            seed_ids, depth=self.spread_depth, activation_threshold=self.activation_threshold
        )

        activated_traces = []
        activation_values = []
        for trace_id, activation in activations.items():
            trace = self.hippocampus.traces.get(trace_id)
            if trace is not None:
                activated_traces.append(trace)
                activation_values.append(activation)
                trace.reactivate()

        schema_results = self.neocortex.retrieve_relevant(query_embedding, top_k=3)
        relevant_schemas = [schema for schema, _ in schema_results]

        fragments = self._assemble_fragments(activated_traces, activation_values)

        distortion_score = self._compute_dual_channel_distortion(
            query_embedding, activated_traces, relevant_schemas, lambda_param
        )

        schacter_sins = self._compute_schacter_sins(activated_traces, relevant_schemas)

        reconstructed_narrative = await self._llm_reconstruct(
            query, fragments, relevant_schemas
        )

        confidence = self._compute_confidence(activation_values, relevant_schemas)
        emotional_tone = self._compute_emotional_tone(activated_traces)

        return ReconstructedMemory(
            query=query,
            reconstructed_narrative=reconstructed_narrative,
            source_traces=[t.trace_id for t in activated_traces],
            source_schemas=[s.schema_id for s in relevant_schemas],
            confidence=confidence,
            distortion_score=distortion_score,
            emotional_tone=emotional_tone,
        )

    def _compute_dual_channel_distortion(
        self,
        query_embedding: np.ndarray,
        activated_traces: List[EpisodicTrace],
        relevant_schemas: List[SemanticSchema],
        lambda_param: Optional[float] = None,
    ) -> float:
        if not activated_traces:
            return 1.0

        activations = [t.memory_strength for t in activated_traces]
        consolidation_levels = [t.consolidation_level for t in activated_traces]
        n_spread = max(0, len(activated_traces) - 1)
        n_schemas = len(relevant_schemas)

        content_distortion = SchacterSinsMapping.compute_distortion(
            activations=activations,
            consolidation_levels=consolidation_levels,
            n_spread_traces=n_spread,
            n_schemas=n_schemas,
        )

        lam = lambda_param if lambda_param is not None else self.hippocampus.bam.lambda_param
        sparsity_ratio = self.hippocampus.bam.sparsity_ratio

        combined_bound = ReconstructiveDistortionBound.compute_combined_distortion(
            content_distortion=content_distortion,
            barcode_distortion=content_distortion * 0.1,
            lambda_param=lam,
            sparsity_ratio=sparsity_ratio,
        )

        return float(min(1.0, combined_bound))

    def _assemble_fragments(
        self,
        traces: List[EpisodicTrace],
        activations: List[float],
    ) -> List[Dict]:
        fragments = []
        for trace, activation in zip(traces, activations):
            detail_level = SchacterSinsMapping.compute_fragment_detail(
                activation=activation,
                importance=trace.importance,
                emotional_valence=trace.emotional_valence,
                alpha_full=self.alpha_full,
                alpha_gist=self.alpha_gist,
            )

            if detail_level == "full":
                fragment_content = trace.content
            elif detail_level == "gist":
                fragment_content = self._extract_gist(trace.content)
            else:
                fragment_content = self._extract_keywords(trace.content)

            fragments.append({
                "trace_id": trace.trace_id,
                "content": fragment_content,
                "detail_level": detail_level,
                "activation": activation,
                "importance": trace.importance,
                "emotional_valence": trace.emotional_valence,
                "timestamp": trace.timestamp.isoformat(),
            })

        return fragments

    def _compute_schacter_sins(
        self,
        traces: List[EpisodicTrace],
        schemas: List[SemanticSchema],
    ) -> Dict[str, float]:
        if not traces:
            return {}

        from datetime import datetime, timezone

        avg_age = np.mean([
            (datetime.now(timezone.utc) - t.timestamp).total_seconds() / 3600.0
            for t in traces
        ])
        avg_decay = np.mean([t.decay_rate for t in traces])
        avg_activation = np.mean([t.memory_strength for t in traces])
        avg_emotion = np.mean([t.emotional_valence for t in traces])
        avg_importance = np.mean([t.importance for t in traces])

        return SchacterSinsMapping.compute_schacter_sins(
            trace_age_hours=avg_age,
            decay_rate=avg_decay,
            encoding_gate=1.0,
            activation=avg_activation,
            detail_level="full",
            n_spread=max(0, len(traces) - 1),
            n_schemas=len(schemas),
            emotional_valence=avg_emotion,
            importance=avg_importance,
        )

    async def _llm_reconstruct(
        self,
        query: str,
        fragments: List[Dict],
        schemas: List[SemanticSchema],
    ) -> str:
        fragment_text = "\n".join(
            f"[{f['detail_level'].upper()}] (activation={f['activation']:.2f}): {f['content']}"
            for f in fragments
        )

        schema_text = "\n".join(
            f"Schema (confidence={s.confidence:.2f}): {s.gist}"
            for s in schemas
        )

        prompt = (
            f"Based on the following memory fragments and schemas, reconstruct a coherent "
            f"response to the query: '{query}'\n\n"
            f"Memory Fragments:\n{fragment_text}\n\n"
            f"Relevant Schemas:\n{schema_text}\n\n"
            f"Reconstruct the memory, filling in gaps naturally while staying faithful "
            f"to the available evidence. Note which parts are reconstructed vs. directly recalled."
        )

        try:
            return await self.llm_engine.generate(prompt, max_tokens=512, temperature=0.5)
        except Exception as e:
            logger.error(f"LLM reconstruction failed: {e}")
            return " ".join(f["content"] for f in fragments)

    @staticmethod
    def _extract_gist(content: str) -> str:
        sentences = content.split(". ")
        if len(sentences) <= 2:
            return content
        return ". ".join(sentences[:2]) + ("." if not sentences[1].endswith(".") else "")

    @staticmethod
    def _extract_keywords(content: str) -> str:
        words = content.split()
        stop_words = {"the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                      "on", "with", "at", "by", "from", "as", "into", "through", "during",
                      "before", "after", "above", "below", "between", "and", "but", "or",
                      "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
                      "every", "all", "any", "few", "more", "most", "other", "some", "such",
                      "than", "too", "very", "just", "because", "if", "when", "where", "how",
                      "what", "which", "who", "whom", "this", "that", "these", "those", "it",
                      "its", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
                      "she", "her", "they", "them", "their"}
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        return " ".join(keywords[:15])

    @staticmethod
    def _compute_confidence(activations: List[float], schemas: List[SemanticSchema]) -> float:
        if not activations:
            return 0.0
        act_confidence = float(np.mean(activations))
        schema_confidence = float(np.mean([s.confidence for s in schemas])) if schemas else 0.5
        return float(0.7 * act_confidence + 0.3 * schema_confidence)

    @staticmethod
    def _compute_emotional_tone(traces: List[EpisodicTrace]) -> float:
        if not traces:
            return 0.0
        return float(np.mean([t.emotional_valence for t in traces]))
