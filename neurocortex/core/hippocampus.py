from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurocortex.core.barcode import BarcodeAssociativeMemory
from neurocortex.core.memory_trace import ContextTag, EpisodicTrace, MemoryPhase
from neurocortex.core.theory import SeparationCompletionDuality

logger = logging.getLogger(__name__)


class Hippocampus:
    def __init__(
        self,
        barcode_dim: int = 256,
        barcode_sparsity: int = 32,
        content_dim: int = 128,
        lambda_param: float = 0.5,
        temperature: float = 10.0,
        max_traces: int = 10000,
        decay_threshold: float = 0.01,
        use_projection: bool = True,
        soft_wta: bool = True,
    ):
        self.bam = BarcodeAssociativeMemory(
            barcode_dim=barcode_dim,
            barcode_sparsity=barcode_sparsity,
            content_dim=content_dim,
            lambda_param=lambda_param,
            temperature=temperature,
            use_projection=use_projection,
            soft_wta=soft_wta,
        )
        self.max_traces = max_traces
        self.decay_threshold = decay_threshold
        self.traces: Dict[str, EpisodicTrace] = {}
        self.association_graph: Dict[str, Dict[str, float]] = {}

    def encode(
        self,
        content: str,
        embedding: np.ndarray,
        context: ContextTag,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        novelty_score: float = 0.0,
        source: str = "",
        encoding_gate: float = 1.0,
    ) -> EpisodicTrace:
        if encoding_gate < 0.3:
            logger.info("Encoding gate too low, skipping encoding")
            return None

        barcode = self.bam.generate_barcode(content_vector=embedding)

        trace = EpisodicTrace(
            content=content,
            embedding=embedding.copy(),
            barcode=barcode,
            context=context,
            importance=importance,
            emotional_valence=emotional_valence,
            novelty_score=novelty_score,
            source=source,
            phase=MemoryPhase.EPISODIC,
        )

        self.traces[trace.trace_id] = trace
        self._form_associations(trace)
        self._prune_decayed()

        logger.info(
            f"Encoded trace {trace.trace_id} with barcode "
            f"(dim={self.bam.barcode_dim}, sparsity={self.bam.barcode_sparsity})"
        )
        return trace

    def retrieve_by_cue(
        self,
        cue_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.3,
        lambda_param: Optional[float] = None,
    ) -> List[Tuple[EpisodicTrace, float]]:
        if not self.traces:
            return []

        content_matrix, barcode_matrix, trace_list = self._get_matrices()
        if content_matrix is None:
            return []

        results = self.bam.retrieve(
            query=cue_embedding,
            content_embeddings=content_matrix,
            barcodes=barcode_matrix,
            top_k=top_k,
            lambda_param=lambda_param,
        )

        output = []
        for idx, content_score, barcode_score, combined_score in results:
            trace = trace_list[idx]
            strength_weight = trace.memory_strength
            weighted_sim = combined_score * strength_weight
            if weighted_sim >= min_similarity:
                output.append((trace, weighted_sim))

        output.sort(key=lambda x: x[1], reverse=True)
        return output[:top_k]

    def retrieve_content_only(
        self,
        cue_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> List[Tuple[EpisodicTrace, float]]:
        if not self.traces:
            return []

        content_matrix, _, trace_list = self._get_matrices()
        if content_matrix is None:
            return []

        results = self.bam.retrieve_content_only(
            query=cue_embedding,
            content_embeddings=content_matrix,
            top_k=top_k,
        )

        output = []
        for idx, score in results:
            trace = trace_list[idx]
            strength_weight = trace.memory_strength
            weighted_sim = score * strength_weight
            if weighted_sim >= min_similarity:
                output.append((trace, weighted_sim))

        output.sort(key=lambda x: x[1], reverse=True)
        return output[:top_k]

    def spread_activation(
        self,
        seed_trace_ids: List[str],
        depth: int = 2,
        activation_threshold: float = 0.1,
    ) -> Dict[str, float]:
        activations: Dict[str, float] = {tid: 1.0 for tid in seed_trace_ids if tid in self.traces}
        frontier = list(activations.keys())

        for _ in range(depth):
            next_frontier = []
            for trace_id in frontier:
                current_activation = activations.get(trace_id, 0.0)
                if current_activation < activation_threshold:
                    continue

                neighbors = self.association_graph.get(trace_id, {})
                for neighbor_id, weight in neighbors.items():
                    if neighbor_id not in self.traces:
                        continue
                    spread = current_activation * weight * 0.5
                    if neighbor_id in activations:
                        activations[neighbor_id] = max(activations[neighbor_id], spread)
                    else:
                        activations[neighbor_id] = spread
                    if spread >= activation_threshold:
                        next_frontier.append(neighbor_id)

            frontier = next_frontier

        return {tid: act for tid, act in activations.items() if act >= activation_threshold}

    def _form_associations(self, new_trace: EpisodicTrace) -> None:
        self.association_graph.setdefault(new_trace.trace_id, {})

        for trace_id, existing in self.traces.items():
            if trace_id == new_trace.trace_id:
                continue

            similarity = self._cosine_similarity(new_trace.embedding, existing.embedding)
            context_overlap = new_trace.context.overlap(existing.context)
            association_strength = 0.6 * similarity + 0.4 * context_overlap

            if association_strength > 0.2:
                self.association_graph[new_trace.trace_id][trace_id] = association_strength
                self.association_graph.setdefault(trace_id, {})[new_trace.trace_id] = association_strength

    def _prune_decayed(self) -> None:
        if len(self.traces) <= self.max_traces:
            return

        to_remove = []
        for trace_id, trace in self.traces.items():
            if trace.memory_strength < self.decay_threshold:
                to_remove.append(trace_id)

        for trace_id in to_remove:
            del self.traces[trace_id]
            self.association_graph.pop(trace_id, None)
            for neighbors in self.association_graph.values():
                neighbors.pop(trace_id, None)

        if len(self.traces) > self.max_traces:
            sorted_traces = sorted(
                self.traces.items(), key=lambda x: x[1].memory_strength
            )
            for trace_id, _ in sorted_traces[: len(self.traces) - self.max_traces]:
                del self.traces[trace_id]
                self.association_graph.pop(trace_id, None)
                for neighbors in self.association_graph.values():
                    neighbors.pop(trace_id, None)

    def _get_matrices(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[EpisodicTrace]]:
        trace_list = [t for t in self.traces.values() if t.embedding.size > 0 and t.barcode.size > 0]
        if not trace_list:
            return None, None, []

        content_matrix = np.stack([t.embedding for t in trace_list]).astype(np.float32)
        barcode_matrix = np.stack([t.barcode for t in trace_list]).astype(np.float32)
        return content_matrix, barcode_matrix, trace_list

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
