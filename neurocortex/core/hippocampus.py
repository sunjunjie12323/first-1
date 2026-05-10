from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .memory_trace import ContextTag, EpisodicTrace, MemoryPhase

logger = logging.getLogger(__name__)


class Hippocampus:
    """
    Hippocampus-inspired episodic memory module.

    INNOVATION 3: Dentate Gyrus Adaptive Pattern Separation

    Implements two key computational principles:
    1. DG pattern separation (INNOVATION): similar inputs produce
       distinct representations through adaptive noise injection,
       where noise magnitude is proportional to max similarity to
       existing traces.

       Formal definition:
         x' = x + epsilon * max_sim(x, W) * N(0, I)
         x' = x' / ||x'|| * ||x||

       Properties:
       - Low max_sim (novel input): minimal noise, x' approx x
       - High max_sim (similar to existing): strong noise, x' pushed apart
       - Adaptive: separation strength scales with similarity

       Differentiation from:
       - HeLa-Mem (Zhu et al., 2026): Hebbian learning strengthens connections;
         pattern separation is the OPPOSITE: makes similar inputs MORE distinct
       - CA3Mem (Zhang et al., AAAI 2026): CA3 autoassociation; DG separation
         is the COMPLEMENT that happens BEFORE CA3 recall
       - Standard vector DB: no pattern separation at all

    2. CA3 autoassociative recall: partial cues can retrieve complete
       episodic traces through spreading activation (not claimed as novel;
       see HeLa-Mem, CA3Mem for similar mechanisms).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        pattern_separation_strength: float = 0.3,
        ca3_recurrent_strength: float = 0.6,
        max_traces: int = 10000,
        association_threshold: float = 0.7,
    ):
        self.embedding_dim = embedding_dim
        self.pattern_separation_strength = pattern_separation_strength
        self.ca3_recurrent_strength = ca3_recurrent_strength
        self.max_traces = max_traces
        self.association_threshold = association_threshold

        self._traces: Dict[str, EpisodicTrace] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._trace_ids_ordered: List[str] = []
        self._association_graph: Dict[str, set] = {}

    def encode(
        self,
        content: str,
        embedding: np.ndarray,
        context: Optional[ContextTag] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        source: str = "user",
        novelty_score: float = 0.0,
        reward_score: float = 0.0,
    ) -> EpisodicTrace:
        """
        Encode a new episodic trace with DG pattern separation.
        Pattern separation ensures that similar experiences are stored
        as distinct traces, preventing catastrophic interference.
        """
        separated_embedding = self._pattern_separation(embedding)

        trace = EpisodicTrace(
            content=content,
            embedding=separated_embedding,
            context=context or ContextTag(),
            importance=importance,
            emotional_valence=emotional_valence,
            source=source,
            phase=MemoryPhase.EPISODIC,
            novelty_score=novelty_score,
            reward_score=reward_score,
        )

        self._traces[trace.trace_id] = trace
        self._trace_ids_ordered.append(trace.trace_id)
        self._association_graph[trace.trace_id] = set()

        self._update_embeddings_matrix()

        self._form_associations(trace)

        if len(self._traces) > self.max_traces:
            self._prune_weak_traces()

        logger.info(
            f"Hippocampus encoded trace {trace.trace_id[:8]}... "
            f"(importance={importance:.2f}, novelty={novelty_score:.2f})"
        )

        return trace

    def retrieve_by_cue(
        self,
        cue_embedding: np.ndarray,
        top_k: int = 5,
        min_strength: float = 0.1,
    ) -> List[Tuple[EpisodicTrace, float]]:
        """
        CA3-inspired autoassociative retrieval.
        Given a partial cue, retrieve the most strongly activated
        episodic traces through pattern completion.
        """
        if self._embeddings_matrix is None or len(self._traces) == 0:
            return []

        separated_cue = self._pattern_separation(cue_embedding)

        similarities = self._compute_similarities(separated_cue)

        scored_traces = []
        for idx, sim in enumerate(similarities):
            if idx >= len(self._trace_ids_ordered):
                break
            trace_id = self._trace_ids_ordered[idx]
            trace = self._traces.get(trace_id)
            if trace is None:
                continue

            activation = sim * trace.memory_strength
            if activation >= min_strength:
                scored_traces.append((trace, float(activation)))

        scored_traces.sort(key=lambda x: x[1], reverse=True)

        results = scored_traces[:top_k]

        for trace, _ in results:
            trace.reactivate()

        return results

    def spread_activation(
        self,
        seed_trace_ids: List[str],
        depth: int = 2,
        decay: float = 0.5,
    ) -> Dict[str, float]:
        """
        CA3 recurrent collateral simulation: spreading activation
        through the association graph. This allows recall of related
        memories that were not directly cued, mimicking how human
        memory chains through associations.
        """
        activation: Dict[str, float] = {}

        current_seeds = {tid: 1.0 for tid in seed_trace_ids if tid in self._traces}

        for level in range(depth):
            next_seeds: Dict[str, float] = {}
            for trace_id, strength in current_seeds.items():
                if trace_id in activation:
                    continue
                activation[trace_id] = strength * (decay ** level)

                for associated_id in self._association_graph.get(trace_id, set()):
                    if associated_id not in activation and associated_id in self._traces:
                        trace = self._traces[associated_id]
                        propagated = strength * self.ca3_recurrent_strength * (decay ** level)
                        propagated *= trace.memory_strength
                        if associated_id in next_seeds:
                            next_seeds[associated_id] = max(
                                next_seeds[associated_id], propagated
                            )
                        else:
                            next_seeds[associated_id] = propagated

            current_seeds = next_seeds
            if not current_seeds:
                break

        return activation

    def get_consolidation_candidates(self, min_age_hours: float = 1.0) -> List[EpisodicTrace]:
        """
        Select episodic traces that are candidates for neocortical consolidation.
        Mimics the hippocampal-neocortical dialogue during sleep.
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(hours=min_age_hours)
        candidates = [
            t
            for t in self._traces.values()
            if t.timestamp < cutoff
            and t.consolidation_level < 1.0
            and not t.is_decayed
            and t.phase == MemoryPhase.EPISODIC
        ]
        candidates.sort(key=lambda t: t.importance * t.memory_strength, reverse=True)
        return candidates

    def mark_consolidated(self, trace_id: str, consolidation_level: float = 1.0) -> None:
        if trace_id in self._traces:
            trace = self._traces[trace_id]
            trace.consolidation_level = consolidation_level
            if consolidation_level >= 0.8:
                trace.phase = MemoryPhase.CONSOLIDATING

    def remove_trace(self, trace_id: str) -> None:
        if trace_id in self._traces:
            del self._traces[trace_id]
            if trace_id in self._association_graph:
                for assoc in self._association_graph.pop(trace_id, set()):
                    if assoc in self._association_graph:
                        self._association_graph[assoc].discard(trace_id)
            if trace_id in self._trace_ids_ordered:
                self._trace_ids_ordered.remove(trace_id)
            self._embeddings_matrix = None

    def get_trace(self, trace_id: str) -> Optional[EpisodicTrace]:
        return self._traces.get(trace_id)

    def get_all_traces(self) -> List[EpisodicTrace]:
        return list(self._traces.values())

    def get_active_traces(self, min_strength: float = 0.05) -> List[EpisodicTrace]:
        return [t for t in self._traces.values() if t.memory_strength >= min_strength]

    @property
    def trace_count(self) -> int:
        return len(self._traces)

    def _pattern_separation(self, embedding: np.ndarray) -> np.ndarray:
        """
        INNOVATION 3: Dentate Gyrus Adaptive Pattern Separation

        Formal definition:
          x' = x + epsilon * max_sim(x, W) * N(0, I)
          x' = x' / ||x'|| * ||x||

        where:
          - x: input embedding
          - W: existing embeddings matrix
          - epsilon: pattern_separation_strength (hyperparameter)
          - max_sim(x, W): maximum cosine similarity to existing traces
          - N(0, I): standard Gaussian noise

        Key property: The separation is ADAPTIVE. When the input is highly
        similar to existing traces (max_sim close to 1), the noise is strong,
        pushing the representation apart. When the input is novel (max_sim
        close to 0), the noise is minimal, preserving the original signal.

        This is fundamentally different from:
        - HeLa-Mem's Hebbian learning: strengthens connections (convergent)
        - Our DG separation: pushes similar inputs apart (divergent)
        These are complementary operations, not competing ones.
        """
        if self._embeddings_matrix is None or len(self._traces) == 0:
            return embedding.copy()

        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return embedding.copy()

        normalized = embedding / norm

        mat_norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
        mat_norms = np.maximum(mat_norms, 1e-8)
        normalized_matrix = self._embeddings_matrix / mat_norms

        sims = normalized_matrix @ normalized

        max_sim = float(np.max(sims))

        separation_noise = np.random.randn(len(embedding)) * self.pattern_separation_strength
        separation_noise *= max_sim

        separated = embedding + separation_noise
        separated_norm = np.linalg.norm(separated)
        if separated_norm > 1e-8:
            separated = separated / separated_norm * norm

        return separated

    def _form_associations(self, new_trace: EpisodicTrace) -> None:
        """
        Form associations between the new trace and existing traces
        based on embedding similarity and contextual overlap.
        """
        if self._embeddings_matrix is None or len(self._traces) <= 1:
            return

        new_embedding = new_trace.embedding
        if new_embedding is None:
            return

        new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
        sims = self._embeddings_matrix @ new_norm

        for idx, sim in enumerate(sims):
            if idx >= len(self._trace_ids_ordered):
                break
            other_id = self._trace_ids_ordered[idx]
            if other_id == new_trace.trace_id:
                continue

            other_trace = self._traces.get(other_id)
            if other_trace is None:
                continue

            context_overlap = self._compute_context_overlap(
                new_trace.context, other_trace.context
            )

            association_strength = float(sim) * 0.7 + context_overlap * 0.3

            if association_strength >= self.association_threshold:
                self._association_graph[new_trace.trace_id].add(other_id)
                self._association_graph[other_id].add(new_trace.trace_id)
                new_trace.associations.append(other_id)
                other_trace.associations.append(new_trace.trace_id)

    def _compute_context_overlap(self, ctx1: ContextTag, ctx2: ContextTag) -> float:
        overlap = 0.0
        count = 0
        for attr in ["spatial", "temporal_period", "interlocutor", "activity"]:
            v1 = getattr(ctx1, attr)
            v2 = getattr(ctx2, attr)
            if v1 is not None and v2 is not None:
                overlap += 1.0 if v1 == v2 else 0.0
                count += 1
        return overlap / max(count, 1)

    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        if self._embeddings_matrix is None:
            return np.array([])

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        matrix_norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
        matrix_norms = np.maximum(matrix_norms, 1e-8)
        normalized_matrix = self._embeddings_matrix / matrix_norms

        return (normalized_matrix @ query_norm)

    def _update_embeddings_matrix(self) -> None:
        if not self._trace_ids_ordered:
            self._embeddings_matrix = None
            return

        embeddings = []
        valid_ids = []
        for tid in self._trace_ids_ordered:
            trace = self._traces.get(tid)
            if trace and trace.embedding is not None:
                embeddings.append(trace.embedding)
                valid_ids.append(tid)

        if embeddings:
            self._embeddings_matrix = np.stack(embeddings)
            self._trace_ids_ordered = valid_ids
        else:
            self._embeddings_matrix = None

    def _prune_weak_traces(self) -> None:
        traces = list(self._traces.values())
        traces.sort(key=lambda t: t.memory_strength)

        to_remove = len(self._traces) - self.max_traces
        for i in range(min(to_remove, len(traces))):
            if traces[i].is_decayed or traces[i].importance < 0.3:
                self.remove_trace(traces[i].trace_id)

    def get_status(self) -> Dict:
        active = [t for t in self._traces.values() if not t.is_decayed]
        return {
            "total_traces": len(self._traces),
            "active_traces": len(active),
            "total_associations": sum(len(v) for v in self._association_graph.values()) // 2,
            "avg_memory_strength": (
                np.mean([t.memory_strength for t in active]) if active else 0.0
            ),
        }
