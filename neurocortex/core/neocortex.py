from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .memory_trace import SemanticSchema

logger = logging.getLogger(__name__)


class Neocortex:
    """
    Neocortex-inspired semantic memory module.

    Implements slow, interleaved learning that extracts semantic schemas
    from episodic experiences. Unlike the hippocampus's fast, one-shot
    encoding, the neocortex gradually builds generalized knowledge
    representations through repeated exposure and consolidation.

    Key innovation: Uses the LLM itself as the consolidation engine,
    extracting semantic gist from replayed episodic traces - analogous
    to how the brain replays experiences during sleep to extract
    generalizable knowledge.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_schemas: int = 5000,
        schema_merge_threshold: float = 0.75,
    ):
        self.embedding_dim = embedding_dim
        self.max_schemas = max_schemas
        self.schema_merge_threshold = schema_merge_threshold

        self._schemas: Dict[str, SemanticSchema] = {}
        self._schema_embeddings: Optional[np.ndarray] = None
        self._schema_ids_ordered: List[str] = []

    def create_schema(
        self,
        gist: str,
        embedding: np.ndarray,
        source_traces: List[str],
        key_entities: Optional[List[str]] = None,
        abstract_concepts: Optional[List[str]] = None,
        confidence: float = 0.3,
    ) -> SemanticSchema:
        """
        Create a new semantic schema from consolidated episodic traces.
        Before creating, check if an existing schema is similar enough
        to merge with (neocortical schema integration).
        """
        if self._schema_embeddings is not None and len(self._schemas) > 0:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                query = embedding / norm
                mat_norms = np.linalg.norm(self._schema_embeddings, axis=1, keepdims=True)
                mat_norms = np.maximum(mat_norms, 1e-8)
                sims = (self._schema_embeddings / mat_norms) @ query

                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

                if best_sim >= self.schema_merge_threshold and best_idx < len(self._schema_ids_ordered):
                    existing_id = self._schema_ids_ordered[best_idx]
                    existing = self._schemas[existing_id]
                    merged = self._merge_schemas(
                        existing, gist, embedding, source_traces,
                        key_entities or [], abstract_concepts or [],
                    )
                    logger.info(
                        f"Neocortex merged into existing schema {existing_id[:8]}... "
                        f"(similarity={best_sim:.3f})"
                    )
                    return merged

        schema = SemanticSchema(
            gist=gist,
            embedding=embedding.copy(),
            source_traces=source_traces,
            key_entities=key_entities or [],
            abstract_concepts=abstract_concepts or [],
            confidence=confidence,
        )

        self._schemas[schema.schema_id] = schema
        self._schema_ids_ordered.append(schema.schema_id)
        self._update_schema_matrix()

        logger.info(
            f"Neocortex created new schema {schema.schema_id[:8]}... "
            f"(confidence={confidence:.2f})"
        )

        return schema

    def retrieve_relevant(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        min_confidence: float = 0.1,
    ) -> List[Tuple[SemanticSchema, float]]:
        """
        Retrieve semantically relevant schemas for a given query.
        Confidence-weighted retrieval ensures that well-consolidated
        schemas are prioritized over nascent ones.
        """
        if self._schema_embeddings is None or len(self._schemas) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        mat_norms = np.linalg.norm(self._schema_embeddings, axis=1, keepdims=True)
        mat_norms = np.maximum(mat_norms, 1e-8)
        normalized = self._schema_embeddings / mat_norms

        sims = normalized @ query_norm

        results = []
        for idx, sim in enumerate(sims):
            if idx >= len(self._schema_ids_ordered):
                break
            schema_id = self._schema_ids_ordered[idx]
            schema = self._schemas.get(schema_id)
            if schema is None or schema.confidence < min_confidence:
                continue

            score = float(sim) * schema.maturity
            results.append((schema, score))

        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        for schema, _ in top_results:
            schema.reinforce()

        return top_results

    def get_schema(self, schema_id: str) -> Optional[SemanticSchema]:
        return self._schemas.get(schema_id)

    def get_all_schemas(self) -> List[SemanticSchema]:
        return list(self._schemas.values())

    def update_schema_embedding(self, schema_id: str, new_embedding: np.ndarray) -> None:
        if schema_id in self._schemas:
            old = self._schemas[schema_id].embedding
            if old is not None:
                alpha = 0.3
                self._schemas[schema_id].embedding = alpha * new_embedding + (1 - alpha) * old
            else:
                self._schemas[schema_id].embedding = new_embedding.copy()
            self._update_schema_matrix()

    @property
    def schema_count(self) -> int:
        return len(self._schemas)

    def _merge_schemas(
        self,
        existing: SemanticSchema,
        new_gist: str,
        new_embedding: np.ndarray,
        new_sources: List[str],
        new_entities: List[str],
        new_concepts: List[str],
    ) -> SemanticSchema:
        """
        Merge new information into an existing schema.
        This implements the neocortical slow learning principle:
        new experiences gradually modify existing knowledge
        representations rather than creating entirely new ones.
        """
        existing.gist = new_gist if len(new_gist) > len(existing.gist) else existing.gist

        if existing.embedding is not None:
            alpha = 0.3
            existing.embedding = alpha * new_embedding + (1 - alpha) * existing.embedding

        for src in new_sources:
            if src not in existing.source_traces:
                existing.source_traces.append(src)

        for entity in new_entities:
            if entity not in existing.key_entities:
                existing.key_entities.append(entity)

        for concept in new_concepts:
            if concept not in existing.abstract_concepts:
                existing.abstract_concepts.append(concept)

        existing.confidence = min(1.0, existing.confidence + 0.1)
        existing.consolidation_rounds += 1
        from datetime import datetime
        existing.updated = datetime.now()

        self._update_schema_matrix()

        return existing

    def _update_schema_matrix(self) -> None:
        if not self._schema_ids_ordered:
            self._schema_embeddings = None
            return

        embeddings = []
        valid_ids = []
        for sid in self._schema_ids_ordered:
            schema = self._schemas.get(sid)
            if schema and schema.embedding is not None:
                embeddings.append(schema.embedding)
                valid_ids.append(sid)

        if embeddings:
            self._schema_embeddings = np.stack(embeddings)
            self._schema_ids_ordered = valid_ids
        else:
            self._schema_embeddings = None

    def get_status(self) -> Dict:
        return {
            "total_schemas": len(self._schemas),
            "avg_confidence": (
                np.mean([s.confidence for s in self._schemas.values()])
                if self._schemas
                else 0.0
            ),
            "avg_maturity": (
                np.mean([s.maturity for s in self._schemas.values()])
                if self._schemas
                else 0.0
            ),
        }
