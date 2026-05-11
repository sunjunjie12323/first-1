from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurocortex.core.memory_trace import SemanticSchema

logger = logging.getLogger(__name__)


class Neocortex:
    def __init__(self, merge_similarity_threshold: float = 0.75, max_schemas: int = 5000):
        self.merge_similarity_threshold = merge_similarity_threshold
        self.max_schemas = max_schemas
        self.schemas: Dict[str, SemanticSchema] = {}

    def create_schema(
        self,
        gist: str,
        embedding: np.ndarray,
        source_trace_id: str,
        key_entities: Optional[List[str]] = None,
        initial_confidence: float = 0.3,
    ) -> SemanticSchema:
        similar_schema = self._find_similar_schema(embedding)

        if similar_schema is not None:
            similar_schema.gist = self._merge_gists(similar_schema.gist, gist)
            if source_trace_id not in similar_schema.source_traces:
                similar_schema.source_traces.append(source_trace_id)
            if key_entities:
                for entity in key_entities:
                    if entity not in similar_schema.key_entities:
                        similar_schema.key_entities.append(entity)
            similar_schema.reinforce(0.05)
            self._update_schema_embedding(similar_schema, embedding)
            logger.info(f"Merged into existing schema {similar_schema.schema_id}")
            return similar_schema

        schema = SemanticSchema(
            gist=gist,
            embedding=embedding.copy(),
            source_traces=[source_trace_id],
            key_entities=key_entities or [],
            confidence=initial_confidence,
        )
        self.schemas[schema.schema_id] = schema
        logger.info(f"Created new schema {schema.schema_id}")
        return schema

    def retrieve_relevant(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_confidence: float = 0.2,
    ) -> List[Tuple[SemanticSchema, float]]:
        if not self.schemas:
            return []

        results = []
        for schema in self.schemas.values():
            if schema.confidence < min_confidence:
                continue
            similarity = self._cosine_similarity(query_embedding, schema.embedding)
            confidence_weight = schema.confidence
            score = similarity * confidence_weight
            if score > 0.1:
                results.append((schema, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def reinforce_schema(self, schema_id: str, additional_confidence: float = 0.1) -> bool:
        schema = self.schemas.get(schema_id)
        if schema is None:
            return False
        schema.reinforce(additional_confidence)
        return True

    def _find_similar_schema(self, embedding: np.ndarray) -> Optional[SemanticSchema]:
        best_schema = None
        best_similarity = 0.0

        for schema in self.schemas.values():
            if schema.embedding.size == 0:
                continue
            similarity = self._cosine_similarity(embedding, schema.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_schema = schema

        if best_similarity >= self.merge_similarity_threshold:
            return best_schema
        return None

    def _merge_gists(self, existing_gist: str, new_gist: str) -> str:
        if new_gist in existing_gist:
            return existing_gist
        return f"{existing_gist}; {new_gist}"

    def _update_schema_embedding(self, schema: SemanticSchema, new_embedding: np.ndarray) -> None:
        if schema.embedding.size == 0:
            schema.embedding = new_embedding.copy()
        else:
            alpha = 1.0 / (1.0 + schema.reinforcement_count)
            schema.embedding = (1.0 - alpha) * schema.embedding + alpha * new_embedding
            norm = np.linalg.norm(schema.embedding)
            if norm > 1e-8:
                schema.embedding = schema.embedding / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
