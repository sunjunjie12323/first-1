from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurocortex.core.theory import (
    BarcodeCapacityTheorem,
    SeparationCompletionDuality,
)

logger = logging.getLogger(__name__)


class BarcodeAssociativeMemory:
    def __init__(
        self,
        barcode_dim: int = 256,
        barcode_sparsity: int = 32,
        content_dim: int = 128,
        lambda_param: float = 0.5,
        temperature: float = 10.0,
        attractor_steps: int = 3,
        attractor_rate: float = 0.5,
        use_projection: bool = True,
        projection_seed: int = 123,
        soft_wta: bool = True,
    ):
        self.barcode_dim = barcode_dim
        self.barcode_sparsity = barcode_sparsity
        self.content_dim = content_dim
        self.lambda_param = lambda_param
        self.temperature = temperature
        self.attractor_steps = attractor_steps
        self.attractor_rate = attractor_rate
        self.use_projection = use_projection
        self.soft_wta = soft_wta
        self._barcode_counter = 0
        self._rng = np.random.RandomState(42)

        if use_projection and content_dim > 0:
            proj_rng = np.random.RandomState(projection_seed)
            self._projection = proj_rng.randn(barcode_dim, content_dim).astype(np.float32)
            row_norms = np.linalg.norm(self._projection, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-8)
            self._projection = self._projection / row_norms
        else:
            self._projection = None

    def project_to_barcode(self, content_vector: np.ndarray) -> np.ndarray:
        if self._projection is None:
            raise ValueError("Projection matrix not available (use_projection=False or content_dim=0)")
        projected = self._projection @ content_vector.astype(np.float32)
        return SeparationCompletionDuality.wta_sparsify(
            projected, self.barcode_sparsity, soft=self.soft_wta
        )

    def generate_barcode(self, content_vector: Optional[np.ndarray] = None) -> np.ndarray:
        if self.use_projection and content_vector is not None and self._projection is not None:
            barcode = self.project_to_barcode(content_vector)
        else:
            barcode = BarcodeCapacityTheorem.generate_sparse_barcode(
                dim=self.barcode_dim,
                sparsity=self.barcode_sparsity,
                rng=self._rng,
            )
        self._barcode_counter += 1
        return barcode

    def retrieve(
        self,
        query: np.ndarray,
        content_embeddings: np.ndarray,
        barcodes: np.ndarray,
        top_k: int = 5,
        lambda_param: Optional[float] = None,
    ) -> List[Tuple[int, float, float, float]]:
        lam = lambda_param if lambda_param is not None else self.lambda_param

        if len(content_embeddings) == 0:
            return []

        content_scores = SeparationCompletionDuality.compute_content_scores(
            query, content_embeddings
        )

        if self.use_projection and self._projection is not None:
            query_barcode = self.project_to_barcode(query)
            barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                query_barcode, barcodes
            )
        else:
            barcode_activation = SeparationCompletionDuality.compute_barcode_activation(
                content_scores, barcodes, temperature=self.temperature
            )
            barcode_scores = SeparationCompletionDuality.compute_barcode_scores(
                barcode_activation, barcodes,
                attractor_steps=self.attractor_steps,
                attractor_rate=self.attractor_rate,
            )

        combined_scores = SeparationCompletionDuality.compute_combined_scores(
            content_scores, barcode_scores, lam
        )

        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(content_scores[idx]),
                float(barcode_scores[idx]),
                float(combined_scores[idx]),
            ))

        return results

    def retrieve_content_only(
        self,
        query: np.ndarray,
        content_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        if len(content_embeddings) == 0:
            return []

        content_scores = SeparationCompletionDuality.compute_content_scores(
            query, content_embeddings
        )

        top_indices = np.argsort(content_scores)[::-1][:top_k]

        return [(int(idx), float(content_scores[idx])) for idx in top_indices]

    def compute_retrieval_accuracy(
        self,
        queries: np.ndarray,
        content_embeddings: np.ndarray,
        barcodes: np.ndarray,
        true_indices: np.ndarray,
        lambda_param: Optional[float] = None,
    ) -> Dict[str, float]:
        lam = lambda_param if lambda_param is not None else self.lambda_param

        if len(queries) == 0:
            return {"accuracy": 0.0, "content_accuracy": 0.0, "barcode_accuracy": 0.0}

        dual_correct = 0
        content_correct = 0
        barcode_correct = 0

        for i in range(len(queries)):
            content_scores = SeparationCompletionDuality.compute_content_scores(
                queries[i], content_embeddings
            )

            if self.use_projection and self._projection is not None:
                query_barcode = self.project_to_barcode(queries[i])
                barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                    query_barcode, barcodes
                )
            else:
                barcode_activation = SeparationCompletionDuality.compute_barcode_activation(
                    content_scores, barcodes, temperature=self.temperature
                )
                barcode_scores = SeparationCompletionDuality.compute_barcode_scores(
                    barcode_activation, barcodes,
                    attractor_steps=self.attractor_steps,
                    attractor_rate=self.attractor_rate,
                )

            combined = SeparationCompletionDuality.compute_combined_scores(
                content_scores, barcode_scores, lam
            )

            if int(np.argmax(combined)) == true_indices[i]:
                dual_correct += 1
            if int(np.argmax(content_scores)) == true_indices[i]:
                content_correct += 1
            if int(np.argmax(barcode_scores)) == true_indices[i]:
                barcode_correct += 1

        n = len(queries)
        return {
            "accuracy": dual_correct / n,
            "content_accuracy": content_correct / n,
            "barcode_accuracy": barcode_correct / n,
        }

    @property
    def sparsity_ratio(self) -> float:
        return self.barcode_sparsity / self.barcode_dim if self.barcode_dim > 0 else 0.0
