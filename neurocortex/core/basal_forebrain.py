from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .memory_trace import NeuromodulatoryState

logger = logging.getLogger(__name__)


class BasalForebrain:
    """
    Basal Forebrain-inspired neuromodulatory control module.

    Implements artificial neuromodulatory signals that gate
    information flow between brain regions, inspired by the
    cholinergic and dopaminergic systems:

    - Acetylcholine (ACh): Signals novelty → promotes new encoding
      in the hippocampus. High ACh = more new memories stored.
    - Dopamine (DA): Signals reward/success → promotes consolidation
      of rewarded experiences. High DA = stronger consolidation.
    - Serotonin (5-HT): Modulates social memory importance.
    - Norepinephrine (NE): Controls arousal level → affects
      encoding precision and attention.

    Innovation: These neuromodulators create a dynamic, context-sensitive
    gating mechanism that determines WHEN and HOW STRONGLY memories
    are formed - a principle absent from all existing LLM memory systems.
    """

    def __init__(
        self,
        novelty_decay: float = 0.95,
        reward_decay: float = 0.9,
        baseline_ach: float = 0.5,
        baseline_da: float = 0.5,
        baseline_5ht: float = 0.5,
        baseline_ne: float = 0.5,
    ):
        self.novelty_decay = novelty_decay
        self.reward_decay = reward_decay

        self._state = NeuromodulatoryState(
            acetylcholine=baseline_ach,
            dopamine=baseline_da,
            serotonin=baseline_5ht,
            norepinephrine=baseline_ne,
        )

        self._baseline = NeuromodulatoryState(
            acetylcholine=baseline_ach,
            dopamine=baseline_da,
            serotonin=baseline_5ht,
            norepinephrine=baseline_ne,
        )

    def compute_novelty(
        self,
        embedding: np.ndarray,
        existing_embeddings: Optional[np.ndarray],
    ) -> float:
        """
        Compute novelty signal based on how different the current
        input is from all existing memory representations.

        High novelty → high ACh release → promotes hippocampal encoding.
        Low novelty → low ACh → promotes neocortical retrieval.
        """
        if existing_embeddings is None or len(existing_embeddings) == 0:
            return 1.0

        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return 0.0

        query = embedding / norm
        mat_norms = np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
        mat_norms = np.maximum(mat_norms, 1e-8)
        sims = (existing_embeddings / mat_norms) @ query

        max_similarity = float(np.max(sims))
        novelty = 1.0 - max_similarity

        self._state.acetylcholine = min(
            1.0, self._baseline.acetylcholine + novelty * 0.5
        )

        if novelty > 0.7:
            self._state.norepinephrine = min(
                1.0, self._baseline.norepinephrine + 0.3
            )

        return float(novelty)

    def compute_reward(self, feedback: float) -> float:
        """
        Compute reward signal from interaction feedback.

        Positive feedback → dopamine release → promotes consolidation.
        Negative feedback → dopamine suppression → promotes forgetting.
        """
        reward = max(0.0, min(1.0, 0.5 + feedback * 0.5))

        self._state.dopamine = (
            self._state.dopamine * self.reward_decay
            + reward * (1.0 - self.reward_decay)
        )

        return reward

    def update_social_signal(self, is_social_interaction: bool) -> None:
        """
        Update serotonin levels based on social interaction detection.
        Social interactions boost serotonin, which enhances social
        memory encoding.
        """
        if is_social_interaction:
            self._state.serotonin = min(
                1.0, self._state.serotonin + 0.2
            )
        else:
            self._state.serotonin = (
                self._state.serotonin * 0.95
                + self._baseline.serotonin * 0.05
            )

    def decay_to_baseline(self) -> None:
        """
        Gradually decay neuromodulatory levels back to baseline,
        mimicking the homeostatic regulation in the brain.
        """
        decay = 0.98
        self._state.acetylcholine = (
            self._state.acetylcholine * decay
            + self._baseline.acetylcholine * (1 - decay)
        )
        self._state.dopamine = (
            self._state.dopamine * decay
            + self._baseline.dopamine * (1 - decay)
        )
        self._state.norepinephrine = (
            self._state.norepinephrine * decay
            + self._baseline.norepinephrine * (1 - decay)
        )

    @property
    def state(self) -> NeuromodulatoryState:
        return self._state

    @property
    def encoding_gate(self) -> float:
        return self._state.encoding_gate

    @property
    def consolidation_gate(self) -> float:
        return self._state.consolidation_gate

    def get_status(self) -> Dict:
        return self._state.to_dict()
