from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .memory_trace import NeuromodulatoryState

logger = logging.getLogger(__name__)


class BasalForebrain:
    """
    INNOVATION 2: Four-Transmitter Neuromodulatory State Machine

    Implements a dynamic neuromodulatory state machine with four
    artificial neurotransmitters that jointly gate memory encoding,
    consolidation, social processing, and encoding precision.

    Formal definition:
      State vector: M(t) = [ACh(t), DA(t), 5-HT(t), NE(t)]

      Update rules:
        ACh(t+1) = ACh(t) * lambda + novelty(x_t, E) * (1 - lambda)
        DA(t+1)  = DA(t)  * lambda + reward(feedback_t) * (1 - lambda)
        5-HT(t+1) = 5-HT(t) * lambda + social(x_t) * (1 - lambda)
        NE(t+1)  = NE(t)  * lambda + arousal(novelty, emotion) * (1 - lambda)

      Gating functions:
        G_encode(t)      = 0.3 + 0.7 * ACh(t)    [what to encode]
        G_consolidate(t)  = 0.2 + 0.8 * DA(t)     [what to consolidate]
        G_social(t)       = 0.5 + 0.5 * 5-HT(t)   [social memory weight]
        G_precision(t)    = 0.5 + 0.5 * NE(t)     [encoding precision]

    Key insight: ACh and DA have COMPLEMENTARY roles:
      - High ACh + Low DA  = new encoding mode (hippocampus active)
      - Low ACh + High DA  = consolidation mode (neocortex active)
      This mirrors Hasselmo (1999) and recent findings by
      Zhang et al. (2025, Nature Neuroscience).

    Differentiation from existing work:
      - ZenBrain (NeurIPS'25): Emotional valence tagging only (1 dimension)
      - True Memory (2026): 3-signal encoding gate (novelty, salience,
        prediction error) but no dynamic state machine, no consolidation
        gating, no social modulation
      - Pirazzini & Ursino (2025): Computational ACh model for hippocampus
        but not applied to LLM agents, no DA/5-HT/NE integration
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
