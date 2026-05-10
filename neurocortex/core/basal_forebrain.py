from __future__ import annotations

import logging
from typing import Dict

from neurocortex.core.memory_trace import NeuromodulatoryState

logger = logging.getLogger(__name__)


class BasalForebrain:
    def __init__(self):
        self.state = NeuromodulatoryState()
        self._ach_baseline = 0.5
        self._da_baseline = 0.5
        self._5ht_baseline = 0.5
        self._ne_baseline = 0.5
        self._decay_rate = 0.05

    def compute_novelty(self, novelty_score: float) -> None:
        target_ach = self._ach_baseline + 0.5 * novelty_score
        self.state.acetylcholine = self._lerp(self.state.acetylcholine, target_ach, 0.3)
        target_ne = self._ne_baseline + 0.3 * novelty_score
        self.state.norepinephrine = self._lerp(self.state.norepinephrine, target_ne, 0.2)
        logger.debug(f"Novelty={novelty_score:.2f} -> ACh={self.state.acetylcholine:.3f}, NE={self.state.norepinephrine:.3f}")

    def compute_reward(self, reward_signal: float) -> None:
        target_da = self._da_baseline + 0.5 * max(0.0, reward_signal)
        self.state.dopamine = self._lerp(self.state.dopamine, target_da, 0.3)
        target_5ht = self._5ht_baseline - 0.2 * max(0.0, reward_signal)
        self.state.serotonin = self._lerp(self.state.serotonin, target_5ht, 0.2)
        logger.debug(f"Reward={reward_signal:.2f} -> DA={self.state.dopamine:.3f}, 5-HT={self.state.serotonin:.3f}")

    def get_encoding_gate(self) -> float:
        return self.state.encoding_gate

    def get_consolidation_gate(self) -> float:
        return self.state.consolidation_gate

    def homeostatic_decay(self) -> None:
        self.state.acetylcholine = self._lerp(self.state.acetylcholine, self._ach_baseline, self._decay_rate)
        self.state.dopamine = self._lerp(self.state.dopamine, self._da_baseline, self._decay_rate)
        self.state.serotonin = self._lerp(self.state.serotonin, self._5ht_baseline, self._decay_rate)
        self.state.norepinephrine = self._lerp(self.state.norepinephrine, self._ne_baseline, self._decay_rate)

    def get_state(self) -> NeuromodulatoryState:
        return self.state

    @staticmethod
    def _lerp(current: float, target: float, rate: float) -> float:
        return float(current + rate * (target - current))
