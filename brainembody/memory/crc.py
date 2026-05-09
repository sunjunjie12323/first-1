"""
Counterfactual Replay Consolidation (CRC)
反事实回放巩固

Neuroscience Basis:
    Wilson & McNaughton (1994) - Reactivation of hippocampal memory traces during sleep
    Dragoi & Tonegawa (2011) - Preplay of future experiences in hippocampal circuits
    Gupta et al. (2010) - Hippocampal replay generates novel sequences
    Foster & Wilson (2006) - Reverse replay of behavioral sequences

Core Idea:
    During hippocampal replay, the brain doesn't just replay past experiences verbatim.
    It generates:
    1. Forward replay: sequential replay of past experience
    2. Reverse replay: replay in reverse order (for credit assignment)
    3. PREPLAY: novel sequences that haven't been experienced yet
    4. Counterfactual replay: alternative outcomes of past experiences

    CRC implements all four replay modes for memory consolidation:
    - Forward replay strengthens temporal associations
    - Reverse replay strengthens outcome-to-cause associations
    - Preplay creates anticipatory memories for future scenarios
    - Counterfactual replay creates robust memories that generalize to environmental changes

    This is fundamentally different from:
    - ZenBrain's sleep: only does forward replay + selection (no counterfactual/preplay)
    - RL experience replay: just re-samples past experiences (no generation)
    - Data augmentation: random perturbation (no structured counterfactual generation)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ReplayMode(Enum):
    FORWARD = "forward"
    REVERSE = "reverse"
    PREPLAY = "preplay"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class ReplayExperience:
    memory_id: str
    content_embedding: np.ndarray
    content_text: str = ""
    context_embedding: Optional[np.ndarray] = None
    outcome_embedding: Optional[np.ndarray] = None
    temporal_position: float = 0.0
    importance: float = 0.5
    replay_count: int = 0
    prediction_error: float = 0.0


@dataclass
class CounterfactualVariant:
    source_id: str
    variant_embedding: np.ndarray
    variant_text: str = ""
    perturbation_type: str = ""
    perturbation_magnitude: float = 0.0
    replay_mode: ReplayMode = ReplayMode.COUNTERFACTUAL


@dataclass
class CRCConfig:
    n_forward_replay: int = 5
    n_reverse_replay: int = 3
    n_preplay: int = 3
    n_counterfactual: int = 5
    counterfactal_magnitude: float = 0.3
    preplay_extrapolation: float = 0.2
    consolidation_learning_rate: float = 0.1
    replay_temperature: float = 1.0
    min_replay_importance: float = 0.2
    max_total_replay: int = 50


class CounterfactualReplayConsolidation:
    """
    Counterfactual Replay Consolidation (CRC)

    Four-mode hippocampal replay for memory consolidation:
    forward, reverse, preplay, and counterfactual.
    """

    def __init__(self, embedding_dim: int = 384, config: CRCConfig = None,
                 embedder=None):
        self.embedding_dim = embedding_dim
        self.config = config or CRCConfig()
        self.embedder = embedder

        self.experiences: Dict[str, ReplayExperience] = {}
        self.temporal_sequence: List[str] = []
        self.global_time = 0.0

        self.replay_log: List[Dict] = []
        self.counterfactual_store: Dict[str, CounterfactualVariant] = {}

    def register_experience(self, memory_id: str, content_embedding: np.ndarray,
                           content_text: str = "",
                           context_embedding: Optional[np.ndarray] = None,
                           outcome_embedding: Optional[np.ndarray] = None,
                           importance: float = 0.5,
                           prediction_error: float = 0.0):
        self.global_time += 1.0

        exp = ReplayExperience(
            memory_id=memory_id,
            content_embedding=content_embedding.copy(),
            content_text=content_text,
            context_embedding=context_embedding.copy() if context_embedding is not None else None,
            outcome_embedding=outcome_embedding.copy() if outcome_embedding is not None else None,
            temporal_position=self.global_time,
            importance=importance,
            prediction_error=prediction_error
        )
        self.experiences[memory_id] = exp
        self.temporal_sequence.append(memory_id)

    def _select_replay_candidates(self, n: int, mode: ReplayMode) -> List[str]:
        if not self.experiences:
            return []

        candidates = []
        for mid, exp in self.experiences.items():
            score = exp.importance

            if mode == ReplayMode.FORWARD:
                recency = np.exp(-0.01 * (self.global_time - exp.temporal_position))
                score = exp.importance * 0.5 + recency * 0.3 + exp.prediction_error * 0.2

            elif mode == ReplayMode.REVERSE:
                score = exp.importance * 0.4 + exp.prediction_error * 0.4 + (1.0 / (exp.replay_count + 1)) * 0.2

            elif mode == ReplayMode.PREPLAY:
                score = exp.importance * 0.3 + exp.prediction_error * 0.5 + (1.0 / (exp.replay_count + 1)) * 0.2

            elif mode == ReplayMode.COUNTERFACTUAL:
                score = exp.prediction_error * 0.5 + exp.importance * 0.3 + (1.0 / (exp.replay_count + 1)) * 0.2

            candidates.append((mid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:n]]

    def _generate_forward_replay(self, candidate_ids: List[str]) -> List[Dict]:
        replays = []
        for mid in candidate_ids:
            if mid not in self.experiences:
                continue
            exp = self.experiences[mid]

            next_idx = self.temporal_sequence.index(mid) + 1 if mid in self.temporal_sequence else -1
            next_mid = self.temporal_sequence[next_idx] if 0 <= next_idx < len(self.temporal_sequence) else None

            association_strength = 0.0
            if next_mid and next_mid in self.experiences:
                next_exp = self.experiences[next_mid]
                association_strength = float(np.dot(exp.content_embedding, next_exp.content_embedding) /
                                           (np.linalg.norm(exp.content_embedding) *
                                            np.linalg.norm(next_exp.content_embedding) + 1e-8))

            replays.append({
                "mode": "forward",
                "source_id": mid,
                "association_with_next": association_strength,
                "strengthen_factor": 1.0 + 0.1 * association_strength
            })
            exp.replay_count += 1

        return replays

    def _generate_reverse_replay(self, candidate_ids: List[str]) -> List[Dict]:
        replays = []
        for mid in candidate_ids:
            if mid not in self.experiences:
                continue
            exp = self.experiences[mid]

            prev_idx = self.temporal_sequence.index(mid) - 1 if mid in self.temporal_sequence else -1
            prev_mid = self.temporal_sequence[prev_idx] if 0 <= prev_idx < len(self.temporal_sequence) else None

            if prev_mid and prev_mid in self.experiences:
                prev_exp = self.experiences[prev_mid]
                outcome_cause_strength = float(np.dot(exp.content_embedding, prev_exp.content_embedding) /
                                              (np.linalg.norm(exp.content_embedding) *
                                               np.linalg.norm(prev_exp.content_embedding) + 1e-8))

                replays.append({
                    "mode": "reverse",
                    "source_id": mid,
                    "cause_id": prev_mid,
                    "outcome_cause_strength": outcome_cause_strength,
                    "strengthen_factor": 1.0 + 0.15 * outcome_cause_strength
                })
                exp.replay_count += 1

        return replays

    def _generate_preplay(self, candidate_ids: List[str]) -> List[CounterfactualVariant]:
        variants = []
        for mid in candidate_ids:
            if mid not in self.experiences:
                continue
            exp = self.experiences[mid]

            if exp.outcome_embedding is not None:
                direction = exp.outcome_embedding - exp.content_embedding
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction /= direction_norm

                for i in range(self.config.n_preplay):
                    magnitude = self.config.preplay_extrapolation * (i + 1)
                    variant_emb = exp.content_embedding + direction * magnitude
                    variant_emb /= (np.linalg.norm(variant_emb) + 1e-8)

                    variant_id = f"preplay_{mid}_{i}"
                    variant = CounterfactualVariant(
                        source_id=mid,
                        variant_embedding=variant_emb,
                        variant_text=f"[preplay:{i}] {exp.content_text[:50]}",
                        perturbation_type="extrapolation",
                        perturbation_magnitude=magnitude,
                        replay_mode=ReplayMode.PREPLAY
                    )
                    self.counterfactual_store[variant_id] = variant
                    variants.append(variant)

            else:
                noise = np.random.randn(self.embedding_dim) * self.config.preplay_extrapolation
                variant_emb = exp.content_embedding + noise
                variant_emb /= (np.linalg.norm(variant_emb) + 1e-8)

                variant_id = f"preplay_{mid}_0"
                variant = CounterfactualVariant(
                    source_id=mid,
                    variant_embedding=variant_emb,
                    variant_text=f"[preplay] {exp.content_text[:50]}",
                    perturbation_type="noise_extrapolation",
                    perturbation_magnitude=self.config.preplay_extrapolation,
                    replay_mode=ReplayMode.PREPLAY
                )
                self.counterfactual_store[variant_id] = variant
                variants.append(variant)

            exp.replay_count += 1

        return variants

    def _generate_counterfactual(self, candidate_ids: List[str]) -> List[CounterfactualVariant]:
        variants = []
        perturbation_types = ["action_perturbation", "outcome_perturbation", "context_perturbation"]

        for mid in candidate_ids:
            if mid not in self.experiences:
                continue
            exp = self.experiences[mid]

            for i in range(self.config.n_counterfactual):
                ptype = perturbation_types[i % len(perturbation_types)]
                magnitude = self.config.counterfactal_magnitude * (0.5 + 0.5 * np.random.random())

                if ptype == "action_perturbation":
                    noise = np.random.randn(self.embedding_dim) * magnitude
                    variant_emb = exp.content_embedding + noise
                    variant_emb /= (np.linalg.norm(variant_emb) + 1e-8)

                elif ptype == "outcome_perturbation":
                    if exp.outcome_embedding is not None:
                        cf_outcome = exp.outcome_embedding + np.random.randn(self.embedding_dim) * magnitude
                        cf_outcome /= (np.linalg.norm(cf_outcome) + 1e-8)
                        variant_emb = 0.5 * exp.content_embedding + 0.5 * cf_outcome
                        variant_emb /= (np.linalg.norm(variant_emb) + 1e-8)
                    else:
                        variant_emb = exp.content_embedding + np.random.randn(self.embedding_dim) * magnitude
                        variant_emb /= (np.linalg.norm(variant_emb) + 1e-8)

                elif ptype == "context_perturbation":
                    if exp.context_embedding is not None:
                        cf_context = exp.context_embedding + np.random.randn(self.embedding_dim) * magnitude
                        cf_context /= (np.linalg.norm(cf_context) + 1e-8)
                        variant_emb = 0.6 * exp.content_embedding + 0.4 * cf_context
                        variant_emb /= (np.linalg.norm(variant_emb) + 1e-8)
                    else:
                        variant_emb = exp.content_embedding + np.random.randn(self.embedding_dim) * magnitude * 0.5
                        variant_emb /= (np.linalg.norm(variant_emb) + 1e-8)

                variant_id = f"cf_{mid}_{ptype}_{i}"
                variant = CounterfactualVariant(
                    source_id=mid,
                    variant_embedding=variant_emb,
                    variant_text=f"[cf:{ptype}] {exp.content_text[:50]}",
                    perturbation_type=ptype,
                    perturbation_magnitude=magnitude,
                    replay_mode=ReplayMode.COUNTERFACTUAL
                )
                self.counterfactual_store[variant_id] = variant
                variants.append(variant)

            exp.replay_count += 1

        return variants

    def consolidate(self) -> Dict:
        if not self.experiences:
            return {"total_replays": 0, "variants_generated": 0}

        total_replays = 0
        all_variants = []

        forward_ids = self._select_replay_candidates(self.config.n_forward_replay, ReplayMode.FORWARD)
        forward_replays = self._generate_forward_replay(forward_ids)
        total_replays += len(forward_replays)

        reverse_ids = self._select_replay_candidates(self.config.n_reverse_replay, ReplayMode.REVERSE)
        reverse_replays = self._generate_reverse_replay(reverse_ids)
        total_replays += len(reverse_replays)

        preplay_ids = self._select_replay_candidates(self.config.n_preplay, ReplayMode.PREPLAY)
        preplay_variants = self._generate_preplay(preplay_ids)
        all_variants.extend(preplay_variants)

        cf_ids = self._select_replay_candidates(self.config.n_counterfactual, ReplayMode.COUNTERFACTUAL)
        cf_variants = self._generate_counterfactual(cf_ids)
        all_variants.extend(cf_variants)

        for replay in forward_replays + reverse_replays:
            source_id = replay["source_id"]
            if source_id in self.experiences:
                strengthen = replay.get("strengthen_factor", 1.0)
                self.experiences[source_id].importance = min(
                    1.0, self.experiences[source_id].importance * strengthen
                )

        consolidation_report = {
            "total_replays": total_replays,
            "forward_replays": len(forward_replays),
            "reverse_replays": len(reverse_replays),
            "preplay_variants": len(preplay_variants),
            "counterfactual_variants": len(cf_variants),
            "total_variants_generated": len(all_variants),
            "total_stored_variants": len(self.counterfactual_store),
        }

        self.replay_log.append({
            "time": self.global_time,
            "report": consolidation_report
        })

        return consolidation_report

    def get_consolidated_embeddings(self) -> Dict[str, np.ndarray]:
        result = {}
        for mid, exp in self.experiences.items():
            result[mid] = exp.content_embedding.copy()

        for vid, variant in self.counterfactual_store.items():
            result[vid] = variant.variant_embedding.copy()

        return result

    def get_consolidation_stats(self) -> Dict:
        if not self.experiences:
            return {"total_experiences": 0}

        return {
            "total_experiences": len(self.experiences),
            "total_counterfactual_variants": len(self.counterfactual_store),
            "avg_replay_count": np.mean([e.replay_count for e in self.experiences.values()]),
            "avg_importance": np.mean([e.importance for e in self.experiences.values()]),
            "avg_prediction_error": np.mean([e.prediction_error for e in self.experiences.values()]),
            "consolidation_rounds": len(self.replay_log),
        }
