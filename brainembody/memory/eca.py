"""
Engram Competition Allocation (ECA)
记忆印迹竞争分配

Neuroscience Basis:
    Han et al. (2007) - CREB-dependent competition for memory allocation in hippocampus
    Silva et al. (2009) - Molecular and cellular mechanisms of engram allocation
    Rashid et al. (2016) - Competition between engrams determines memory allocation

Core Idea:
    In the hippocampus, engram cells compete for memory allocation via CREB-dependent
    lateral inhibition. Cells with higher CREB activity have higher excitability and
    win the competition to become part of the engram. This creates natural capacity
    limits, interference effects, and novelty-based prioritization.

    Unlike all existing memory systems that use:
    - Store-everything (RAG)
    - Hand-crafted importance scores
    - FIFO/LRU eviction
    - Fixed capacity with random replacement

    ECA uses biologically-plausible competitive allocation where:
    - Each memory slot has an "excitability" score (analogous to CREB activity)
    - Novelty/surprise increases excitability
    - Lateral inhibition creates competition between candidate slots
    - Winners form the engram, losers are suppressed
    - This naturally produces: capacity limits, interference, novelty effects,
      and the spacing effect (distributed encoding is stronger than massed)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EngramCell:
    memory_id: str = ""
    content_embedding: np.ndarray = None
    excitability: float = 0.5
    allocation_strength: float = 0.0
    novelty_score: float = 0.0
    access_count: int = 0
    last_access_time: float = 0.0
    is_allocated: bool = False
    age: float = 0.0
    consolidation_level: float = 0.0


@dataclass
class ECAConfig:
    capacity: int = 500
    base_excitability: float = 0.5
    novelty_boost: float = 0.4
    surprise_boost: float = 0.6
    lateral_inhibition_strength: float = 0.3
    decay_rate: float = 0.01
    consolidation_rate: float = 0.05
    competition_temperature: float = 1.0
    reactivation_threshold: float = 0.6
    spacing_effect_strength: float = 0.2


class EngramCompetitionAllocation:
    """
    Engram Competition Allocation (ECA)

    Memory allocation via lateral inhibition competition between neural assemblies,
    inspired by engram cell competition in hippocampus.
    """

    def __init__(self, embedding_dim: int = 384, config: ECAConfig = None):
        self.embedding_dim = embedding_dim
        self.config = config or ECAConfig()

        self.cells: Dict[str, EngramCell] = {}
        self.cell_order: List[str] = []
        self.global_time = 0.0
        self.allocation_history: List[Dict] = []

    def _compute_novelty(self, embedding: np.ndarray, n_sample: int = 50) -> float:
        if not self.cells:
            return 1.0

        allocated = [c for c in self.cells.values() if c.is_allocated and c.content_embedding is not None]
        if not allocated:
            return 1.0

        if len(allocated) > n_sample:
            indices = np.random.choice(len(allocated), n_sample, replace=False)
            sampled = [allocated[i] for i in indices]
        else:
            sampled = allocated

        max_similarity = 0.0
        for cell in sampled:
            sim = float(np.dot(embedding, cell.content_embedding) /
                       (np.linalg.norm(embedding) * np.linalg.norm(cell.content_embedding) + 1e-8))
            max_similarity = max(max_similarity, sim)

        return 1.0 - max_similarity

    def _compute_surprise(self, embedding: np.ndarray, predicted_embedding: Optional[np.ndarray] = None) -> float:
        if predicted_embedding is None:
            if not self.cells:
                return 0.5
            recent_cells = [c for c in self.cells.values()
                          if c.is_allocated and c.content_embedding is not None
                          and self.global_time - c.last_access_time < 10.0]
            if not recent_cells:
                return 0.3

            avg_embedding = np.mean([c.content_embedding for c in recent_cells], axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding /= norm
            predicted_embedding = avg_embedding

        prediction_error = 1.0 - float(np.dot(embedding, predicted_embedding) /
                                        (np.linalg.norm(embedding) * np.linalg.norm(predicted_embedding) + 1e-8))
        return np.clip(prediction_error, 0.0, 1.0)

    def _update_excitabilities(self):
        for cell in self.cells.values():
            time_decay = np.exp(-self.config.decay_rate * (self.global_time - cell.last_access_time))
            access_boost = min(cell.access_count * 0.05, 0.3)
            consolidation_bonus = cell.consolidation_level * 0.2

            cell.excitability = (self.config.base_excitability * time_decay +
                                cell.novelty_score * 0.3 +
                                access_boost +
                                consolidation_bonus)
            cell.excitability = np.clip(cell.excitability, 0.1, 1.0)

    def _lateral_inhibition(self, new_memory_id: str, new_excitability: float,
                            new_embedding: np.ndarray, n_competitors: int = 20) -> Tuple[float, List[Tuple[str, float]]]:
        if not self.cells:
            return new_excitability, []

        allocated = [(mid, c) for mid, c in self.cells.items() if c.is_allocated and c.content_embedding is not None]
        if not allocated:
            return new_excitability, []

        sims = []
        for mid, cell in allocated:
            sim = float(np.dot(new_embedding, cell.content_embedding) /
                       (np.linalg.norm(new_embedding) * np.linalg.norm(cell.content_embedding) + 1e-8))
            sims.append((mid, sim, cell.excitability))

        sims.sort(key=lambda x: x[1], reverse=True)
        top_competitors = sims[:n_competitors]

        total_inhibition = 0.0
        for mid, sim, excit in top_competitors:
            inhibition = self.config.lateral_inhibition_strength * sim * excit
            total_inhibition += inhibition

        inhibited_excitability = new_excitability - total_inhibition
        inhibited_excitability = max(inhibited_excitability, 0.05)

        return inhibited_excitability, top_competitors

    def _spacing_effect(self, embedding: np.ndarray, n_sample: int = 20) -> float:
        if not self.cells:
            return 1.0

        allocated = [c for c in self.cells.values() if c.is_allocated and c.content_embedding is not None]
        if not allocated:
            return 1.0

        if len(allocated) > n_sample:
            indices = np.random.choice(len(allocated), n_sample, replace=False)
            sampled = [allocated[i] for i in indices]
        else:
            sampled = allocated

        similar_cells = []
        for cell in sampled:
            sim = float(np.dot(embedding, cell.content_embedding) /
                       (np.linalg.norm(embedding) * np.linalg.norm(cell.content_embedding) + 1e-8))
            if sim > 0.7:
                time_gap = self.global_time - cell.last_access_time
                similar_cells.append((sim, time_gap))

        if not similar_cells:
            return 1.0

        spacing_bonus = 0.0
        for sim, time_gap in similar_cells:
            spacing_bonus += self.config.spacing_effect_strength * sim * (1.0 - np.exp(-0.1 * time_gap))

        return 1.0 + min(spacing_bonus, 0.5)

    def allocate(self, memory_id: str, embedding: np.ndarray,
                 predicted_embedding: Optional[np.ndarray] = None) -> Tuple[bool, Dict]:
        self.global_time += 1.0

        novelty = self._compute_novelty(embedding)
        surprise = self._compute_surprise(embedding, predicted_embedding)
        spacing = self._spacing_effect(embedding)

        excitability = (self.config.base_excitability +
                       novelty * self.config.novelty_boost +
                       surprise * self.config.surprise_boost) * spacing
        excitability = np.clip(excitability, 0.1, 1.5)

        if memory_id in self.cells:
            cell = self.cells[memory_id]
            cell.access_count += 1
            cell.last_access_time = self.global_time
            cell.excitability = max(cell.excitability, excitability * 0.5)
            cell.consolidation_level = min(1.0, cell.consolidation_level + self.config.consolidation_rate)
            return True, {"action": "reactivated", "excitability": cell.excitability}

        allocated_count = sum(1 for c in self.cells.values() if c.is_allocated)

        if allocated_count < self.config.capacity:
            cell = EngramCell(
                memory_id=memory_id,
                content_embedding=embedding.copy(),
                excitability=excitability,
                allocation_strength=excitability,
                novelty_score=novelty,
                access_count=1,
                last_access_time=self.global_time,
                is_allocated=True,
                age=0.0,
                consolidation_level=0.0
            )
            self.cells[memory_id] = cell
            self.cell_order.append(memory_id)
            return True, {"action": "allocated", "excitability": excitability, "novelty": novelty}

        inhibited_excitability, competitors = self._lateral_inhibition(
            memory_id, excitability, embedding
        )

        if not competitors:
            self.allocation_history.append({
                "time": self.global_time,
                "memory_id": memory_id,
                "action": "rejected",
                "excitability": inhibited_excitability
            })
            return False, {"action": "rejected", "excitability": inhibited_excitability}

        losers = [(mid, cell_excit) for mid, sim, cell_excit in competitors
                  if cell_excit < inhibited_excitability]

        if not losers:
            self.allocation_history.append({
                "time": self.global_time,
                "memory_id": memory_id,
                "action": "rejected",
                "excitability": inhibited_excitability
            })
            return False, {"action": "rejected", "excitability": inhibited_excitability}

        losers.sort(key=lambda x: x[1])
        victim_id, victim_excit = losers[0]

        if victim_excit >= inhibited_excitability:
            return False, {"action": "rejected", "excitability": inhibited_excitability}

        del self.cells[victim_id]
        if victim_id in self.cell_order:
            self.cell_order.remove(victim_id)

        cell = EngramCell(
            memory_id=memory_id,
            content_embedding=embedding.copy(),
            excitability=excitability,
            allocation_strength=excitability,
            novelty_score=novelty,
            access_count=1,
            last_access_time=self.global_time,
            is_allocated=True,
            age=0.0,
            consolidation_level=0.0
        )
        self.cells[memory_id] = cell
        self.cell_order.append(memory_id)

        self.allocation_history.append({
            "time": self.global_time,
            "memory_id": memory_id,
            "action": "displaced",
            "displaced": victim_id,
            "new_excitability": excitability,
            "victim_excitability": victim_excit,
            "novelty": novelty,
            "surprise": surprise
        })

        return True, {"action": "displaced", "displaced": victim_id,
                      "excitability": excitability, "novelty": novelty, "surprise": surprise}

    def reactivate(self, memory_id: str) -> float:
        if memory_id not in self.cells:
            return 0.0

        cell = self.cells[memory_id]
        cell.access_count += 1
        cell.last_access_time = self.global_time
        cell.excitability = min(1.0, cell.excitability + 0.1)
        cell.consolidation_level = min(1.0, cell.consolidation_level + self.config.consolidation_rate)

        return cell.excitability

    def get_allocation_stats(self) -> Dict:
        if not self.cells:
            return {"total_allocated": 0, "avg_excitability": 0, "avg_consolidation": 0}

        allocated = [c for c in self.cells.values() if c.is_allocated]
        return {
            "total_allocated": len(allocated),
            "capacity": self.config.capacity,
            "utilization": len(allocated) / self.config.capacity,
            "avg_excitability": np.mean([c.excitability for c in allocated]) if allocated else 0,
            "avg_consolidation": np.mean([c.consolidation_level for c in allocated]) if allocated else 0,
            "avg_novelty": np.mean([c.novelty_score for c in allocated]) if allocated else 0,
            "displacement_count": sum(1 for h in self.allocation_history if h["action"] == "displaced"),
            "rejection_count": sum(1 for h in self.allocation_history if h["action"] == "rejected"),
        }

    def decay_all(self, dt: float = 1.0):
        self.global_time += dt
        for cell in self.cells.values():
            cell.age += dt
            cell.excitability *= (1.0 - self.config.decay_rate * dt)
            cell.excitability = max(cell.excitability, 0.1)
