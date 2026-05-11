from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class MemoryPhase(Enum):
    SENSORY = "sensory"
    EPISODIC = "episodic"
    CONSOLIDATING = "consolidating"
    SEMANTIC = "semantic"


@dataclass
class ContextTag:
    spatial: Optional[str] = None
    temporal_period: Optional[str] = None
    interlocutor: Optional[str] = None
    activity: Optional[str] = None
    modality: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spatial": self.spatial,
            "temporal_period": self.temporal_period,
            "interlocutor": self.interlocutor,
            "activity": self.activity,
            "modality": self.modality,
        }

    def overlap(self, other: ContextTag) -> float:
        matches = 0
        total = 0
        for attr in ("spatial", "temporal_period", "interlocutor", "activity", "modality"):
            a = getattr(self, attr)
            b = getattr(other, attr)
            if a is not None and b is not None:
                total += 1
                if a == b:
                    matches += 1
        return matches / total if total > 0 else 0.0


@dataclass
class EpisodicTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    barcode: np.ndarray = field(default_factory=lambda: np.array([]))
    context: ContextTag = field(default_factory=ContextTag)
    importance: float = 0.5
    emotional_valence: float = 0.0
    consolidation_level: float = 0.0
    reactivation_count: int = 0
    last_reactivation: Optional[datetime] = None
    decay_rate: float = 0.1
    associations: List[str] = field(default_factory=list)
    source: str = ""
    novelty_score: float = 0.0
    phase: MemoryPhase = MemoryPhase.EPISODIC

    @property
    def memory_strength(self) -> float:
        age_hours = (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 3600.0
        decay_factor = np.exp(-self.decay_rate * age_hours)
        consolidation_factor = 1.0 + self.consolidation_level
        reactivation_factor = 1.0 + np.log1p(self.reactivation_count)
        return float(decay_factor * consolidation_factor * reactivation_factor)

    def reactivate(self) -> None:
        self.reactivation_count += 1
        self.last_reactivation = datetime.now(timezone.utc)
        self.consolidation_level = min(1.0, self.consolidation_level + 0.05)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "embedding_shape": list(self.embedding.shape) if self.embedding.size > 0 else [],
            "barcode_shape": list(self.barcode.shape) if self.barcode.size > 0 else [],
            "context": self.context.to_dict(),
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "consolidation_level": self.consolidation_level,
            "reactivation_count": self.reactivation_count,
            "last_reactivation": self.last_reactivation.isoformat() if self.last_reactivation else None,
            "decay_rate": self.decay_rate,
            "associations": self.associations,
            "source": self.source,
            "novelty_score": self.novelty_score,
            "phase": self.phase.value,
            "memory_strength": self.memory_strength,
        }


@dataclass
class SemanticSchema:
    schema_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gist: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    source_traces: List[str] = field(default_factory=list)
    confidence: float = 0.5
    key_entities: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)
    reinforcement_count: int = 0

    @property
    def maturity(self) -> float:
        return float(min(1.0, self.confidence * (1.0 + np.log1p(self.reinforcement_count)) / 2.0))

    def reinforce(self, additional_confidence: float = 0.1) -> None:
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + additional_confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "gist": self.gist,
            "embedding_shape": list(self.embedding.shape) if self.embedding.size > 0 else [],
            "source_traces": self.source_traces,
            "confidence": self.confidence,
            "key_entities": self.key_entities,
            "associations": self.associations,
            "reinforcement_count": self.reinforcement_count,
            "maturity": self.maturity,
        }


@dataclass
class ReconstructedMemory:
    reconstruction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    reconstructed_narrative: str = ""
    source_traces: List[str] = field(default_factory=list)
    source_schemas: List[str] = field(default_factory=list)
    confidence: float = 0.0
    distortion_score: float = 0.0
    emotional_tone: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconstruction_id": self.reconstruction_id,
            "query": self.query,
            "reconstructed_narrative": self.reconstructed_narrative,
            "source_traces": self.source_traces,
            "source_schemas": self.source_schemas,
            "confidence": self.confidence,
            "distortion_score": self.distortion_score,
            "emotional_tone": self.emotional_tone,
        }


@dataclass
class NeuromodulatoryState:
    acetylcholine: float = 0.5
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.5

    @property
    def encoding_gate(self) -> float:
        return float(0.3 + 0.7 * self.acetylcholine)

    @property
    def consolidation_gate(self) -> float:
        return float(0.2 + 0.8 * self.dopamine)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acetylcholine": self.acetylcholine,
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
            "encoding_gate": self.encoding_gate,
            "consolidation_gate": self.consolidation_gate,
        }
