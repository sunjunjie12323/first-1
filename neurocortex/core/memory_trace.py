from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class MemoryPhase(Enum):
    SENSORY = "sensory"
    EPISODIC = "episodic"
    CONSOLIDATING = "consolidating"
    SEMANTIC = "semantic"


class EmotionalValence(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class ContextTag:
    spatial: Optional[str] = None
    temporal_period: Optional[str] = None
    interlocutor: Optional[str] = None
    activity: Optional[str] = None
    modality: str = "verbal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spatial": self.spatial,
            "temporal_period": self.temporal_period,
            "interlocutor": self.interlocutor,
            "activity": self.activity,
            "modality": self.modality,
        }


@dataclass
class EpisodicTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    embedding: Optional[np.ndarray] = None
    context: ContextTag = field(default_factory=ContextTag)
    importance: float = 0.5
    emotional_valence: float = 0.0
    consolidation_level: float = 0.0
    reactivation_count: int = 0
    last_reactivation: Optional[datetime] = None
    decay_rate: float = 0.1
    associations: List[str] = field(default_factory=list)
    source: str = "unknown"
    phase: MemoryPhase = MemoryPhase.EPISODIC
    novelty_score: float = 0.0
    reward_score: float = 0.0
    compressed_gist: Optional[str] = None

    @property
    def memory_strength(self) -> float:
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600.0
        consolidation_factor = 1.0 + self.consolidation_level * 3.0
        reactivation_factor = 1.0 + np.log1p(self.reactivation_count)
        importance_factor = 0.5 + self.importance * 1.5
        decay = np.exp(-self.decay_rate * age_hours / consolidation_factor)
        return float(decay * reactivation_factor * importance_factor)

    @property
    def is_decayed(self) -> bool:
        return self.memory_strength < 0.05

    def reactivate(self) -> None:
        self.reactivation_count += 1
        self.last_reactivation = datetime.now()
        self.decay_rate *= 0.85

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "context": self.context.to_dict(),
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "consolidation_level": self.consolidation_level,
            "reactivation_count": self.reactivation_count,
            "memory_strength": self.memory_strength,
            "source": self.source,
            "phase": self.phase.value,
            "novelty_score": self.novelty_score,
            "associations": self.associations,
        }


@dataclass
class SemanticSchema:
    schema_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)
    gist: str = ""
    embedding: Optional[np.ndarray] = None
    source_traces: List[str] = field(default_factory=list)
    confidence: float = 0.0
    access_count: int = 0
    associations: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    abstract_concepts: List[str] = field(default_factory=list)
    consolidation_rounds: int = 0

    @property
    def maturity(self) -> float:
        return min(1.0, self.confidence * (1.0 + np.log1p(self.consolidation_rounds) / 5.0))

    def reinforce(self) -> None:
        self.access_count += 1
        self.updated = datetime.now()
        self.confidence = min(1.0, self.confidence + 0.05)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "gist": self.gist,
            "source_traces": self.source_traces,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "key_entities": self.key_entities,
            "abstract_concepts": self.abstract_concepts,
            "maturity": self.maturity,
            "associations": self.associations,
        }


@dataclass
class WorkingMemoryItem:
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[np.ndarray] = None
    attention_weight: float = 1.0
    source: str = "current"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content": self.content,
            "attention_weight": self.attention_weight,
            "source": self.source,
        }


@dataclass
class NeuromodulatoryState:
    acetylcholine: float = 0.5
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.5

    @property
    def novelty_signal(self) -> float:
        return self.acetylcholine

    @property
    def reward_signal(self) -> float:
        return self.dopamine

    @property
    def social_signal(self) -> float:
        return self.serotonin

    @property
    def arousal_signal(self) -> float:
        return self.norepinephrine

    @property
    def encoding_gate(self) -> float:
        return 0.3 + 0.7 * self.novelty_signal

    @property
    def consolidation_gate(self) -> float:
        return 0.2 + 0.8 * self.reward_signal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acetylcholine": self.acetylcholine,
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
            "encoding_gate": self.encoding_gate,
            "consolidation_gate": self.consolidation_gate,
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
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconstruction_id": self.reconstruction_id,
            "query": self.query,
            "reconstructed_narrative": self.reconstructed_narrative,
            "source_traces": self.source_traces,
            "source_schemas": self.source_schemas,
            "confidence": self.confidence,
            "distortion_score": self.distortion_score,
        }
