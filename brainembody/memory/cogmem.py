"""
CogMem: Cognitive Memory for Embodied Intelligence
具身智能认知记忆框架

Three Core Innovations:
1. ECA (Engram Competition Allocation) - 记忆印迹竞争分配
   Memory allocation via lateral inhibition competition, inspired by
   CREB-dependent engram cell competition in hippocampus.

2. CRC (Counterfactual Replay Consolidation) - 反事实回放巩固
   Four-mode hippocampal replay (forward, reverse, preplay, counterfactual)
   for memory consolidation during sleep/idle periods.

3. SSDR (Sensorimotor State-Dependent Retrieval) - 感知运动状态依赖检索
   Context-dependent memory retrieval modulated by the agent's full
   sensorimotor state (position, action, held objects, motor state).

Paper Positioning:
    The first memory framework that integrates engram competition allocation,
    counterfactual replay consolidation, and sensorimotor state-dependent
    retrieval for embodied intelligence agents.

    Unlike existing embodied memory systems (RoboMemory, Memo) that treat
    memory as a database with fixed allocation and semantic-only retrieval,
    CogMem treats memory as a biological system with competitive allocation,
    generative consolidation, and state-dependent retrieval.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from brainembody.memory.eca import EngramCompetitionAllocation, ECAConfig, EngramCell
from brainembody.memory.crc import CounterfactualReplayConsolidation, CRCConfig
from brainembody.memory.ssdr import SensorimotorStateRetrieval, SSDRConfig, SensorimotorState


@dataclass
class CogMemConfig:
    embedding_dim: int = 384
    eca_config: ECAConfig = None
    crc_config: CRCConfig = None
    ssdr_config: SSDRConfig = None
    enable_eca: bool = True
    enable_crc: bool = True
    enable_ssdr: bool = True
    consolidation_interval: int = 50
    auto_consolidate: bool = True


class CogMemMemory:
    """
    CogMem: Cognitive Memory for Embodied Intelligence

    Integrates three biologically-inspired innovations:
    - ECA for memory allocation
    - CRC for memory consolidation
    - SSDR for memory retrieval
    """

    def __init__(self, embedding_dim: int = 384, embedder=None,
                 config: CogMemConfig = None):
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self.config = config or CogMemConfig()
        if self.config.eca_config is None:
            self.config.eca_config = ECAConfig()
        if self.config.crc_config is None:
            self.config.crc_config = CRCConfig()
        if self.config.ssdr_config is None:
            self.config.ssdr_config = SSDRConfig()

        self.eca = EngramCompetitionAllocation(
            embedding_dim=embedding_dim,
            config=self.config.eca_config
        ) if self.config.enable_eca else None

        self.crc = CounterfactualReplayConsolidation(
            embedding_dim=embedding_dim,
            config=self.config.crc_config,
            embedder=embedder
        ) if self.config.enable_crc else None

        self.ssdr = SensorimotorStateRetrieval(
            embedding_dim=embedding_dim,
            config=self.config.ssdr_config
        ) if self.config.enable_ssdr else None

        self.memories: Dict[str, Dict] = {}
        self.encode_count = 0
        self.retrieve_count = 0

    def _get_embedding(self, content: str) -> np.ndarray:
        if self.embedder is not None:
            emb = self.embedder.embed(content)
            if emb.shape[0] != self.embedding_dim:
                if len(emb.shape) == 1:
                    padded = np.zeros(self.embedding_dim)
                    padded[:min(len(emb), self.embedding_dim)] = emb[:self.embedding_dim]
                    return padded
            return emb
        return np.random.randn(self.embedding_dim) * 0.1

    def encode(self, content: str,
               sensorimotor_state: Optional[SensorimotorState] = None,
               importance: float = 0.5,
               prediction_error: float = 0.0,
               memory_id: Optional[str] = None,
               context_text: str = "",
               outcome_text: str = "") -> str:
        self.encode_count += 1

        if memory_id is None:
            memory_id = f"mem_{self.encode_count}"

        embedding = self._get_embedding(content)

        allocated = True
        eca_info = {}

        if self.eca is not None:
            predicted_emb = None
            if outcome_text:
                predicted_emb = self._get_embedding(outcome_text)

            allocated, eca_info = self.eca.allocate(
                memory_id=memory_id,
                embedding=embedding,
                predicted_embedding=predicted_emb
            )

            if not allocated:
                return None

        if self.crc is not None:
            context_emb = self._get_embedding(context_text) if context_text else None
            outcome_emb = self._get_embedding(outcome_text) if outcome_text else None

            self.crc.register_experience(
                memory_id=memory_id,
                content_embedding=embedding,
                content_text=content[:200],
                context_embedding=context_emb,
                outcome_embedding=outcome_emb,
                importance=importance,
                prediction_error=prediction_error
            )

        if self.ssdr is not None:
            self.ssdr.register_memory(
                memory_id=memory_id,
                content_embedding=embedding,
                sensorimotor_state=sensorimotor_state
            )

        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "importance": importance,
            "prediction_error": prediction_error,
            "sensorimotor_state": sensorimotor_state,
            "eca_info": eca_info,
            "encode_time": self.encode_count,
        }

        if (self.config.auto_consolidate and
            self.encode_count % self.config.consolidation_interval == 0 and
            self.crc is not None):
            self.consolidate()

        return memory_id

    def retrieve(self, query: str, top_k: int = 5,
                 sensorimotor_state: Optional[SensorimotorState] = None,
                 query_embedding: Optional[np.ndarray] = None) -> List[Dict]:
        self.retrieve_count += 1

        if query_embedding is None:
            query_embedding = self._get_embedding(query)

        if self.ssdr is not None:
            if sensorimotor_state is not None:
                self.ssdr.set_current_state(sensorimotor_state)

            results = self.ssdr.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                query_state=sensorimotor_state
            )

            output = []
            for r in results:
                mid = r["id"]
                if mid in self.memories:
                    entry = self.memories[mid].copy()
                    entry["score"] = r["score"]
                    entry["retrieval_details"] = r.get("details", {})
                    output.append(entry)
                else:
                    for vid, variant in self.crc.counterfactual_store.items() if self.crc else []:
                        if vid == mid:
                            output.append({
                                "id": vid,
                                "content": variant.variant_text,
                                "embedding": variant.variant_embedding,
                                "score": r["score"],
                                "retrieval_details": r.get("details", {}),
                                "is_counterfactual": True,
                                "source_id": variant.source_id,
                            })
                            break

            return output

        scores = []
        for mid, mem in self.memories.items():
            mem_emb = mem["embedding"]
            sim = float(np.dot(query_embedding, mem_emb) /
                       (np.linalg.norm(query_embedding) * np.linalg.norm(mem_emb) + 1e-8))
            scores.append((mid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mid, score in scores[:top_k]:
            entry = self.memories[mid].copy()
            entry["score"] = score
            results.append(entry)

        return results

    def consolidate(self) -> Dict:
        report = {}

        if self.crc is not None:
            crc_report = self.crc.consolidate()
            report["crc"] = crc_report

            for mid, exp in self.crc.experiences.items():
                if mid in self.memories:
                    self.memories[mid]["embedding"] = exp.content_embedding.copy()
                    if self.ssdr is not None and mid in self.ssdr.memory_embeddings:
                        self.ssdr.memory_embeddings[mid] = exp.content_embedding.copy()

            for vid, variant in self.crc.counterfactual_store.items():
                if vid not in self.memories:
                    self.memories[vid] = {
                        "id": vid,
                        "content": variant.variant_text,
                        "embedding": variant.variant_embedding.copy(),
                        "importance": 0.3,
                        "prediction_error": 0.0,
                        "sensorimotor_state": None,
                        "is_counterfactual": True,
                        "source_id": variant.source_id,
                        "encode_time": self.encode_count,
                    }
                    if self.ssdr is not None:
                        source_state = self.ssdr.memory_states.get(variant.source_id)
                        self.ssdr.register_memory(
                            memory_id=vid,
                            content_embedding=variant.variant_embedding.copy(),
                            sensorimotor_state=source_state
                        )

        if self.eca is not None:
            self.eca._update_excitabilities()
            report["eca"] = self.eca.get_allocation_stats()

        if self.ssdr is not None:
            report["ssdr"] = self.ssdr.get_retrieval_stats()

        return report

    def sleep_consolidate(self, n_rounds: int = 3) -> Dict:
        total_report = {"rounds": []}

        for i in range(n_rounds):
            round_report = self.consolidate()
            round_report["round"] = i + 1
            total_report["rounds"].append(round_report)

        if self.eca is not None:
            for mid, cell in self.eca.cells.items():
                if cell.is_allocated:
                    cell.consolidation_level = min(1.0, cell.consolidation_level + 0.2)

        total_report["summary"] = {
            "total_memories": len(self.memories),
            "total_encode": self.encode_count,
            "total_retrieve": self.retrieve_count,
            "eca_stats": self.eca.get_allocation_stats() if self.eca else None,
            "crc_stats": self.crc.get_consolidation_stats() if self.crc else None,
            "ssdr_stats": self.ssdr.get_retrieval_stats() if self.ssdr else None,
        }

        return total_report

    def get_stats(self) -> Dict:
        return {
            "total_memories": len(self.memories),
            "encode_count": self.encode_count,
            "retrieve_count": self.retrieve_count,
            "eca_enabled": self.eca is not None,
            "crc_enabled": self.crc is not None,
            "ssdr_enabled": self.ssdr is not None,
            "eca_stats": self.eca.get_allocation_stats() if self.eca else None,
            "crc_stats": self.crc.get_consolidation_stats() if self.crc else None,
            "ssdr_stats": self.ssdr.get_retrieval_stats() if self.ssdr else None,
        }
