"""
Sensorimotor State-Dependent Retrieval (SSDR)
感知运动状态依赖检索

Neuroscience Basis:
    Godden & Baddeley (1975) - Context-dependent memory (diving experiment)
    Smith & Vela (2001) - Environmental context-dependent memory meta-analysis
    Tulving & Thomson (1973) - Encoding specificity principle
    Ritchey et al. (2015) - Neural pattern similarity reveals context-dependent retrieval

Core Idea:
    Human memory retrieval is profoundly influenced by the current physical and
    cognitive state. The classic Godden & Baddeley (1975) experiment showed that
    divers recall words better when tested in the same environment where they
    learned them (underwater vs. on land). This is the encoding specificity
    principle: memory retrieval is most effective when the retrieval context
    matches the encoding context.

    For embodied agents, the "context" is not just spatial location (as in
    RoboMemory's spatial memory module) but the FULL sensorimotor state:
    - Body position and orientation (proprioception)
    - Current action being performed
    - Objects being held or manipulated
    - Nearby objects and their affordances
    - Motor state (moving, stationary, reaching, etc.)
    - Environmental features (lighting, obstacles, terrain)

    SSDR integrates this full sensorimotor context into the retrieval scoring
    function, creating truly embodied memory retrieval.

    This is fundamentally different from:
    - RoboMemory: spatial memory is a separate module, not integrated into retrieval scoring
    - SynapticRAG: uses temporal triggers, not sensorimotor state
    - Affordance RAG: only considers manipulation affordance, not full sensorimotor state
    - Standard RAG: purely semantic similarity, no state dependency
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SensorimotorState:
    position: np.ndarray = None
    orientation: np.ndarray = None
    velocity: np.ndarray = None
    current_action: str = ""
    held_object: str = ""
    nearby_objects: List[str] = field(default_factory=list)
    motor_state: str = ""
    environmental_features: Dict = field(default_factory=dict)

    def to_vector(self, dim: int = 64) -> np.ndarray:
        vec = np.zeros(dim)
        idx = 0

        if self.position is not None:
            n = min(len(self.position), 3)
            vec[idx:idx+n] = self.position[:n]
            idx += 3

        if self.orientation is not None:
            n = min(len(self.orientation), 3)
            vec[idx:idx+n] = self.orientation[:n]
            idx += 3

        if self.velocity is not None:
            n = min(len(self.velocity), 3)
            vec[idx:idx+n] = self.velocity[:n]
            idx += 3

        if self.current_action:
            action_hash = hash(self.current_action) % (2**31)
            vec[idx] = (action_hash % 1000) / 1000.0
            idx += 1

        if self.held_object:
            obj_hash = hash(self.held_object) % (2**31)
            vec[idx] = (obj_hash % 1000) / 1000.0
            idx += 1

        for i, obj in enumerate(self.nearby_objects[:5]):
            if idx < dim:
                obj_hash = hash(obj) % (2**31)
                vec[idx] = (obj_hash % 1000) / 1000.0
                idx += 1

        if self.motor_state:
            ms_hash = hash(self.motor_state) % (2**31)
            if idx < dim:
                vec[idx] = (ms_hash % 1000) / 1000.0
                idx += 1

        for key, val in list(self.environmental_features.items())[:5]:
            if idx < dim:
                if isinstance(val, (int, float)):
                    vec[idx] = float(val)
                elif isinstance(val, str):
                    vec[idx] = (hash(val) % 1000) / 1000.0
                idx += 1

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


@dataclass
class SSDRConfig:
    semantic_weight: float = 0.5
    spatial_weight: float = 0.2
    action_weight: float = 0.15
    context_weight: float = 0.15
    state_vector_dim: int = 64
    context_decay_rate: float = 0.05
    min_state_similarity: float = 0.1
    enable_adaptive_weights: bool = True
    adaptation_learning_rate: float = 0.01


class SensorimotorStateRetrieval:
    """
    Sensorimotor State-Dependent Retrieval (SSDR)

    Retrieves memories not just by semantic similarity, but also by how well
    the current sensorimotor state matches the encoding context.
    """

    def __init__(self, embedding_dim: int = 384, config: SSDRConfig = None):
        self.embedding_dim = embedding_dim
        self.config = config or SSDRConfig()

        self.memory_states: Dict[str, SensorimotorState] = {}
        self.memory_embeddings: Dict[str, np.ndarray] = {}
        self.memory_state_vectors: Dict[str, np.ndarray] = {}

        self.current_state: Optional[SensorimotorState] = None
        self.current_state_vector: Optional[np.ndarray] = None

        self.retrieval_history: List[Dict] = []
        self.weight_adaptation_buffer: List[Dict] = []

        self._adaptive_weights = {
            "semantic": self.config.semantic_weight,
            "spatial": self.config.spatial_weight,
            "action": self.config.action_weight,
            "context": self.config.context_weight,
        }

    def set_current_state(self, state: SensorimotorState):
        self.current_state = state
        self.current_state_vector = state.to_vector(self.config.state_vector_dim)

    def register_memory(self, memory_id: str, content_embedding: np.ndarray,
                       sensorimotor_state: Optional[SensorimotorState] = None):
        self.memory_embeddings[memory_id] = content_embedding.copy()

        if sensorimotor_state is not None:
            self.memory_states[memory_id] = sensorimotor_state
            self.memory_state_vectors[memory_id] = sensorimotor_state.to_vector(
                self.config.state_vector_dim
            )
        else:
            default_state = SensorimotorState()
            self.memory_states[memory_id] = default_state
            self.memory_state_vectors[memory_id] = default_state.to_vector(
                self.config.state_vector_dim
            )

    def _compute_semantic_similarity(self, query_embedding: np.ndarray,
                                     memory_embedding: np.ndarray) -> float:
        norm_q = np.linalg.norm(query_embedding)
        norm_m = np.linalg.norm(memory_embedding)
        if norm_q < 1e-8 or norm_m < 1e-8:
            return 0.0
        return float(np.dot(query_embedding, memory_embedding) / (norm_q * norm_m))

    def _compute_spatial_proximity(self, query_state: SensorimotorState,
                                   memory_state: SensorimotorState) -> float:
        if query_state.position is None or memory_state.position is None:
            return 0.5

        distance = np.linalg.norm(query_state.position - memory_state.position)
        spatial_sim = np.exp(-0.1 * distance)
        return float(spatial_sim)

    def _compute_action_relevance(self, query_state: SensorimotorState,
                                  memory_state: SensorimotorState) -> float:
        score = 0.0

        if query_state.current_action and memory_state.current_action:
            if query_state.current_action == memory_state.current_action:
                score += 0.5
            elif any(w in memory_state.current_action
                    for w in query_state.current_action.split('_')):
                score += 0.3

        if query_state.held_object and memory_state.held_object:
            if query_state.held_object == memory_state.held_object:
                score += 0.3
            elif any(w in memory_state.held_object
                    for w in query_state.held_object.split()):
                score += 0.15

        if query_state.motor_state and memory_state.motor_state:
            if query_state.motor_state == memory_state.motor_state:
                score += 0.2

        return min(score, 1.0)

    def _compute_context_similarity(self, query_state_vec: np.ndarray,
                                    memory_state_vec: np.ndarray) -> float:
        norm_q = np.linalg.norm(query_state_vec)
        norm_m = np.linalg.norm(memory_state_vec)
        if norm_q < 1e-8 or norm_m < 1e-8:
            return 0.0
        return float(np.dot(query_state_vec, memory_state_vec) / (norm_q * norm_m))

    def _state_informativeness(self, state: SensorimotorState) -> float:
        info = 0.0
        generic_actions = {"conversing", "questioning", "idle", "waiting", "none", ""}
        generic_motor = {"stationary", "idle", "none", ""}

        if state.position is not None and np.linalg.norm(state.position) > 0.5:
            info += 0.3

        if state.current_action and state.current_action.lower() not in generic_actions:
            info += 0.25

        if state.held_object:
            info += 0.2

        if state.nearby_objects and len(state.nearby_objects) > 0:
            info += 0.1

        if state.motor_state and state.motor_state.lower() not in generic_motor:
            info += 0.15

        return min(info, 1.0)

    def compute_retrieval_score(self, query_embedding: np.ndarray,
                                memory_id: str,
                                query_state: Optional[SensorimotorState] = None) -> Tuple[float, Dict]:
        if memory_id not in self.memory_embeddings:
            return 0.0, {}

        memory_embedding = self.memory_embeddings[memory_id]
        memory_state = self.memory_states.get(memory_id)
        memory_state_vec = self.memory_state_vectors.get(memory_id,
                          np.zeros(self.config.state_vector_dim))

        semantic_sim = self._compute_semantic_similarity(query_embedding, memory_embedding)

        if query_state is None:
            query_state = self.current_state

        if query_state is None or memory_state is None:
            return semantic_sim, {"semantic": semantic_sim, "state_dependent": False}

        spatial_sim = self._compute_spatial_proximity(query_state, memory_state)
        action_sim = self._compute_action_relevance(query_state, memory_state)

        query_state_vec = query_state.to_vector(self.config.state_vector_dim)
        context_sim = self._compute_context_similarity(query_state_vec, memory_state_vec)

        w = dict(self._adaptive_weights)
        query_info = self._state_informativeness(query_state) if query_state else 0.0
        memory_info = self._state_informativeness(memory_state) if memory_state else 0.0
        state_confidence = min(query_info, memory_info)

        if state_confidence < 0.3:
            w["semantic"] = 0.9
            w["spatial"] = 0.03
            w["action"] = 0.03
            w["context"] = 0.04
        elif state_confidence < 0.6:
            scale = state_confidence / 0.6
            w["semantic"] = 0.9 - 0.4 * scale
            w["spatial"] = 0.03 + 0.17 * scale
            w["action"] = 0.03 + 0.12 * scale
            w["context"] = 0.04 + 0.11 * scale

        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}

        total_score = (w["semantic"] * semantic_sim +
                      w["spatial"] * spatial_sim +
                      w["action"] * action_sim +
                      w["context"] * context_sim)

        score_details = {
            "semantic": semantic_sim,
            "spatial": spatial_sim,
            "action": action_sim,
            "context": context_sim,
            "total": total_score,
            "state_dependent": True
        }

        return total_score, score_details

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5,
                 query_state: Optional[SensorimotorState] = None) -> List[Dict]:
        if not self.memory_embeddings:
            return []

        scores = []
        for mid in self.memory_embeddings:
            score, details = self.compute_retrieval_score(
                query_embedding, mid, query_state
            )
            scores.append((mid, score, details))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mid, score, details in scores[:top_k]:
            results.append({
                "id": mid,
                "score": score,
                "details": details
            })

        self.retrieval_history.append({
            "n_candidates": len(self.memory_embeddings),
            "top_k": top_k,
            "top_score": results[0]["score"] if results else 0,
            "state_dependent": query_state is not None or self.current_state is not None
        })

        return results

    def adapt_weights(self, feedback: List[Dict]):
        if not self.config.enable_adaptive_weights:
            return

        for entry in feedback:
            relevant = entry.get("relevant", False)
            details = entry.get("details", {})

            if not details or not details.get("state_dependent"):
                continue

            lr = self.config.adaptation_learning_rate
            sign = 1.0 if relevant else -1.0

            for key in ["semantic", "spatial", "action", "context"]:
                if key in details and details[key] > 0.1:
                    self._adaptive_weights[key] += lr * sign * details[key]
                    self._adaptive_weights[key] = max(0.05, min(0.8, self._adaptive_weights[key]))

            total = sum(self._adaptive_weights.values())
            for key in self._adaptive_weights:
                self._adaptive_weights[key] /= total

    def get_retrieval_stats(self) -> Dict:
        if not self.retrieval_history:
            return {"total_retrievals": 0}

        state_dep_count = sum(1 for h in self.retrieval_history if h.get("state_dependent"))

        return {
            "total_retrievals": len(self.retrieval_history),
            "state_dependent_retrievals": state_dep_count,
            "adaptive_weights": dict(self._adaptive_weights),
            "registered_memories": len(self.memory_embeddings),
            "memories_with_state": len(self.memory_states),
        }
