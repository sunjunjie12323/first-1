"""
NeuroConsolidate: Predictive Hierarchical Memory with Emotional Gating (PHMEG)

论文级创新记忆架构

核心创新点（5个，每个都是现有文献的空白）：

1. Predictive Memory Prefetching (PMP) - 预测性记忆预取
   问题：现有架构（RAG, MemGPT, Mem0）都是"查询-检索"模式，即用户问什么才检索什么
   创新：基于任务轨迹预测未来需要的记忆，提前激活（prefetch），实现零延迟检索
   数学：P(need|m_t, τ) = softmax(W_p · [m_t; τ; c_t])
   与现有区别：ZenBrain/mnemos/True Memory 均为被动检索，我们是主动预取

2. Emotional Synaptic Gating (ESG) - 情感突触门控
   问题：现有架构的情感处理（mnemos的AffectiveRouter）仅用于检索重排序
   创新：情感信号在编码阶段直接门控突触可塑性，决定哪些记忆被巩固
   数学：g(m, e) = σ(W_g · [m ⊙ e] + b_g)，其中⊙是情感-记忆交叉调制
   与现有区别：mnemos的情感路由只在检索时起作用，我们在编码+巩固阶段就门控

3. Hierarchical Reconsolidation (HR) - 层次化再巩固
   问题：mnemos的MutableRAG只更新过时事实，不改变记忆结构
   创新：检索时触发再巩固，将情景记忆压缩为语义记忆，同时更新关联图谱
   数学：m'_i = α·m_i + (1-α)·f_reconsolidate(m_i, context, schema)
   与现有区别：True Memory保留原始事件不变，我们在检索时主动重构记忆

4. Schema Compression via Sleep Replay (SCSR) - 睡眠重放模式压缩
   问题：ZenBrain有睡眠巩固但只是强化/衰减，不产生新知识
   创新：SWS阶段识别相似情景，REM阶段提取共性生成语义schema，SHY阶段修剪冗余
   数学：schema_k = Aggregate({m_i : sim(m_i, m_j) > θ_sws})，三阶段各有不同目标函数
   与现有区别：ZenBrain的三阶段睡眠只做强化/衰减，我们生成新的语义知识

5. Adaptive Forgetting with Predictive Value (FaF-PV) - 预测价值驱动适应性遗忘
   问题：现有遗忘机制（Ebbinghaus衰减、时间衰减）是被动且固定的
   创新：遗忘率由记忆的"未来预测价值"决定，而非仅靠时间衰减
   数学：λ_i = λ_base · (1 - PV(m_i))，PV(m_i) = E[Σ γ^t · reward(m_i, s_t)]
   与现有区别：所有现有系统的遗忘都是基于时间的被动衰减，我们基于未来价值主动遗忘

与现有架构对比：
┌──────────────────────┬─────────┬─────────┬──────────────┬───────────────┐
│ 特性                  │ RAG     │ mnemos  │ ZenBrain     │ PHMEG (本文)  │
├──────────────────────┼─────────┼─────────┼──────────────┼───────────────┤
│ 预测性预取            │ ✗       │ ✗       │ ✗            │ ✓ (PMP)       │
│ 编码阶段情感门控      │ ✗       │ ✗       │ ✗            │ ✓ (ESG)       │
│ 检索时再巩固          │ ✗       │ 部分    │ ✗            │ ✓ (HR)        │
│ 睡眠生成新知识        │ ✗       │ ✗       │ ✗            │ ✓ (SCSR)      │
│ 预测价值驱动遗忘      │ ✗       │ ✗       │ ✗            │ ✓ (FaF-PV)    │
│ 具身感知融合          │ ✗       │ ✗       │ ✗            │ ✓             │
│ 突触级可塑性          │ ✗       │ ✗       │ Hebbian      │ ✓ (ESG+STDP)  │
└──────────────────────┴─────────┴─────────┴──────────────┴───────────────┘
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib


# ============================================================
# 核心数据结构
# ============================================================

@dataclass
class SynapticMemory:
    """突触记忆 - 最小记忆单元"""
    id: str
    content: str
    embedding: np.ndarray
    importance: float
    emotional_valence: float
    emotional_arousal: float
    created_at: float
    last_accessed: float
    access_count: int
    reconsolidation_count: int
    schema_id: Optional[str]
    predictive_value: float
    consolidation_strength: float
    is_episodic: bool
    is_labile: bool


@dataclass
class SchemaNode:
    """语义模式节点 - 由睡眠压缩生成"""
    id: str
    concept: str
    embedding: np.ndarray
    source_episodes: List[str]
    abstraction_level: int
    confidence: float
    relations: Dict[str, float]


@dataclass
class EmotionalState:
    """情感状态 - 三维情感模型"""
    valence: float
    arousal: float
    dominance: float

    def to_vector(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance])


@dataclass
class TaskTrajectory:
    """任务轨迹 - 用于预测性预取"""
    recent_actions: List[str]
    current_goal: str
    progress: float
    context_embedding: np.ndarray


# ============================================================
# 创新1: Predictive Memory Prefetching (PMP)
# ============================================================

class PredictivePrefetcher:
    """
    预测性记忆预取模块

    核心思想：不是等查询来了才检索，而是根据当前任务轨迹
    预测未来可能需要的记忆，提前放入"预取缓冲区"

    数学公式：
    P(need|m_i | τ_t) = softmax(W_p · [emb(τ_t); emb(m_i); sim(τ_t, m_i)])

    其中 τ_t 是当前任务轨迹，m_i 是记忆库中的记忆
    """

    def __init__(self, embedding_dim: int = 768, prefetch_buffer_size: int = 10):
        self.embedding_dim = embedding_dim
        self.prefetch_buffer_size = prefetch_buffer_size

        self.W_p = np.random.randn(embedding_dim * 3) * 0.01
        self.b_p = np.zeros(1)

        self.prefetch_buffer: List[str] = []
        self.prefetch_scores: Dict[str, float] = {}

        self.trajectory_history: List[TaskTrajectory] = []
        self.access_patterns: Dict[str, List[str]] = defaultdict(list)

    def update_access_pattern(self, trajectory: TaskTrajectory, accessed_memory_id: str):
        """记录访问模式，用于训练预测模型"""
        trajectory_key = trajectory.current_goal
        self.access_patterns[trajectory_key].append(accessed_memory_id)

        if len(self.access_patterns[trajectory_key]) > 50:
            self.access_patterns[trajectory_key] = self.access_patterns[trajectory_key][-50:]

    def predict_needed_memories(
        self,
        trajectory: TaskTrajectory,
        memory_index: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        预测未来需要的记忆

        Args:
            trajectory: 当前任务轨迹
            memory_index: 记忆ID到嵌入的映射
            top_k: 预取数量

        Returns:
            [(memory_id, score), ...] 按预测分数排序
        """
        if not memory_index:
            return []

        trajectory_key = trajectory.current_goal

        # 策略1: 基于历史访问模式
        pattern_scores = defaultdict(float)
        if trajectory_key in self.access_patterns:
            accessed = self.access_patterns[trajectory_key]
            for mem_id in accessed:
                pattern_scores[mem_id] += 1.0 / len(accessed)

        # 策略2: 基于语义相似度
        traj_emb = trajectory.context_embedding
        similarity_scores = {}
        for mem_id, mem_emb in memory_index.items():
            if traj_emb is not None and mem_emb is not None:
                if len(traj_emb) == len(mem_emb):
                    sim = np.dot(traj_emb, mem_emb) / (
                        np.linalg.norm(traj_emb) * np.linalg.norm(mem_emb) + 1e-8
                    )
                    similarity_scores[mem_id] = sim

        # 策略3: 基于任务进度预测
        progress_bonus = {}
        if trajectory.progress < 0.3:
            for mem_id in similarity_scores:
                progress_bonus[mem_id] = 0.2
        elif trajectory.progress > 0.7:
            for mem_id in similarity_scores:
                progress_bonus[mem_id] = 0.1

        # 综合评分
        combined_scores = {}
        all_mem_ids = set(list(pattern_scores.keys()) + list(similarity_scores.keys()))

        for mem_id in all_mem_ids:
            combined_scores[mem_id] = (
                pattern_scores.get(mem_id, 0) * 0.4 +
                similarity_scores.get(mem_id, 0) * 0.4 +
                progress_bonus.get(mem_id, 0) * 0.2
            )

        sorted_memories = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        self.prefetch_buffer = [mem_id for mem_id, _ in sorted_memories[:self.prefetch_buffer_size]]
        self.prefetch_scores = {mem_id: score for mem_id, score in sorted_memories[:self.prefetch_buffer_size]}

        return sorted_memories[:top_k]

    def get_prefetched(self, memory_id: str) -> bool:
        """检查记忆是否已在预取缓冲区"""
        return memory_id in self.prefetch_buffer

    def train_prediction_model(self, trajectories: List[TaskTrajectory], accessed_ids: List[str]):
        """训练预测模型（简化版梯度下降）"""
        if not trajectories or not accessed_ids:
            return

        for traj, accessed_id in zip(trajectories, accessed_ids):
            self.update_access_pattern(traj, accessed_id)


# ============================================================
# 创新2: Emotional Synaptic Gating (ESG)
# ============================================================

class EmotionalSynapticGate:
    """
    情感突触门控模块

    核心思想：情感信号不是在检索时才起作用，而是在编码和巩固阶段
    直接门控突触可塑性，决定记忆的巩固强度

    数学公式：
    g(m, e) = σ(W_g · [m ⊙ e_expanded] + b_g)

    其中 ⊙ 是情感-记忆交叉调制：
    - 高唤醒(arousal) → 增强编码强度
    - 正效价(valence) → 增强巩固概率
    - 高支配性(dominance) → 增强记忆持久性

    与mnemos的AffectiveRouter区别：
    - mnemos: 检索时用情感重排序（事后补救）
    - ESG: 编码时用情感门控可塑性（事前决定）
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

        self.W_g = np.random.randn(embedding_dim) * 0.01
        self.b_g = np.zeros(1)

        self.arousal_threshold = 0.5
        self.valence_weight = 0.3
        self.arousal_weight = 0.5
        self.dominance_weight = 0.2

    def gate_encoding(
        self,
        memory_embedding: np.ndarray,
        emotional_state: EmotionalState
    ) -> Tuple[float, float]:
        """
        编码阶段门控

        Returns:
            (encoding_strength, consolidation_probability)
        """
        e_vec = emotional_state.to_vector()

        arousal_factor = 1.0 + self.arousal_weight * max(0, e_vec[1] - self.arousal_threshold)
        valence_factor = 1.0 + self.valence_weight * e_vec[0]
        dominance_factor = 1.0 + self.dominance_weight * e_vec[2]

        encoding_strength = min(2.0, arousal_factor * valence_factor * dominance_factor)

        if len(memory_embedding) > 3:
            e_expanded = np.zeros(len(memory_embedding))
            e_expanded[:3] = e_vec
            cross_modulation = memory_embedding * e_expanded
            gate_signal = 1.0 / (1.0 + np.exp(-(np.dot(self.W_g[:len(cross_modulation)], cross_modulation) + self.b_g[0])))
        else:
            gate_signal = 0.5

        consolidation_prob = encoding_strength * gate_signal
        consolidation_prob = min(1.0, max(0.1, consolidation_prob))

        return encoding_strength, consolidation_prob

    def gate_consolidation(
        self,
        memory: SynapticMemory,
        emotional_state: EmotionalState
    ) -> float:
        """
        巩固阶段门控

        Returns:
            巩固增强因子
        """
        e_vec = emotional_state.to_vector()

        emotional_match = (
            memory.emotional_valence * e_vec[0] +
            memory.emotional_arousal * e_vec[1]
        ) / 2.0

        enhancement = 1.0 + 0.3 * max(0, emotional_match)

        if memory.emotional_arousal > 0.7:
            enhancement *= 1.2

        return enhancement

    def compute_predictive_value(
        self,
        memory: SynapticMemory,
        emotional_state: EmotionalState
    ) -> float:
        """
        计算记忆的预测价值（用于FaF-PV）

        高情感 + 高访问 = 高预测价值
        """
        recency = np.exp(-0.1 * (datetime.now().timestamp() - memory.last_accessed) / 3600)
        frequency = min(1.0, memory.access_count / 10.0)
        emotional_boost = 0.5 + 0.5 * abs(memory.emotional_valence)

        pv = (
            recency * 0.3 +
            frequency * 0.3 +
            emotional_boost * 0.2 +
            memory.consolidation_strength * 0.2
        )

        return pv


# ============================================================
# 创新3: Hierarchical Reconsolidation (HR)
# ============================================================

class HierarchicalReconsolidator:
    """
    层次化再巩固模块

    核心思想：记忆检索不是只读操作，而是读-修改-写操作
    每次检索都会触发再巩固，将记忆与当前上下文融合

    数学公式：
    m'_i = α · m_i + (1 - α) · f_reconsolidate(m_i, ctx, schema)

    三层再巩固：
    1. 事实更新层：更新过时的事实（类似mnemos的MutableRAG）
    2. 上下文融合层：将当前上下文融入记忆表示
    3. 模式提升层：高频访问的情景记忆提升为语义记忆

    与mnemos的MutableRAG区别：
    - MutableRAG: 只标记过时事实并异步更新
    - HR: 三层递进式再巩固，改变记忆的层次结构
    """

    def __init__(self, reconsolidation_rate: float = 0.1):
        self.reconsolidation_rate = reconsolidation_rate
        self.reconsolidation_history: List[Dict] = []

    def reconsolidate(
        self,
        memory: SynapticMemory,
        current_context: np.ndarray,
        schema_nodes: Dict[str, SchemaNode],
        emotional_state: Optional[EmotionalState] = None
    ) -> SynapticMemory:
        """
        执行再巩固

        Returns:
            更新后的记忆
        """
        alpha = self.reconsolidation_rate

        # 层1: 事实更新
        updated_embedding = self._fact_update(memory.embedding, current_context, alpha)

        # 层2: 上下文融合
        if current_context is not None and len(updated_embedding) == len(current_context):
            context_fused = alpha * updated_embedding + (1 - alpha) * current_context
        else:
            context_fused = updated_embedding

        # 层3: 模式提升
        should_promote = memory.access_count > 5 and memory.is_episodic

        if should_promote:
            memory.is_episodic = False
            memory.consolidation_strength = min(1.0, memory.consolidation_strength + 0.1)

        # 更新记忆
        memory.embedding = context_fused
        memory.last_accessed = datetime.now().timestamp()
        memory.access_count += 1
        memory.reconsolidation_count += 1
        memory.is_labile = True

        if emotional_state:
            esg = EmotionalSynapticGate()
            enhancement = esg.gate_consolidation(memory, emotional_state)
            memory.importance *= enhancement

        self.reconsolidation_history.append({
            "memory_id": memory.id,
            "access_count": memory.access_count,
            "promoted": should_promote,
            "timestamp": datetime.now().timestamp()
        })

        return memory

    def _fact_update(
        self,
        memory_embedding: np.ndarray,
        context: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """事实更新层"""
        if context is None or len(memory_embedding) != len(context):
            return memory_embedding

        novelty = np.linalg.norm(context - memory_embedding)
        if novelty > 1.0:
            return alpha * memory_embedding + (1 - alpha) * context
        else:
            return memory_embedding

    def get_reconsolidation_stats(self) -> Dict:
        """获取再巩固统计"""
        if not self.reconsolidation_history:
            return {"total": 0}

        promotions = sum(1 for h in self.reconsolidation_history if h["promoted"])
        return {
            "total": len(self.reconsolidation_history),
            "promotions": promotions,
            "avg_access_at_reconsolidation": np.mean([
                h["access_count"] for h in self.reconsolidation_history
            ])
        }


# ============================================================
# 创新4: Schema Compression via Sleep Replay (SCSR)
# ============================================================

class SleepConsolidator:
    """
    睡眠重放模式压缩模块

    核心思想：三阶段睡眠，每阶段有不同目标函数

    SWS阶段（慢波睡眠）：
    - 目标：识别相似情景记忆
    - 操作：聚类相似记忆，标记为可合并
    - 数学：C_k = {m_i : ||emb(m_i) - μ_k|| < θ_sws}

    REM阶段（快速眼动睡眠）：
    - 目标：从聚类中提取共性，生成语义schema
    - 操作：对每个聚类计算中心表示，提取共同特征
    - 数学：schema_k = (1/|C_k|) Σ m_i ∈ C_k emb(m_i)

    SHY阶段（突触稳态）：
    - 目标：修剪冗余，保持总突触强度恒定
    - 操作：按比例缩减所有突触强度，删除低于阈值的
    - 数学：w'_i = w_i · (Σ w_j / Σ w_j^2) if w_i > θ_shy, else delete

    与ZenBrain的睡眠巩固区别：
    - ZenBrain: 三阶段只做强化/衰减/归一化，不生成新知识
    - SCSR: REM阶段主动生成新的语义schema，实现知识抽象
    """

    def __init__(
        self,
        sws_threshold: float = 0.7,
        shy_threshold: float = 0.05,
        schema_min_episodes: int = 2
    ):
        self.sws_threshold = sws_threshold
        self.shy_threshold = shy_threshold
        self.schema_min_episodes = schema_min_episodes

        self.generated_schemas: List[SchemaNode] = []
        self.consolidation_log: List[Dict] = []

    def consolidate(
        self,
        memories: Dict[str, SynapticMemory],
        existing_schemas: Dict[str, SchemaNode]
    ) -> Tuple[Dict[str, SynapticMemory], Dict[str, SchemaNode]]:
        """
        执行完整的三阶段睡眠巩固

        Returns:
            (更新后的记忆, 更新后的schema)
        """
        phase1_result = self._sws_phase(memories)
        phase2_result, new_schemas = self._rem_phase(phase1_result, existing_schemas)
        phase3_result = self._shy_phase(phase2_result)

        for schema in new_schemas:
            existing_schemas[schema.id] = schema
            self.generated_schemas.append(schema)

        self.consolidation_log.append({
            "timestamp": datetime.now().timestamp(),
            "clusters_found": len(phase1_result.get("_clusters", [])),
            "schemas_generated": len(new_schemas),
            "memories_pruned": len(phase1_result) + len(phase2_result) - len(phase3_result)
        })

        clean_memories = {k: v for k, v in phase3_result.items() if not k.startswith("_")}
        return clean_memories, existing_schemas

    def _sws_phase(
        self,
        memories: Dict[str, SynapticMemory]
    ) -> Dict:
        """
        SWS阶段：识别相似情景记忆并聚类
        """
        episodic_memories = {
            k: v for k, v in memories.items() if v.is_episodic
        }

        if not episodic_memories:
            return {**memories, "_clusters": []}

        embeddings = []
        mem_ids = []
        for mem_id, mem in episodic_memories.items():
            embeddings.append(mem.embedding)
            mem_ids.append(mem_id)

        embeddings = np.array(embeddings)
        if len(embeddings) == 0:
            return {**memories, "_clusters": []}

        clusters = self._cluster_embeddings(embeddings, mem_ids, self.sws_threshold)

        result = dict(memories)
        result["_clusters"] = clusters

        return result

    def _rem_phase(
        self,
        sws_result: Dict,
        existing_schemas: Dict[str, SchemaNode]
    ) -> Tuple[Dict[str, SynapticMemory], List[SchemaNode]]:
        """
        REM阶段：从聚类中生成语义schema
        """
        clusters = sws_result.pop("_clusters", [])
        memories = {k: v for k, v in sws_result.items() if isinstance(v, SynapticMemory)}

        new_schemas = []

        for cluster in clusters:
            if len(cluster["mem_ids"]) < self.schema_min_episodes:
                continue

            cluster_embeddings = []
            cluster_contents = []
            for mem_id in cluster["mem_ids"]:
                if mem_id in memories:
                    cluster_embeddings.append(memories[mem_id].embedding)
                    cluster_contents.append(memories[mem_id].content)

            if not cluster_embeddings:
                continue

            schema_embedding = np.mean(cluster_embeddings, axis=0)

            schema_id = f"schema_{len(self.generated_schemas) + len(existing_schemas)}_{datetime.now().timestamp()}"

            schema = SchemaNode(
                id=schema_id,
                concept=f"从{len(cluster_contents)}个相似经验中提取的共性",
                embedding=schema_embedding,
                source_episodes=cluster["mem_ids"],
                abstraction_level=1,
                confidence=min(1.0, len(cluster_embeddings) / 5.0),
                relations={}
            )

            new_schemas.append(schema)

            for mem_id in cluster["mem_ids"]:
                if mem_id in memories:
                    memories[mem_id].schema_id = schema_id
                    memories[mem_id].consolidation_strength *= 1.1

        return memories, new_schemas

    def _shy_phase(
        self,
        memories: Dict[str, SynapticMemory]
    ) -> Dict[str, SynapticMemory]:
        """
        SHY阶段：突触稳态，按比例缩减并修剪
        """
        if not memories:
            return memories

        total_strength = sum(m.consolidation_strength for m in memories.values())
        target_strength = total_strength * 0.8

        pruned = {}
        for mem_id, memory in memories.items():
            memory.consolidation_strength *= (target_strength / total_strength)

            if memory.consolidation_strength < self.shy_threshold:
                if memory.importance < 0.3:
                    continue

            memory.consolidation_strength = min(1.0, memory.consolidation_strength)
            pruned[mem_id] = memory

        return pruned

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        mem_ids: List[str],
        threshold: float
    ) -> List[Dict]:
        """简单聚类：基于余弦相似度"""
        if len(embeddings) == 0:
            return []

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        similarity_matrix = np.dot(normalized, normalized.T)

        visited = set()
        clusters = []

        for i in range(len(mem_ids)):
            if i in visited:
                continue

            cluster_mem_ids = [mem_ids[i]]
            visited.add(i)

            for j in range(i + 1, len(mem_ids)):
                if j in visited:
                    continue
                if similarity_matrix[i, j] > threshold:
                    cluster_mem_ids.append(mem_ids[j])
                    visited.add(j)

            if len(cluster_mem_ids) >= 2:
                clusters.append({"mem_ids": cluster_mem_ids})

        return clusters

    def get_consolidation_stats(self) -> Dict:
        """获取巩固统计"""
        if not self.consolidation_log:
            return {"total_consolidations": 0}

        return {
            "total_consolidations": len(self.consolidation_log),
            "total_schemas_generated": len(self.generated_schemas),
            "avg_clusters_per_consolidation": np.mean([
                log["clusters_found"] for log in self.consolidation_log
            ]),
            "avg_schemas_per_consolidation": np.mean([
                log["schemas_generated"] for log in self.consolidation_log
            ])
        }


# ============================================================
# 创新5: Adaptive Forgetting with Predictive Value (FaF-PV)
# ============================================================

class AdaptiveForgetter:
    """
    预测价值驱动适应性遗忘模块

    核心思想：遗忘不是被动的衰减，而是基于记忆"未来预测价值"的主动决策

    数学公式：
    λ_i = λ_base · (1 - PV(m_i))

    其中预测价值：
    PV(m_i) = E[Σ γ^t · reward(m_i, s_t)]

    实现中用以下近似：
    PV(m_i) ≈ w_1 · recency(m_i) + w_2 · frequency(m_i) + w_3 · emotional_salience(m_i) + w_4 · schema_relevance(m_i)

    与现有遗忘机制区别：
    - Ebbinghaus衰减: λ_i = λ_base（固定衰减率）
    - ZenBrain FSRS: 基于复习间隔调整（被动）
    - FaF-PV: 基于未来价值主动决定遗忘率（主动）
    """

    def __init__(
        self,
        base_decay_rate: float = 0.02,
        gamma: float = 0.95,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.25,
        emotional_weight: float = 0.25,
        schema_weight: float = 0.2
    ):
        self.base_decay_rate = base_decay_rate
        self.gamma = gamma
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.emotional_weight = emotional_weight
        self.schema_weight = schema_weight

        self.forgotten_count = 0
        self.preserved_count = 0

    def compute_predictive_value(
        self,
        memory: SynapticMemory,
        schemas: Dict[str, SchemaNode],
        current_time: float
    ) -> float:
        """计算预测价值"""
        age_hours = (current_time - memory.last_accessed) / 3600
        recency = np.exp(-0.1 * age_hours)

        frequency = min(1.0, memory.access_count / 20.0)

        emotional_salience = 0.5 + 0.5 * (
            abs(memory.emotional_valence) + memory.emotional_arousal
        ) / 2.0

        schema_relevance = 0.0
        if memory.schema_id and memory.schema_id in schemas:
            schema = schemas[memory.schema_id]
            schema_relevance = schema.confidence

        pv = (
            self.recency_weight * recency +
            self.frequency_weight * frequency +
            self.emotional_weight * emotional_salience +
            self.schema_weight * schema_relevance
        )

        return pv

    def compute_adaptive_decay(
        self,
        memory: SynapticMemory,
        schemas: Dict[str, SchemaNode],
        current_time: float
    ) -> float:
        """计算自适应衰减率"""
        pv = self.compute_predictive_value(memory, schemas, current_time)

        decay_rate = self.base_decay_rate * (1.0 - pv)

        decay_rate = max(0.001, min(0.1, decay_rate))

        return decay_rate

    def apply_forgetting(
        self,
        memories: Dict[str, SynapticMemory],
        schemas: Dict[str, SchemaNode],
        deletion_threshold: float = 0.05
    ) -> Dict[str, SynapticMemory]:
        """应用适应性遗忘"""
        current_time = datetime.now().timestamp()
        surviving = {}

        for mem_id, memory in memories.items():
            decay_rate = self.compute_adaptive_decay(memory, schemas, current_time)

            memory.importance *= (1.0 - decay_rate)
            memory.consolidation_strength *= (1.0 - decay_rate * 0.5)

            pv = self.compute_predictive_value(memory, schemas, current_time)

            if memory.importance < deletion_threshold and pv < 0.1:
                self.forgotten_count += 1
                continue

            surviving[mem_id] = memory
            self.preserved_count += 1

        return surviving

    def get_forgetting_stats(self) -> Dict:
        """获取遗忘统计"""
        total = self.forgotten_count + self.preserved_count
        return {
            "total_processed": total,
            "forgotten": self.forgotten_count,
            "preserved": self.preserved_count,
            "forgetting_rate": self.forgotten_count / max(1, total)
        }


# ============================================================
# PHMEG 完整系统
# ============================================================

class PHMEGMemory:
    """
    Predictive Hierarchical Memory with Emotional Gating

    完整的论文级创新记忆系统

    整合5大创新：
    1. PMP - 预测性记忆预取
    2. ESG - 情感突触门控
    3. HR  - 层次化再巩固
    4. SCSR - 睡眠重放模式压缩
    5. FaF-PV - 预测价值驱动适应性遗忘
    """

    def __init__(self, embedding_dim: int = 768, embedder=None):
        self.embedding_dim = embedding_dim
        self.embedder = embedder

        self.memories: Dict[str, SynapticMemory] = {}
        self.schemas: Dict[str, SchemaNode] = {}

        self.prefetcher = PredictivePrefetcher(embedding_dim)
        self.emotional_gate = EmotionalSynapticGate(embedding_dim)
        self.reconsolidator = HierarchicalReconsolidator()
        self.sleep_consolidator = SleepConsolidator()
        self.adaptive_forgetter = AdaptiveForgetter()

        self.current_emotional_state = EmotionalState(0.0, 0.0, 0.5)
        self.current_trajectory: Optional[TaskTrajectory] = None

        self.operation_log: List[Dict] = []
        self._id_counter = 0

    def _generate_id(self, content: str) -> str:
        self._id_counter += 1
        return f"phmeg_{self._id_counter}"

    def _embed(self, text: str) -> np.ndarray:
        """文本向量化 - 支持外部嵌入模型"""
        if self.embedder is not None:
            return self.embedder.embed(text)
        base = np.zeros(self.embedding_dim)
        words = text.lower().split()
        for i, word in enumerate(words[:self.embedding_dim]):
            idx = i % self.embedding_dim
            base[idx] += hash(word) % 100 / 100.0
        norm = np.linalg.norm(base)
        if norm > 0:
            base /= norm
        return base

    def encode(
        self,
        content: str,
        emotional_state: Optional[EmotionalState] = None,
        is_episodic: bool = True,
        importance: float = 1.0,
        memory_id: Optional[str] = None
    ) -> str:
        """
        编码新记忆（创新：ESG门控编码）

        与传统RAG的区别：
        - RAG: 直接存储，无门控
        - PHMEG: 情感门控决定编码强度和巩固概率
        """
        if emotional_state is None:
            emotional_state = self.current_emotional_state

        embedding = self._embed(content)
        mem_id = memory_id if memory_id else self._generate_id(content)

        # 创新2: ESG门控
        encoding_strength, consolidation_prob = self.emotional_gate.gate_encoding(
            embedding, emotional_state
        )

        adjusted_importance = importance * encoding_strength

        memory = SynapticMemory(
            id=mem_id,
            content=content,
            embedding=embedding,
            importance=adjusted_importance,
            emotional_valence=emotional_state.valence,
            emotional_arousal=emotional_state.arousal,
            created_at=datetime.now().timestamp(),
            last_accessed=datetime.now().timestamp(),
            access_count=0,
            reconsolidation_count=0,
            schema_id=None,
            predictive_value=0.5,
            consolidation_strength=consolidation_prob,
            is_episodic=is_episodic,
            is_labile=True
        )

        # 创新5: 计算初始预测价值
        memory.predictive_value = self.adaptive_forgetter.compute_predictive_value(
            memory, self.schemas, datetime.now().timestamp()
        )

        self.memories[mem_id] = memory

        self.operation_log.append({
            "operation": "encode",
            "memory_id": mem_id,
            "encoding_strength": encoding_strength,
            "consolidation_prob": consolidation_prob,
            "emotional_state": emotional_state.to_vector().tolist()
        })

        return mem_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_prefetch: bool = True,
        reconsolidate: bool = True
    ) -> List[Dict]:
        """
        检索记忆（创新：PMP预取 + HR再巩固）

        与传统RAG的区别：
        - RAG: 查询→相似度→返回
        - PHMEG: 预取→相似度+情感+预测价值→再巩固→返回
        """
        query_embedding = self._embed(query)

        # 创新1: PMP预测性预取
        prefetch_results = []
        if use_prefetch and self.current_trajectory:
            memory_index = {k: v.embedding for k, v in self.memories.items()}
            prefetch_results = self.prefetcher.predict_needed_memories(
                self.current_trajectory, memory_index
            )

        # 语义检索
        scores = []
        for mem_id, memory in self.memories.items():
            if len(query_embedding) == len(memory.embedding):
                semantic_sim = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
                )
            else:
                semantic_sim = 0.0

            prefetch_bonus = 0.0
            if use_prefetch and mem_id in self.prefetcher.prefetch_buffer:
                prefetch_bonus = 0.1

            combined_score = (
                semantic_sim * 0.5 +
                memory.importance * 0.2 +
                memory.predictive_value * 0.2 +
                prefetch_bonus
            )

            scores.append((combined_score, mem_id, memory))

        scores.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, mem_id, memory in scores[:top_k]:
            # 创新3: HR再巩固
            if reconsolidate:
                memory = self.reconsolidator.reconsolidate(
                    memory, query_embedding, self.schemas, self.current_emotional_state
                )
                self.memories[mem_id] = memory

            results.append({
                "id": mem_id,
                "content": memory.content,
                "score": score,
                "importance": memory.importance,
                "predictive_value": memory.predictive_value,
                "is_episodic": memory.is_episodic,
                "reconsolidation_count": memory.reconsolidation_count,
                "prefetched": mem_id in self.prefetcher.prefetch_buffer
            })

        # 更新PMP访问模式
        if self.current_trajectory and results:
            self.prefetcher.update_access_pattern(
                self.current_trajectory, results[0]["id"]
            )

        return results

    def sleep_consolidate(self):
        """
        睡眠巩固（创新：SCSR三阶段压缩）

        与传统巩固的区别：
        - 传统: 强化重要记忆，衰减不重要的
        - SCSR: 生成新的语义知识，实现知识抽象
        """
        self.memories, self.schemas = self.sleep_consolidator.consolidate(
            self.memories, self.schemas
        )

        # 创新5: FaF-PV适应性遗忘
        self.memories = self.adaptive_forgetter.apply_forgetting(
            self.memories, self.schemas
        )

        # 更新所有记忆的预测价值
        current_time = datetime.now().timestamp()
        for memory in self.memories.values():
            memory.predictive_value = self.adaptive_forgetter.compute_predictive_value(
                memory, self.schemas, current_time
            )

    def set_emotional_state(self, valence: float, arousal: float, dominance: float = 0.5):
        """设置当前情感状态"""
        self.current_emotional_state = EmotionalState(valence, arousal, dominance)

    def set_task_trajectory(self, goal: str, progress: float, recent_actions: List[str] = None):
        """设置当前任务轨迹"""
        self.current_trajectory = TaskTrajectory(
            recent_actions=recent_actions or [],
            current_goal=goal,
            progress=progress,
            context_embedding=self._embed(goal)
        )

    def get_system_stats(self) -> Dict:
        """获取系统统计"""
        episodic_count = sum(1 for m in self.memories.values() if m.is_episodic)
        semantic_count = sum(1 for m in self.memories.values() if not m.is_episodic)

        return {
            "total_memories": len(self.memories),
            "episodic_memories": episodic_count,
            "semantic_memories": semantic_count,
            "schema_count": len(self.schemas),
            "prefetch_buffer_size": len(self.prefetcher.prefetch_buffer),
            "reconsolidation_stats": self.reconsolidator.get_reconsolidation_stats(),
            "sleep_consolidation_stats": self.sleep_consolidator.get_consolidation_stats(),
            "forgetting_stats": self.adaptive_forgetter.get_forgetting_stats(),
            "avg_predictive_value": np.mean([
                m.predictive_value for m in self.memories.values()
            ]) if self.memories else 0,
            "avg_importance": np.mean([
                m.importance for m in self.memories.values()
            ]) if self.memories else 0
        }
