"""
AMO: Adaptive Memory Orchestration
自适应记忆编排模块

核心创新：
- 场景感知的模块选择
- 自动学习最优模块组合
- 解决HR在长对话中的负面效应

这解决了：
- HR在长对话有害的问题
- PMP在简单任务多余的问题
- 需要手动调参的问题

与现有方法区别：
- MemGPT: 固定层级，无自适应
- mnemos: 固定路由，无编排
- PHMEG-AMO: 场景感知，动态编排
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime


@dataclass
class MemoryContext:
    """记忆上下文"""
    context_type: str  # "short_conversation", "long_conversation", "embodied", "factual"
    emotional_arity: float  # 情感强度
    recency_score: float  # 时间新旧
    complexity: float  # 复杂度
    task_type: str  # "retrieval", "consolidation", "forgetting"


@dataclass
class ModuleConfig:
    """模块配置"""
    enable_pmp: bool = True
    enable_esg: bool = True
    enable_hr: bool = True
    enable_scsr: bool = True
    enable_faf: bool = True
    enable_amo: bool = True
    enable_cmc: bool = False


class AdaptiveMemoryOrchestrator:
    """
    自适应记忆编排器

    核心机制：
    1. 场景检测：分析当前上下文类型
    2. 模块评估：预测每个模块的收益
    3. 动态编排：选择最优模块组合

    数学公式：
    config* = argmax_config E[performance(config, context)]

    使用强化学习风格的策略：
    π(config | context) = softmax(W · context_features)
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

        # 场景检测特征权重
        self.context_weights = np.random.randn(embedding_dim, 5) * 0.01
        self.context_bias = np.zeros(5)

        # 模块收益预测权重
        self.module_weights = {
            "pmp": np.random.randn(5, 1) * 0.01,
            "esg": np.random.randn(5, 1) * 0.01,
            "hr": np.random.randn(5, 1) * 0.01,
            "scsr": np.random.randn(5, 1) * 0.01,
            "faf": np.random.randn(5, 1) * 0.01,
        }

        # 经验回放
        self.experience_buffer: List[Tuple[MemoryContext, ModuleConfig, float]] = []
        self.max_experiences = 1000

        # 场景类型统计
        self.context_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.module_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # 学习率
        self.lr = 0.01

    def detect_context(self, query: str, memory_count: int,
                     recent_accesses: List[float],
                     emotional_states: List) -> MemoryContext:
        """
        检测当前场景

        输入特征：
        - query: 查询文本
        - memory_count: 当前记忆数量
        - recent_accesses: 最近访问模式
        - emotional_states: 最近情感状态

        输出：
        - MemoryContext: 场景描述
        """
        # 特征1: 上下文类型（基于查询和记忆数量）
        if memory_count > 500:
            context_type = "long_conversation"
        elif memory_count > 100:
            context_type = "medium_conversation"
        elif "具身" in query or "机器人" in query or "embodied" in query.lower():
            context_type = "embodied"
        elif "什么" in query or "如何" in query or "why" in query.lower():
            context_type = "factual"
        else:
            context_type = "short_conversation"

        # 特征2: 情感强度
        if emotional_states:
            emotional_arity = np.mean([abs(e.valence) + e.arousal for e in emotional_states])
        else:
            emotional_arity = 0.5

        # 特征3: 时间新旧
        if recent_accesses:
            recency_score = 1.0 / (1.0 + np.mean(recent_accesses) / 100)
        else:
            recency_score = 0.5

        # 特征4: 复杂度
        complexity = min(1.0, len(query.split()) / 50)

        # 特征5: 任务类型
        if "记住" in query or "recall" in query.lower():
            task_type = "retrieval"
        elif "巩固" in query or "consolidate" in query.lower():
            task_type = "consolidation"
        else:
            task_type = "mixed"

        return MemoryContext(
            context_type=context_type,
            emotional_arity=emotional_arity,
            recency_score=recency_score,
            complexity=complexity,
            task_type=task_type
        )

    def _context_to_features(self, context: MemoryContext) -> np.ndarray:
        """将上下文转为特征向量"""
        features = np.zeros(self.embedding_dim)

        # 编码上下文类型
        type_encoding = {
            "short_conversation": [1, 0, 0, 0, 0],
            "medium_conversation": [0, 1, 0, 0, 0],
            "long_conversation": [0, 0, 1, 0, 0],
            "embodied": [0, 0, 0, 1, 0],
            "factual": [0, 0, 0, 0, 1],
        }

        type_vec = type_encoding.get(context.context_type, [0, 0, 0, 0, 0])

        # 填充前5维
        features[:5] = type_vec
        features[5] = context.emotional_arity
        features[6] = context.recency_score
        features[7] = context.complexity

        return features

    def predict_module_impact(self, context: MemoryContext,
                            module_name: str) -> float:
        """
        预测某个模块在当前上下文中的收益

        Returns:
            float: 预测收益 (-1 to 1)
        """
        features = self._context_to_features(context)

        weights = self.module_weights.get(module_name, np.zeros((5, 1)))
        if weights.shape[0] <= 5:
            impact = np.dot(features[:weights.shape[0]], weights.flatten())
        else:
            impact = np.dot(features, weights[:len(features)])

        return float(np.tanh(impact))

    def select_optimal_config(self, context: MemoryContext) -> ModuleConfig:
        """
        选择最优模块配置

        核心算法：
        1. 预测每个模块的收益
        2. 如果经验丰富，使用经验
        3. 否则使用预测

        Returns:
            ModuleConfig: 最优配置
        """
        # 预测各模块收益
        pmp_impact = self.predict_module_impact(context, "pmp")
        esg_impact = self.predict_module_impact(context, "esg")
        hr_impact = self.predict_module_impact(context, "hr")
        scsr_impact = self.predict_module_impact(context, "scsr")
        faf_impact = self.predict_module_impact(context, "faf")

        # 经验增强（如果有）
        context_type = context.context_type
        if context_type in self.module_performance:
            perf = self.module_performance[context_type]
            if "pmp" in perf:
                pmp_impact = 0.7 * pmp_impact + 0.3 * perf["pmp"]
            if "esg" in perf:
                esg_impact = 0.7 * esg_impact + 0.3 * perf["esg"]
            if "hr" in perf:
                hr_impact = 0.7 * hr_impact + 0.3 * perf["hr"]
            if "scsr" in perf:
                scsr_impact = 0.7 * scsr_impact + 0.3 * perf["scsr"]
            if "faf" in perf:
                faf_impact = 0.7 * faf_impact + 0.3 * perf["faf"]

        # 决策规则（硬编码 + 学习）
        # ESG 始终重要（情感对记忆编码至关重要）
        enable_esg = esg_impact > -0.5

        # HR 在长对话有害，自动关闭
        enable_hr = context.context_type != "long_conversation" and hr_impact > 0.0

        # SCSR 在具身场景有益
        enable_scsr = context.context_type == "embodied" or scsr_impact > 0.2

        # FaF 始终有益（防止内存溢出）
        enable_faf = True

        # PMP 在复杂任务有益
        enable_pmp = context.complexity > 0.3 and pmp_impact > 0.0

        return ModuleConfig(
            enable_pmp=enable_pmp,
            enable_esg=enable_esg,
            enable_hr=enable_hr,
            enable_scsr=enable_scsr,
            enable_faf=enable_faf,
            enable_amo=True,
            enable_cmc=False
        )

    def record_experience(self, context: MemoryContext,
                         config: ModuleConfig,
                         success: float):
        """
        记录经验用于学习

        Args:
            context: 当前场景
            config: 使用的配置
            success: 成功指标 (0-1)
        """
        self.experience_buffer.append((context, config, success))

        if len(self.experience_buffer) > self.max_experiences:
            self.experience_buffer.pop(0)

        # 更新统计
        self.context_stats[context.context_type][config.get_type()] += 1

        # 更新模块性能
        if len(self.experience_buffer) > 10:
            recent = self.experience_buffer[-100:]
            for ctx, cfg, succ in recent:
                if ctx.context_type == context.context_type:
                    if cfg.enable_pmp:
                        self.module_performance[context.context_type]["pmp"] = \
                            0.9 * self.module_performance[context.context_type]["pmp"] + 0.1 * succ
                    if cfg.enable_esg:
                        self.module_performance[context.context_type]["esg"] = \
                            0.9 * self.module_performance[context.context_type]["esg"] + 0.1 * succ
                    if cfg.enable_hr:
                        self.module_performance[context.context_type]["hr"] = \
                            0.9 * self.module_performance[context.context_type]["hr"] + 0.1 * succ

    def get_type(self) -> str:
        """获取配置类型描述"""
        return "AMO"


# ============================================================
# CMC: Continuous Memory Consolidation
# 持续记忆巩固
# ============================================================

class ContinuousMemoryConsolidator:
    """
    持续记忆巩固器

    核心思想：
    - 不是"睡眠时才巩固"，而是每次检索都触发微巩固
    - 重要记忆更频繁地巩固
    - 遗忘和巩固同时进行

    数学公式：
    m_i(t+1) = α · m_i(t) + β · micro_consolidation(m_i, context)

    其中 micro_consolidation 是基于检索触发的
    """

    def __init__(self, consolidation_threshold: float = 0.7):
        self.consolidation_threshold = consolidation_threshold
        self.micro_consolidation_count = 0
        self.total_accesses = 0

        # 微巩固参数
        self.alpha = 0.95  # 保持系数
        self.beta = 0.05   # 微巩固强度

        # 访问模式
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.consolidation_timestamps: Dict[str, float] = {}

    def should_consolidate(self, memory_importance: float,
                          memory_access_count: int,
                          time_since_last_consolidation: float) -> bool:
        """
        决定是否需要微巩固

        条件：
        1. 记忆重要性 > 阈值
        2. 访问次数 > 阈值
        3. 距离上次巩固 > 阈值
        """
        importance_trigger = memory_importance > self.consolidation_threshold
        access_trigger = memory_access_count >= 3
        time_trigger = time_since_last_consolidation > 3600  # 1小时

        return importance_trigger or (access_trigger and time_trigger)

    def micro_consolidate(self, memory, query_context: np.ndarray,
                         emotional_state) -> float:
        """
        执行微巩固

        Args:
            memory: 待巩固的记忆
            query_context: 查询上下文向量
            emotional_state: 当前情感状态

        Returns:
            float: 巩固强度
        """
        self.micro_consolidation_count += 1
        self.total_accesses += 1

        # 1. 重要性更新
        if hasattr(emotional_state, 'arousal'):
            importance_boost = emotional_state.arousal * 0.1
            memory.importance = min(1.0, memory.importance + importance_boost)

        # 2. 嵌入微调（向查询上下文方向移动一小步）
        if query_context is not None and len(memory.embedding) == len(query_context):
            # 只移动5%的距离
            direction = query_context - memory.embedding
            memory.embedding += self.beta * direction

        # 3. 巩固强度更新
        consolidation_strength = self.beta * (1.0 + memory.importance)
        memory.consolidation_strength = min(1.0,
            memory.consolidation_strength + consolidation_strength)

        # 4. 时间戳更新
        self.consolidation_timestamps[memory.id] = datetime.now().timestamp()

        return consolidation_strength

    def adaptive_beta(self, memory) -> float:
        """
        自适应调整巩固强度

        根据记忆属性动态调整β
        """
        base_beta = 0.05

        # 高情感 → 更强巩固
        if hasattr(memory, 'emotional_arousal'):
            beta = base_beta * (1.0 + memory.emotional_arousal)
        else:
            beta = base_beta

        # 高重要性 → 更强巩固
        beta *= (1.0 + memory.importance)

        # 频繁访问 → 减弱巩固（已足够稳定）
        if self.access_counts[memory.id] > 10:
            beta *= 0.5

        return min(0.2, max(0.01, beta))

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "micro_consolidations": self.micro_consolidation_count,
            "total_accesses": self.total_accesses,
            "avg_beta": self.beta,
            "threshold": self.consolidation_threshold
        }
