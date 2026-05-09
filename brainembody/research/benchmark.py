"""
研究级基准测试框架
包含：合成基准 + 消融实验 + 基线对比 + 统计检验
"""

import sys
import os
import numpy as np
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainembody.memory.phmeg import PHMEGMemory, EmotionalState, TaskTrajectory
from brainembody.research.embedder import ResearchEmbedder
from brainembody.research.metrics import (
    MetricsCalculator, SignificanceTester, RetrievalResult
)


# ============================================================
# 合成基准数据集
# ============================================================

class SyntheticBenchmark:
    """
    合成基准数据集

    模拟真实场景：
    - 多轮对话记忆
    - 长期知识检索
    - 情感增强回忆
    - 时间衰减记忆
    - 跨会话依赖
    """

    def __init__(self, embedder: ResearchEmbedder, seed: int = 42):
        self.embedder = embedder
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def generate_conversation_dataset(
        self, n_conversations: int = 20, n_turns_per_conv: int = 10
    ) -> Dict:
        """生成多轮对话数据集"""
        np.random.seed(self.seed)

        user_profiles = [
            {"name": "张三", "role": "AI研究员", "interest": "类脑计算"},
            {"name": "李四", "role": "机器人工程师", "interest": "具身智能"},
            {"name": "王五", "role": "数据科学家", "interest": "深度学习"},
        ]

        conversation_topics = [
            "导航", "避障", "目标识别", "路径规划", "传感器融合",
            "强化学习", "记忆系统", "情感计算", "多模态感知", "人机交互"
        ]

        conversations = []
        all_memories = []
        queries = []

        for conv_idx in range(n_conversations):
            profile = user_profiles[conv_idx % len(user_profiles)]
            topic = conversation_topics[conv_idx % len(conversation_topics)]

            conv_memories = []
            for turn_idx in range(n_turns_per_conv):
                content = f"{profile['name']}讨论了{topic}的第{turn_idx+1}个方面"
                if turn_idx % 3 == 0:
                    content += f"，提到了{profile['interest']}相关内容"

                emotional_state = EmotionalState(
                    valence=self.rng.uniform(-0.5, 1.0),
                    arousal=self.rng.uniform(0.1, 0.9),
                    dominance=self.rng.uniform(0.3, 0.8)
                )

                memory_entry = {
                    "id": f"conv{conv_idx}_turn{turn_idx}",
                    "content": content,
                    "emotional_state": emotional_state,
                    "conversation_id": conv_idx,
                    "turn_index": turn_idx,
                    "user_profile": profile,
                    "topic": topic
                }

                conv_memories.append(memory_entry)
                all_memories.append(memory_entry)

            conversations.append({
                "id": conv_idx,
                "memories": conv_memories,
                "profile": profile,
                "topic": topic
            })

            for turn_idx in range(0, n_turns_per_conv, 3):
                query_entry = {
                    "id": f"q_conv{conv_idx}_turn{turn_idx}",
                    "query": f"关于{topic}的讨论",
                    "relevant_ids": [m["id"] for m in conv_memories[max(0, turn_idx-2):turn_idx+3]],
                    "conversation_id": conv_idx
                }
                queries.append(query_entry)

        return {
            "conversations": conversations,
            "memories": all_memories,
            "queries": queries,
            "metadata": {
                "n_conversations": n_conversations,
                "n_turns_per_conv": n_turns_per_conv,
                "n_total_memories": len(all_memories),
                "n_queries": len(queries)
            }
        }

    def generate_embodied_dataset(
        self, n_episodes: int = 50, n_steps_per_episode: int = 20
    ) -> Dict:
        """生成具身智能经验数据集"""
        np.random.seed(self.seed)

        scenarios = [
            "室内导航", "物体抓取", "避障行驶", "目标搜索", "路径规划"
        ]

        all_experiences = []
        queries = []

        for ep_idx in range(n_episodes):
            scenario = scenarios[ep_idx % len(scenarios)]

            ep_experiences = []
            for step_idx in range(n_steps_per_episode):
                success = self.rng.random() > 0.3
                content = f"在{scenario}场景中，机器人执行了第{step_idx+1}步动作"
                if success:
                    content += "，成功完成了子任务"
                else:
                    content += "，遇到了障碍需要调整策略"

                emotional_state = EmotionalState(
                    valence=0.8 if success else -0.3,
                    arousal=0.7 if success else 0.9,
                    dominance=0.6 if success else 0.3
                )

                exp = {
                    "id": f"ep{ep_idx}_step{step_idx}",
                    "content": content,
                    "emotional_state": emotional_state,
                    "episode_id": ep_idx,
                    "step_index": step_idx,
                    "scenario": scenario,
                    "success": success
                }

                ep_experiences.append(exp)
                all_experiences.append(exp)

            queries.append({
                "id": f"q_ep{ep_idx}",
                "query": f"在{scenario}场景中如何行动",
                "relevant_ids": [e["id"] for e in ep_experiences if e["success"]][:5],
                "episode_id": ep_idx
            })

        return {
            "experiences": all_experiences,
            "queries": queries,
            "metadata": {
                "n_episodes": n_episodes,
                "n_steps_per_episode": n_steps_per_episode,
                "n_total_experiences": len(all_experiences),
                "n_queries": len(queries)
            }
        }


# ============================================================
# 基线系统
# ============================================================

class BaselineRAG:
    """基线1: 标准RAG（向量存储 + 余弦检索）"""

    def __init__(self, embedder: ResearchEmbedder):
        self.embedder = embedder
        self.memories = {}

    def encode(self, content: str, **kwargs) -> str:
        mem_id = f"rag_{len(self.memories)}"
        self.memories[mem_id] = {
            "id": mem_id,
            "content": content,
            "embedding": self.embedder.embed(content)
        }
        return mem_id

    def encode_with_id(self, mem_id: str, content: str, **kwargs) -> str:
        self.memories[mem_id] = {
            "id": mem_id,
            "content": content,
            "embedding": self.embedder.embed(content)
        }
        return mem_id

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        query_emb = self.embedder.embed(query)
        scores = []
        for mem_id, mem in self.memories.items():
            sim = np.dot(query_emb, mem["embedding"]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem["embedding"]) + 1e-8
            )
            scores.append((sim, mem))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s}
                for s, m in scores[:top_k]]

    def consolidate(self):
        pass


class BaselineTimeDecay:
    """基线2: 时间衰减RAG"""

    def __init__(self, embedder: ResearchEmbedder, decay_rate: float = 0.02):
        self.embedder = embedder
        self.decay_rate = decay_rate
        self.memories = {}

    def encode(self, content: str, **kwargs) -> str:
        mem_id = f"td_{len(self.memories)}"
        self.memories[mem_id] = {
            "id": mem_id,
            "content": content,
            "embedding": self.embedder.embed(content),
            "importance": 1.0,
            "created_at": time.time()
        }
        return mem_id

    def encode_with_id(self, mem_id: str, content: str, **kwargs) -> str:
        self.memories[mem_id] = {
            "id": mem_id,
            "content": content,
            "embedding": self.embedder.embed(content),
            "importance": 1.0,
            "created_at": time.time()
        }
        return mem_id

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        query_emb = self.embedder.embed(query)
        current_time = time.time()
        scores = []
        for mem_id, mem in self.memories.items():
            sim = np.dot(query_emb, mem["embedding"]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem["embedding"]) + 1e-8
            )
            age = current_time - mem["created_at"]
            time_factor = np.exp(-self.decay_rate * age / 3600)
            score = sim * 0.7 + mem["importance"] * time_factor * 0.3
            scores.append((score, mem))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s}
                for s, m in scores[:top_k]]

    def consolidate(self):
        current_time = time.time()
        to_remove = []
        for mem_id, mem in self.memories.items():
            age = current_time - mem["created_at"]
            mem["importance"] *= np.exp(-self.decay_rate * age / 3600)
            if mem["importance"] < 0.05:
                to_remove.append(mem_id)
        for mem_id in to_remove:
            del self.memories[mem_id]


class BaselineEmotionalRAG:
    """基线3: 情感增强RAG（类似mnemos的AffectiveRouter）"""

    def __init__(self, embedder: ResearchEmbedder):
        self.embedder = embedder
        self.memories = {}

    def encode(self, content: str, emotional_state: EmotionalState = None, **kwargs) -> str:
        mem_id = f"erag_{len(self.memories)}"
        self.memories[mem_id] = {
            "id": mem_id,
            "content": content,
            "embedding": self.embedder.embed(content),
            "emotional_valence": emotional_state.valence if emotional_state else 0,
            "emotional_arousal": emotional_state.arousal if emotional_state else 0,
        }
        return mem_id

    def encode_with_id(self, mem_id: str, content: str, emotional_state: EmotionalState = None, **kwargs) -> str:
        self.memories[mem_id] = {
            "id": mem_id,
            "content": content,
            "embedding": self.embedder.embed(content),
            "emotional_valence": emotional_state.valence if emotional_state else 0,
            "emotional_arousal": emotional_state.arousal if emotional_state else 0,
        }
        return mem_id

    def retrieve(self, query: str, top_k: int = 5, emotional_state: EmotionalState = None, **kwargs) -> List[Dict]:
        query_emb = self.embedder.embed(query)
        scores = []
        for mem_id, mem in self.memories.items():
            sim = np.dot(query_emb, mem["embedding"]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem["embedding"]) + 1e-8
            )
            emotional_match = 0.0
            if emotional_state:
                emotional_match = (
                    abs(mem["emotional_valence"] - emotional_state.valence) +
                    abs(mem["emotional_arousal"] - emotional_state.arousal)
                ) / 2.0
                emotional_match = 1.0 - emotional_match

            score = sim * 0.7 + emotional_match * 0.3
            scores.append((score, mem))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s}
                for s, m in scores[:top_k]]

    def consolidate(self):
        pass


# ============================================================
# 消融实验配置
# ============================================================

@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    enable_pmp: bool = True
    enable_esg: bool = True
    enable_hr: bool = True
    enable_scsr: bool = True
    enable_faf: bool = True


ABLATION_CONFIGS = [
    AblationConfig("PHMEG (Full)", True, True, True, True, True),
    AblationConfig("w/o PMP", False, True, True, True, True),
    AblationConfig("w/o ESG", True, False, True, True, True),
    AblationConfig("w/o HR", True, True, False, True, True),
    AblationConfig("w/o SCSR", True, True, True, False, True),
    AblationConfig("w/o FaF-PV", True, True, True, True, False),
    AblationConfig("w/o PMP+ESG", False, False, True, True, True),
    AblationConfig("w/o HR+SCSR", True, True, False, False, True),
]


# ============================================================
# 完整实验运行器
# ============================================================

class ExperimentRunner:
    """完整实验运行器"""

    def __init__(self, embedder: ResearchEmbedder, seed: int = 42):
        self.embedder = embedder
        self.seed = seed
        self.benchmark = SyntheticBenchmark(embedder, seed)
        self.metrics_calc = MetricsCalculator()
        self.sig_tester = SignificanceTester()

    def _load_dataset_into_system(self, system, dataset: Dict) -> Dict:
        """将数据集加载到系统中"""
        id_mapping = {}
        items = dataset.get("memories", dataset.get("experiences", []))
        for memory in items:
            original_id = memory["id"]
            if isinstance(system, PHMEGMemory):
                new_id = system.encode(
                    memory["content"],
                    emotional_state=memory.get("emotional_state"),
                    is_episodic=True,
                    memory_id=original_id
                )
            else:
                if hasattr(system, 'encode_with_id'):
                    new_id = system.encode_with_id(
                        original_id,
                        memory["content"],
                        emotional_state=memory.get("emotional_state")
                    )
                else:
                    new_id = system.encode(
                        memory["content"],
                        emotional_state=memory.get("emotional_state")
                    )
            id_mapping[original_id] = new_id
        return id_mapping

    def _evaluate_system(self, system, dataset: Dict, id_mapping: Dict,
                         top_k: int = 5) -> List[RetrievalResult]:
        """评估系统"""
        results = []

        for query in dataset["queries"]:
            if isinstance(system, PHMEGMemory):
                retrieved = system.retrieve(query["query"], top_k=top_k)
                retrieved_ids = [r["id"] for r in retrieved]
            else:
                retrieved = system.retrieve(query["query"], top_k=top_k)
                retrieved_ids = [r["id"] for r in retrieved]

            relevant_mapped = []
            for orig_id in query["relevant_ids"]:
                if orig_id in id_mapping:
                    relevant_mapped.append(id_mapping[orig_id])

            results.append(RetrievalResult(
                query_id=query["id"],
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_mapped,
                scores=[]
            ))

        return results

    def run_baseline_comparison(self, dataset_type: str = "conversation") -> Dict:
        """运行基线对比实验"""
        print(f"\n{'='*70}")
        print(f"基线对比实验 - {dataset_type}")
        print(f"{'='*70}")

        if dataset_type == "conversation":
            dataset = self.benchmark.generate_conversation_dataset(
                n_conversations=20, n_turns_per_conv=10
            )
        else:
            dataset = self.benchmark.generate_embodied_dataset(
                n_episodes=50, n_steps_per_episode=20
            )

        print(f"\n数据集: {dataset['metadata']}")

        systems = {
            "RAG": BaselineRAG(self.embedder),
            "TimeDecay": BaselineTimeDecay(self.embedder),
            "EmotionalRAG": BaselineEmotionalRAG(self.embedder),
            "PHMEG": PHMEGMemory(embedding_dim=self.embedder.embedding_dim, embedder=self.embedder),
        }

        all_results = {}
        all_metrics = {}

        for name, system in systems.items():
            print(f"\n评估 {name}...")

            id_mapping = self._load_dataset_into_system(system, dataset)

            if isinstance(system, PHMEGMemory):
                system.sleep_consolidate()
                surviving_ids = set(system.memories.keys())
                id_mapping = {k: v for k, v in id_mapping.items() if v in surviving_ids}

            retrieval_results = self._evaluate_system(system, dataset, id_mapping)
            metrics = self.metrics_calc.compute_all_metrics(retrieval_results)

            all_results[name] = retrieval_results
            all_metrics[name] = metrics

            print(f"  F1@5: {metrics.get('F1@5', 0):.4f}")
            print(f"  Recall@5: {metrics.get('Recall@5', 0):.4f}")
            print(f"  MRR: {metrics.get('MRR', 0):.4f}")
            print(f"  HitRate: {metrics.get('HitRate', 0):.4f}")

        # 统计显著性检验
        print(f"\n{'='*70}")
        print("统计显著性检验 (PHMEG vs 各基线)")
        print(f"{'='*70}")

        significance_results = {}
        phmeg_f1_scores = [1.0 if any(r in res.relevant_ids for r in res.retrieved_ids[:5]) else 0.0
                          for res in all_results["PHMEG"]]

        for baseline_name in ["RAG", "TimeDecay", "EmotionalRAG"]:
            baseline_f1_scores = [1.0 if any(r in res.relevant_ids for r in res.retrieved_ids[:5]) else 0.0
                                 for res in all_results[baseline_name]]

            comparison = self.sig_tester.full_comparison(
                phmeg_f1_scores, baseline_f1_scores,
                name_a="PHMEG", name_b=baseline_name
            )
            significance_results[baseline_name] = comparison

            print(f"\nPHMEG vs {baseline_name}:")
            print(f"  t-test p-value: {comparison['paired_t_test']['p_value']:.4f}")
            print(f"  Cohen's d: {comparison['effect_size']['cohens_d']:.3f} ({comparison['effect_size']['magnitude']})")
            print(f"  改进: {comparison['effect_size']['improvement']:.1f}%")

        return {
            "metrics": all_metrics,
            "significance": significance_results
        }

    def run_ablation_study(self, dataset_type: str = "conversation") -> Dict:
        """运行消融实验"""
        print(f"\n{'='*70}")
        print(f"消融实验 - {dataset_type}")
        print(f"{'='*70}")

        if dataset_type == "conversation":
            dataset = self.benchmark.generate_conversation_dataset(
                n_conversations=20, n_turns_per_conv=10
            )
        else:
            dataset = self.benchmark.generate_embodied_dataset(
                n_episodes=50, n_steps_per_episode=20
            )

        ablation_results = {}

        for config in ABLATION_CONFIGS:
            print(f"\n评估 {config.name}...")

            system = PHMEGMemory(
                embedding_dim=self.embedder.embedding_dim,
                embedder=self.embedder,
                enable_pmp=config.enable_pmp,
                enable_esg=config.enable_esg,
                enable_hr=config.enable_hr,
                enable_scsr=config.enable_scsr,
                enable_faf=config.enable_faf
            )

            id_mapping = self._load_dataset_into_system(system, dataset)

            if config.enable_scsr:
                system.sleep_consolidate()
                surviving_ids = set(system.memories.keys())
                id_mapping = {k: v for k, v in id_mapping.items() if v in surviving_ids}

            retrieval_results = self._evaluate_system(system, dataset, id_mapping)
            metrics = self.metrics_calc.compute_all_metrics(retrieval_results)

            ablation_results[config.name] = metrics

            print(f"  F1@5: {metrics.get('F1@5', 0):.4f}")
            print(f"  Recall@5: {metrics.get('Recall@5', 0):.4f}")
            print(f"  MRR: {metrics.get('MRR', 0):.4f}")

        return ablation_results

    def run_full_experiment(self) -> Dict:
        """运行完整实验"""
        print("=" * 70)
        print("PHMEG 完整研究实验")
        print("Predictive Hierarchical Memory with Emotional Gating")
        print("=" * 70)

        results = {}

        results["conversation_baselines"] = self.run_baseline_comparison("conversation")
        results["embodied_baselines"] = self.run_baseline_comparison("embodied")
        results["conversation_ablation"] = self.run_ablation_study("conversation")
        results["embodied_ablation"] = self.run_ablation_study("embodied")

        return results


# ============================================================
# 论文级结果表格生成
# ============================================================

def format_paper_table(metrics: Dict, title: str = "Main Results"):
    """生成论文格式表格"""
    print(f"\n{'='*80}")
    print(f"Table: {title}")
    print(f"{'='*80}")
    print(f"{'System':<20} {'P@5':>8} {'R@5':>8} {'F1@5':>8} {'MRR':>8} {'HitRate':>8} {'NDCG@5':>8}")
    print("-" * 80)

    for name, m in metrics.items():
        print(f"{name:<20} "
              f"{m.get('Precision@5', 0):>8.4f} "
              f"{m.get('Recall@5', 0):>8.4f} "
              f"{m.get('F1@5', 0):>8.4f} "
              f"{m.get('MRR', 0):>8.4f} "
              f"{m.get('HitRate', 0):>8.4f} "
              f"{m.get('NDCG@5', 0):>8.4f}")

    print("-" * 80)


def format_significance_table(sig_results: Dict):
    """生成显著性检验表格"""
    print(f"\n{'='*80}")
    print("Table: Statistical Significance Tests (PHMEG vs Baselines)")
    print(f"{'='*80}")
    print(f"{'Comparison':<25} {'p-value':>10} {'Cohen-d':>10} {'Magnitude':>12} {'Improvement':>12}")
    print("-" * 80)

    for baseline, result in sig_results.items():
        t_test = result["paired_t_test"]
        effect = result["effect_size"]
        print(f"PHMEG vs {baseline:<15} "
              f"{t_test['p_value']:>10.4f} "
              f"{effect['cohens_d']:>10.3f} "
              f"{effect['magnitude']:>12} "
              f"{effect['improvement']:>11.1f}%")

    print("-" * 80)


# ============================================================
# 主函数
# ============================================================

def main():
    print("初始化研究级嵌入模型...")
    embedder = ResearchEmbedder()

    runner = ExperimentRunner(embedder, seed=42)

    results = runner.run_full_experiment()

    # 生成论文表格
    if "conversation_baselines" in results:
        format_paper_table(
            results["conversation_baselines"]["metrics"],
            "Conversation Benchmark - Baseline Comparison"
        )
        format_significance_table(
            results["conversation_baselines"]["significance"]
        )

    if "embodied_baselines" in results:
        format_paper_table(
            results["embodied_baselines"]["metrics"],
            "Embodied Benchmark - Baseline Comparison"
        )

    if "conversation_ablation" in results:
        format_paper_table(
            results["conversation_ablation"],
            "Ablation Study - Conversation Benchmark"
        )

    if "embodied_ablation" in results:
        format_paper_table(
            results["embodied_ablation"],
            "Ablation Study - Embodied Benchmark"
        )

    print("\n" + "=" * 80)
    print("实验完成！以上结果可直接用于论文。")
    print("=" * 80)


if __name__ == "__main__":
    main()
