"""
AMO + CMC 新模块验证实验
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainembody.memory.phmeg import PHMEGMemory, EmotionalState
from brainembody.research.embedder import ResearchEmbedder
from brainembody.research.metrics import MetricsCalculator, RetrievalResult


class SimpleRAG:
    def __init__(self, embedder):
        self.embedder = embedder
        self.memories = {}

    def encode_with_id(self, mem_id, content, emotional_state=None):
        self.memories[mem_id] = {
            "id": mem_id, "content": content,
            "embedding": self.embedder.embed(content)
        }
        return mem_id

    def retrieve(self, query, top_k=5):
        q_emb = self.embedder.embed(query)
        scores = []
        for mid, m in self.memories.items():
            sim = np.dot(q_emb, m["embedding"]) / (
                np.linalg.norm(q_emb) * np.linalg.norm(m["embedding"]) + 1e-8)
            scores.append((sim, m))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s} for s, m in scores[:top_k]]


def create_dataset():
    """创建测试数据集"""
    np.random.seed(42)
    memories = []
    queries = []
    mem_id_set = set()

    for i in range(50):
        mem_id = f"short_{i}"
        content = f"短对话关于机器人导航内容编号{i}"
        memories.append({'id': mem_id, 'content': content, 'emotional_state': EmotionalState(0.3, 0.5, 0.5)})
        mem_id_set.add(mem_id)
        if i % 5 == 0:
            queries.append({'id': f"q_short_{i}", 'query': "机器人导航", 'relevant_ids': [mem_id]})

    # 长对话场景
    for i in range(600):
        mem_id = f"long_{i}"
        content = f"长对话关于具身智能的复杂任务编号{i}"
        memories.append({'id': mem_id, 'content': content, 'emotional_state': EmotionalState(0.0, 0.3, 0.5)})
        mem_id_set.add(mem_id)
        if i % 10 == 0:
            queries.append({'id': f"q_long_{i}", 'query': "具身智能", 'relevant_ids': [mem_id]})

    # 具身场景
    for i in range(100):
        mem_id = f"embodied_{i}"
        content = f"具身任务机器人在复杂环境中执行动作编号{i}"
        memories.append({'id': mem_id, 'content': content, 'emotional_state': EmotionalState(0.7, 0.8, 0.6)})
        mem_id_set.add(mem_id)
        if i % 5 == 0:
            queries.append({'id': f"q_embodied_{i}", 'query': "机器人动作", 'relevant_ids': [mem_id]})

    return {'memories': memories, 'queries': queries}


def run_experiment():
    """运行对比实验"""
    print("=" * 80)
    print("AMO + CMC 新模块验证实验")
    print("=" * 80)

    embedder = ResearchEmbedder()
    dataset = create_dataset()
    metrics_calc = MetricsCalculator()

    systems = {
        "RAG (Baseline)": SimpleRAG(embedder),
        "PHMEG (no AMO)": PHMEGMemory(embedder.embedding_dim, embedder, enable_hr=False, enable_scsr=False),
        "PHMEG + AMO": PHMEGMemory(embedder.embedding_dim, embedder, enable_hr=False, enable_scsr=False, enable_amo=True),
        "PHMEG + CMC": PHMEGMemory(embedder.embedding_dim, embedder, enable_hr=False, enable_scsr=False, enable_cmc=True),
        "PHMEG + AMO+CMC": PHMEGMemory(embedder.embedding_dim, embedder, enable_hr=False, enable_scsr=False, enable_amo=True, enable_cmc=True),
    }

    all_metrics = {}

    for name, system in systems.items():
        print(f"\n{'='*40}")
        print(f"评估 {name}")
        print(f"{'='*40}")

        for memory in dataset["memories"]:
            if isinstance(system, PHMEGMemory):
                system.encode(memory["content"], memory_id=memory["id"], emotional_state=memory.get("emotional_state"))
            else:
                system.encode_with_id(memory["id"], memory["content"])

        if isinstance(system, PHMEGMemory):
            surviving = set(system.memories.keys())
            id_map = {m['id']: m['id'] for m in dataset["memories"] if m['id'] in surviving}
        else:
            id_map = {m['id']: m['id'] for m in dataset["memories"]}

        results = []
        for query in dataset["queries"]:
            retrieved = system.retrieve(query["query"], top_k=5)
            retrieved_ids = [r["id"] for r in retrieved]
            relevant = [id_map.get(rid, rid) for rid in query["relevant_ids"] if rid in id_map]

            if relevant:
                results.append(RetrievalResult(
                    query_id=query["id"],
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant,
                    scores=[]
                ))

        metrics = metrics_calc.compute_all_metrics(results)
        all_metrics[name] = metrics

        print(f"  记忆数: {len(system.memories) if isinstance(system, PHMEGMemory) else len(system.memories)}")
        print(f"  F1@5: {metrics.get('F1@5', 0):.4f}")
        print(f"  Recall@5: {metrics.get('Recall@5', 0):.4f}")
        print(f"  MRR: {metrics.get('MRR', 0):.4f}")
        print(f"  HitRate: {metrics.get('HitRate', 0):.4f}")

    # 结果表格
    print(f"\n{'='*80}")
    print("Table: AMO + CMC Ablation Results")
    print(f"{'='*80}")
    print(f"{'System':<25} {'F1@5':>8} {'R@5':>8} {'MRR':>8} {'HitRate':>8}")
    print("-" * 80)
    for name, m in all_metrics.items():
        print(f"{name:<25} {m.get('F1@5', 0):>8.4f} {m.get('Recall@5', 0):>8.4f} {m.get('MRR', 0):>8.4f} {m.get('HitRate', 0):>8.4f}")
    print("-" * 80)

    # AMO场景检测示例
    if "PHMEG + AMO" in systems or "PHMEG + AMO+CMC" in systems:
        system = systems.get("PHMEG + AMO+CMC", systems.get("PHMEG + AMO"))
        if hasattr(system, 'orchestrator') and system.orchestrator:
            print(f"\n{'='*40}")
            print("AMO 场景检测示例")
            print(f"{'='*40}")
            test_queries = ["机器人导航怎么做", "关于具身智能的研究", "简单的对话"]
            for q in test_queries:
                ctx = system.orchestrator.detect_context(q, 600, [], [EmotionalState(0.3, 0.5, 0.5)])
                cfg = system.orchestrator.select_optimal_config(ctx)
                print(f"  查询: '{q}'")
                print(f"    场景: {ctx.context_type}")
                print(f"    HR开启: {cfg.enable_hr}, ESG开启: {cfg.enable_esg}, SCSR开启: {cfg.enable_scsr}")
                print()


if __name__ == "__main__":
    run_experiment()
