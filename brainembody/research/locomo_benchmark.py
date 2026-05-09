"""
LoCoMo 基准测试适配器
ACL 2024: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from brainembody.memory.phmeg import PHMEGMemory, EmotionalState
from brainembody.research.embedder import ResearchEmbedder
from brainembody.research.metrics import MetricsCalculator, SignificanceTester, RetrievalResult


@dataclass
class LoCoMoSample:
    """LoCoMo 数据样本"""
    sample_id: str
    sessions: List[Dict]
    qa_pairs: List[Dict]
    event_summaries: List[Dict]


class LoCoMoAdapter:
    """
    LoCoMo 基准适配器

    将 LoCoMo 数据集转换为 PHMEG 评估格式
    """

    def __init__(self, data_path: str, embedder: ResearchEmbedder):
        self.data_path = data_path
        self.embedder = embedder
        self.samples: List[LoCoMoSample] = []
        self._load_data()

    def _load_data(self):
        """加载数据"""
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)

        for item in raw_data:
            sample_id = item.get('sample_id', 'unknown')
            conversation = item.get('conversation', {})
            qa_pairs = item.get('qa', [])
            event_summaries = item.get('event_summary', [])

            sessions = []
            session_keys = sorted([
                k for k in conversation.keys()
                if k.startswith('session_') and not any(
                    x in k for x in ['date', 'observation', 'summary']
                )
            ])

            for session_key in session_keys:
                session_data = conversation[session_key]
                if isinstance(session_data, list):
                    turns = []
                    for turn in session_data:
                        if isinstance(turn, dict) and 'text' in turn:
                            turns.append({
                                'speaker': turn.get('speaker', 'unknown'),
                                'text': turn.get('text', ''),
                                'dia_id': turn.get('dia_id', '')
                            })
                    sessions.append({
                        'session_id': session_key,
                        'turns': turns
                    })

            self.samples.append(LoCoMoSample(
                sample_id=str(sample_id),
                sessions=sessions,
                qa_pairs=qa_pairs,
                event_summaries=event_summaries
            ))

        total_qa = sum(len(s.qa_pairs) for s in self.samples)
        total_sessions = sum(len(s.sessions) for s in self.samples)
        print(f"✓ LoCoMo 加载完成: {len(self.samples)} 个对话, "
              f"{total_sessions} 个会话, {total_qa} 个QA对")

    def build_memory_dataset(self, max_samples: int = 3, max_qa_per_sample: int = 50) -> Dict:
        """构建记忆数据集（限制规模避免超时）"""
        memories = []
        queries = []
        mem_id_set = set()

        for sample in self.samples[:max_samples]:
            for session in sample.sessions:
                for turn_idx, turn in enumerate(session['turns'][:30]):
                    mem_id = f"{sample.sample_id}_{session['session_id']}_t{turn_idx}"
                    memories.append({
                        'id': mem_id,
                        'content': f"{turn['speaker']}: {turn['text'][:200]}",
                        'emotional_state': EmotionalState(0.0, 0.3, 0.5),
                        'sample_id': sample.sample_id,
                        'session_id': session['session_id'],
                        'dia_id': turn.get('dia_id', '')
                    })
                    mem_id_set.add(mem_id)

            for qa_idx, qa in enumerate(sample.qa_pairs[:max_qa_per_sample]):
                # 将evidence dia_id映射到mem_id
                evidence_ids = []
                if 'evidence' in qa and qa['evidence']:
                    for ev in qa['evidence']:
                        # ev格式如 "D1:3" -> 找到对应的mem_id
                        for mem in memories:
                            if mem['sample_id'] == sample.sample_id:
                                if mem.get('dia_id', '') == ev:
                                    evidence_ids.append(mem['id'])
                                    break

                # 如果没有精确匹配，用内容相似度匹配
                if not evidence_ids and qa.get('answer', ''):
                    answer_text = qa['answer'].lower()
                    for mem in memories:
                        if mem['sample_id'] == sample.sample_id:
                            if any(word in mem['content'].lower() for word in answer_text.split()[:3] if len(word) > 1):
                                evidence_ids.append(mem['id'])
                                if len(evidence_ids) >= 3:
                                    break

                if evidence_ids:
                    queries.append({
                        'id': f"q_{sample.sample_id}_{qa_idx}",
                        'query': qa['question'],
                        'answer': qa.get('answer', ''),
                        'relevant_ids': evidence_ids,
                        'category': qa.get('category', 0),
                        'sample_id': sample.sample_id
                    })

        return {
            'memories': memories,
            'queries': queries,
            'metadata': {
                'n_samples': min(max_samples, len(self.samples)),
                'n_memories': len(memories),
                'n_queries': len(queries)
            }
        }


class BaselineRAG:
    def __init__(self, embedder):
        self.embedder = embedder
        self.memories = {}

    def encode_with_id(self, mem_id, content, **kwargs):
        self.memories[mem_id] = {
            "id": mem_id, "content": content,
            "embedding": self.embedder.embed(content)
        }
        return mem_id

    def retrieve(self, query, top_k=5, **kwargs):
        q_emb = self.embedder.embed(query)
        scores = []
        for mid, m in self.memories.items():
            sim = np.dot(q_emb, m["embedding"]) / (
                np.linalg.norm(q_emb) * np.linalg.norm(m["embedding"]) + 1e-8)
            scores.append((sim, m))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s} for s, m in scores[:top_k]]

    def consolidate(self):
        pass


class BaselineEmotionalRAG:
    def __init__(self, embedder):
        self.embedder = embedder
        self.memories = {}

    def encode_with_id(self, mem_id, content, emotional_state=None, **kwargs):
        self.memories[mem_id] = {
            "id": mem_id, "content": content,
            "embedding": self.embedder.embed(content),
            "emotional_valence": emotional_state.valence if emotional_state else 0,
            "emotional_arousal": emotional_state.arousal if emotional_state else 0,
        }
        return mem_id

    def retrieve(self, query, top_k=5, emotional_state=None, **kwargs):
        q_emb = self.embedder.embed(query)
        scores = []
        for mid, m in self.memories.items():
            sim = np.dot(q_emb, m["embedding"]) / (
                np.linalg.norm(q_emb) * np.linalg.norm(m["embedding"]) + 1e-8)
            emotional_match = 0.0
            if emotional_state:
                emotional_match = (1.0 - abs(m["emotional_valence"] - emotional_state.valence)) * 0.3
            score = sim * 0.7 + emotional_match
            scores.append((score, m))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s} for s, m in scores[:top_k]]

    def consolidate(self):
        pass


def run_locomo_experiment(data_path: str, embedder: ResearchEmbedder):
    """运行 LoCoMo 实验"""
    print("=" * 80)
    print("LoCoMo 基准测试 (ACL 2024)")
    print("Evaluating Very Long-Term Conversational Memory")
    print("=" * 80)

    adapter = LoCoMoAdapter(data_path, embedder)
    dataset = adapter.build_memory_dataset()

    print(f"\n数据集: {dataset['metadata']}")

    metrics_calc = MetricsCalculator()
    sig_tester = SignificanceTester()

    systems = {
        "RAG": BaselineRAG(embedder),
        "EmotionalRAG": BaselineEmotionalRAG(embedder),
        "PHMEG": PHMEGMemory(embedding_dim=embedder.embedding_dim, embedder=embedder,
                              enable_hr=False, enable_scsr=False),
        "PHMEG (Full)": PHMEGMemory(embedding_dim=embedder.embedding_dim, embedder=embedder),
    }

    all_results = {}
    all_metrics = {}

    for name, system in systems.items():
        print(f"\n加载 {name}...")

        id_mapping = {}
        for memory in dataset["memories"]:
            original_id = memory["id"]
            if isinstance(system, PHMEGMemory):
                new_id = system.encode(
                    memory["content"],
                    emotional_state=memory.get("emotional_state"),
                    is_episodic=True,
                    memory_id=original_id
                )
            else:
                new_id = system.encode_with_id(
                    original_id,
                    memory["content"],
                    emotional_state=memory.get("emotional_state")
                )
            id_mapping[original_id] = new_id

        if isinstance(system, PHMEGMemory):
            system.sleep_consolidate()
            surviving_ids = set(system.memories.keys())
            id_mapping = {k: v for k, v in id_mapping.items() if v in surviving_ids}

        print(f"  记忆数: {len(system.memories) if isinstance(system, PHMEGMemory) else len(system.memories)}")

        # 评估
        retrieval_results = []
        for query in dataset["queries"]:
            retrieved = system.retrieve(query["query"], top_k=5)
            retrieved_ids = [r["id"] for r in retrieved]
            relevant_mapped = [id_mapping[rid] for rid in query["relevant_ids"] if rid in id_mapping]

            if relevant_mapped:
                retrieval_results.append(RetrievalResult(
                    query_id=query["id"],
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_mapped,
                    scores=[]
                ))

        if not retrieval_results:
            print(f"  ⚠ 无有效评估数据")
            continue

        metrics = metrics_calc.compute_all_metrics(retrieval_results)
        all_results[name] = retrieval_results
        all_metrics[name] = metrics

        print(f"  F1@5: {metrics.get('F1@5', 0):.4f}")
        print(f"  Recall@5: {metrics.get('Recall@5', 0):.4f}")
        print(f"  MRR: {metrics.get('MRR', 0):.4f}")
        print(f"  HitRate: {metrics.get('HitRate', 0):.4f}")

    # 统计检验
    if "PHMEG" in all_results and "RAG" in all_results:
        print(f"\n{'='*80}")
        print("统计显著性检验")
        print(f"{'='*80}")

        phmeg_hits = [1.0 if any(r in res.relevant_ids for r in res.retrieved_ids[:5]) else 0.0
                      for res in all_results["PHMEG"]]
        rag_hits = [1.0 if any(r in res.relevant_ids for r in res.retrieved_ids[:5]) else 0.0
                    for res in all_results["RAG"]]

        comparison = sig_tester.full_comparison(phmeg_hits, rag_hits, "PHMEG", "RAG")
        print(f"\nPHMEG vs RAG:")
        print(f"  p-value: {comparison['paired_t_test']['p_value']:.4f}")
        print(f"  Cohen's d: {comparison['effect_size']['cohens_d']:.3f}")
        print(f"  改进: {comparison['effect_size']['improvement']:.1f}%")

    # 论文表格
    print(f"\n{'='*80}")
    print("Table: LoCoMo Benchmark Results")
    print(f"{'='*80}")
    print(f"{'System':<20} {'P@5':>8} {'R@5':>8} {'F1@5':>8} {'MRR':>8} {'HitRate':>8} {'NDCG@5':>8}")
    print("-" * 80)
    for name, m in all_metrics.items():
        print(f"{name:<20} "
              f"{m.get('Precision@5', 0):>8.4f} "
              f"{m.get('Recall@5', 0):>8.4f} "
              f"{m.get('F1@5', 0):>8.4f} "
              f"{m.get('MRR', 0):>8.4f} "
              f"{m.get('HitRate', 0):>8.4f} "
              f"{m.get('NDCG@5', 0):>8.4f}")
    print("-" * 80)

    return all_metrics


def main():
    import sys
    data_path = "/workspace/locomo10.json"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)

    print("初始化嵌入模型...")
    embedder = ResearchEmbedder()

    results = run_locomo_experiment(data_path, embedder)

    print("\n" + "=" * 80)
    print("LoCoMo 实验完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
