"""
CogMem Experiment: LoCoMo Benchmark + Ablation Study
用真实数据验证ECA/CRC/SSDR三个创新点
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List

from brainembody.memory.cogmem import CogMemMemory, CogMemConfig
from brainembody.memory.eca import ECAConfig
from brainembody.memory.crc import CRCConfig
from brainembody.memory.ssdr import SSDRConfig, SensorimotorState
from brainembody.research.embedder import ResearchEmbedder
from brainembody.research.metrics import MetricsCalculator, SignificanceTester, RetrievalResult


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


def load_locomo_data(data_path: str, max_samples: int = 3, max_qa: int = 50):
    print("Loading LoCoMo dataset...", flush=True)
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    print(f"  Loaded {len(raw_data)} raw samples", flush=True)

    memories = []
    queries = []
    mem_id_set = set()

    for item in raw_data[:max_samples]:
        sample_id = str(item.get('sample_id', 'unknown'))
        conversation = item.get('conversation', {})
        qa_pairs = item.get('qa', [])

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
                sessions.append({'session_id': session_key, 'turns': turns})

        for session in sessions:
            for turn_idx, turn in enumerate(session['turns'][:30]):
                mem_id = f"{sample_id}_{session['session_id']}_t{turn_idx}"
                session_num = 0
                for ch in session['session_id'].split('_'):
                    if ch.isdigit():
                        session_num = int(ch)
                        break

                sm_state = SensorimotorState(
                    position=np.array([session_num * 2.0, turn_idx * 0.5, 0.0]),
                    orientation=np.array([0.0, 0.0, 1.0]),
                    current_action="conversing",
                    motor_state="stationary",
                    environmental_features={"session": session_num, "turn": turn_idx}
                )

                memories.append({
                    'id': mem_id,
                    'content': f"{turn['speaker']}: {turn['text'][:200]}",
                    'sensorimotor_state': sm_state,
                    'sample_id': sample_id,
                    'session_id': session['session_id'],
                    'dia_id': turn.get('dia_id', ''),
                    'session_num': session_num,
                    'turn_idx': turn_idx,
                })
                mem_id_set.add(mem_id)

        for qa_idx, qa in enumerate(qa_pairs[:max_qa]):
            evidence_ids = []
            if 'evidence' in qa and qa['evidence']:
                for ev in qa['evidence']:
                    for mem in memories:
                        if mem['sample_id'] == sample_id:
                            if mem.get('dia_id', '') == ev:
                                evidence_ids.append(mem['id'])
                                break

            if not evidence_ids and qa.get('answer', ''):
                answer_text = qa['answer'].lower()
                for mem in memories:
                    if mem['sample_id'] == sample_id:
                        if any(word in mem['content'].lower()
                               for word in answer_text.split()[:3] if len(word) > 1):
                            evidence_ids.append(mem['id'])
                            if len(evidence_ids) >= 3:
                                break

            if evidence_ids:
                queries.append({
                    'id': f"q_{sample_id}_{qa_idx}",
                    'query': qa['question'],
                    'answer': qa.get('answer', ''),
                    'relevant_ids': evidence_ids,
                    'category': qa.get('category', 0),
                    'sample_id': sample_id
                })

    return {'memories': memories, 'queries': queries,
            'metadata': {'n_memories': len(memories), 'n_queries': len(queries)}}


def evaluate_system(system, dataset, system_name: str, is_cogmem: bool = False):
    print(f"  Encoding {len(dataset['memories'])} memories...", flush=True)
    id_mapping = {}

    for memory in dataset["memories"]:
        original_id = memory["id"]
        if is_cogmem:
            new_id = system.encode(
                content=memory["content"],
                sensorimotor_state=memory.get("sensorimotor_state"),
                importance=0.5,
                memory_id=original_id
            )
            if new_id is not None:
                id_mapping[original_id] = new_id
        else:
            new_id = system.encode_with_id(original_id, memory["content"])
            id_mapping[original_id] = new_id

    if is_cogmem:
        system.sleep_consolidate(n_rounds=2)
        surviving_ids = set(system.memories.keys())
        id_mapping = {k: v for k, v in id_mapping.items() if v in surviving_ids}

    results = []
    for query in dataset["queries"]:
        if is_cogmem:
            query_state = SensorimotorState(
                position=np.array([0.0, 0.0, 0.0]),
                current_action="questioning",
                motor_state="stationary"
            )
            retrieved = system.retrieve(
                query=query["query"],
                top_k=5,
                sensorimotor_state=query_state
            )
        else:
            retrieved = system.retrieve(query["query"], top_k=5)

        retrieved_ids = [r["id"] for r in retrieved]
        relevant_mapped = [id_mapping[rid] for rid in query["relevant_ids"] if rid in id_mapping]

        if relevant_mapped:
            results.append(RetrievalResult(
                query_id=query["id"],
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_mapped,
                scores=[r.get("score", 0) for r in retrieved]
            ))

    return results, id_mapping


def main():
    data_path = "/workspace/locomo10.json"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print("=" * 80)
    print("CogMem Experiment: LoCoMo Benchmark + Ablation Study")
    print("=" * 80)

    print("\nInitializing embedder...")
    embedder = ResearchEmbedder()

    print("\nLoading LoCoMo dataset...")
    dataset = load_locomo_data(data_path, max_samples=3, max_qa=50)
    print(f"  Memories: {dataset['metadata']['n_memories']}")
    print(f"  Queries: {dataset['metadata']['n_queries']}")

    metrics_calc = MetricsCalculator()
    sig_tester = SignificanceTester()

    systems = {}

    print("\n[1/6] Baseline RAG...")
    systems["RAG"] = BaselineRAG(embedder)

    print("[2/6] CogMem (ECA only)...")
    systems["CogMem-ECA"] = CogMemMemory(
        embedding_dim=embedder.embedding_dim, embedder=embedder,
        config=CogMemConfig(enable_eca=True, enable_crc=False, enable_ssdr=False,
                           eca_config=ECAConfig(capacity=500))
    )

    print("[3/6] CogMem (CRC only)...")
    systems["CogMem-CRC"] = CogMemMemory(
        embedding_dim=embedder.embedding_dim, embedder=embedder,
        config=CogMemConfig(enable_eca=False, enable_crc=True, enable_ssdr=False,
                           crc_config=CRCConfig())
    )

    print("[4/6] CogMem (SSDR only)...")
    systems["CogMem-SSDR"] = CogMemMemory(
        embedding_dim=embedder.embedding_dim, embedder=embedder,
        config=CogMemConfig(enable_eca=False, enable_crc=False, enable_ssdr=True,
                           ssdr_config=SSDRConfig())
    )

    print("[5/6] CogMem (ECA+CRC)...")
    systems["CogMem-ECA+CRC"] = CogMemMemory(
        embedding_dim=embedder.embedding_dim, embedder=embedder,
        config=CogMemConfig(enable_eca=True, enable_crc=True, enable_ssdr=False,
                           eca_config=ECAConfig(capacity=500),
                           crc_config=CRCConfig())
    )

    print("[6/6] CogMem (Full: ECA+CRC+SSDR)...")
    systems["CogMem-Full"] = CogMemMemory(
        embedding_dim=embedder.embedding_dim, embedder=embedder,
        config=CogMemConfig(enable_eca=True, enable_crc=True, enable_ssdr=True,
                           eca_config=ECAConfig(capacity=500),
                           crc_config=CRCConfig(),
                           ssdr_config=SSDRConfig())
    )

    all_results = {}
    all_metrics = {}

    for name, system in systems.items():
        is_cogmem = isinstance(system, CogMemMemory)
        print(f"\nEvaluating {name}...")

        try:
            results, id_mapping = evaluate_system(system, dataset, name, is_cogmem)
            if not results:
                print(f"  No valid results")
                continue

            metrics = metrics_calc.compute_all_metrics(results)
            all_results[name] = results
            all_metrics[name] = metrics

            print(f"  F1@5: {metrics.get('F1@5', 0):.4f}")
            print(f"  Recall@5: {metrics.get('Recall@5', 0):.4f}")
            print(f"  MRR: {metrics.get('MRR', 0):.4f}")
            print(f"  HitRate: {metrics.get('HitRate', 0):.4f}")
            print(f"  NDCG@5: {metrics.get('NDCG@5', 0):.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Table 1: Main Results on LoCoMo Benchmark")
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

    comparisons_to_run = []
    if "CogMem-ECA+CRC" in all_results and "RAG" in all_results:
        comparisons_to_run.append(("CogMem-ECA+CRC", "RAG"))
    if "CogMem-ECA" in all_results and "RAG" in all_results:
        comparisons_to_run.append(("CogMem-ECA", "RAG"))
    if "CogMem-Full" in all_results and "RAG" in all_results:
        comparisons_to_run.append(("CogMem-Full", "RAG"))

    for name_a, name_b in comparisons_to_run:
        results_a = all_results[name_a]
        results_b = all_results[name_b]

        common_qids = set(r.query_id for r in results_a) & set(r.query_id for r in results_b)
        if not common_qids:
            continue

        a_by_qid = {r.query_id: r for r in results_a}
        b_by_qid = {r.query_id: r for r in results_b}

        hits_a = [1.0 if any(r in a_by_qid[qid].relevant_ids for r in a_by_qid[qid].retrieved_ids[:5]) else 0.0
                  for qid in common_qids]
        hits_b = [1.0 if any(r in b_by_qid[qid].relevant_ids for r in b_by_qid[qid].retrieved_ids[:5]) else 0.0
                  for qid in common_qids]

        print(f"\n{'='*80}")
        print(f"Statistical Significance: {name_a} vs {name_b}")
        print(f"{'='*80}")

        comparison = sig_tester.full_comparison(hits_a, hits_b, name_a, name_b)
        print(f"  Paired t-test p-value: {comparison['paired_t_test']['p_value']:.4f}")
        print(f"  Wilcoxon p-value: {comparison['wilcoxon']['p_value']:.4f}")
        print(f"  Cohen's d: {comparison['effect_size']['cohens_d']:.3f} ({comparison['effect_size']['magnitude']})")
        print(f"  Improvement: {comparison['effect_size']['improvement']:.1f}%")

    if len(all_results) > 2:
        print(f"\n{'='*80}")
        print("Table 2: Ablation Study")
        print(f"{'='*80}")
        print(f"{'System':<20} {'F1@5':>8} {'R@5':>8} {'MRR':>8} {'NDCG@5':>8}")
        print("-" * 60)
        ablation_order = ["RAG", "CogMem-ECA", "CogMem-CRC", "CogMem-SSDR",
                         "CogMem-ECA+CRC", "CogMem-Full"]
        for name in ablation_order:
            if name in all_metrics:
                m = all_metrics[name]
                print(f"{name:<20} "
                      f"{m.get('F1@5', 0):>8.4f} "
                      f"{m.get('Recall@5', 0):>8.4f} "
                      f"{m.get('MRR', 0):>8.4f} "
                      f"{m.get('NDCG@5', 0):>8.4f}")
        print("-" * 60)

    output_path = "/workspace/cogmem_results.json"
    serializable_metrics = {}
    for name, m in all_metrics.items():
        serializable_metrics[name] = {k: float(v) for k, v in m.items()}

    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 80)
    print("CogMem Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
