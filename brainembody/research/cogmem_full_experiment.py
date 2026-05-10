"""
CogMem Full Experiment Suite
Complete experiments for paper submission:
1. LoCoMo Benchmark (full 10 samples) - Table 1
2. Ablation Study - Table 2
3. Embodied Synthetic Benchmark for SSDR - Table 3
4. Multi-K evaluation - Table 4
5. Statistical Significance Tests - Table 5
6. Stability across random seeds
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

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
            sim = float(np.dot(q_emb, m["embedding"]) /
                       (np.linalg.norm(q_emb) * np.linalg.norm(m["embedding"]) + 1e-8))
            scores.append((sim, m))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s} for s, m in scores[:top_k]]

    def consolidate(self):
        pass


class TimeDecayRAG:
    def __init__(self, embedder, decay_rate=0.01):
        self.embedder = embedder
        self.memories = {}
        self.time_counter = 0
        self.decay_rate = decay_rate

    def encode_with_id(self, mem_id, content, **kwargs):
        self.time_counter += 1
        self.memories[mem_id] = {
            "id": mem_id, "content": content,
            "embedding": self.embedder.embed(content),
            "timestamp": self.time_counter
        }
        return mem_id

    def retrieve(self, query, top_k=5, **kwargs):
        q_emb = self.embedder.embed(query)
        scores = []
        for mid, m in self.memories.items():
            sem_sim = float(np.dot(q_emb, m["embedding"]) /
                           (np.linalg.norm(q_emb) * np.linalg.norm(m["embedding"]) + 1e-8))
            recency = np.exp(-self.decay_rate * (self.time_counter - m["timestamp"]))
            score = 0.7 * sem_sim + 0.3 * recency
            scores.append((score, m))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s} for s, m in scores[:top_k]]

    def consolidate(self):
        pass


class EmotionalRAG:
    def __init__(self, embedder):
        self.embedder = embedder
        self.memories = {}

    def _detect_emotion(self, text):
        positive_words = ["happy", "great", "love", "wonderful", "excellent", "amazing",
                         "good", "best", "fantastic", "beautiful", "enjoy", "glad"]
        negative_words = ["sad", "bad", "hate", "terrible", "awful", "horrible",
                         "worst", "angry", "disappointed", "frustrated", "annoyed", "upset"]
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        if pos_count > neg_count:
            return 0.5 + min(0.5, pos_count * 0.1)
        elif neg_count > pos_count:
            return 0.5 - min(0.3, neg_count * 0.1)
        return 0.5

    def encode_with_id(self, mem_id, content, **kwargs):
        emotion_score = self._detect_emotion(content)
        self.memories[mem_id] = {
            "id": mem_id, "content": content,
            "embedding": self.embedder.embed(content),
            "emotion_score": emotion_score
        }
        return mem_id

    def retrieve(self, query, top_k=5, **kwargs):
        q_emb = self.embedder.embed(query)
        q_emotion = self._detect_emotion(query)
        scores = []
        for mid, m in self.memories.items():
            sem_sim = float(np.dot(q_emb, m["embedding"]) /
                           (np.linalg.norm(q_emb) * np.linalg.norm(m["embedding"]) + 1e-8))
            emotion_match = 1.0 - abs(q_emotion - m["emotion_score"])
            score = 0.8 * sem_sim + 0.2 * emotion_match
            scores.append((score, m))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m["id"], "content": m["content"], "score": s} for s, m in scores[:top_k]]

    def consolidate(self):
        pass


class RandomRetrieval:
    def __init__(self, embedder, seed=42):
        self.embedder = embedder
        self.memories = {}
        self.rng = np.random.RandomState(seed)

    def encode_with_id(self, mem_id, content, **kwargs):
        self.memories[mem_id] = {
            "id": mem_id, "content": content,
            "embedding": self.embedder.embed(content)
        }
        return mem_id

    def retrieve(self, query, top_k=5, **kwargs):
        all_ids = list(self.memories.keys())
        if len(all_ids) <= top_k:
            selected = all_ids
        else:
            selected_indices = self.rng.choice(len(all_ids), top_k, replace=False)
            selected = [all_ids[i] for i in selected_indices]
        return [{"id": mid, "content": self.memories[mid]["content"], "score": 1.0} for mid in selected]

    def consolidate(self):
        pass


def load_locomo_data(data_path: str, max_samples: int = 10, max_qa: int = 100):
    print("Loading LoCoMo dataset...", flush=True)
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    print(f"  Loaded {len(raw_data)} raw samples", flush=True)

    memories = []
    queries = []

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
                answer_text = qa.get('answer', '').lower()
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

    print(f"  Memories: {len(memories)}, Queries: {len(queries)}", flush=True)
    return {'memories': memories, 'queries': queries,
            'metadata': {'n_memories': len(memories), 'n_queries': len(queries)}}


def create_embodied_benchmark(embedder, n_scenes=5, n_objects_per_scene=8, n_queries_per_scene=10):
    """
    Create synthetic embodied benchmark for SSDR validation.
    Each scene has a location, objects, and actions.
    Queries test whether SSDR can leverage sensorimotor state.
    """
    scenes = [
        {"name": "kitchen", "position": np.array([0.0, 0.0, 0.0]),
         "objects": ["refrigerator", "stove", "sink", "knife", "plate", "cup", "spoon", "pan"],
         "actions": ["cooking", "washing", "cutting", "pouring"]},
        {"name": "bedroom", "position": np.array([5.0, 0.0, 0.0]),
         "objects": ["bed", "pillow", "blanket", "wardrobe", "lamp", "alarm", "book", "phone"],
         "actions": ["sleeping", "reading", "dressing", "resting"]},
        {"name": "garage", "position": np.array([10.0, 0.0, 0.0]),
         "objects": ["car", "toolbox", "wrench", "tire", "jack", "oil", "battery", "helmet"],
         "actions": ["repairing", "driving", "parking", "lifting"]},
        {"name": "garden", "position": np.array([0.0, 5.0, 0.0]),
         "objects": ["flower", "shovel", "watering_can", "seed", "soil", "fence", "hose", "pruner"],
         "actions": ["planting", "watering", "pruning", "digging"]},
        {"name": "office", "position": np.array([5.0, 5.0, 0.0]),
         "objects": ["computer", "keyboard", "monitor", "printer", "desk", "chair", "pen", "paper"],
         "actions": ["typing", "printing", "writing", "meeting"]},
    ][:n_scenes]

    memories = []
    queries = []

    for scene in scenes:
        for obj_idx, obj in enumerate(scene["objects"]):
            action = scene["actions"][obj_idx % len(scene["actions"])]
            content = f"I used the {obj} in the {scene['name']} while {action}"

            sm_state = SensorimotorState(
                position=scene["position"] + np.array([obj_idx * 0.3, 0.1, 0.0]),
                orientation=np.array([0.0, 0.0, 1.0]),
                current_action=action,
                held_object=obj,
                nearby_objects=scene["objects"][:3],
                motor_state="active",
                environmental_features={"scene": scene["name"], "object_idx": obj_idx}
            )

            mem_id = f"emb_{scene['name']}_{obj}"
            memories.append({
                'id': mem_id,
                'content': content,
                'sensorimotor_state': sm_state,
                'scene': scene['name'],
                'object': obj,
                'action': action,
            })

    for scene in scenes:
        for q_idx in range(n_queries_per_scene):
            obj = scene["objects"][q_idx % len(scene["objects"])]
            action = scene["actions"][q_idx % len(scene["actions"])]

            query_text = f"Where is the {obj}?"

            query_state = SensorimotorState(
                position=scene["position"] + np.array([0.1, 0.1, 0.0]),
                orientation=np.array([0.0, 0.0, 1.0]),
                current_action=action,
                motor_state="active",
                environmental_features={"scene": scene["name"]}
            )

            relevant_id = f"emb_{scene['name']}_{obj}"

            queries.append({
                'id': f"eq_{scene['name']}_{q_idx}",
                'query': query_text,
                'relevant_ids': [relevant_id],
                'query_state': query_state,
                'scene': scene['name'],
            })

    print(f"  Embodied benchmark: {len(memories)} memories, {len(queries)} queries", flush=True)
    return {'memories': memories, 'queries': queries,
            'metadata': {'n_memories': len(memories), 'n_queries': len(queries)}}


def evaluate_system(system, dataset, system_name: str, is_cogmem: bool = False,
                    use_query_states: bool = False):
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

    print(f"  Retrieving for {len(dataset['queries'])} queries...", flush=True)
    results = []
    for qi, query in enumerate(dataset["queries"]):
        if qi % 50 == 0 and qi > 0:
            print(f"    Query {qi}/{len(dataset['queries'])}...", flush=True)

        if is_cogmem:
            query_state = query.get("query_state") if use_query_states else None
            if query_state is None:
                query_state = SensorimotorState(
                    position=np.array([0.0, 0.0, 0.0]),
                    current_action="questioning",
                    motor_state="stationary"
                )
            retrieved = system.retrieve(
                query=query["query"],
                top_k=10,
                sensorimotor_state=query_state
            )
        else:
            retrieved = system.retrieve(query["query"], top_k=10)

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


def print_table(title, headers, rows, col_widths=None):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    if col_widths is None:
        col_widths = [max(len(h), 12) for h in headers]
    header_line = "  ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        line = "  ".join(f"{str(v):>{w}}" for v, w in zip(row, col_widths))
        print(line)
    print("-" * len(header_line))


def main():
    data_path = "/workspace/locomo10.json"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print("=" * 80)
    print("CogMem Full Experiment Suite")
    print("=" * 80)

    print("\n[1/8] Initializing embedder...", flush=True)
    embedder = ResearchEmbedder()

    print("\n[2/8] Loading LoCoMo dataset (full 10 samples)...", flush=True)
    locomo_data = load_locomo_data(data_path, max_samples=10, max_qa=100)

    print("\n[3/8] Creating embodied benchmark...", flush=True)
    embodied_data = create_embodied_benchmark(embedder)

    metrics_calc = MetricsCalculator()
    sig_tester = SignificanceTester()

    all_locomo_results = {}
    all_locomo_metrics = {}
    all_embodied_results = {}
    all_embodied_metrics = {}

    systems_config = {
        "Random": ("baseline", None),
        "RAG": ("baseline", None),
        "TimeDecay": ("baseline", None),
        "EmotionalRAG": ("baseline", None),
        "CogMem-ECA": ("cogmem", CogMemConfig(
            enable_eca=True, enable_crc=False, enable_ssdr=False,
            eca_config=ECAConfig(capacity=2000)
        )),
        "CogMem-CRC": ("cogmem", CogMemConfig(
            enable_eca=False, enable_crc=True, enable_ssdr=False,
            crc_config=CRCConfig()
        )),
        "CogMem-SSDR": ("cogmem", CogMemConfig(
            enable_eca=False, enable_crc=False, enable_ssdr=True,
            ssdr_config=SSDRConfig()
        )),
        "CogMem-ECA+CRC": ("cogmem", CogMemConfig(
            enable_eca=True, enable_crc=True, enable_ssdr=False,
            eca_config=ECAConfig(capacity=2000),
            crc_config=CRCConfig()
        )),
        "CogMem-ECA+SSDR": ("cogmem", CogMemConfig(
            enable_eca=True, enable_crc=False, enable_ssdr=True,
            eca_config=ECAConfig(capacity=2000),
            ssdr_config=SSDRConfig()
        )),
        "CogMem-Full": ("cogmem", CogMemConfig(
            enable_eca=True, enable_crc=True, enable_ssdr=True,
            eca_config=ECAConfig(capacity=2000),
            crc_config=CRCConfig(),
            ssdr_config=SSDRConfig()
        )),
    }

    system_order = list(systems_config.keys())

    print(f"\n[4/8] Running LoCoMo experiments ({len(systems_config)} systems)...", flush=True)
    for idx, name in enumerate(system_order):
        stype, config = systems_config[name]
        print(f"\n  [{idx+1}/{len(system_order)}] {name}...", flush=True)

        if stype == "baseline":
            if name == "Random":
                system = RandomRetrieval(embedder)
            elif name == "RAG":
                system = BaselineRAG(embedder)
            elif name == "TimeDecay":
                system = TimeDecayRAG(embedder)
            elif name == "EmotionalRAG":
                system = EmotionalRAG(embedder)
            is_cogmem = False
        else:
            system = CogMemMemory(
                embedding_dim=embedder.embedding_dim,
                embedder=embedder,
                config=config
            )
            is_cogmem = True

        try:
            results, id_mapping = evaluate_system(system, locomo_data, name, is_cogmem)
            if not results:
                print(f"    No valid results")
                continue

            metrics = metrics_calc.compute_all_metrics(results)
            all_locomo_results[name] = results
            all_locomo_metrics[name] = metrics

            print(f"    F1@5={metrics.get('F1@5', 0):.4f}  R@5={metrics.get('Recall@5', 0):.4f}  "
                  f"MRR={metrics.get('MRR', 0):.4f}  HitRate={metrics.get('HitRate', 0):.4f}  "
                  f"NDCG@5={metrics.get('NDCG@5', 0):.4f}", flush=True)
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[5/8] Running Embodied benchmark (SSDR validation)...", flush=True)
    embodied_systems = {
        "RAG": ("baseline", None),
        "CogMem-SSDR": ("cogmem", CogMemConfig(
            enable_eca=False, enable_crc=False, enable_ssdr=True,
            ssdr_config=SSDRConfig()
        )),
        "CogMem-ECA+SSDR": ("cogmem", CogMemConfig(
            enable_eca=True, enable_crc=False, enable_ssdr=True,
            eca_config=ECAConfig(capacity=200),
            ssdr_config=SSDRConfig()
        )),
        "CogMem-Full": ("cogmem", CogMemConfig(
            enable_eca=True, enable_crc=True, enable_ssdr=True,
            eca_config=ECAConfig(capacity=200),
            crc_config=CRCConfig(),
            ssdr_config=SSDRConfig()
        )),
    }

    for name, (stype, config) in embodied_systems.items():
        print(f"\n  {name} (embodied)...", flush=True)
        if stype == "baseline":
            system = BaselineRAG(embedder)
            is_cogmem = False
        else:
            system = CogMemMemory(
                embedding_dim=embedder.embedding_dim,
                embedder=embedder,
                config=config
            )
            is_cogmem = True

        try:
            results, id_mapping = evaluate_system(
                system, embodied_data, name, is_cogmem,
                use_query_states=True
            )
            if not results:
                print(f"    No valid results")
                continue

            metrics = metrics_calc.compute_all_metrics(results)
            all_embodied_results[name] = results
            all_embodied_metrics[name] = metrics

            print(f"    F1@5={metrics.get('F1@5', 0):.4f}  R@5={metrics.get('Recall@5', 0):.4f}  "
                  f"MRR={metrics.get('MRR', 0):.4f}  HitRate={metrics.get('HitRate', 0):.4f}", flush=True)
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    # ===== TABLE 1: Main Results on LoCoMo =====
    print("\n[6/8] Generating tables...", flush=True)
    main_systems = ["Random", "RAG", "TimeDecay", "EmotionalRAG",
                    "CogMem-ECA", "CogMem-CRC", "CogMem-SSDR",
                    "CogMem-ECA+CRC", "CogMem-Full"]
    rows = []
    for name in main_systems:
        if name in all_locomo_metrics:
            m = all_locomo_metrics[name]
            rows.append([
                name,
                f"{m.get('Precision@5', 0):.4f}",
                f"{m.get('Recall@5', 0):.4f}",
                f"{m.get('F1@5', 0):.4f}",
                f"{m.get('MRR', 0):.4f}",
                f"{m.get('HitRate', 0):.4f}",
                f"{m.get('NDCG@5', 0):.4f}",
            ])
    print_table("Table 1: Main Results on LoCoMo Benchmark",
                ["System", "P@5", "R@5", "F1@5", "MRR", "HitRate", "NDCG@5"],
                rows, [18, 8, 8, 8, 8, 8, 8])

    # ===== TABLE 2: Ablation Study =====
    ablation_systems = ["RAG", "CogMem-ECA", "CogMem-CRC", "CogMem-SSDR",
                        "CogMem-ECA+CRC", "CogMem-ECA+SSDR", "CogMem-Full"]
    rows = []
    for name in ablation_systems:
        if name in all_locomo_metrics:
            m = all_locomo_metrics[name]
            rag_m = all_locomo_metrics.get("RAG", {})
            delta_f1 = m.get('F1@5', 0) - rag_m.get('F1@5', 0)
            delta_mrr = m.get('MRR', 0) - rag_m.get('MRR', 0)
            rows.append([
                name,
                f"{m.get('F1@5', 0):.4f}",
                f"{delta_f1:+.4f}",
                f"{m.get('MRR', 0):.4f}",
                f"{delta_mrr:+.4f}",
                f"{m.get('NDCG@5', 0):.4f}",
            ])
    print_table("Table 2: Ablation Study (vs RAG baseline)",
                ["System", "F1@5", "ΔF1", "MRR", "ΔMRR", "NDCG@5"],
                rows, [18, 8, 8, 8, 8, 8])

    # ===== TABLE 3: Embodied Benchmark (SSDR Validation) =====
    rows = []
    for name in ["RAG", "CogMem-SSDR", "CogMem-ECA+SSDR", "CogMem-Full"]:
        if name in all_embodied_metrics:
            m = all_embodied_metrics[name]
            rows.append([
                name,
                f"{m.get('F1@5', 0):.4f}",
                f"{m.get('Recall@5', 0):.4f}",
                f"{m.get('MRR', 0):.4f}",
                f"{m.get('HitRate', 0):.4f}",
                f"{m.get('NDCG@5', 0):.4f}",
            ])
    print_table("Table 3: Embodied Benchmark (SSDR Validation)",
                ["System", "F1@5", "R@5", "MRR", "HitRate", "NDCG@5"],
                rows, [18, 8, 8, 8, 8, 8])

    # ===== TABLE 4: Multi-K Evaluation =====
    if "CogMem-ECA+CRC" in all_locomo_results and "RAG" in all_locomo_results:
        rows = []
        for k in [1, 3, 5, 10]:
            eca_crc_m = all_locomo_metrics.get("CogMem-ECA+CRC", {})
            rag_m = all_locomo_metrics.get("RAG", {})
            full_m = all_locomo_metrics.get("CogMem-Full", {})
            rows.append([
                f"K={k}",
                f"{rag_m.get(f'F1@{k}', 0):.4f}",
                f"{eca_crc_m.get(f'F1@{k}', 0):.4f}",
                f"{full_m.get(f'F1@{k}', 0):.4f}",
                f"{rag_m.get(f'NDCG@{k}', 0):.4f}",
                f"{eca_crc_m.get(f'NDCG@{k}', 0):.4f}",
            ])
        print_table("Table 4: Multi-K Evaluation (F1@K and NDCG@K)",
                    ["K", "RAG F1", "ECA+CRC F1", "Full F1", "RAG NDCG", "ECA+CRC NDCG"],
                    rows, [6, 10, 10, 10, 10, 10])

    # ===== TABLE 5: Statistical Significance =====
    print(f"\n{'='*80}")
    print("Table 5: Statistical Significance Tests")
    print(f"{'='*80}")

    comparisons = [
        ("CogMem-ECA+CRC", "RAG"),
        ("CogMem-ECA", "RAG"),
        ("CogMem-Full", "RAG"),
        ("CogMem-ECA+CRC", "TimeDecay"),
        ("CogMem-ECA+CRC", "EmotionalRAG"),
    ]

    for name_a, name_b in comparisons:
        if name_a not in all_locomo_results or name_b not in all_locomo_results:
            continue

        results_a = all_locomo_results[name_a]
        results_b = all_locomo_results[name_b]

        common_qids = set(r.query_id for r in results_a) & set(r.query_id for r in results_b)
        if not common_qids:
            continue

        a_by_qid = {r.query_id: r for r in results_a}
        b_by_qid = {r.query_id: r for r in results_b}

        for metric_name, metric_func in [
            ("HitRate", lambda r: 1.0 if any(rid in r.relevant_ids for rid in r.retrieved_ids[:5]) else 0.0),
            ("MRR", lambda r: next((1.0/(i+1) for i, rid in enumerate(r.retrieved_ids) if rid in r.relevant_ids), 0.0)),
            ("F1@5", lambda r: MetricsCalculator.f1_at_k(r.retrieved_ids, r.relevant_ids, 5)),
        ]:
            scores_a = [metric_func(a_by_qid[qid]) for qid in common_qids]
            scores_b = [metric_func(b_by_qid[qid]) for qid in common_qids]

            comparison = sig_tester.full_comparison(scores_a, scores_b, name_a, name_b)
            d = comparison['effect_size']['cohens_d']
            p = comparison['paired_t_test']['p_value']
            imp = comparison['effect_size']['improvement']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

            print(f"  {name_a} vs {name_b} ({metric_name}): p={p:.4f} {sig}, d={d:.3f}, Δ={imp:+.1f}%")

    # ===== Embodied benchmark significance =====
    if "CogMem-Full" in all_embodied_results and "RAG" in all_embodied_results:
        print(f"\n  --- Embodied Benchmark Significance ---")
        results_a = all_embodied_results["CogMem-Full"]
        results_b = all_embodied_results["RAG"]
        common_qids = set(r.query_id for r in results_a) & set(r.query_id for r in results_b)
        if common_qids:
            a_by_qid = {r.query_id: r for r in results_a}
            b_by_qid = {r.query_id: r for r in results_b}
            scores_a = [1.0 if any(rid in a_by_qid[qid].relevant_ids for rid in a_by_qid[qid].retrieved_ids[:5]) else 0.0
                       for qid in common_qids]
            scores_b = [1.0 if any(rid in b_by_qid[qid].relevant_ids for rid in b_by_qid[qid].retrieved_ids[:5]) else 0.0
                       for qid in common_qids]
            comparison = sig_tester.full_comparison(scores_a, scores_b, "CogMem-Full", "RAG")
            d = comparison['effect_size']['cohens_d']
            p = comparison['paired_t_test']['p_value']
            imp = comparison['effect_size']['improvement']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  CogMem-Full vs RAG (HitRate): p={p:.4f} {sig}, d={d:.3f}, Δ={imp:+.1f}%")

    # ===== Save all results =====
    print(f"\n[7/8] Saving results...", flush=True)
    output = {
        "locomo_metrics": {name: {k: float(v) for k, v in m.items()}
                          for name, m in all_locomo_metrics.items()},
        "embodied_metrics": {name: {k: float(v) for k, v in m.items()}
                            for name, m in all_embodied_metrics.items()},
        "metadata": {
            "locomo_memories": locomo_data['metadata']['n_memories'],
            "locomo_queries": locomo_data['metadata']['n_queries'],
            "embodied_memories": embodied_data['metadata']['n_memories'],
            "embodied_queries": embodied_data['metadata']['n_queries'],
        }
    }

    with open("/workspace/cogmem_full_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[8/8] Experiment suite complete!", flush=True)
    print(f"Results saved to /workspace/cogmem_full_results.json")


if __name__ == "__main__":
    main()
