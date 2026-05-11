from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class DGModule:
    def __init__(self, input_dim=128, output_dim=256, sparsity=32, seed=0):
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(output_dim, input_dim).astype(np.float32)
        row_norms = np.linalg.norm(self.projection, axis=1, keepdims=True)
        self.projection = self.projection / np.maximum(row_norms, 1e-8)
        self.sparsity = sparsity

    def separate(self, x):
        projected = self.projection @ x.astype(np.float32)
        barcode = np.zeros_like(projected)
        if self.sparsity >= len(projected):
            return np.maximum(projected, 0.0)
        top_idx = np.argpartition(projected, -self.sparsity)[-self.sparsity:]
        barcode[top_idx] = np.maximum(projected[top_idx], 0.0)
        return barcode


class AdaptiveModularDG:
    def __init__(self, input_dim=128, output_dim=256, sparsity=32, base_seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.base_seed = base_seed
        self.modules: Dict[int, DGModule] = {}
        self._counter = 0

    def get_or_create_module(self, task_id):
        if task_id not in self.modules:
            self.modules[task_id] = DGModule(
                self.input_dim, self.output_dim, self.sparsity,
                self.base_seed + self._counter * 1000)
            self._counter += 1
        return self.modules[task_id]

    def encode(self, x, task_id):
        return self.get_or_create_module(task_id).separate(x)

    def infer_module(self, query, stored_emb, stored_module_ids, top_k=5):
        if not stored_emb:
            return 0
        q = query.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return 0
        q = q / q_norm
        emb = np.stack(stored_emb).astype(np.float32)
        norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        sims = (emb / norms) @ q
        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        votes = {}
        for idx in top_idx:
            mid = stored_module_ids[idx]
            votes[mid] = votes.get(mid, 0) + 1
        return max(votes, key=votes.get)


class BrainMemoryNetwork:
    def __init__(self, embedding_dim=128, barcode_dim=256, barcode_sparsity=32,
                 lambda_param=0.8, n_replay=5, use_barcode=True, use_modular=True,
                 routing_top_k=5, seed=42):
        self.embedding_dim = embedding_dim
        self.lambda_param = lambda_param
        self.n_replay = n_replay
        self.use_barcode = use_barcode
        self.use_modular = use_modular
        self.routing_top_k = routing_top_k
        self.seed = seed

        if use_barcode:
            self.dg = AdaptiveModularDG(embedding_dim, barcode_dim, barcode_sparsity, seed)
            if not use_modular:
                self.dg.get_or_create_module(0)

        self.stored_emb: List[np.ndarray] = []
        self.stored_bc: List[np.ndarray] = []
        self.stored_lbl: List[int] = []
        self.stored_task: List[int] = []
        self.stored_module: List[int] = []
        self._rng = np.random.RandomState(seed)

    def learn_task(self, embeddings, labels, task_id):
        for i in range(len(embeddings)):
            if self.use_barcode:
                tid = task_id if self.use_modular else 0
                bc = self.dg.encode(embeddings[i], tid)
                self.stored_bc.append(bc)
                self.stored_module.append(tid)
            self.stored_emb.append(embeddings[i].copy())
            self.stored_lbl.append(labels[i])
            self.stored_task.append(task_id)

        if self.n_replay > 0 and self.use_barcode and len(self.stored_emb) > len(embeddings):
            n_old = len(self.stored_emb) - len(embeddings)
            for idx in self._rng.choice(n_old, min(self.n_replay, n_old), replace=False):
                tid = self.stored_task[idx] if self.use_modular else 0
                self.stored_bc[idx] = self.dg.encode(self.stored_emb[idx], tid)

    def predict(self, query, task_id=None, lambda_param=None):
        if not self.stored_emb:
            return -1
        lam = lambda_param if lambda_param is not None else self.lambda_param
        c_scores = self._content_scores(query)
        if self.use_barcode:
            if task_id is not None:
                tid = task_id if self.use_modular else 0
                bc = self.dg.encode(query, tid)
            else:
                mid = self.dg.infer_module(query, self.stored_emb, self.stored_module, self.routing_top_k)
                bc = self.dg.encode(query, mid)
            b_scores = self._barcode_scores(bc)
            combined = self._combine(c_scores, b_scores, lam)
        else:
            combined = c_scores
        return self.stored_lbl[int(np.argmax(combined))]

    def evaluate(self, test_emb, test_lbl, test_task_ids=None, lambda_param=None):
        correct = 0
        for i in range(len(test_emb)):
            tid = int(test_task_ids[i]) if test_task_ids is not None else None
            if self.predict(test_emb[i], tid, lambda_param) == test_lbl[i]:
                correct += 1
        return {"accuracy": correct / len(test_emb)}

    def _content_scores(self, q):
        q = q.astype(np.float32)
        n = np.linalg.norm(q)
        if n < 1e-8:
            return np.zeros(len(self.stored_emb), dtype=np.float32)
        q = q / n
        emb = np.stack(self.stored_emb)
        norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        return (emb / norms @ q).astype(np.float32)

    def _barcode_scores(self, q_bc):
        n = np.linalg.norm(q_bc)
        if n < 1e-8:
            return np.zeros(len(self.stored_bc), dtype=np.float32)
        q = q_bc / n
        bc = np.stack(self.stored_bc)
        norms = np.maximum(np.linalg.norm(bc, axis=1, keepdims=True), 1e-8)
        return (bc / norms @ q).astype(np.float32)

    def _combine(self, c, b, lam):
        c_min, c_max = float(np.min(c)), float(np.max(c))
        cr = c_max - c_min
        nc = (c - c_min) / cr if cr > 1e-8 else np.ones_like(c) / len(c)
        b_min, b_max = float(np.min(b)), float(np.max(b))
        br = b_max - b_min
        nb = (b - b_min) / br if br > 1e-8 else np.ones_like(b) / len(b)
        return (lam * nc + (1.0 - lam) * nb).astype(np.float32)


CATEGORIES = {
    0: {"name": "Sports", "sentences": [
        "The team won the championship game last night with a spectacular goal",
        "The player scored a hat trick in the final match of the season",
        "The coach announced the starting lineup for the playoff game",
        "The stadium was packed with enthusiastic fans cheering loudly",
        "The athlete broke the world record in the hundred meter dash",
        "The referee made a controversial penalty call in overtime",
        "The basketball team trained hard for the upcoming tournament",
        "The soccer match ended in a dramatic penalty shootout draw",
        "The star striker missed the crucial penalty kick wide",
        "The defense held strong throughout the entire championship game",
        "The tennis player served an ace to win the match point",
        "The swimming champion set a new personal best time",
        "The football quarterback threw a touchdown pass in the final seconds",
        "The baseball pitcher threw a perfect game with no hits",
        "The hockey team scored a power play goal in the third period",
    ]},
    1: {"name": "Technology", "sentences": [
        "The company released a new smartphone model with advanced features",
        "The software update fixed several critical security bugs",
        "The AI model achieved state of the art results on the benchmark",
        "The startup raised significant funding for their cloud platform",
        "The cloud service experienced a major outage affecting users",
        "The new chip delivers significantly faster processing speeds",
        "The mobile app was downloaded millions of times in the first week",
        "The hackers exploited a serious security vulnerability in the system",
        "The research lab published a breakthrough paper on quantum computing",
        "The device features a high resolution OLED display panel",
        "The social media platform introduced new privacy controls",
        "The electric vehicle company announced a new battery technology",
        "The robotics team demonstrated their autonomous navigation system",
        "The semiconductor factory began producing next generation chips",
        "The virtual reality headset offers immersive gaming experiences",
    ]},
    2: {"name": "Science", "sentences": [
        "The researchers discovered a new species in the Amazon rainforest",
        "The experiment confirmed the theoretical hypothesis about dark matter",
        "The space telescope captured stunning images of a distant galaxy",
        "The clinical drug trial showed promising results for patients",
        "The laboratory developed a new superconducting material",
        "The study was published in a prestigious peer reviewed journal",
        "The physics theory was validated by new experimental observations",
        "The electron microscope revealed intricate cellular structures",
        "The chemical reaction produced unexpected but useful byproducts",
        "The statistical data analysis revealed a significant correlation trend",
        "The marine biologists documented a new deep sea ecosystem",
        "The climate model predicted rising temperatures over the next decade",
        "The genetic sequencing identified a rare mutation in the DNA",
        "The particle accelerator detected evidence of a new subatomic particle",
        "The neuroscience study mapped brain activity during learning tasks",
    ]},
    3: {"name": "Politics", "sentences": [
        "The president signed the new legislation into federal law today",
        "The election results were announced after a close vote count",
        "The senator proposed a comprehensive new healthcare policy",
        "The government increased the education budget significantly",
        "The diplomatic talks between nations reached a historic agreement",
        "The parliament debated the controversial reform bill for hours",
        "The presidential candidate campaigned extensively in swing states",
        "The international treaty was ratified by the senate committee",
        "The cabinet reshuffle surprised many political analysts today",
        "The new tax policy change affected millions of middle class citizens",
        "The supreme court issued a landmark ruling on civil rights",
        "The trade negotiations resulted in a new tariff agreement",
        "The opposition party challenged the government economic plan",
        "The governor declared a state of emergency after the disaster",
        "The congressional committee launched an investigation into spending",
    ]},
    4: {"name": "Health", "sentences": [
        "The patient made a full recovery after the surgical treatment",
        "The hospital implemented strict new infection control protocols",
        "The vaccine was approved for emergency distribution nationwide",
        "The patient symptoms improved significantly with new medication",
        "The doctor recommended important dietary and lifestyle changes",
        "The clinical trial enrolled five hundred new participants this month",
        "The health insurance plan covered the expensive medical procedure",
        "The disease outbreak was contained quickly through contact tracing",
        "The nutrition research study linked Mediterranean diet to longevity",
        "The mental health program expanded services to rural communities",
        "The pharmaceutical company developed a novel antibiotic treatment",
        "The rehabilitation center introduced innovative physical therapy",
        "The public health campaign raised awareness about heart disease",
        "The medical device received regulatory approval for patient use",
        "The nursing staff provided exceptional round the clock care",
    ]},
}


def generate_real_text_tasks(n_tasks=5, n_train=50, n_test=20, dim=128, seed=42,
                             cross_category_noise=0.6):
    rng = np.random.RandomState(seed)

    all_sentences = []
    all_labels = []
    all_task_ids = []

    for task_id in range(min(n_tasks, len(CATEGORIES))):
        cat = CATEGORIES[task_id]
        sentences = cat["sentences"]
        for i in range(n_train + n_test):
            s = sentences[i % len(sentences)]
            words = s.split()
            rng_words = np.random.RandomState(seed + task_id * 10000 + i)
            n_swap = rng_words.randint(1, 3)
            for _ in range(n_swap):
                if len(words) > 3:
                    pos = rng_words.randint(0, len(words) - 1)
                    synonyms_map = {
                        "new": "novel", "significant": "substantial", "important": "crucial",
                        "major": "significant", "announced": "revealed", "developed": "created",
                        "achieved": "attained", "proposed": "introduced", "published": "released",
                        "confirmed": "verified", "discovered": "identified", "produced": "generated",
                        "implemented": "established", "recommended": "suggested", "increased": "raised",
                        "fixed": "resolved", "launched": "initiated", "demonstrated": "showcased",
                    }
                    if words[pos].lower() in synonyms_map:
                        words[pos] = synonyms_map[words[pos].lower()]
            all_sentences.append(" ".join(words))
            all_labels.append(task_id)
            all_task_ids.append(task_id)

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_sentences).astype(np.float32)

    svd = TruncatedSVD(n_components=dim, random_state=seed)
    embeddings = svd.fit_transform(tfidf_matrix).astype(np.float32)

    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    embeddings = embeddings / norms

    shared_direction = rng.randn(dim).astype(np.float32)
    shared_direction /= np.linalg.norm(shared_direction)
    alpha = cross_category_noise
    embeddings = (1 - alpha) * embeddings + alpha * shared_direction
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    embeddings = embeddings / norms

    train_tasks = []
    test_tasks = []
    for task_id in range(min(n_tasks, len(CATEGORIES))):
        task_mask = np.array(all_task_ids) == task_id
        task_emb = embeddings[task_mask]
        task_lbl = np.array(all_labels)[task_mask]

        train_tasks.append({
            "task_id": task_id,
            "name": CATEGORIES[task_id]["name"],
            "emb": task_emb[:n_train],
            "lbl": task_lbl[:n_train],
        })
        test_tasks.append({
            "task_id": task_id,
            "name": CATEGORIES[task_id]["name"],
            "emb": task_emb[n_train:n_train + n_test],
            "lbl": task_lbl[n_train:n_train + n_test],
        })

    return train_tasks, test_tasks, dim


def run_real_data_experiments():
    print("=" * 76)
    print("  REAL DATA EXPERIMENTS")
    print("  TF-IDF + SVD embeddings on 5 topic categories")
    print("=" * 76)

    train_tasks, test_tasks, emb_dim = generate_real_text_tasks(
        n_tasks=5, n_train=50, n_test=20, dim=128, seed=42,
        cross_category_noise=0.85
    )

    print(f"\n  Embedding: TF-IDF + SVD, dim={emb_dim}")
    for t in train_tasks:
        print(f"  Task {t['task_id']}: {t['name']} ({len(t['emb'])} train, "
              f"{len(test_tasks[t['task_id']]['emb'])} test)")

    all_e = np.concatenate([t["emb"] for t in test_tasks])
    all_l = np.concatenate([t["lbl"] for t in test_tasks])
    all_tid = np.concatenate([np.full(len(t["lbl"]), t["task_id"]) for t in test_tasks])

    # ============================================================
    # Experiment 1: Task-Agnostic Comparison
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 1: Task-Agnostic Comparison (Real Text)")
    print("=" * 76)

    configs = {
        "Nearest-Neighbor": dict(use_barcode=False, n_replay=0, lambda_param=1.0),
        "Shared-DG+Dual": dict(use_barcode=True, use_modular=False,
                               n_replay=5, lambda_param=0.8),
        "Modular-DG+Dual (ours)": dict(use_barcode=True, use_modular=True,
                                       n_replay=5, lambda_param=0.8, routing_top_k=5),
    }

    print(f"\n  {'Method':>25s} | {'Agnostic':>9s} | {'Aware':>6s} | {'vs NN':>7s}")
    print(f"  {'-'*25} | {'-'*9} | {'-'*6} | {'-'*7}")

    nn_acc = 0
    for name, cfg in configs.items():
        model = BrainMemoryNetwork(seed=42, embedding_dim=emb_dim,
                                   barcode_dim=256, barcode_sparsity=32, **cfg)
        for td in train_tasks:
            model.learn_task(td["emb"], td["lbl"], td["task_id"])
        ag = model.evaluate(all_e, all_l)["accuracy"]
        aw = model.evaluate(all_e, all_l, all_tid)["accuracy"]
        if name == "Nearest-Neighbor":
            nn_acc = ag
        vs_nn = (ag - nn_acc) / max(nn_acc, 1e-8) * 100
        print(f"  {name:>25s} | {ag:9.4f} | {aw:6.4f} | {vs_nn:+6.1f}%")

    # ============================================================
    # Experiment 2: Per-Task Forgetting
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 2: Per-Task Forgetting (Real Text)")
    print("=" * 76)

    nn_m = BrainMemoryNetwork(seed=42, embedding_dim=emb_dim,
                              use_barcode=False, n_replay=0, lambda_param=1.0)
    ours_m = BrainMemoryNetwork(seed=42, embedding_dim=emb_dim,
                                use_barcode=True, use_modular=True,
                                n_replay=5, lambda_param=0.8, routing_top_k=5)

    for td in train_tasks:
        nn_m.learn_task(td["emb"], td["lbl"], td["task_id"])
        ours_m.learn_task(td["emb"], td["lbl"], td["task_id"])

    print(f"\n  {'Task':>6s} | {'Category':>12s} | {'NN':>8s} | {'Ours':>8s} | {'Improv':>8s}")
    print(f"  {'-'*6} | {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8}")

    for t in test_tasks:
        nn_a = nn_m.evaluate(t["emb"], t["lbl"])["accuracy"]
        o_a = ours_m.evaluate(t["emb"], t["lbl"])["accuracy"]
        imp = (o_a - nn_a) / max(nn_a, 1e-8) * 100
        print(f"  {t['task_id']+1:6d} | {t['name']:>12s} | {nn_a:8.4f} | {o_a:8.4f} | {imp:+7.1f}%")

    # ============================================================
    # Experiment 3: Ablation
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 3: Ablation (Real Text)")
    print("=" * 76)

    ablation = {
        "Full (Modular+Dual+Route)": dict(use_barcode=True, use_modular=True,
                                          n_replay=5, lambda_param=0.8, routing_top_k=5),
        "- Modular (shared DG)":     dict(use_barcode=True, use_modular=False,
                                          n_replay=5, lambda_param=0.8),
        "- Dual Channel (λ=1)":      dict(use_barcode=True, use_modular=True,
                                          n_replay=5, lambda_param=1.0, routing_top_k=5),
        "- Barcode entirely":        dict(use_barcode=False, n_replay=0, lambda_param=1.0),
    }

    print(f"\n  {'Configuration':>35s} | {'Agnostic':>8s} | {'vs Full':>8s}")
    print(f"  {'-'*35} | {'-'*8} | {'-'*8}")

    full_acc = None
    for name, cfg in ablation.items():
        model = BrainMemoryNetwork(seed=42, embedding_dim=emb_dim,
                                   barcode_dim=256, barcode_sparsity=32, **cfg)
        for td in train_tasks:
            model.learn_task(td["emb"], td["lbl"], td["task_id"])
        acc = model.evaluate(all_e, all_l)["accuracy"]
        if full_acc is None:
            full_acc = acc
        diff = acc - full_acc
        print(f"  {name:>35s} | {acc:8.4f} | {diff:+8.4f}")

    # ============================================================
    # Experiment 4: Lambda Sensitivity
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 4: Lambda Sensitivity (Real Text)")
    print("=" * 76)

    print(f"\n  {'Lambda':>8s} | {'Agnostic':>10s} | {'Aware':>8s}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*8}")

    for lam in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        model = BrainMemoryNetwork(seed=42, embedding_dim=emb_dim,
                                   use_barcode=True, use_modular=True,
                                   n_replay=5, lambda_param=lam, routing_top_k=5)
        for td in train_tasks:
            model.learn_task(td["emb"], td["lbl"], td["task_id"])
        ag = model.evaluate(all_e, all_l)["accuracy"]
        aw = model.evaluate(all_e, all_l, all_tid)["accuracy"]
        print(f"  {lam:8.1f} | {ag:10.4f} | {aw:8.4f}")

    # ============================================================
    # Experiment 5: Scalability
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 5: Scalability with More Tasks (Real Text)")
    print("=" * 76)

    for n_tasks in [3, 5]:
        tr, te, dim = generate_real_text_tasks(n_tasks=n_tasks, n_train=50, n_test=20, dim=128, seed=42)
        all_e2 = np.concatenate([t["emb"] for t in te])
        all_l2 = np.concatenate([t["lbl"] for t in te])

        nn2 = BrainMemoryNetwork(seed=42, embedding_dim=dim, use_barcode=False, n_replay=0, lambda_param=1.0)
        ours2 = BrainMemoryNetwork(seed=42, embedding_dim=dim, use_barcode=True, use_modular=True,
                                   n_replay=5, lambda_param=0.8, routing_top_k=5)
        for td in tr:
            nn2.learn_task(td["emb"], td["lbl"], td["task_id"])
            ours2.learn_task(td["emb"], td["lbl"], td["task_id"])

        nn_a2 = nn2.evaluate(all_e2, all_l2)["accuracy"]
        o_a2 = ours2.evaluate(all_e2, all_l2)["accuracy"]
        print(f"  {n_tasks} tasks: NN={nn_a2:.4f}, Ours={o_a2:.4f} ({(o_a2-nn_a2)/max(nn_a2,1e-8)*100:+.1f}%)")

    print("\n" + "=" * 76)
    print("  REAL DATA EXPERIMENTS COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    run_real_data_experiments()
