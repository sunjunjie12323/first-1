from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


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
        module = self.get_or_create_module(task_id)
        return module.separate(x)

    def infer_module_content_guided(self, query, stored_emb, stored_module_ids, top_k=5):
        """
        Content-guided module inference (EC→DG routing).
        
        Uses content similarity to determine which module a query belongs to.
        This is biologically motivated: EC provides content information to DG,
        helping it route to the correct granule cell module.
        
        Mechanism: Find top-K content matches → majority vote on module ID.
        """
        if not stored_emb:
            return 0

        q = query.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return 0
        q = q / q_norm

        emb_matrix = np.stack(stored_emb).astype(np.float32)
        emb_norms = np.maximum(np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-8)
        emb_n = emb_matrix / emb_norms
        sims = emb_n @ q

        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]

        module_votes = {}
        for idx in top_indices:
            mid = stored_module_ids[idx]
            module_votes[mid] = module_votes.get(mid, 0) + 1

        return max(module_votes, key=module_votes.get)

    def separate_with_inference(self, query, stored_emb, stored_module_ids, top_k=5):
        module_idx = self.infer_module_content_guided(query, stored_emb, stored_module_ids, top_k)
        module = self.modules.get(module_idx)
        if module is None:
            module = list(self.modules.values())[0]
            module_idx = list(self.modules.keys())[0]
        bc = module.separate(query)
        return bc, module_idx


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
                if self.use_modular:
                    bc = self.dg.encode(embeddings[i], task_id)
                    module_idx = task_id
                else:
                    bc = self.dg.encode(embeddings[i], 0)
                    module_idx = 0
                self.stored_bc.append(bc)
                self.stored_module.append(module_idx)
            self.stored_emb.append(embeddings[i].copy())
            self.stored_lbl.append(labels[i])
            self.stored_task.append(task_id)

        if self.n_replay > 0 and self.use_barcode and len(self.stored_emb) > len(embeddings):
            n_old = len(self.stored_emb) - len(embeddings)
            for idx in self._rng.choice(n_old, min(self.n_replay, n_old), replace=False):
                tid = self.stored_task[idx] if self.use_modular else 0
                bc = self.dg.encode(self.stored_emb[idx], tid)
                self.stored_bc[idx] = bc

    def predict_task_aware(self, query, task_id, lambda_param=None):
        if not self.stored_emb:
            return -1
        lam = lambda_param if lambda_param is not None else self.lambda_param
        c_scores = self._content_scores(query)
        if self.use_barcode:
            tid = task_id if self.use_modular else 0
            bc = self.dg.encode(query, tid)
            b_scores = self._barcode_scores(bc)
            combined = self._combine(c_scores, b_scores, lam)
        else:
            combined = c_scores
        return self.stored_lbl[int(np.argmax(combined))]

    def predict_task_agnostic(self, query, lambda_param=None):
        if not self.stored_emb:
            return -1
        lam = lambda_param if lambda_param is not None else self.lambda_param
        c_scores = self._content_scores(query)
        if self.use_barcode:
            bc, module_idx = self.dg.separate_with_inference(
                query, self.stored_emb, self.stored_module, self.routing_top_k
            )
            b_scores = self._barcode_scores(bc)
            combined = self._combine(c_scores, b_scores, lam)
        else:
            combined = c_scores
        return self.stored_lbl[int(np.argmax(combined))]

    def evaluate_task_aware(self, test_emb, test_lbl, test_task_ids, lambda_param=None):
        correct = 0
        for i in range(len(test_emb)):
            if self.predict_task_aware(test_emb[i], int(test_task_ids[i]), lambda_param) == test_lbl[i]:
                correct += 1
        return {"accuracy": correct / len(test_emb)}

    def evaluate_task_agnostic(self, test_emb, test_lbl, lambda_param=None):
        correct = 0
        for i in range(len(test_emb)):
            if self.predict_task_agnostic(test_emb[i], lambda_param) == test_lbl[i]:
                correct += 1
        return {"accuracy": correct / len(test_emb)}

    def evaluate_module_detection(self, test_emb, test_task_ids):
        correct = 0
        for i in range(len(test_emb)):
            detected = self.dg.infer_module_content_guided(
                test_emb[i], self.stored_emb, self.stored_module, self.routing_top_k
            )
            true_task = int(test_task_ids[i])
            if detected == true_task:
                correct += 1
        return {"detection_accuracy": correct / len(test_emb)}

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


def generate_tasks(n_tasks=5, n_classes=2, n_train=50, n_test=20,
                   dim=128, cross_sim=0.3, spread=0.3, seed=42):
    rng = np.random.RandomState(seed)
    shared = rng.randn(dim).astype(np.float32)
    shared /= np.linalg.norm(shared)
    train_t, test_t = [], []
    for task in range(n_tasks):
        centers = []
        for c in range(n_classes):
            center = rng.randn(dim).astype(np.float32)
            center /= np.linalg.norm(center)
            center = cross_sim * shared + (1 - cross_sim) * center
            center /= np.linalg.norm(center)
            centers.append(center)
        tr_e, tr_l, te_e, te_l = [], [], [], []
        for c in range(n_classes):
            label = task * n_classes + c
            for _ in range(n_train):
                noise = rng.randn(dim).astype(np.float32) * spread
                e = centers[c] + noise
                e /= np.linalg.norm(e)
                tr_e.append(e); tr_l.append(label)
            for _ in range(n_test):
                noise = rng.randn(dim).astype(np.float32) * spread
                e = centers[c] + noise
                e /= np.linalg.norm(e)
                te_e.append(e); te_l.append(label)
        train_t.append({"task_id": task, "emb": np.stack(tr_e), "lbl": np.array(tr_l)})
        test_t.append({"task_id": task, "emb": np.stack(te_e), "lbl": np.array(te_l)})
    return train_t, test_t


def run_optimal_experiments():
    print("=" * 76)
    print("  OPTIMAL VERSION: Content-Guided Modular DG")
    print("  for Task-Agnostic Continual Learning")
    print("=" * 76)
    print("""
  KEY IMPROVEMENT:
  Module inference uses CONTENT-GUIDED ROUTING (EC→DG pathway).
  
  Previous problem: Pure barcode matching failed (29% detection).
  Fix: Use content similarity to vote on which module to use.
  This is biologically motivated: EC provides context to DG.
  
  Innovation chain:
  1. Modular DG = adult neurogenesis (new module per task)
  2. Content-guided routing = EC→DG pathway (module selection)
  3. Dual-channel retrieval = DG+CA3 (separation+completion)
  4. Task-agnostic = no task ID needed at test time
""")

    # ============================================================
    # Experiment 1: Task-Agnostic vs Task-Aware
    # ============================================================
    print("=" * 76)
    print("  EXPERIMENT 1: Task-Agnostic vs Task-Aware")
    print("=" * 76)

    for cross_sim in [0.2, 0.3, 0.4]:
        tr, te = generate_tasks(n_tasks=5, n_classes=2, n_train=50, n_test=20,
                                cross_sim=cross_sim, seed=42)

        model = BrainMemoryNetwork(seed=42, use_barcode=True, use_modular=True,
                                   lambda_param=0.8, n_replay=5, routing_top_k=5)
        for td in tr:
            model.learn_task(td["emb"], td["lbl"], td["task_id"])

        all_e = np.concatenate([t["emb"] for t in te])
        all_l = np.concatenate([t["lbl"] for t in te])
        all_tid = np.concatenate([np.full(len(t["lbl"]), t["task_id"]) for t in te])

        nn_m = BrainMemoryNetwork(seed=42, use_barcode=False, n_replay=0, lambda_param=1.0)
        for td in tr:
            nn_m.learn_task(td["emb"], td["lbl"], td["task_id"])

        aware = model.evaluate_task_aware(all_e, all_l, all_tid)["accuracy"]
        agnostic = model.evaluate_task_agnostic(all_e, all_l)["accuracy"]
        detection = model.evaluate_module_detection(all_e, all_tid)["detection_accuracy"]
        nn_acc = nn_m.evaluate_task_agnostic(all_e, all_l)["accuracy"]

        print(f"  Cross-sim={cross_sim}:")
        print(f"    NN (no barcode):           {nn_acc:.4f}")
        print(f"    Ours Task-AWARE:           {aware:.4f}")
        print(f"    Ours Task-AGNOSTIC:        {agnostic:.4f}")
        print(f"    Module detection:          {detection:.4f}")
        print(f"    Agnostic/Aware:            {agnostic/max(aware,1e-8):.1%}")
        print(f"    Agnostic vs NN:            {(agnostic-nn_acc)/max(nn_acc,1e-8)*100:+.1f}%")
        print()

    # ============================================================
    # Experiment 2: Fair Comparison (all task-agnostic)
    # ============================================================
    print("=" * 76)
    print("  EXPERIMENT 2: Fair Task-Agnostic Comparison")
    print("=" * 76)

    for n_tasks in [5, 10, 20]:
        tr, te = generate_tasks(n_tasks=n_tasks, n_classes=2, n_train=30,
                                n_test=10, cross_sim=0.3, seed=42)

        nn_m = BrainMemoryNetwork(seed=42, use_barcode=False, n_replay=0, lambda_param=1.0)
        shared_m = BrainMemoryNetwork(seed=42, use_barcode=True, use_modular=False,
                                      n_replay=5, lambda_param=0.8)
        modular_m = BrainMemoryNetwork(seed=42, use_barcode=True, use_modular=True,
                                       n_replay=5, lambda_param=0.8, routing_top_k=5)

        for td in tr:
            nn_m.learn_task(td["emb"], td["lbl"], td["task_id"])
            shared_m.learn_task(td["emb"], td["lbl"], td["task_id"])
            modular_m.learn_task(td["emb"], td["lbl"], td["task_id"])

        all_e = np.concatenate([t["emb"] for t in te])
        all_l = np.concatenate([t["lbl"] for t in te])

        nn_a = nn_m.evaluate_task_agnostic(all_e, all_l)["accuracy"]
        sh_a = shared_m.evaluate_task_agnostic(all_e, all_l)["accuracy"]
        mo_a = modular_m.evaluate_task_agnostic(all_e, all_l)["accuracy"]

        print(f"  {n_tasks:2d} tasks: NN={nn_a:.4f}, Shared-DG={sh_a:.4f}, "
              f"Modular-DG={mo_a:.4f} "
              f"(vs NN: {(mo_a-nn_a)/max(nn_a,1e-8)*100:+.0f}%, "
              f"vs Shared: {(mo_a-sh_a)/max(sh_a,1e-8)*100:+.0f}%)")

    # ============================================================
    # Experiment 3: Ablation (task-agnostic)
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 3: Ablation (Task-Agnostic)")
    print("=" * 76)

    tr, te = generate_tasks(n_tasks=5, n_classes=2, n_train=50, n_test=20,
                            cross_sim=0.3, seed=42)
    all_e = np.concatenate([t["emb"] for t in te])
    all_l = np.concatenate([t["lbl"] for t in te])

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
        model = BrainMemoryNetwork(seed=42, **cfg)
        for td in tr:
            model.learn_task(td["emb"], td["lbl"], td["task_id"])
        acc = model.evaluate_task_agnostic(all_e, all_l)["accuracy"]
        if full_acc is None:
            full_acc = acc
        diff = acc - full_acc
        print(f"  {name:>35s} | {acc:8.4f} | {diff:+8.4f}")

    # ============================================================
    # Experiment 4: Per-Task Forgetting
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 4: Per-Task Forgetting (Task-Agnostic)")
    print("=" * 76)

    nn_m = BrainMemoryNetwork(seed=42, use_barcode=False, n_replay=0, lambda_param=1.0)
    ours_m = BrainMemoryNetwork(seed=42, use_barcode=True, use_modular=True,
                                n_replay=5, lambda_param=0.8, routing_top_k=5)

    for td in tr:
        nn_m.learn_task(td["emb"], td["lbl"], td["task_id"])
        ours_m.learn_task(td["emb"], td["lbl"], td["task_id"])

    print(f"\n  {'Task':>6s} | {'NN':>8s} | {'Ours':>8s} | {'Improvement':>12s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*12}")
    for t in te:
        nn_a = nn_m.evaluate_task_agnostic(t["emb"], t["lbl"])["accuracy"]
        o_a = ours_m.evaluate_task_agnostic(t["emb"], t["lbl"])["accuracy"]
        imp = (o_a - nn_a) / max(nn_a, 1e-8) * 100
        print(f"  {t['task_id']+1:6d} | {nn_a:8.4f} | {o_a:8.4f} | {imp:+11.1f}%")

    # ============================================================
    # Experiment 5: Routing top-K sensitivity
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 5: Routing Top-K Sensitivity")
    print("=" * 76)

    all_tid = np.concatenate([np.full(len(t["lbl"]), t["task_id"]) for t in te])

    print(f"\n  {'Top-K':>6s} | {'Agnostic':>10s} | {'Aware':>8s} | {'Detection':>10s}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*8} | {'-'*10}")

    for k in [1, 3, 5, 7, 10, 15, 20]:
        model = BrainMemoryNetwork(seed=42, use_barcode=True, use_modular=True,
                                   n_replay=5, lambda_param=0.8, routing_top_k=k)
        for td in tr:
            model.learn_task(td["emb"], td["lbl"], td["task_id"])
        ag = model.evaluate_task_agnostic(all_e, all_l)["accuracy"]
        aw = model.evaluate_task_aware(all_e, all_l, all_tid)["accuracy"]
        det = model.evaluate_module_detection(all_e, all_tid)["detection_accuracy"]
        print(f"  {k:6d} | {ag:10.4f} | {aw:8.4f} | {det:10.4f}")

    # ============================================================
    # Experiment 6: Scalability
    # ============================================================
    print("\n" + "=" * 76)
    print("  EXPERIMENT 6: Scalability (Task-Agnostic)")
    print("=" * 76)

    print(f"\n  {'Tasks':>6s} | {'NN':>8s} | {'Ours':>8s} | {'Improvement':>12s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*12}")

    for n_tasks in [5, 10, 20]:
        tr, te = generate_tasks(n_tasks=n_tasks, n_classes=2, n_train=30,
                                n_test=10, cross_sim=0.3, seed=42)
        nn_m = BrainMemoryNetwork(seed=42, use_barcode=False, n_replay=0, lambda_param=1.0)
        ours_m = BrainMemoryNetwork(seed=42, use_barcode=True, use_modular=True,
                                    n_replay=5, lambda_param=0.8, routing_top_k=5)
        for td in tr:
            nn_m.learn_task(td["emb"], td["lbl"], td["task_id"])
            ours_m.learn_task(td["emb"], td["lbl"], td["task_id"])
        all_e = np.concatenate([t["emb"] for t in te])
        all_l = np.concatenate([t["lbl"] for t in te])
        nn_a = nn_m.evaluate_task_agnostic(all_e, all_l)["accuracy"]
        o_a = ours_m.evaluate_task_agnostic(all_e, all_l)["accuracy"]
        imp = (o_a - nn_a) / max(nn_a, 1e-8) * 100
        print(f"  {n_tasks:6d} | {nn_a:8.4f} | {o_a:8.4f} | {imp:+11.1f}%")

    # ============================================================
    # FINAL ASSESSMENT
    # ============================================================
    print("\n\n" + "=" * 76)
    print("  FINAL HONEST ASSESSMENT")
    print("=" * 76)


if __name__ == "__main__":
    run_optimal_experiments()
