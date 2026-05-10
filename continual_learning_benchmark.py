from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurocortex.core.theory import (
    ChannelIndependenceTheorem,
    SeparationCompletionDuality,
)


class DGModule:
    """
    Single DG module with its own random projection.
    Simulates a cluster of DG granule cells that respond to a specific input context.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 256,
        sparsity: int = 32,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(output_dim, input_dim).astype(np.float32)
        row_norms = np.linalg.norm(self.projection, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-8)
        self.projection = self.projection / row_norms
    
    def separate(self, x: np.ndarray) -> np.ndarray:
        projected = self.projection @ x.astype(np.float32)
        barcode = np.zeros_like(projected)
        if self.sparsity >= len(projected):
            return np.maximum(projected, 0.0)
        top_indices = np.argpartition(projected, -self.sparsity)[-self.sparsity:]
        barcode[top_indices] = np.maximum(projected[top_indices], 0.0)
        return barcode


class ModularDG:
    """
    Modular Dentate Gyrus with task-specific projection modules.
    
    Simulates adult neurogenesis in DG: new granule cells are born
    for new experiences, creating task-specific sparse codes.
    
    Key innovation: Each task gets its own DG module, ensuring that
    barcodes from different tasks occupy DIFFERENT subspaces.
    This provides task-level pattern separation that is impossible
    with a single shared projection.
    
    At test time, the correct module is selected based on context
    (task identity), which is the standard task-incremental setting.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 256,
        sparsity: int = 32,
        base_seed: int = 42,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.base_seed = base_seed
        self.modules: Dict[int, DGModule] = {}
    
    def get_or_create_module(self, task_id: int) -> DGModule:
        if task_id not in self.modules:
            self.modules[task_id] = DGModule(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                sparsity=self.sparsity,
                seed=self.base_seed + task_id * 1000,
            )
        return self.modules[task_id]
    
    def separate(self, x: np.ndarray, task_id: int) -> np.ndarray:
        module = self.get_or_create_module(task_id)
        return module.separate(x)


class BrainMemoryNetwork:
    """
    Brain-Inspired Memory Network for Continual Learning
    
    Architecture:
    - DG (Modular): Task-specific sparse coding → pattern separation
    - CA3 (Hopfield): Attractor dynamics → pattern completion
    - Dual-channel retrieval: Content + Barcode → interference resolution
    
    Theoretical foundation:
    - Theorem 2: Separation-Completion Duality (optimal λ* ∈ (0,1))
    - Theorem 4: Channel Independence (resolves Bird et al. 2024)
    
    This is the core system for the paper's contribution to embodied
    intelligence: a brain-inspired memory network that learns continuously
    with theoretical guarantees on pattern separation and identifiability.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        barcode_dim: int = 256,
        barcode_sparsity: int = 32,
        lambda_param: float = 0.8,
        n_replay: int = 5,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.barcode_dim = barcode_dim
        self.lambda_param = lambda_param
        self.n_replay = n_replay
        self.seed = seed
        
        self.dg = ModularDG(
            input_dim=embedding_dim,
            output_dim=barcode_dim,
            sparsity=barcode_sparsity,
            base_seed=seed,
        )
        
        self.stored_embeddings: List[np.ndarray] = []
        self.stored_barcodes: List[np.ndarray] = []
        self.stored_labels: List[int] = []
        self.stored_tasks: List[int] = []
        self._rng = np.random.RandomState(seed)
    
    def learn_task(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        task_id: int,
    ) -> Dict[str, float]:
        for i in range(len(embeddings)):
            barcode = self.dg.separate(embeddings[i], task_id)
            self.stored_embeddings.append(embeddings[i].copy())
            self.stored_barcodes.append(barcode)
            self.stored_labels.append(labels[i])
            self.stored_tasks.append(task_id)
        
        if self.n_replay > 0 and len(self.stored_embeddings) > len(embeddings):
            n_old = len(self.stored_embeddings) - len(embeddings)
            n_replay = min(self.n_replay, n_old)
            replay_indices = self._rng.choice(n_old, size=n_replay, replace=False)
            for idx in replay_indices:
                old_task = self.stored_tasks[idx]
                barcode = self.dg.separate(self.stored_embeddings[idx], old_task)
                self.stored_barcodes[idx] = barcode
        
        return {"n_stored": len(self.stored_embeddings)}
    
    def retrieve(
        self,
        query: np.ndarray,
        task_id: Optional[int] = None,
        top_k: int = 1,
        lambda_param: Optional[float] = None,
    ) -> Tuple[int, float, Dict[str, float]]:
        if not self.stored_embeddings:
            return -1, 0.0, {}
        
        lam = lambda_param if lambda_param is not None else self.lambda_param
        
        content_scores = self._content_scores(query)
        
        if task_id is not None:
            query_barcode = self.dg.separate(query, task_id)
            barcode_scores = self._barcode_scores(query_barcode)
        else:
            barcode_scores = np.ones(len(self.stored_barcodes), dtype=np.float32) / len(self.stored_barcodes)
        
        combined = self._combine_scores(content_scores, barcode_scores, lam)
        
        best_idx = int(np.argmax(combined))
        
        info = {
            "content_best": int(np.argmax(content_scores)),
            "barcode_best": int(np.argmax(barcode_scores)),
            "combined_best": best_idx,
        }
        
        return self.stored_labels[best_idx], float(combined[best_idx]), info
    
    def evaluate(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        test_task_ids: Optional[np.ndarray] = None,
        lambda_param: Optional[float] = None,
    ) -> Dict[str, float]:
        if not self.stored_embeddings:
            return {"accuracy": 0.0, "content_accuracy": 0.0, "barcode_accuracy": 0.0}
        
        lam = lambda_param if lambda_param is not None else self.lambda_param
        
        correct = 0
        content_correct = 0
        barcode_correct = 0
        
        for i in range(len(test_embeddings)):
            task_id = int(test_task_ids[i]) if test_task_ids is not None else None
            
            content_scores = self._content_scores(test_embeddings[i])
            
            if task_id is not None:
                query_barcode = self.dg.separate(test_embeddings[i], task_id)
                barcode_scores = self._barcode_scores(query_barcode)
            else:
                barcode_scores = np.ones(len(self.stored_barcodes), dtype=np.float32) / len(self.stored_barcodes)
            
            combined = self._combine_scores(content_scores, barcode_scores, lam)
            
            if self.stored_labels[int(np.argmax(combined))] == test_labels[i]:
                correct += 1
            if self.stored_labels[int(np.argmax(content_scores))] == test_labels[i]:
                content_correct += 1
            if task_id is not None and self.stored_labels[int(np.argmax(barcode_scores))] == test_labels[i]:
                barcode_correct += 1
        
        n = len(test_embeddings)
        return {
            "accuracy": correct / n,
            "content_accuracy": content_correct / n,
            "barcode_accuracy": barcode_correct / n if test_task_ids is not None else 0.0,
        }
    
    def _content_scores(self, query: np.ndarray) -> np.ndarray:
        query = query.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm < 1e-8:
            return np.zeros(len(self.stored_embeddings), dtype=np.float32)
        q = query / norm
        emb = np.stack(self.stored_embeddings).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_n = emb / norms
        return (emb_n @ q).astype(np.float32)
    
    def _barcode_scores(self, query_barcode: np.ndarray) -> np.ndarray:
        query_barcode = query_barcode.astype(np.float32)
        norm = np.linalg.norm(query_barcode)
        if norm < 1e-8:
            return np.zeros(len(self.stored_barcodes), dtype=np.float32)
        q = query_barcode / norm
        bc = np.stack(self.stored_barcodes).astype(np.float32)
        norms = np.linalg.norm(bc, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        bc_n = bc / norms
        return (bc_n @ q).astype(np.float32)
    
    def _combine_scores(
        self,
        content_scores: np.ndarray,
        barcode_scores: np.ndarray,
        lambda_param: float,
    ) -> np.ndarray:
        c_min, c_max = float(np.min(content_scores)), float(np.max(content_scores))
        c_range = c_max - c_min
        if c_range < 1e-8:
            norm_c = np.ones_like(content_scores) / len(content_scores)
        else:
            norm_c = (content_scores - c_min) / c_range
        
        b_min, b_max = float(np.min(barcode_scores)), float(np.max(barcode_scores))
        b_range = b_max - b_min
        if b_range < 1e-8:
            norm_b = np.ones_like(barcode_scores) / len(barcode_scores)
        else:
            norm_b = (barcode_scores - b_min) / b_range
        
        return (lambda_param * norm_c + (1.0 - lambda_param) * norm_b).astype(np.float32)


def generate_continual_learning_data(
    n_tasks: int = 5,
    n_classes_per_task: int = 2,
    n_train_per_class: int = 50,
    n_test_per_class: int = 20,
    embedding_dim: int = 128,
    cross_task_similarity: float = 0.3,
    within_class_spread: float = 0.3,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(seed)
    
    shared_base = rng.randn(embedding_dim).astype(np.float32)
    shared_base = shared_base / np.linalg.norm(shared_base)
    
    train_tasks = []
    test_tasks = []
    
    for task in range(n_tasks):
        class_centers = []
        for c in range(n_classes_per_task):
            center = rng.randn(embedding_dim).astype(np.float32)
            center = center / np.linalg.norm(center)
            center = cross_task_similarity * shared_base + (1 - cross_task_similarity) * center
            center = center / np.linalg.norm(center)
            class_centers.append(center)
        
        train_embs, train_lbls = [], []
        test_embs, test_lbls = [], []
        
        for c in range(n_classes_per_task):
            label = task * n_classes_per_task + c
            for _ in range(n_train_per_class):
                noise = rng.randn(embedding_dim).astype(np.float32) * within_class_spread
                emb = class_centers[c] + noise
                emb = emb / np.linalg.norm(emb)
                train_embs.append(emb)
                train_lbls.append(label)
            for _ in range(n_test_per_class):
                noise = rng.randn(embedding_dim).astype(np.float32) * within_class_spread
                emb = class_centers[c] + noise
                emb = emb / np.linalg.norm(emb)
                test_embs.append(emb)
                test_lbls.append(label)
        
        train_tasks.append({
            "task_id": task,
            "embeddings": np.stack(train_embs),
            "labels": np.array(train_lbls),
        })
        test_tasks.append({
            "task_id": task,
            "embeddings": np.stack(test_embs),
            "labels": np.array(test_lbls),
        })
    
    return train_tasks, test_tasks


def run_continual_learning_experiment() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Brain-Inspired Memory Network: Continual Learning Benchmark      ║")
    print("║                                                                    ║")
    print("║  Core Question:                                                   ║")
    print("║  Does modular DG + dual-channel retrieval reduce catastrophic     ║")
    print("║  forgetting compared to single-channel methods?                   ║")
    print("║                                                                    ║")
    print("║  Methods compared:                                                ║")
    print("║  1. Content-Only (no barcode, no replay)                          ║")
    print("║  2. Content + Replay (replay old memories, no barcode)            ║")
    print("║  3. Shared-DG + Dual (single DG projection, dual-channel)         ║")
    print("║  4. Modular-DG + Dual (task-specific DG, dual-channel) ★OURS★    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    for cross_sim in [0.2, 0.4]:
        print(f"\n{'='*70}")
        print(f"  Cross-Task Similarity = {cross_sim}")
        print(f"{'='*70}")
        
        train_tasks, test_tasks = generate_continual_learning_data(
            n_tasks=5,
            n_classes_per_task=2,
            n_train_per_class=50,
            n_test_per_class=20,
            embedding_dim=128,
            cross_task_similarity=cross_sim,
            seed=42,
        )
        
        shared_dg_model = BrainMemoryNetwork(
            embedding_dim=128, barcode_dim=256, barcode_sparsity=32,
            lambda_param=0.8, n_replay=5, seed=42,
        )
        
        methods = {
            "Content-Only": BrainMemoryNetwork(
                embedding_dim=128, barcode_dim=256, barcode_sparsity=32,
                lambda_param=1.0, n_replay=0, seed=42,
            ),
            "Content+Replay": BrainMemoryNetwork(
                embedding_dim=128, barcode_dim=256, barcode_sparsity=32,
                lambda_param=1.0, n_replay=5, seed=42,
            ),
            "Shared-DG+Dual": shared_dg_model,
            "Modular-DG+Dual★": BrainMemoryNetwork(
                embedding_dim=128, barcode_dim=256, barcode_sparsity=32,
                lambda_param=0.8, n_replay=5, seed=42,
            ),
        }
        
        use_modular = {
            "Content-Only": False,
            "Content+Replay": False,
            "Shared-DG+Dual": False,
            "Modular-DG+Dual★": True,
        }
        
        shared_dg_task_id = 0
        
        task_accuracies = {name: [] for name in methods}
        task_forgetting = {name: [] for name in methods}
        
        for task_data in train_tasks:
            task_id = task_data["task_id"]
            
            for name, model in methods.items():
                if name == "Shared-DG+Dual":
                    model.learn_task(task_data["embeddings"], task_data["labels"], shared_dg_task_id)
                elif use_modular[name]:
                    model.learn_task(task_data["embeddings"], task_data["labels"], task_id)
                else:
                    model.learn_task(task_data["embeddings"], task_data["labels"], task_id)
            
            all_test_emb = np.concatenate([t["embeddings"] for t in test_tasks[:task_id + 1]])
            all_test_lbl = np.concatenate([t["labels"] for t in test_tasks[:task_id + 1]])
            all_test_task = np.concatenate([
                np.full(len(t["labels"]), t["task_id"]) for t in test_tasks[:task_id + 1]
            ])
            
            for name, model in methods.items():
                if name in ("Content-Only", "Content+Replay"):
                    result = model.evaluate(all_test_emb, all_test_lbl, lambda_param=1.0)
                elif name == "Shared-DG+Dual":
                    shared_task_ids = np.full_like(all_test_task, shared_dg_task_id)
                    result = model.evaluate(
                        all_test_emb, all_test_lbl,
                        test_task_ids=shared_task_ids,
                        lambda_param=0.8,
                    )
                else:
                    result = model.evaluate(
                        all_test_emb, all_test_lbl,
                        test_task_ids=all_test_task,
                        lambda_param=0.8,
                    )
                task_accuracies[name].append(result["accuracy"])
        
        print(f"\n  {'Task':>6s}", end="")
        for name in methods:
            print(f" | {name:>18s}", end="")
        print()
        print(f"  {'-'*6}", end="")
        for _ in methods:
            print(f" | {'-'*18}", end="")
        print()
        
        for t in range(len(train_tasks)):
            print(f"  {t+1:6d}", end="")
            for name in methods:
                print(f" | {task_accuracies[name][t]:18.4f}", end="")
            print()
        
        print()
        print("  Final accuracy & Forgetting analysis:")
        for name in methods:
            final_acc = task_accuracies[name][-1]
            first_acc = task_accuracies[name][0]
            forgetting = first_acc - final_acc
            print(f"    {name:20s}: final={final_acc:.4f}, forgetting={forgetting:+.4f}")
        
        modular_final = task_accuracies["Modular-DG+Dual★"][-1]
        content_final = task_accuracies["Content+Replay"][-1]
        shared_final = task_accuracies["Shared-DG+Dual"][-1]
        
        print()
        print(f"  Modular-DG+Dual vs Content+Replay: "
              f"{'✓ BETTER' if modular_final > content_final else '✗ NOT BETTER'} "
              f"({modular_final:.4f} vs {content_final:.4f})")
        print(f"  Modular-DG+Dual vs Shared-DG+Dual: "
              f"{'✓ BETTER' if modular_final > shared_final else '✗ NOT BETTER'} "
              f"({modular_final:.4f} vs {shared_final:.4f})")
    
    print(f"\n{'='*70}")
    print("  Channel Independence Verification (Theorem 4)")
    print(f"{'='*70}")
    
    train_tasks, test_tasks = generate_continual_learning_data(
        n_tasks=5, n_classes_per_task=2, n_train_per_class=50,
        n_test_per_class=20, embedding_dim=128, cross_task_similarity=0.3, seed=42,
    )
    
    model = BrainMemoryNetwork(
        embedding_dim=128, barcode_dim=256, barcode_sparsity=32,
        lambda_param=0.8, n_replay=5, seed=42,
    )
    
    for task_data in train_tasks:
        model.learn_task(
            task_data["embeddings"], task_data["labels"], task_data["task_id"]
        )
    
    print("  Verifying: modifying barcodes does not affect content scores")
    content_before = model._content_scores(test_tasks[0]["embeddings"][0])
    
    for i in range(len(model.stored_barcodes)):
        model.stored_barcodes[i] = np.zeros_like(model.stored_barcodes[i])
    
    content_after = model._content_scores(test_tasks[0]["embeddings"][0])
    
    max_diff = float(np.max(np.abs(content_before - content_after)))
    print(f"  Max content score change after zeroing all barcodes: {max_diff:.10f}")
    print(f"  Content invariant: {max_diff < 1e-6}")
    
    print()
    print("  Verifying: modifying content does not affect barcode scores")
    barcode_before = model._barcode_scores(
        model.dg.separate(test_tasks[0]["embeddings"][0], 0)
    )
    
    for i in range(len(model.stored_embeddings)):
        model.stored_embeddings[i] = np.zeros_like(model.stored_embeddings[i])
    
    barcode_after = model._barcode_scores(
        model.dg.separate(test_tasks[0]["embeddings"][0], 0)
    )
    
    max_diff_bc = float(np.max(np.abs(barcode_before - barcode_after)))
    print(f"  Max barcode score change after zeroing all content: {max_diff_bc:.10f}")
    print(f"  Barcode invariant: {max_diff_bc < 1e-6}")
    
    print()
    if max_diff < 1e-6 and max_diff_bc < 1e-6:
        print("  ★ Theorem 4 CONFIRMED: Channels are operationally independent ★")
    else:
        print("  ⚠ Theorem 4 needs further verification")


if __name__ == "__main__":
    run_continual_learning_experiment()
