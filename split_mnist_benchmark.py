from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def log(msg=""):
    print(msg, flush=True)


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, n_tasks=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_tasks)])
        self.n_tasks = n_tasks

    def forward(self, x, task_id):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.heads[task_id](x)

    def get_features(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DGModule:
    def __init__(self, input_dim=400, output_dim=512, sparsity=64, seed=0):
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(output_dim, input_dim).astype(np.float32)
        row_norms = np.linalg.norm(self.projection, axis=1, keepdims=True)
        self.projection = self.projection / np.maximum(row_norms, 1e-8)
        self.sparsity = sparsity

    def separate_batch(self, X):
        projected = (self.projection @ X.T.astype(np.float32)).T
        barcode = np.zeros_like(projected)
        if self.sparsity >= projected.shape[1]:
            return np.maximum(projected, 0.0)
        top_idx = np.argpartition(projected, -self.sparsity, axis=1)[:, -self.sparsity:]
        for i in range(len(barcode)):
            barcode[i, top_idx[i]] = np.maximum(projected[i, top_idx[i]], 0.0)
        return barcode


class ModularDG:
    def __init__(self, input_dim=400, output_dim=512, sparsity=64, base_seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.base_seed = base_seed
        self.modules: Dict[int, DGModule] = {}
        self._counter = 0

    def get_or_create(self, task_id):
        if task_id not in self.modules:
            self.modules[task_id] = DGModule(
                self.input_dim, self.output_dim, self.sparsity,
                self.base_seed + self._counter * 1000)
            self._counter += 1
        return self.modules[task_id]

    def encode_batch(self, X, task_id):
        return self.get_or_create(task_id).separate_batch(X)

    def infer_modules_batch(self, queries, stored_features, stored_task_ids, top_k=5):
        q = queries.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        q = q / q_norms
        emb = stored_features.astype(np.float32)
        emb_norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        emb = emb / emb_norms
        sims = q @ emb.T
        inferred = np.zeros(len(queries), dtype=np.int32)
        for i in range(len(queries)):
            k = min(top_k, sims.shape[1])
            top_idx = np.argpartition(sims[i], -k)[-k:]
            votes = {}
            for idx in top_idx:
                tid = stored_task_ids[idx]
                votes[tid] = votes.get(tid, 0) + 1
            inferred[i] = max(votes, key=votes.get)
        return inferred


class DualChannelMemory:
    def __init__(self, feature_dim=400, barcode_dim=512, barcode_sparsity=64,
                 lambda_param=0.7, use_modular=True, seed=42):
        self.feature_dim = feature_dim
        self.lambda_param = lambda_param
        self.use_modular = use_modular
        self.dg = ModularDG(feature_dim, barcode_dim, barcode_sparsity, seed)
        if not use_modular:
            self.dg.get_or_create(0)
        self.stored_features = np.zeros((0, feature_dim), dtype=np.float32)
        self.stored_barcodes = np.zeros((0, barcode_dim), dtype=np.float32)
        self.stored_labels = np.zeros(0, dtype=np.int32)
        self.stored_tasks = np.zeros(0, dtype=np.int32)
        self.stored_raw = []
        self._rng = np.random.RandomState(seed)

    def store(self, features, labels, task_id, raw_images=None):
        tid = task_id if self.use_modular else 0
        barcodes = self.dg.encode_batch(features, tid)
        self.stored_features = np.concatenate([self.stored_features, features], axis=0)
        self.stored_barcodes = np.concatenate([self.stored_barcodes, barcodes], axis=0)
        self.stored_labels = np.concatenate([self.stored_labels, labels], axis=0)
        self.stored_tasks = np.concatenate(
            [self.stored_tasks, np.full(len(labels), task_id, dtype=np.int32)], axis=0)
        if raw_images is not None:
            self.stored_raw.extend(raw_images)

    def predict_batch(self, query_features, task_ids=None, lambda_param=None):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        lam = lambda_param if lambda_param is not None else self.lambda_param
        c_scores = self._content_scores_batch(query_features)
        if task_ids is not None:
            barcodes = np.zeros((len(query_features), self.dg.output_dim), dtype=np.float32)
            for tid_val in np.unique(task_ids):
                mask = task_ids == tid_val
                t = tid_val if self.use_modular else 0
                barcodes[mask] = self.dg.encode_batch(query_features[mask], t)
        else:
            inferred = self.dg.infer_modules_batch(
                query_features, self.stored_features, self.stored_tasks, top_k=5)
            barcodes = np.zeros((len(query_features), self.dg.output_dim), dtype=np.float32)
            for tid_val in np.unique(inferred):
                mask = inferred == tid_val
                t = tid_val if self.use_modular else 0
                barcodes[mask] = self.dg.encode_batch(query_features[mask], t)
        b_scores = self._barcode_scores_batch(barcodes)
        combined = self._combine_batch(c_scores, b_scores, lam)
        return self.stored_labels[np.argmax(combined, axis=1)]

    def predict_content_only_batch(self, query_features):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        c_scores = self._content_scores_batch(query_features)
        return self.stored_labels[np.argmax(c_scores, axis=1)]

    def _content_scores_batch(self, Q):
        Q = Q.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / q_norms
        E = self.stored_features.astype(np.float32)
        e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / e_norms
        return (Q @ E.T).astype(np.float32)

    def _barcode_scores_batch(self, Q_bc):
        q_norms = np.maximum(np.linalg.norm(Q_bc, axis=1, keepdims=True), 1e-8)
        Q = Q_bc / q_norms
        B = self.stored_barcodes.astype(np.float32)
        b_norms = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-8)
        B = B / b_norms
        return (Q @ B.T).astype(np.float32)

    def _combine_batch(self, C, B, lam):
        c_min = C.min(axis=1, keepdims=True)
        c_max = C.max(axis=1, keepdims=True)
        cr = c_max - c_min
        nc = np.where(cr > 1e-8, (C - c_min) / cr, np.ones_like(C) / C.shape[1])
        b_min = B.min(axis=1, keepdims=True)
        b_max = B.max(axis=1, keepdims=True)
        br = b_max - b_min
        nb = np.where(br > 1e-8, (B - b_min) / br, np.ones_like(B) / B.shape[1])
        return (lam * nc + (1.0 - lam) * nb).astype(np.float32)


class KNNMemory:
    def __init__(self, feature_dim=400, k=5):
        self.k = k
        self.feature_dim = feature_dim
        self.stored_features = np.zeros((0, feature_dim), dtype=np.float32)
        self.stored_labels = np.zeros(0, dtype=np.int32)

    def store(self, features, labels, task_id=None, **kwargs):
        self.stored_features = np.concatenate([self.stored_features, features], axis=0)
        self.stored_labels = np.concatenate([self.stored_labels, labels], axis=0)

    def predict_batch(self, query_features, **kwargs):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        Q = query_features.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / q_norms
        E = self.stored_features.astype(np.float32)
        e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / e_norms
        sims = Q @ E.T
        preds = np.zeros(len(Q), dtype=np.int32)
        for i in range(len(Q)):
            k = min(self.k, sims.shape[1])
            top_idx = np.argpartition(sims[i], -k)[-k:]
            top_labels = self.stored_labels[top_idx]
            top_sims = sims[i, top_idx]
            label_scores = {}
            for j, lbl in enumerate(top_labels):
                label_scores[lbl] = label_scores.get(lbl, 0.0) + top_sims[j]
            preds[i] = max(label_scores, key=label_scores.get)
        return preds


class EWC:
    def __init__(self, model: nn.Module, lambda_ewc=5000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def compute_fisher(self, dataloader, task_id, model_forward):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        self.model.eval()
        for data, target in dataloader:
            self.model.zero_grad()
            output = model_forward(data, task_id)
            target_in_task = target % 2
            loss = F.cross_entropy(output, target_in_task)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * len(data)
        n_samples = len(dataloader.dataset)
        for n in fisher:
            fisher[n] /= max(n_samples, 1)
        return fisher

    def update(self, dataloader, task_id, model_forward):
        new_fisher = self.compute_fisher(dataloader, task_id, model_forward)
        if not self.fisher:
            self.fisher = new_fisher
        else:
            for n in self.fisher:
                self.fisher[n] += new_fisher[n]
        self.optimal_params = {n: p.clone() for n, p in self.model.named_parameters()
                               if p.requires_grad}

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher and p.requires_grad:
                loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.lambda_ewc * loss


def select_exemplars(features, labels, n_per_class=20, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    selected_features = []
    selected_labels = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) <= n_per_class:
            selected_features.append(features[idx])
            selected_labels.append(labels[idx])
        else:
            chosen = rng.choice(idx, n_per_class, replace=False)
            selected_features.append(features[chosen])
            selected_labels.append(labels[chosen])
    return np.concatenate(selected_features), np.concatenate(selected_labels)


def extract_features(model, dataloader):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for data, target in dataloader:
            feat = model.get_features(data)
            features_list.append(feat.numpy())
            labels_list.append(target.numpy())
    return np.concatenate(features_list).astype(np.float32), np.concatenate(labels_list).astype(np.int32)


def run_split_mnist():
    log("=" * 76)
    log("  SPLIT-MNIST: Standard Continual Learning Benchmark")
    log("  With Per-Task Heads + Replay + Memory-Based Classification")
    log("=" * 76)
    log()
    log("  Key Design Choices:")
    log("  - Per-task output heads (2-class each) - standard in CL literature")
    log("  - Replay: stored exemplars mixed into training of new tasks")
    log("  - Memory-based classification at test time (task-agnostic)")
    log("  - Comparison: kNN vs Dual-Channel retrieval")
    log()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    log("  Loading MNIST...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    n_tasks = 5
    n_exemplars = 25
    n_epochs = 3
    hidden_dim = 400

    task_train_data = []
    task_test_data = []
    for task_id, (label_a, label_b) in enumerate(task_labels):
        train_idx = [i for i, (_, y) in enumerate(train_dataset) if y in (label_a, label_b)]
        test_idx = [i for i, (_, y) in enumerate(test_dataset) if y in (label_a, label_b)]
        task_train_data.append(train_idx)
        task_test_data.append(test_idx)
        log(f"  Task {task_id+1} ({label_a},{label_b}): {len(train_idx)} train, {len(test_idx)} test")

    all_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    results = {}

    # ============================================================
    # Method 1: Fine-tuning with per-task heads
    # ============================================================
    log("\n  --- Method 1: Fine-tuning (Naive) ---")
    model_naive = MLP(hidden_dim=hidden_dim, n_tasks=n_tasks)
    optimizer_naive = optim.Adam(model_naive.parameters(), lr=0.001)
    naive_accs = []

    for task_id in range(n_tasks):
        train_idx = task_train_data[task_id]
        subset = torch.utils.data.Subset(train_dataset, train_idx)
        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)

        model_naive.train()
        for epoch in range(n_epochs):
            for data, target in loader:
                optimizer_naive.zero_grad()
                target_in_task = target % 2
                output = model_naive(data, task_id)
                loss = F.cross_entropy(output, target_in_task)
                loss.backward()
                optimizer_naive.step()

        task_accs = []
        for t in range(task_id + 1):
            test_idx = task_test_data[t]
            test_subset = torch.utils.data.Subset(test_dataset, test_idx)
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1000, shuffle=False)
            model_naive.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model_naive(data, t)
                    pred = output.argmax(dim=1)
                    correct += (pred == (target % 2)).sum().item()
                    total += len(target)
            task_accs.append(correct / max(total, 1))

        aa = np.mean(task_accs)
        naive_accs.append(aa)
        log(f"    After task {task_id+1}: AA={aa:.4f}, per-task={[f'{a:.3f}' for a in task_accs]}")

    results["Fine-tune"] = naive_accs

    # ============================================================
    # Method 2: EWC with per-task heads
    # ============================================================
    log("\n  --- Method 2: EWC ---")
    model_ewc = MLP(hidden_dim=hidden_dim, n_tasks=n_tasks)
    optimizer_ewc = optim.Adam(model_ewc.parameters(), lr=0.001)
    ewc = EWC(model_ewc, lambda_ewc=5000)
    ewc_accs = []

    for task_id in range(n_tasks):
        train_idx = task_train_data[task_id]
        subset = torch.utils.data.Subset(train_dataset, train_idx)
        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)

        model_ewc.train()
        for epoch in range(n_epochs):
            for data, target in loader:
                optimizer_ewc.zero_grad()
                target_in_task = target % 2
                output = model_ewc(data, task_id)
                loss = F.cross_entropy(output, target_in_task) + ewc.penalty()
                loss.backward()
                optimizer_ewc.step()

        ewc.update(loader, task_id, lambda d, t: model_ewc(d, t))

        task_accs = []
        for t in range(task_id + 1):
            test_idx = task_test_data[t]
            test_subset = torch.utils.data.Subset(test_dataset, test_idx)
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1000, shuffle=False)
            model_ewc.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model_ewc(data, t)
                    pred = output.argmax(dim=1)
                    correct += (pred == (target % 2)).sum().item()
                    total += len(target)
            task_accs.append(correct / max(total, 1))

        aa = np.mean(task_accs)
        ewc_accs.append(aa)
        log(f"    After task {task_id+1}: AA={aa:.4f}, per-task={[f'{a:.3f}' for a in task_accs]}")

    results["EWC"] = ewc_accs

    # ============================================================
    # Method 3: Replay + kNN Memory (task-agnostic)
    # ============================================================
    log("\n  --- Method 3: Replay + kNN Memory ---")
    model_knn = MLP(hidden_dim=hidden_dim, n_tasks=n_tasks)
    optimizer_knn = optim.Adam(model_knn.parameters(), lr=0.001)
    knn_memory = KNNMemory(feature_dim=hidden_dim, k=5)
    knn_accs = []
    exemplar_rng = np.random.RandomState(42)

    for task_id in range(n_tasks):
        train_idx = task_train_data[task_id]
        subset = torch.utils.data.Subset(train_dataset, train_idx)
        loader = torch.utils.data.Subset(subset, range(len(subset)))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)

        model_knn.train()
        for epoch in range(n_epochs):
            for data, target in train_loader:
                optimizer_knn.zero_grad()
                target_in_task = target % 2
                output = model_knn(data, task_id)
                loss = F.cross_entropy(output, target_in_task)

                if len(knn_memory.stored_features) > 0:
                    n_replay = min(32, len(knn_memory.stored_features))
                    replay_idx = exemplar_rng.choice(len(knn_memory.stored_features), n_replay, replace=False)
                    replay_feat = torch.tensor(knn_memory.stored_features[replay_idx], dtype=torch.float32)
                    replay_lbl = torch.tensor(knn_memory.stored_labels[replay_idx], dtype=torch.long)
                    replay_tasks = knn_memory.stored_tasks[replay_idx] if hasattr(knn_memory, 'stored_tasks') else np.zeros(n_replay, dtype=np.int32)

                loss.backward()
                optimizer_knn.step()

        train_feat, train_lbl = extract_features(model_knn, train_loader)
        train_feat = train_feat / np.maximum(np.linalg.norm(train_feat, axis=1, keepdims=True), 1e-8)
        exem_feat, exem_lbl = select_exemplars(train_feat, train_lbl, n_per_class=n_exemplars, rng=exemplar_rng)
        knn_memory.store(exem_feat, exem_lbl, task_id)

        all_feat, all_lbl = extract_features(model_knn, all_test_loader)
        all_feat = all_feat / np.maximum(np.linalg.norm(all_feat, axis=1, keepdims=True), 1e-8)
        preds = knn_memory.predict_batch(all_feat)

        task_accs = []
        for t in range(task_id + 1):
            label_a, label_b = task_labels[t]
            mask = np.isin(all_lbl, [label_a, label_b])
            if mask.sum() > 0:
                task_accs.append(np.mean(preds[mask] == all_lbl[mask]))
            else:
                task_accs.append(0.0)

        aa = np.mean(task_accs)
        knn_accs.append(aa)
        log(f"    After task {task_id+1}: AA={aa:.4f}, per-task={[f'{a:.3f}' for a in task_accs]}")

    results["kNN-Mem"] = knn_accs

    # ============================================================
    # Method 4: Replay + Ours (Modular-DG + Dual-Channel)
    # ============================================================
    log("\n  --- Method 4: Replay + Ours (Modular-DG + Dual-Channel) ---")
    model_ours = MLP(hidden_dim=hidden_dim, n_tasks=n_tasks)
    optimizer_ours = optim.Adam(model_ours.parameters(), lr=0.001)
    memory = DualChannelMemory(
        feature_dim=hidden_dim, barcode_dim=512, barcode_sparsity=64,
        lambda_param=0.7, use_modular=True, seed=42
    )
    ours_accs = []
    ours_content_accs = []
    exemplar_rng2 = np.random.RandomState(42)

    for task_id in range(n_tasks):
        train_idx = task_train_data[task_id]
        subset = torch.utils.data.Subset(train_dataset, train_idx)
        train_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)

        model_ours.train()
        for epoch in range(n_epochs):
            for data, target in train_loader:
                optimizer_ours.zero_grad()
                target_in_task = target % 2
                output = model_ours(data, task_id)
                loss = F.cross_entropy(output, target_in_task)
                loss.backward()
                optimizer_ours.step()

        train_feat, train_lbl = extract_features(model_ours, train_loader)
        train_feat = train_feat / np.maximum(np.linalg.norm(train_feat, axis=1, keepdims=True), 1e-8)
        exem_feat, exem_lbl = select_exemplars(train_feat, train_lbl, n_per_class=n_exemplars, rng=exemplar_rng2)
        memory.store(exem_feat, exem_lbl, task_id)

        all_feat, all_lbl = extract_features(model_ours, all_test_loader)
        all_feat = all_feat / np.maximum(np.linalg.norm(all_feat, axis=1, keepdims=True), 1e-8)

        preds_dual = memory.predict_batch(all_feat, task_ids=None, lambda_param=0.7)
        preds_content = memory.predict_content_only_batch(all_feat)

        task_accs_dual = []
        task_accs_content = []
        for t in range(task_id + 1):
            label_a, label_b = task_labels[t]
            mask = np.isin(all_lbl, [label_a, label_b])
            if mask.sum() > 0:
                task_accs_dual.append(np.mean(preds_dual[mask] == all_lbl[mask]))
                task_accs_content.append(np.mean(preds_content[mask] == all_lbl[mask]))
            else:
                task_accs_dual.append(0.0)
                task_accs_content.append(0.0)

        aa_dual = np.mean(task_accs_dual)
        aa_content = np.mean(task_accs_content)
        ours_accs.append(aa_dual)
        ours_content_accs.append(aa_content)
        log(f"    After task {task_id+1}: dual_AA={aa_dual:.4f}, content_AA={aa_content:.4f}")
        log(f"      per-task dual: {[f'{a:.3f}' for a in task_accs_dual]}")

    results["Ours"] = ours_accs

    # ============================================================
    # Method 5: Replay + Shared-DG (ablation)
    # ============================================================
    log("\n  --- Method 5: Replay + Shared-DG (ablation) ---")
    model_shared = MLP(hidden_dim=hidden_dim, n_tasks=n_tasks)
    optimizer_shared = optim.Adam(model_shared.parameters(), lr=0.001)
    memory_shared = DualChannelMemory(
        feature_dim=hidden_dim, barcode_dim=512, barcode_sparsity=64,
        lambda_param=0.7, use_modular=False, seed=42
    )
    shared_accs = []
    exemplar_rng3 = np.random.RandomState(42)

    for task_id in range(n_tasks):
        train_idx = task_train_data[task_id]
        subset = torch.utils.data.Subset(train_dataset, train_idx)
        train_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)

        model_shared.train()
        for epoch in range(n_epochs):
            for data, target in train_loader:
                optimizer_shared.zero_grad()
                target_in_task = target % 2
                output = model_shared(data, task_id)
                loss = F.cross_entropy(output, target_in_task)
                loss.backward()
                optimizer_shared.step()

        train_feat, train_lbl = extract_features(model_shared, train_loader)
        train_feat = train_feat / np.maximum(np.linalg.norm(train_feat, axis=1, keepdims=True), 1e-8)
        exem_feat, exem_lbl = select_exemplars(train_feat, train_lbl, n_per_class=n_exemplars, rng=exemplar_rng3)
        memory_shared.store(exem_feat, exem_lbl, task_id)

        all_feat, all_lbl = extract_features(model_shared, all_test_loader)
        all_feat = all_feat / np.maximum(np.linalg.norm(all_feat, axis=1, keepdims=True), 1e-8)
        preds = memory_shared.predict_batch(all_feat, task_ids=None, lambda_param=0.7)

        task_accs = []
        for t in range(task_id + 1):
            label_a, label_b = task_labels[t]
            mask = np.isin(all_lbl, [label_a, label_b])
            if mask.sum() > 0:
                task_accs.append(np.mean(preds[mask] == all_lbl[mask]))
            else:
                task_accs.append(0.0)

        aa = np.mean(task_accs)
        shared_accs.append(aa)
        log(f"    After task {task_id+1}: AA={aa:.4f}, per-task={[f'{a:.3f}' for a in task_accs]}")

    results["Shared-DG"] = shared_accs

    # ============================================================
    # Results Summary
    # ============================================================
    log("\n" + "=" * 76)
    log("  RESULTS: Split-MNIST Continual Learning")
    log("=" * 76)

    methods = ["Fine-tune", "EWC", "kNN-Mem", "Shared-DG", "Ours"]
    header = f"  {'After Task':>12s}"
    for m in methods:
        header += f" | {m:>10s}"
    log(header)
    log("  " + "-" * (12 + 13 * len(methods)))

    for t in range(n_tasks):
        row = f"  {t+1:>12d}"
        for m in methods:
            row += f" | {results[m][t]:10.4f}"
        log(row)

    log(f"\n  Final Average Accuracy (after all 5 tasks):")
    for m in methods:
        log(f"    {m:12s}: {results[m][-1]:.4f}")

    # ============================================================
    # Per-Task Accuracy after all tasks (Ours)
    # ============================================================
    log(f"\n  Per-Task Accuracy (Ours, after all 5 tasks):")
    all_feat, all_lbl = extract_features(model_ours, all_test_loader)
    all_feat = all_feat / np.maximum(np.linalg.norm(all_feat, axis=1, keepdims=True), 1e-8)
    preds_dual = memory.predict_batch(all_feat, task_ids=None, lambda_param=0.7)
    preds_content = memory.predict_content_only_batch(all_feat)

    for task_id in range(n_tasks):
        label_a, label_b = task_labels[task_id]
        mask = np.isin(all_lbl, [label_a, label_b])
        dual_acc = np.mean(preds_dual[mask] == all_lbl[mask])
        content_acc = np.mean(preds_content[mask] == all_lbl[mask])
        log(f"    Task {task_id+1} ({label_a},{label_b}): dual={dual_acc:.4f}, content_only={content_acc:.4f}")

    # ============================================================
    # Module Detection Accuracy
    # ============================================================
    log(f"\n  Module Detection Accuracy (content-guided routing):")
    inferred = memory.dg.infer_modules_batch(
        all_feat, memory.stored_features, memory.stored_tasks, top_k=5)
    true_tasks = np.zeros(len(all_lbl), dtype=np.int32)
    for task_id in range(n_tasks):
        label_a, label_b = task_labels[task_id]
        mask = np.isin(all_lbl, [label_a, label_b])
        true_tasks[mask] = task_id
    detection_acc = np.mean(inferred == true_tasks)
    log(f"    Overall: {detection_acc:.4f}")

    # ============================================================
    # KEY METRICS
    # ============================================================
    log(f"\n  KEY METRICS FOR PAPER:")
    log(f"  1. Ours vs Fine-tune:  {results['Ours'][-1] - results['Fine-tune'][-1]:+.4f}")
    log(f"  2. Ours vs kNN-Mem:    {results['Ours'][-1] - results['kNN-Mem'][-1]:+.4f}")
    log(f"  3. Ours vs Shared-DG:  {results['Ours'][-1] - results['Shared-DG'][-1]:+.4f}")
    log(f"  4. Ours vs EWC:        {results['Ours'][-1] - results['EWC'][-1]:+.4f}")
    log(f"  5. Module detection:    {detection_acc:.4f}")

    return results


if __name__ == "__main__":
    run_split_mnist()
