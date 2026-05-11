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
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_features(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DGModule:
    def __init__(self, input_dim, output_dim=512, sparsity=64, seed=0):
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(output_dim, input_dim).astype(np.float32)
        row_norms = np.linalg.norm(self.projection, axis=1, keepdims=True)
        self.projection = self.projection / np.maximum(row_norms, 1e-8)
        self.sparsity = sparsity
        self.input_dim = input_dim

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
    def __init__(self, input_dim, output_dim=512, sparsity=64, base_seed=42):
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

    def encode_all_modules_batch(self, X):
        results = {}
        for tid in self.modules:
            results[tid] = self.modules[tid].separate_batch(X)
        return results


class DualChannelMemoryV2:
    def __init__(self, feature_dim, barcode_dim=512, barcode_sparsity=64,
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
        self._rng = np.random.RandomState(seed)

    def store(self, features, labels, task_id):
        tid = task_id if self.use_modular else 0
        barcodes = self.dg.encode_batch(features, tid)
        self.stored_features = np.concatenate([self.stored_features, features], axis=0)
        self.stored_barcodes = np.concatenate([self.stored_barcodes, barcodes], axis=0)
        self.stored_labels = np.concatenate([self.stored_labels, labels], axis=0)
        self.stored_tasks = np.concatenate(
            [self.stored_tasks, np.full(len(labels), task_id, dtype=np.int32)], axis=0)

    def predict_multi_module(self, query_features, lambda_param=None):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        lam = lambda_param if lambda_param is not None else self.lambda_param

        c_scores = self._content_scores_batch(query_features)

        if not self.use_modular:
            b_scores = self._barcode_scores_batch(
                self.dg.encode_batch(query_features, 0))
            combined = self._combine_batch(c_scores, b_scores, lam)
            return self.stored_labels[np.argmax(combined, axis=1)]

        all_barcodes = self.dg.encode_all_modules_batch(query_features)

        n_query = len(query_features)
        n_stored = len(self.stored_features)
        best_combined = np.full((n_query, n_stored), -np.inf, dtype=np.float32)

        for tid, query_barcodes in all_barcodes.items():
            task_mask = self.stored_tasks == tid
            if not task_mask.any():
                continue

            b_sub = np.zeros((n_query, n_stored), dtype=np.float32)
            stored_bc = self.stored_barcodes[task_mask]
            q_norms = np.maximum(np.linalg.norm(query_barcodes, axis=1, keepdims=True), 1e-8)
            Q = query_barcodes / q_norms
            b_norms = np.maximum(np.linalg.norm(stored_bc, axis=1, keepdims=True), 1e-8)
            B = stored_bc / b_norms
            sims = (Q @ B.T).astype(np.float32)
            b_sub[:, task_mask] = sims

            combined = self._combine_batch(c_scores, b_sub, lam)
            best_combined = np.maximum(best_combined, combined)

        return self.stored_labels[np.argmax(best_combined, axis=1)]

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


def normalize_features(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X / norms


def add_noise(X, noise_level=0.0, rng=None):
    if noise_level <= 0:
        return X.copy()
    if rng is None:
        rng = np.random.RandomState(42)
    noise = rng.randn(*X.shape).astype(np.float32) * noise_level
    return X + noise


def knn_predict(query_features, stored_features, stored_labels, k=5):
    Q = query_features.astype(np.float32)
    q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
    Q = Q / q_norms
    E = stored_features.astype(np.float32)
    e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
    E = E / e_norms
    sims = Q @ E.T
    preds = np.zeros(len(Q), dtype=np.int32)
    for i in range(len(Q)):
        kk = min(k, sims.shape[1])
        top_idx = np.argpartition(sims[i], -kk)[-kk:]
        top_labels = stored_labels[top_idx]
        top_sims = sims[i, top_idx]
        label_scores = {}
        for j, lbl in enumerate(top_labels):
            label_scores[lbl] = label_scores.get(lbl, 0.0) + top_sims[j]
        preds[i] = max(label_scores, key=label_scores.get)
    return preds


def run_comprehensive_experiment():
    log("=" * 76)
    log("  COMPREHENSIVE EXPERIMENT: Dual-Channel Associative Memory")
    log("  Multi-Module Barcode Matching + Context-Dependent Retrieval")
    log("=" * 76)
    log()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    log("  Loading MNIST...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    log("  Pre-training feature extractor...")
    model = MLP(hidden_dim=256, output_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        n_batches = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        log(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    log("  Extracting features...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    all_features, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            feat = model.get_features(data)
            all_features.append(feat.numpy())
            all_labels.append(target.numpy())
    all_features = normalize_features(np.concatenate(all_features).astype(np.float32))
    all_labels = np.concatenate(all_labels).astype(np.int32)
    log(f"  Features: {all_features.shape}")

    rng = np.random.RandomState(42)

    # ============================================================
    # Experiment A: Cross-Task Interference (Sequential Task Addition)
    # With multi-module barcode matching
    # ============================================================
    log("\n" + "=" * 76)
    log("  Experiment A: Cross-Task Interference (Fixed Features + Noise)")
    log("=" * 76)

    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    noise_level = 0.3
    n_items_per_class = 50

    memory_ours = DualChannelMemoryV2(
        feature_dim=256, barcode_dim=512, barcode_sparsity=64,
        lambda_param=0.7, use_modular=True, seed=42)
    memory_shared = DualChannelMemoryV2(
        feature_dim=256, barcode_dim=512, barcode_sparsity=64,
        lambda_param=0.7, use_modular=False, seed=42)
    knn_stored_feat = np.zeros((0, 256), dtype=np.float32)
    knn_stored_lbl = np.zeros(0, dtype=np.int32)

    results_a = {"kNN": [], "Content": [], "Ours-V2": [], "Shared-DG": []}

    for task_id, (label_a, label_b) in enumerate(task_labels):
        for lbl in [label_a, label_b]:
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_items_per_class]
            feat = all_features[idx]
            labels = all_labels[idx]
            memory_ours.store(feat, labels, task_id)
            memory_shared.store(feat, labels, 0)
            knn_stored_feat = np.concatenate([knn_stored_feat, feat], axis=0)
            knn_stored_lbl = np.concatenate([knn_stored_lbl, labels], axis=0)

        query_idx = []
        for t in range(task_id + 1):
            for lbl in task_labels[t]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                query_idx.extend(class_idx[n_items_per_class:n_items_per_class + 100])

        query_feat = all_features[query_idx]
        query_lbl = all_labels[query_idx]
        query_noisy = add_noise(query_feat, noise_level, rng)

        preds_knn = knn_predict(query_noisy, knn_stored_feat, knn_stored_lbl, k=5)
        preds_content = memory_ours.predict_content_only_batch(query_noisy)
        preds_ours = memory_ours.predict_multi_module(query_noisy, lambda_param=0.7)
        preds_shared = memory_shared.predict_multi_module(query_noisy, lambda_param=0.7)

        acc_knn = np.mean(preds_knn == query_lbl)
        acc_content = np.mean(preds_content == query_lbl)
        acc_ours = np.mean(preds_ours == query_lbl)
        acc_shared = np.mean(preds_shared == query_lbl)

        results_a["kNN"].append(acc_knn)
        results_a["Content"].append(acc_content)
        results_a["Ours-V2"].append(acc_ours)
        results_a["Shared-DG"].append(acc_shared)

        log(f"    After task {task_id+1}: kNN={acc_knn:.4f}, Content={acc_content:.4f}, "
            f"Ours-V2={acc_ours:.4f}, Shared={acc_shared:.4f}")

    # ============================================================
    # Experiment B: Lambda Sensitivity (V2)
    # ============================================================
    log("\n  --- Lambda Sensitivity (V2, after 5 tasks) ---")

    query_idx_b = []
    for task_id, (label_a, label_b) in enumerate(task_labels):
        for lbl in [label_a, label_b]:
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            query_idx_b.extend(class_idx[n_items_per_class:n_items_per_class + 100])

    query_feat_b = all_features[query_idx_b]
    query_lbl_b = all_labels[query_idx_b]
    query_noisy_b = add_noise(query_feat_b, noise_level, rng)

    lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]

    log(f"    {'lambda':>8s} | {'Ours-V2':>10s} | {'Shared':>10s} | {'kNN':>8s}")
    log(f"    {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8}")

    best_lam = 0.5
    best_acc = 0
    for lam in lambda_values:
        preds_ours = memory_ours.predict_multi_module(query_noisy_b, lambda_param=lam)
        preds_shared = memory_shared.predict_multi_module(query_noisy_b, lambda_param=lam)
        acc_ours = np.mean(preds_ours == query_lbl_b)
        acc_shared = np.mean(preds_shared == query_lbl_b)
        acc_knn = np.mean(knn_predict(query_noisy_b, knn_stored_feat, knn_stored_lbl, k=5) == query_lbl_b)
        log(f"    {lam:>8.2f} | {acc_ours:>10.4f} | {acc_shared:>10.4f} | {acc_knn:>8.4f}")
        if acc_ours > best_acc:
            best_acc = acc_ours
            best_lam = lam

    # ============================================================
    # Experiment C: Varying Noise Levels
    # ============================================================
    log("\n  --- Varying Noise Levels (after 5 tasks) ---")

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    log(f"    {'noise':>8s} | {'kNN':>8s} | {'Content':>8s} | {'Ours-V2':>10s} | {'Shared':>8s}")
    log(f"    {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8}")

    results_c = {"kNN": [], "Content": [], "Ours-V2": [], "Shared-DG": []}
    for noise in noise_levels:
        query_noisy_c = add_noise(query_feat_b, noise, rng)
        preds_knn = knn_predict(query_noisy_c, knn_stored_feat, knn_stored_lbl, k=5)
        preds_content = memory_ours.predict_content_only_batch(query_noisy_c)
        preds_ours = memory_ours.predict_multi_module(query_noisy_c, lambda_param=best_lam)
        preds_shared = memory_shared.predict_multi_module(query_noisy_c, lambda_param=best_lam)

        acc_knn = np.mean(preds_knn == query_lbl_b)
        acc_content = np.mean(preds_content == query_lbl_b)
        acc_ours = np.mean(preds_ours == query_lbl_b)
        acc_shared = np.mean(preds_shared == query_lbl_b)

        results_c["kNN"].append(acc_knn)
        results_c["Content"].append(acc_content)
        results_c["Ours-V2"].append(acc_ours)
        results_c["Shared-DG"].append(acc_shared)

        log(f"    {noise:>8.1f} | {acc_knn:>8.4f} | {acc_content:>8.4f} | "
            f"{acc_ours:>10.4f} | {acc_shared:>8.4f}")

    # ============================================================
    # Experiment D: Varying Memory Size
    # ============================================================
    log("\n  --- Varying Memory Size (noise=0.3) ---")

    items_per_class_list = [5, 10, 20, 50, 100, 200]

    log(f"    {'items':>8s} | {'kNN':>8s} | {'Content':>8s} | {'Ours-V2':>10s} | {'Shared':>8s}")
    log(f"    {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8}")

    results_d = {"kNN": [], "Content": [], "Ours-V2": [], "Shared-DG": []}
    for n_items in items_per_class_list:
        stored_idx = []
        query_idx_d = []
        for lbl in range(10):
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            stored_idx.extend(class_idx[:n_items])
            query_idx_d.extend(class_idx[n_items:n_items + 100])

        stored_feat = all_features[stored_idx]
        stored_lbl = all_labels[stored_idx]
        query_feat_d = all_features[query_idx_d]
        query_lbl_d = all_labels[query_idx_d]
        query_noisy_d = add_noise(query_feat_d, 0.3, rng)
        task_ids = stored_lbl // 2

        mem_ours = DualChannelMemoryV2(
            feature_dim=256, barcode_dim=512, barcode_sparsity=64,
            lambda_param=best_lam, use_modular=True, seed=42)
        mem_shared = DualChannelMemoryV2(
            feature_dim=256, barcode_dim=512, barcode_sparsity=64,
            lambda_param=best_lam, use_modular=False, seed=42)

        for tid in np.unique(task_ids):
            mask = task_ids == tid
            mem_ours.store(stored_feat[mask], stored_lbl[mask], tid)
        mem_shared.store(stored_feat, stored_lbl, 0)

        preds_knn = knn_predict(query_noisy_d, stored_feat, stored_lbl, k=5)
        preds_content = mem_ours.predict_content_only_batch(query_noisy_d)
        preds_ours = mem_ours.predict_multi_module(query_noisy_d, lambda_param=best_lam)
        preds_shared = mem_shared.predict_multi_module(query_noisy_d, lambda_param=best_lam)

        acc_knn = np.mean(preds_knn == query_lbl_d)
        acc_content = np.mean(preds_content == query_lbl_d)
        acc_ours = np.mean(preds_ours == query_lbl_d)
        acc_shared = np.mean(preds_shared == query_lbl_d)

        results_d["kNN"].append(acc_knn)
        results_d["Content"].append(acc_content)
        results_d["Ours-V2"].append(acc_ours)
        results_d["Shared-DG"].append(acc_shared)

        log(f"    {n_items:>8d} | {acc_knn:>8.4f} | {acc_content:>8.4f} | "
            f"{acc_ours:>10.4f} | {acc_shared:>8.4f}")

    # ============================================================
    # Experiment E: Varying Barcode Dimension and Sparsity
    # ============================================================
    log("\n  --- Varying Barcode Parameters (5 tasks, noise=0.3) ---")

    barcode_configs = [
        (256, 32), (256, 64), (512, 64), (512, 128), (1024, 128), (1024, 256)
    ]

    log(f"    {'bc_dim':>8s} | {'sparsity':>8s} | {'Ours-V2':>10s} | {'Shared':>8s} | {'kNN':>8s}")
    log(f"    {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*8}")

    for bc_dim, sparsity in barcode_configs:
        mem_ours_e = DualChannelMemoryV2(
            feature_dim=256, barcode_dim=bc_dim, barcode_sparsity=sparsity,
            lambda_param=best_lam, use_modular=True, seed=42)
        mem_shared_e = DualChannelMemoryV2(
            feature_dim=256, barcode_dim=bc_dim, barcode_sparsity=sparsity,
            lambda_param=best_lam, use_modular=False, seed=42)

        for task_id, (label_a, label_b) in enumerate(task_labels):
            for lbl in [label_a, label_b]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_items_per_class]
                mem_ours_e.store(all_features[idx], all_labels[idx], task_id)
            for lbl in [label_a, label_b]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_items_per_class]
            mem_shared_e.store(
                all_features[np.where(np.isin(all_labels, [label_a, label_b]))[0][:n_items_per_class*2]],
                all_labels[np.where(np.isin(all_labels, [label_a, label_b]))[0][:n_items_per_class*2]],
                0)

        preds_ours = mem_ours_e.predict_multi_module(query_noisy_b, lambda_param=best_lam)
        preds_shared = mem_shared_e.predict_multi_module(query_noisy_b, lambda_param=best_lam)
        acc_ours = np.mean(preds_ours == query_lbl_b)
        acc_shared = np.mean(preds_shared == query_lbl_b)
        acc_knn = np.mean(knn_predict(query_noisy_b, knn_stored_feat, knn_stored_lbl, k=5) == query_lbl_b)

        log(f"    {bc_dim:>8d} | {sparsity:>8d} | {acc_ours:>10.4f} | {acc_shared:>8.4f} | {acc_knn:>8.4f}")

    # ============================================================
    # Summary
    # ============================================================
    log("\n" + "=" * 76)
    log("  COMPREHENSIVE SUMMARY")
    log("=" * 76)

    log("\n  Experiment A: Cross-Task Interference")
    log(f"    {'Task':>6s} | {'kNN':>8s} | {'Content':>8s} | {'Ours-V2':>10s} | {'Shared':>8s}")
    for i in range(5):
        log(f"    {i+1:>6d} | {results_a['kNN'][i]:>8.4f} | {results_a['Content'][i]:>8.4f} | "
            f"{results_a['Ours-V2'][i]:>10.4f} | {results_a['Shared-DG'][i]:>8.4f}")

    log(f"\n  Experiment C: Noise Robustness")
    log(f"    {'Noise':>8s} | {'kNN':>8s} | {'Content':>8s} | {'Ours-V2':>10s} | {'Shared':>8s}")
    for i, n in enumerate(noise_levels):
        log(f"    {n:>8.1f} | {results_c['kNN'][i]:>8.4f} | {results_c['Content'][i]:>8.4f} | "
            f"{results_c['Ours-V2'][i]:>10.4f} | {results_c['Shared-DG'][i]:>8.4f}")

    log(f"\n  Experiment D: Memory Size")
    log(f"    {'Items':>8s} | {'kNN':>8s} | {'Content':>8s} | {'Ours-V2':>10s} | {'Shared':>8s}")
    for i, n in enumerate(items_per_class_list):
        log(f"    {n:>8d} | {results_d['kNN'][i]:>8.4f} | {results_d['Content'][i]:>8.4f} | "
            f"{results_d['Ours-V2'][i]:>10.4f} | {results_d['Shared-DG'][i]:>8.4f}")

    log(f"\n  KEY METRICS:")
    log(f"  Best lambda: {best_lam}")
    log(f"  Ours-V2 vs kNN (5 tasks, noise=0.3):     {results_a['Ours-V2'][-1] - results_a['kNN'][-1]:+.4f}")
    log(f"  Ours-V2 vs Content-only:                  {results_a['Ours-V2'][-1] - results_a['Content'][-1]:+.4f}")
    log(f"  Ours-V2 vs Shared-DG:                     {results_a['Ours-V2'][-1] - results_a['Shared-DG'][-1]:+.4f}")

    return results_a, results_c, results_d


if __name__ == "__main__":
    run_comprehensive_experiment()
