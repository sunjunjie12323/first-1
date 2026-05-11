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


def run_experiment_1_retrieval_quality():
    log("=" * 76)
    log("  EXPERIMENT 1: Retrieval Quality Under Interference")
    log("  (Fixed feature extractor, varying memory load & noise)")
    log("=" * 76)
    log()
    log("  Core question: Does dual-channel retrieval outperform")
    log("  content-only (kNN) retrieval when memory is large and noisy?")
    log()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    log("  Loading MNIST...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    log("  Pre-training feature extractor on ALL MNIST digits...")
    model = MLP(hidden_dim=256, output_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

    log("  Extracting features...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    all_features, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            feat = model.get_features(data)
            all_features.append(feat.numpy())
            all_labels.append(target.numpy())
    all_features = np.concatenate(all_features).astype(np.float32)
    all_labels = np.concatenate(all_labels).astype(np.int32)
    all_features = normalize_features(all_features)
    log(f"  Features: {all_features.shape}, Labels: {all_labels.shape}")

    rng = np.random.RandomState(42)

    # ============================================================
    # Test 1a: Varying memory size (items per class)
    # ============================================================
    log("\n  --- Test 1a: Varying Memory Size ---")
    log("  (10 classes, noise=0.0, query=200 per class)")

    items_per_class_list = [5, 10, 20, 50, 100, 200]
    noise_level = 0.0
    n_query_per_class = 200

    results_1a = {"kNN": [], "Content": [], "Ours": [], "Shared-DG": []}

    for n_items in items_per_class_list:
        stored_idx = []
        query_idx = []
        for lbl in range(10):
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            stored_idx.extend(class_idx[:n_items])
            query_idx.extend(class_idx[n_items:n_items + n_query_per_class])

        stored_feat = all_features[stored_idx]
        stored_lbl = all_labels[stored_idx]
        query_feat = all_features[query_idx]
        query_lbl = all_labels[query_idx]

        query_feat_noisy = add_noise(query_feat, noise_level, rng)

        task_ids = stored_lbl // 2

        memory_ours = DualChannelMemory(
            feature_dim=256, barcode_dim=512, barcode_sparsity=64,
            lambda_param=0.7, use_modular=True, seed=42)
        memory_ours.store(stored_feat, stored_lbl, task_ids[0])
        for i in range(1, len(stored_feat)):
            memory_ours.stored_tasks[i] = task_ids[i]
        memory_ours.stored_barcodes = np.zeros((len(stored_feat), 512), dtype=np.float32)
        for tid in np.unique(task_ids):
            mask = task_ids == tid
            memory_ours.stored_barcodes[mask] = memory_ours.dg.encode_batch(stored_feat[mask], tid)

        preds_ours = memory_ours.predict_batch(query_feat_noisy, task_ids=None, lambda_param=0.7)
        preds_content = memory_ours.predict_content_only_batch(query_feat_noisy)

        memory_shared = DualChannelMemory(
            feature_dim=256, barcode_dim=512, barcode_sparsity=64,
            lambda_param=0.7, use_modular=False, seed=42)
        memory_shared.store(stored_feat, stored_lbl, 0)

        preds_shared = memory_shared.predict_batch(query_feat_noisy, task_ids=None, lambda_param=0.7)

        Q = query_feat_noisy.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / q_norms
        E = stored_feat.astype(np.float32)
        e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / e_norms
        sims = Q @ E.T
        preds_knn = np.zeros(len(Q), dtype=np.int32)
        for i in range(len(Q)):
            k = min(5, sims.shape[1])
            top_idx = np.argpartition(sims[i], -k)[-k:]
            top_labels = stored_lbl[top_idx]
            top_sims = sims[i, top_idx]
            label_scores = {}
            for j, lbl in enumerate(top_labels):
                label_scores[lbl] = label_scores.get(lbl, 0.0) + top_sims[j]
            preds_knn[i] = max(label_scores, key=label_scores.get)

        acc_knn = np.mean(preds_knn == query_lbl)
        acc_content = np.mean(preds_content == query_lbl)
        acc_ours = np.mean(preds_ours == query_lbl)
        acc_shared = np.mean(preds_shared == query_lbl)

        results_1a["kNN"].append(acc_knn)
        results_1a["Content"].append(acc_content)
        results_1a["Ours"].append(acc_ours)
        results_1a["Shared-DG"].append(acc_shared)

        log(f"    {n_items:>4d}/class: kNN={acc_knn:.4f}, Content={acc_content:.4f}, "
            f"Ours={acc_ours:.4f}, Shared={acc_shared:.4f}")

    # ============================================================
    # Test 1b: Varying noise level
    # ============================================================
    log("\n  --- Test 1b: Varying Query Noise ---")
    log("  (10 classes, 50 items/class, query=200 per class)")

    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    n_items = 50

    stored_idx = []
    query_idx = []
    for lbl in range(10):
        class_idx = np.where(all_labels == lbl)[0]
        rng.shuffle(class_idx)
        stored_idx.extend(class_idx[:n_items])
        query_idx.extend(class_idx[n_items:n_items + n_query_per_class])

    stored_feat = all_features[stored_idx]
    stored_lbl = all_labels[stored_idx]
    query_feat = all_features[query_idx]
    query_lbl = all_labels[query_idx]
    task_ids = stored_lbl // 2

    results_1b = {"kNN": [], "Content": [], "Ours": [], "Shared-DG": []}

    for noise in noise_levels:
        query_feat_noisy = add_noise(query_feat, noise, rng)

        memory_ours = DualChannelMemory(
            feature_dim=256, barcode_dim=512, barcode_sparsity=64,
            lambda_param=0.7, use_modular=True, seed=42)
        memory_ours.store(stored_feat, stored_lbl, 0)
        for i in range(len(stored_feat)):
            memory_ours.stored_tasks[i] = task_ids[i]
        memory_ours.stored_barcodes = np.zeros((len(stored_feat), 512), dtype=np.float32)
        for tid in np.unique(task_ids):
            mask = task_ids == tid
            memory_ours.stored_barcodes[mask] = memory_ours.dg.encode_batch(stored_feat[mask], tid)

        preds_ours = memory_ours.predict_batch(query_feat_noisy, task_ids=None, lambda_param=0.7)
        preds_content = memory_ours.predict_content_only_batch(query_feat_noisy)

        memory_shared = DualChannelMemory(
            feature_dim=256, barcode_dim=512, barcode_sparsity=64,
            lambda_param=0.7, use_modular=False, seed=42)
        memory_shared.store(stored_feat, stored_lbl, 0)

        preds_shared = memory_shared.predict_batch(query_feat_noisy, task_ids=None, lambda_param=0.7)

        Q = query_feat_noisy.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / q_norms
        E = stored_feat.astype(np.float32)
        e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / e_norms
        sims = Q @ E.T
        preds_knn = np.zeros(len(Q), dtype=np.int32)
        for i in range(len(Q)):
            k = min(5, sims.shape[1])
            top_idx = np.argpartition(sims[i], -k)[-k:]
            top_labels = stored_lbl[top_idx]
            top_sims = sims[i, top_idx]
            label_scores = {}
            for j, lbl in enumerate(top_labels):
                label_scores[lbl] = label_scores.get(lbl, 0.0) + top_sims[j]
            preds_knn[i] = max(label_scores, key=label_scores.get)

        acc_knn = np.mean(preds_knn == query_lbl)
        acc_content = np.mean(preds_content == query_lbl)
        acc_ours = np.mean(preds_ours == query_lbl)
        acc_shared = np.mean(preds_shared == query_lbl)

        results_1b["kNN"].append(acc_knn)
        results_1b["Content"].append(acc_content)
        results_1b["Ours"].append(acc_ours)
        results_1b["Shared-DG"].append(acc_shared)

        log(f"    noise={noise:.1f}: kNN={acc_knn:.4f}, Content={acc_content:.4f}, "
            f"Ours={acc_ours:.4f}, Shared={acc_shared:.4f}")

    # ============================================================
    # Test 1c: Cross-task interference (simulated CL)
    # ============================================================
    log("\n  --- Test 1c: Cross-Task Interference ---")
    log("  (Sequential task addition, fixed features)")
    log("  (5 tasks x 2 digits, 50 items/class, noise=0.3)")

    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    noise_level_c = 0.3
    n_items_c = 50

    results_1c = {"kNN": [], "Content": [], "Ours": [], "Shared-DG": []}

    memory_ours_c = DualChannelMemory(
        feature_dim=256, barcode_dim=512, barcode_sparsity=64,
        lambda_param=0.7, use_modular=True, seed=42)
    memory_shared_c = DualChannelMemory(
        feature_dim=256, barcode_dim=512, barcode_sparsity=64,
        lambda_param=0.7, use_modular=False, seed=42)
    knn_stored_feat = np.zeros((0, 256), dtype=np.float32)
    knn_stored_lbl = np.zeros(0, dtype=np.int32)

    for task_id, (label_a, label_b) in enumerate(task_labels):
        for lbl in [label_a, label_b]:
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_items_c]
            feat = all_features[idx]
            labels = all_labels[idx]

            memory_ours_c.store(feat, labels, task_id)
            memory_shared_c.store(feat, labels, 0)
            knn_stored_feat = np.concatenate([knn_stored_feat, feat], axis=0)
            knn_stored_lbl = np.concatenate([knn_stored_lbl, labels], axis=0)

        query_idx_c = []
        for t in range(task_id + 1):
            for lbl in task_labels[t]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                query_idx_c.extend(class_idx[n_items_c:n_items_c + 100])

        query_feat_c = all_features[query_idx_c]
        query_lbl_c = all_labels[query_idx_c]
        query_feat_noisy_c = add_noise(query_feat_c, noise_level_c, rng)

        preds_ours = memory_ours_c.predict_batch(query_feat_noisy_c, task_ids=None, lambda_param=0.7)
        preds_content = memory_ours_c.predict_content_only_batch(query_feat_noisy_c)
        preds_shared = memory_shared_c.predict_batch(query_feat_noisy_c, task_ids=None, lambda_param=0.7)

        Q = query_feat_noisy_c.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / q_norms
        E = knn_stored_feat.astype(np.float32)
        e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / e_norms
        sims = Q @ E.T
        preds_knn = np.zeros(len(Q), dtype=np.int32)
        for i in range(len(Q)):
            k = min(5, sims.shape[1])
            top_idx = np.argpartition(sims[i], -k)[-k:]
            top_labels = knn_stored_lbl[top_idx]
            top_sims = sims[i, top_idx]
            label_scores = {}
            for j, lbl in enumerate(top_labels):
                label_scores[lbl] = label_scores.get(lbl, 0.0) + top_sims[j]
            preds_knn[i] = max(label_scores, key=label_scores.get)

        acc_knn = np.mean(preds_knn == query_lbl_c)
        acc_content = np.mean(preds_content == query_lbl_c)
        acc_ours = np.mean(preds_ours == query_lbl_c)
        acc_shared = np.mean(preds_shared == query_lbl_c)

        results_1c["kNN"].append(acc_knn)
        results_1c["Content"].append(acc_content)
        results_1c["Ours"].append(acc_ours)
        results_1c["Shared-DG"].append(acc_shared)

        log(f"    After task {task_id+1}: kNN={acc_knn:.4f}, Content={acc_content:.4f}, "
            f"Ours={acc_ours:.4f}, Shared={acc_shared:.4f}")

    # ============================================================
    # Test 1d: Lambda sensitivity
    # ============================================================
    log("\n  --- Test 1d: Lambda Sensitivity ---")
    log("  (5 tasks, 50 items/class, noise=0.3)")

    lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]

    query_idx_d = []
    for task_id, (label_a, label_b) in enumerate(task_labels):
        for lbl in [label_a, label_b]:
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            query_idx_d.extend(class_idx[n_items_c:n_items_c + 100])

    query_feat_d = all_features[query_idx_d]
    query_lbl_d = all_labels[query_idx_d]
    query_feat_noisy_d = add_noise(query_feat_d, noise_level_c, rng)

    log(f"    {'lambda':>8s} | {'Ours':>10s} | {'Shared':>10s}")
    log(f"    {'-'*8} | {'-'*10} | {'-'*10}")

    best_lam_ours = 0
    best_acc_ours = 0
    for lam in lambda_values:
        preds_ours = memory_ours_c.predict_batch(query_feat_noisy_d, task_ids=None, lambda_param=lam)
        preds_shared = memory_shared_c.predict_batch(query_feat_noisy_d, task_ids=None, lambda_param=lam)
        acc_ours = np.mean(preds_ours == query_lbl_d)
        acc_shared = np.mean(preds_shared == query_lbl_d)
        log(f"    {lam:>8.1f} | {acc_ours:>10.4f} | {acc_shared:>10.4f}")
        if acc_ours > best_acc_ours:
            best_acc_ours = acc_ours
            best_lam_ours = lam

    # ============================================================
    # Summary
    # ============================================================
    log("\n" + "=" * 76)
    log("  SUMMARY: Experiment 1 Results")
    log("=" * 76)

    log("\n  Test 1a: Memory Size vs Accuracy")
    log(f"    {'Items/cls':>10s} | {'kNN':>8s} | {'Content':>8s} | {'Ours':>8s} | {'Shared':>8s}")
    for i, n in enumerate(items_per_class_list):
        log(f"    {n:>10d} | {results_1a['kNN'][i]:>8.4f} | {results_1a['Content'][i]:>8.4f} | "
            f"{results_1a['Ours'][i]:>8.4f} | {results_1a['Shared-DG'][i]:>8.4f}")

    log("\n  Test 1b: Noise Level vs Accuracy")
    log(f"    {'Noise':>8s} | {'kNN':>8s} | {'Content':>8s} | {'Ours':>8s} | {'Shared':>8s}")
    for i, n in enumerate(noise_levels):
        log(f"    {n:>8.1f} | {results_1b['kNN'][i]:>8.4f} | {results_1b['Content'][i]:>8.4f} | "
            f"{results_1b['Ours'][i]:>8.4f} | {results_1b['Shared-DG'][i]:>8.4f}")

    log("\n  Test 1c: Cross-Task Interference (Sequential)")
    log(f"    {'Task':>6s} | {'kNN':>8s} | {'Content':>8s} | {'Ours':>8s} | {'Shared':>8s}")
    for i in range(5):
        log(f"    {i+1:>6d} | {results_1c['kNN'][i]:>8.4f} | {results_1c['Content'][i]:>8.4f} | "
            f"{results_1c['Ours'][i]:>8.4f} | {results_1c['Shared-DG'][i]:>8.4f}")

    log(f"\n  Best lambda for Ours: {best_lam_ours} (acc={best_acc_ours:.4f})")

    log(f"\n  KEY FINDINGS:")
    ours_vs_knn_1c = results_1c['Ours'][-1] - results_1c['kNN'][-1]
    ours_vs_content_1c = results_1c['Ours'][-1] - results_1c['Content'][-1]
    ours_vs_shared_1c = results_1c['Ours'][-1] - results_1c['Shared-DG'][-1]
    log(f"  1. Ours vs kNN (5 tasks, noise=0.3): {ours_vs_knn_1c:+.4f}")
    log(f"  2. Ours vs Content-only:             {ours_vs_content_1c:+.4f}")
    log(f"  3. Ours vs Shared-DG:                {ours_vs_shared_1c:+.4f}")

    return results_1a, results_1b, results_1c


if __name__ == "__main__":
    run_experiment_1_retrieval_quality()
