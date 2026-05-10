from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

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


class OnlineContextDiscovery:
    """
    Online context discovery from environmental signals.
    
    Models the MEC (medial entorhinal cortex) pathway that provides
    spatial/temporal context to the hippocampus. In embodied intelligence,
    this corresponds to robot sensor readings (location, temperature, etc.)
    that indicate which 'context' the agent is operating in.
    
    Key innovation: Discovers contexts ONLINE without task ID,
    using a simple online clustering algorithm on environmental signals.
    When a new signal doesn't match any existing context (novelty detection),
    a new context is automatically created.
    """
    
    def __init__(self, env_dim=32, novelty_threshold=0.5, max_contexts=20, seed=42):
        self.env_dim = env_dim
        self.novelty_threshold = novelty_threshold
        self.max_contexts = max_contexts
        self.rng = np.random.RandomState(seed)
        
        self.context_centers = np.zeros((0, env_dim), dtype=np.float32)
        self.context_counts = np.zeros(0, dtype=np.int32)
        self.n_contexts = 0
    
    def discover(self, env_signal: np.ndarray) -> int:
        if len(self.context_centers) == 0:
            self._add_context(env_signal)
            return 0
        
        sims = self._compute_similarities(env_signal)
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        
        if best_sim < self.novelty_threshold:
            if self.n_contexts < self.max_contexts:
                self._add_context(env_signal)
                return self.n_contexts - 1
            else:
                return best_idx
        
        self.context_counts[best_idx] += 1
        alpha = 1.0 / max(self.context_counts[best_idx], 1)
        self.context_centers[best_idx] = (
            (1 - alpha) * self.context_centers[best_idx] + alpha * env_signal
        )
        return best_idx
    
    def discover_batch(self, env_signals: np.ndarray) -> np.ndarray:
        results = np.zeros(len(env_signals), dtype=np.int32)
        for i in range(len(env_signals)):
            results[i] = self.discover(env_signals[i])
        return results
    
    def _add_context(self, center: np.ndarray):
        new_center = center.reshape(1, -1).astype(np.float32)
        if len(self.context_centers) == 0:
            self.context_centers = new_center
        else:
            self.context_centers = np.concatenate([self.context_centers, new_center], axis=0)
        self.context_counts = np.concatenate([self.context_counts, [1]])
        self.n_contexts += 1
    
    def _compute_similarities(self, signal: np.ndarray) -> np.ndarray:
        signal_norm = np.linalg.norm(signal)
        if signal_norm < 1e-8:
            return np.zeros(len(self.context_centers), dtype=np.float32)
        signal_normalized = signal / signal_norm
        center_norms = np.maximum(
            np.linalg.norm(self.context_centers, axis=1), 1e-8)
        centers_normalized = self.context_centers / center_norms[:, np.newaxis]
        return (centers_normalized @ signal_normalized).astype(np.float32)


class ContextBarcodeEncoder:
    def __init__(self, max_contexts=20, barcode_dim=512, sparsity=64, seed=42):
        self.max_contexts = max_contexts
        self.barcode_dim = barcode_dim
        self.sparsity = sparsity
        self.rng = np.random.RandomState(seed)
        self.barcodes: Dict[int, np.ndarray] = {}
        self._next_seed = seed + 1000
    
    def get_barcode(self, context_id: int) -> np.ndarray:
        if context_id not in self.barcodes:
            rng = np.random.RandomState(self._next_seed + context_id * 137)
            raw = rng.randn(self.barcode_dim).astype(np.float32)
            top_idx = np.argpartition(raw, -self.sparsity)[-self.sparsity:]
            barcode = np.zeros(self.barcode_dim, dtype=np.float32)
            barcode[top_idx] = np.maximum(raw[top_idx], 0.0)
            norm = np.linalg.norm(barcode)
            if norm > 1e-8:
                barcode /= norm
            self.barcodes[context_id] = barcode
        return self.barcodes[context_id]
    
    def get_barcodes_batch(self, context_ids: np.ndarray) -> np.ndarray:
        result = np.zeros((len(context_ids), self.barcode_dim), dtype=np.float32)
        for i, cid in enumerate(context_ids):
            result[i] = self.get_barcode(int(cid))
        return result


class EnvironmentSimulator:
    def __init__(self, n_contexts, env_dim=32, signal_noise=0.1, seed=42):
        self.n_contexts = n_contexts
        self.env_dim = env_dim
        self.signal_noise = signal_noise
        self.rng = np.random.RandomState(seed)
        self.context_signals = self.rng.randn(n_contexts, env_dim).astype(np.float32)
        norms = np.maximum(np.linalg.norm(self.context_signals, axis=1, keepdims=True), 1e-8)
        self.context_signals /= norms
    
    def get_signal(self, context_id: int) -> np.ndarray:
        signal = self.context_signals[context_id % self.n_contexts].copy()
        if self.signal_noise > 0:
            signal += self.rng.randn(self.env_dim).astype(np.float32) * self.signal_noise
        return signal
    
    def get_signals_batch(self, context_ids: np.ndarray) -> np.ndarray:
        signals = self.context_signals[context_ids % self.n_contexts].copy()
        if self.signal_noise > 0:
            signals += self.rng.randn(*signals.shape).astype(np.float32) * self.signal_noise
        return signals


class MEC_DG_Memory:
    """
    MEC-DG Dual-Channel Associative Memory
    
    Architecture (bio-inspired):
    - LEC (lateral entorhinal cortex) → Content features (what)
    - MEC (medial entorhinal cortex) → Environmental context (where/when)
    - DG (dentate gyrus) → Sparse barcode from MEC context
    - CA3 → Dual-channel retrieval (content + barcode)
    
    Key innovation: Context is discovered ONLINE from environmental signals,
    without requiring task ID. This models the brain's ability to use
    spatial/temporal context for memory disambiguation.
    """
    
    def __init__(self, feature_dim=256, barcode_dim=512, sparsity=64,
                 env_dim=32, novelty_threshold=0.5, lambda_param=0.5,
                 max_contexts=20, seed=42):
        self.feature_dim = feature_dim
        self.lambda_param = lambda_param
        self.context_discovery = OnlineContextDiscovery(
            env_dim, novelty_threshold, max_contexts, seed)
        self.barcode_encoder = ContextBarcodeEncoder(
            max_contexts, barcode_dim, sparsity, seed)
        
        self.stored_features = np.zeros((0, feature_dim), dtype=np.float32)
        self.stored_barcodes = np.zeros((0, barcode_dim), dtype=np.float32)
        self.stored_labels = np.zeros(0, dtype=np.int32)
        self.stored_contexts = np.zeros(0, dtype=np.int32)
    
    def store_with_env(self, features, labels, env_signals):
        discovered_contexts = self.context_discovery.discover_batch(env_signals)
        barcodes = self.barcode_encoder.get_barcodes_batch(discovered_contexts)
        
        self.stored_features = np.concatenate([self.stored_features, features], axis=0)
        self.stored_barcodes = np.concatenate([self.stored_barcodes, barcodes], axis=0)
        self.stored_labels = np.concatenate([self.stored_labels, labels], axis=0)
        self.stored_contexts = np.concatenate(
            [self.stored_contexts, discovered_contexts], axis=0)
        
        return discovered_contexts
    
    def store_with_context_id(self, features, labels, context_ids):
        barcodes = self.barcode_encoder.get_barcodes_batch(context_ids)
        self.stored_features = np.concatenate([self.stored_features, features], axis=0)
        self.stored_barcodes = np.concatenate([self.stored_barcodes, barcodes], axis=0)
        self.stored_labels = np.concatenate([self.stored_labels, labels], axis=0)
        self.stored_contexts = np.concatenate(
            [self.stored_contexts, context_ids], axis=0)
    
    def predict_with_env(self, query_features, env_signals, lambda_param=None):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        lam = lambda_param if lambda_param is not None else self.lambda_param
        
        discovered = self.context_discovery.discover_batch(env_signals)
        query_barcodes = self.barcode_encoder.get_barcodes_batch(discovered)
        
        c_scores = self._content_scores(query_features)
        b_scores = self._barcode_scores(query_barcodes)
        combined = self._combine(c_scores, b_scores, lam)
        return self.stored_labels[np.argmax(combined, axis=1)]
    
    def predict_content_only(self, query_features):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        c_scores = self._content_scores(query_features)
        return self.stored_labels[np.argmax(c_scores, axis=1)]
    
    def predict_knn(self, query_features, k=5):
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
            kk = min(k, sims.shape[1])
            top_idx = np.argpartition(sims[i], -kk)[-kk:]
            top_labels = self.stored_labels[top_idx]
            top_sims = sims[i, top_idx]
            label_scores = {}
            for j, lbl in enumerate(top_labels):
                label_scores[lbl] = label_scores.get(lbl, 0.0) + top_sims[j]
            preds[i] = max(label_scores, key=label_scores.get)
        return preds
    
    def predict_conditional_knn(self, query_features, env_signals, k=5):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        
        discovered = self.context_discovery.discover_batch(env_signals)
        
        Q = query_features.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / q_norms
        E = self.stored_features.astype(np.float32)
        e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / e_norms
        sims = Q @ E.T
        
        preds = np.zeros(len(Q), dtype=np.int32)
        for i in range(len(Q)):
            ctx_mask = self.stored_contexts == discovered[i]
            if ctx_mask.any():
                ctx_sims = np.full(sims.shape[1], -np.inf, dtype=np.float32)
                ctx_sims[ctx_mask] = sims[i, ctx_mask]
                kk = min(k, ctx_mask.sum())
                top_idx = np.argpartition(ctx_sims, -kk)[-kk:]
                top_labels = self.stored_labels[top_idx]
                top_sims = ctx_sims[top_idx]
                label_scores = {}
                for j, lbl in enumerate(top_labels):
                    label_scores[lbl] = label_scores.get(lbl, 0.0) + max(top_sims[j], 0)
                preds[i] = max(label_scores, key=label_scores.get) if label_scores else -1
            else:
                kk = min(k, sims.shape[1])
                top_idx = np.argpartition(sims[i], -kk)[-kk:]
                top_labels = self.stored_labels[top_idx]
                top_sims = sims[i, top_idx]
                label_scores = {}
                for j, lbl in enumerate(top_labels):
                    label_scores[lbl] = label_scores.get(lbl, 0.0) + top_sims[j]
                preds[i] = max(label_scores, key=label_scores.get)
        return preds
    
    def _content_scores(self, Q):
        Q = Q.astype(np.float32)
        q_norms = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / q_norms
        E = self.stored_features.astype(np.float32)
        e_norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / e_norms
        return (Q @ E.T).astype(np.float32)
    
    def _barcode_scores(self, Q_bc):
        q_norms = np.maximum(np.linalg.norm(Q_bc, axis=1, keepdims=True), 1e-8)
        Q = Q_bc / q_norms
        B = self.stored_barcodes.astype(np.float32)
        b_norms = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-8)
        B = B / b_norms
        return (Q @ B.T).astype(np.float32)
    
    def _combine(self, C, B, lam):
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
    return X + rng.randn(*X.shape).astype(np.float32) * noise_level


def run_ijcnn_experiment():
    log("=" * 76)
    log("  IJCNN 2026 Submission: MEC-DG Dual-Channel Associative Memory")
    log("  with Online Context Discovery for Embodied Intelligence")
    log("=" * 76)
    log()
    log("  FAIR COMPARISON: All methods receive the SAME information")
    log("  - kNN: content features only (no context)")
    log("  - Conditional kNN: content features + env signal (same info as ours)")
    log("  - Ours: content features + env signal + DG barcode from discovered context")
    log("  - Content-only: content features only (no context)")
    log()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    log("  Loading MNIST...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    log("  Pre-training feature extractor on ALL MNIST (5 epochs)...")
    model = MLP(hidden_dim=256, output_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    model.train()
    for epoch in range(5):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
    
    log("  Extracting test features...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    all_features, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            all_features.append(model.get_features(data).numpy())
            all_labels.append(target.numpy())
    all_features = normalize_features(np.concatenate(all_features).astype(np.float32))
    all_labels = np.concatenate(all_labels).astype(np.int32)
    log(f"  Features: {all_features.shape}")
    
    rng = np.random.RandomState(42)
    
    # ================================================================
    # Experiment 1: Context-Dependent MNIST (FAIR comparison)
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 1: Context-Dependent Retrieval (FAIR)")
    log("  Same visual input, different labels in different contexts")
    log("  ALL context-aware methods get the SAME env signal")
    log("=" * 76)
    
    n_contexts = 5
    base_digits = [0, 1, 2, 3, 4]
    n_store = 40
    n_query = 50
    env_dim = 32
    env_noise = 0.1
    
    label_mappings = {}
    for ctx in range(n_contexts):
        mapping = {}
        for i, d in enumerate(base_digits):
            mapping[d] = (i + ctx) % 10
        label_mappings[ctx] = mapping
    
    env_sim = EnvironmentSimulator(n_contexts, env_dim, env_noise, seed=42)
    
    memory = MEC_DG_Memory(
        feature_dim=256, barcode_dim=512, sparsity=64,
        env_dim=env_dim, novelty_threshold=0.5, lambda_param=0.5,
        max_contexts=20, seed=42)
    
    for ctx in range(n_contexts):
        for digit in base_digits:
            class_idx = np.where(all_labels == digit)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_store]
            feat = all_features[idx]
            labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
            env_signals = env_sim.get_signals_batch(np.full(len(idx), ctx))
            memory.store_with_env(feat, labels, env_signals)
    
    log(f"  Stored {len(memory.stored_features)} items")
    log(f"  Discovered {memory.context_discovery.n_contexts} contexts (true: {n_contexts})")
    
    log(f"\n  Table 1: Context-Dependent Retrieval (FAIR comparison)")
    log(f"  {'Noise':>8s} | {'kNN':>8s} | {'Cond-kNN':>8s} | {'Ours':>8s} | {'Ours-Cond':>10s} | {'Ctx Disc':>8s}")
    log(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8}")
    
    for noise_level in [0.0, 0.1, 0.2, 0.3, 0.5]:
        correct_knn = 0
        correct_cond_knn = 0
        correct_ours = 0
        correct_content = 0
        correct_ctx = 0
        total = 0
        
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_store:n_store + n_query]
                if len(q_idx) < n_query:
                    continue
                q_feat = add_noise(all_features[q_idx], noise_level, rng)
                q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)
                q_env = env_sim.get_signals_batch(np.full(len(q_idx), ctx))
                
                p_knn = memory.predict_knn(q_feat, k=5)
                p_cond = memory.predict_conditional_knn(q_feat, q_env, k=5)
                p_ours = memory.predict_with_env(q_feat, q_env, lambda_param=0.5)
                p_content = memory.predict_content_only(q_feat)
                
                discovered = memory.context_discovery.discover_batch(q_env)
                true_ctx = np.full(len(q_idx), ctx, dtype=np.int32)
                
                correct_knn += np.sum(p_knn == q_lbl)
                correct_cond_knn += np.sum(p_cond == q_lbl)
                correct_ours += np.sum(p_ours == q_lbl)
                correct_content += np.sum(p_content == q_lbl)
                correct_ctx += np.sum(discovered == true_ctx)
                total += len(q_idx)
        
        log(f"  {noise_level:>8.1f} | {correct_knn/total:>8.4f} | {correct_cond_knn/total:>8.4f} | "
            f"{correct_ours/total:>8.4f} | {correct_content/total:>10.4f} | {correct_ctx/total:>8.4f}")
    
    # ================================================================
    # Experiment 2: Standard Split-MNIST (no context conflict)
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 2: Standard Split-MNIST (no context conflict)")
    log("=" * 76)
    
    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    n_items_sm = 50
    
    env_sim_sm = EnvironmentSimulator(5, env_dim, env_noise, seed=100)
    memory_sm = MEC_DG_Memory(
        feature_dim=256, barcode_dim=512, sparsity=64,
        env_dim=env_dim, novelty_threshold=0.5, lambda_param=0.5,
        max_contexts=20, seed=100)
    
    for task_id, (la, lb) in enumerate(task_labels):
        for lbl in [la, lb]:
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_items_sm]
            env_signals = env_sim_sm.get_signals_batch(np.full(len(idx), task_id))
            memory_sm.store_with_env(all_features[idx], all_labels[idx], env_signals)
    
    log(f"  Discovered {memory_sm.context_discovery.n_contexts} contexts")
    
    log(f"\n  Table 2: Split-MNIST with Environmental Context")
    log(f"  {'Noise':>8s} | {'kNN':>8s} | {'Cond-kNN':>8s} | {'Ours':>8s} | {'Ctx Disc':>8s}")
    log(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    
    for noise_level in [0.0, 0.2, 0.3, 0.5]:
        correct_knn = 0
        correct_cond = 0
        correct_ours = 0
        correct_ctx = 0
        total = 0
        
        for task_id, (la, lb) in enumerate(task_labels):
            for lbl in [la, lb]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_items_sm:n_items_sm + 100]
                if len(q_idx) < 50:
                    continue
                q_feat = add_noise(all_features[q_idx], noise_level, rng)
                q_lbl = all_labels[q_idx]
                q_env = env_sim_sm.get_signals_batch(np.full(len(q_idx), task_id))
                
                p_knn = memory_sm.predict_knn(q_feat, k=5)
                p_cond = memory_sm.predict_conditional_knn(q_feat, q_env, k=5)
                p_ours = memory_sm.predict_with_env(q_feat, q_env, lambda_param=0.5)
                
                discovered = memory_sm.context_discovery.discover_batch(q_env)
                true_ctx = np.full(len(q_idx), task_id, dtype=np.int32)
                
                correct_knn += np.sum(p_knn == q_lbl)
                correct_cond += np.sum(p_cond == q_lbl)
                correct_ours += np.sum(p_ours == q_lbl)
                correct_ctx += np.sum(discovered == true_ctx)
                total += len(q_idx)
        
        log(f"  {noise_level:>8.1f} | {correct_knn/total:>8.4f} | {correct_cond/total:>8.4f} | "
            f"{correct_ours/total:>8.4f} | {correct_ctx/total:>8.4f}")
    
    # ================================================================
    # Experiment 3: Permuted-MNIST
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 3: Permuted-MNIST (5 permutation tasks)")
    log("=" * 76)
    
    n_perm_tasks = 5
    n_items_pm = 50
    
    perm_rng = np.random.RandomState(123)
    permutations = [np.arange(784) for _ in range(n_perm_tasks)]
    for i in range(1, n_perm_tasks):
        permutations[i] = perm_rng.permutation(784)
    
    perm_features_list = []
    perm_labels_list = []
    for t in range(n_perm_tasks):
        test_loader_p = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
        p_feats, p_lbls = [], []
        model.eval()
        with torch.no_grad():
            for data, target in test_loader_p:
                flat = data.view(data.size(0), -1)
                perm_data = flat[:, permutations[t]]
                perm_data = perm_data.view(data.size(0), 1, 28, 28)
                feat = model.get_features(perm_data)
                p_feats.append(feat.numpy())
                p_lbls.append(target.numpy())
        perm_features_list.append(normalize_features(np.concatenate(p_feats).astype(np.float32)))
        perm_labels_list.append(np.concatenate(p_lbls).astype(np.int32))
    
    env_sim_pm = EnvironmentSimulator(n_perm_tasks, env_dim, env_noise, seed=200)
    memory_pm = MEC_DG_Memory(
        feature_dim=256, barcode_dim=512, sparsity=64,
        env_dim=env_dim, novelty_threshold=0.5, lambda_param=0.5,
        max_contexts=20, seed=200)
    
    for t in range(n_perm_tasks):
        pf = perm_features_list[t]
        pl = perm_labels_list[t]
        for lbl in range(10):
            class_idx = np.where(pl == lbl)[0]
            perm_rng.shuffle(class_idx)
            idx = class_idx[:5]
            env_signals = env_sim_pm.get_signals_batch(np.full(len(idx), t))
            memory_pm.store_with_env(pf[idx], pl[idx], env_signals)
    
    log(f"  Discovered {memory_pm.context_discovery.n_contexts} contexts")
    log(f"  Stored {len(memory_pm.stored_features)} items")
    
    log(f"\n  Table 3: Permuted-MNIST")
    log(f"  {'Task':>6s} | {'kNN':>8s} | {'Cond-kNN':>8s} | {'Ours':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")
    
    for t in range(n_perm_tasks):
        pf = perm_features_list[t]
        pl = perm_labels_list[t]
        
        correct_knn = 0
        correct_cond = 0
        correct_ours = 0
        total = 0
        
        for lbl in range(10):
            class_idx = np.where(pl == lbl)[0]
            perm_rng.shuffle(class_idx)
            q_idx = class_idx[5:25]
            if len(q_idx) < 10:
                continue
            q_env = env_sim_pm.get_signals_batch(np.full(len(q_idx), t))
            
            p_knn = memory_pm.predict_knn(pf[q_idx], k=5)
            p_cond = memory_pm.predict_conditional_knn(pf[q_idx], q_env, k=5)
            p_ours = memory_pm.predict_with_env(pf[q_idx], q_env, lambda_param=0.5)
            
            correct_knn += np.sum(p_knn == pl[q_idx])
            correct_cond += np.sum(p_cond == pl[q_idx])
            correct_ours += np.sum(p_ours == pl[q_idx])
            total += len(q_idx)
        
        if total > 0:
            log(f"  {t+1:>6d} | {correct_knn/total:>8.4f} | {correct_cond/total:>8.4f} | "
                f"{correct_ours/total:>8.4f}")
    
    # ================================================================
    # Experiment 4: Ablation - Novelty Threshold
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 4: Ablation - Novelty Threshold Sensitivity")
    log("=" * 76)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    log(f"\n  Table 4: Context Discovery vs Novelty Threshold")
    log(f"  {'Thresh':>8s} | {'N Ctx':>6s} | {'Ctx Acc':>8s} | {'Retr Acc':>8s}")
    log(f"  {'-'*8} | {'-'*6} | {'-'*8} | {'-'*8}")
    
    for thresh in thresholds:
        env_sim_ab = EnvironmentSimulator(n_contexts, env_dim, env_noise, seed=42)
        mem_ab = MEC_DG_Memory(
            feature_dim=256, barcode_dim=512, sparsity=64,
            env_dim=env_dim, novelty_threshold=thresh, lambda_param=0.5,
            max_contexts=20, seed=42)
        
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_store]
                feat = all_features[idx]
                labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                env_signals = env_sim_ab.get_signals_batch(np.full(len(idx), ctx))
                mem_ab.store_with_env(feat, labels, env_signals)
        
        correct_ctx = 0
        correct_ours = 0
        total = 0
        
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_store:n_store + n_query]
                if len(q_idx) < n_query:
                    continue
                q_feat = add_noise(all_features[q_idx], 0.2, rng)
                q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)
                q_env = env_sim_ab.get_signals_batch(np.full(len(q_idx), ctx))
                
                discovered = mem_ab.context_discovery.discover_batch(q_env)
                true_ctx = np.full(len(q_idx), ctx, dtype=np.int32)
                p_ours = mem_ab.predict_with_env(q_feat, q_env, lambda_param=0.5)
                
                correct_ctx += np.sum(discovered == true_ctx)
                correct_ours += np.sum(p_ours == q_lbl)
                total += len(q_idx)
        
        log(f"  {thresh:>8.1f} | {mem_ab.context_discovery.n_contexts:>6d} | "
            f"{correct_ctx/total:>8.4f} | {correct_ours/total:>8.4f}")
    
    # ================================================================
    # Experiment 5: Ablation - Lambda
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 5: Ablation - Lambda Sensitivity")
    log("=" * 76)
    
    lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    log(f"\n  Table 5: Retrieval Accuracy vs Lambda (noise=0.2)")
    log(f"  {'Lambda':>8s} | {'Ours':>8s} | {'Note':>20s}")
    log(f"  {'-'*8} | {'-'*8} | {'-'*20}")
    
    for lam in lambda_values:
        correct_ours = 0
        total = 0
        
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_store:n_store + n_query]
                if len(q_idx) < n_query:
                    continue
                q_feat = add_noise(all_features[q_idx], 0.2, rng)
                q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)
                q_env = env_sim.get_signals_batch(np.full(len(q_idx), ctx))
                
                p_ours = memory.predict_with_env(q_feat, q_env, lambda_param=lam)
                correct_ours += np.sum(p_ours == q_lbl)
                total += len(q_idx)
        
        note = ""
        if lam == 0.0:
            note = "barcode only"
        elif lam == 1.0:
            note = "content only"
        elif lam == 0.5:
            note = "balanced"
        log(f"  {lam:>8.1f} | {correct_ours/total:>8.4f} | {note:>20s}")
    
    # ================================================================
    # Experiment 6: Ablation - Env Signal Noise
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 6: Robustness to Environmental Signal Noise")
    log("=" * 76)
    
    env_noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    
    log(f"\n  Table 6: Retrieval vs Env Noise (query noise=0.2)")
    log(f"  {'Env Noise':>10s} | {'Ctx Disc':>8s} | {'Cond-kNN':>8s} | {'Ours':>8s}")
    log(f"  {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
    
    for env_n in env_noise_levels:
        env_sim_rn = EnvironmentSimulator(n_contexts, env_dim, env_n, seed=42)
        mem_rn = MEC_DG_Memory(
            feature_dim=256, barcode_dim=512, sparsity=64,
            env_dim=env_dim, novelty_threshold=0.5, lambda_param=0.5,
            max_contexts=20, seed=42)
        
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_store]
                feat = all_features[idx]
                labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                env_signals = env_sim_rn.get_signals_batch(np.full(len(idx), ctx))
                mem_rn.store_with_env(feat, labels, env_signals)
        
        correct_ctx = 0
        correct_cond = 0
        correct_ours = 0
        total = 0
        
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_store:n_store + n_query]
                if len(q_idx) < n_query:
                    continue
                q_feat = add_noise(all_features[q_idx], 0.2, rng)
                q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)
                q_env = env_sim_rn.get_signals_batch(np.full(len(q_idx), ctx))
                
                discovered = mem_rn.context_discovery.discover_batch(q_env)
                true_ctx = np.full(len(q_idx), ctx, dtype=np.int32)
                p_cond = mem_rn.predict_conditional_knn(q_feat, q_env, k=5)
                p_ours = mem_rn.predict_with_env(q_feat, q_env, lambda_param=0.5)
                
                correct_ctx += np.sum(discovered == true_ctx)
                correct_cond += np.sum(p_cond == q_lbl)
                correct_ours += np.sum(p_ours == q_lbl)
                total += len(q_idx)
        
        log(f"  {env_n:>10.1f} | {correct_ctx/total:>8.4f} | {correct_cond/total:>8.4f} | "
            f"{correct_ours/total:>8.4f}")
    
    # ================================================================
    # Summary
    # ================================================================
    log("\n" + "=" * 76)
    log("  SUMMARY: IJCNN 2026 Submission")
    log("=" * 76)
    log()
    log("  Paper Title:")
    log("  MEC-DG Dual-Channel Associative Memory with Online")
    log("  Context Discovery for Embodied Intelligence")
    log()
    log("  Key Contributions:")
    log("  1. Online context discovery from environmental signals")
    log("     (no task ID required - truly task-agnostic)")
    log("  2. Bio-inspired MEC→DG architecture for context-dependent")
    log("     sparse barcodes (independent of content features)")
    log("  3. Dual-channel retrieval combining content + context barcode")
    log("  4. Fair comparison: all context-aware methods get same env signal")
    log()
    log("  What makes this IJCNN-worthy:")
    log("  - Bio-inspired neural architecture (MEC-DG-CA3 pathway)")
    log("  - Online learning (no pre-training of context)")
    log("  - Task-agnostic (no task ID at test time)")
    log("  - Fair experimental comparison")
    log("  - Multiple datasets + ablation studies")
    log()
    log("  Honest limitations:")
    log("  - Context discovery depends on env signal quality")
    log("  - Barcode is random (not learned)")
    log("  - Only tested on MNIST variants")
    log("  - No comparison with deep CL methods (different problem)")
    
    return True


if __name__ == "__main__":
    run_ijcnn_experiment()
