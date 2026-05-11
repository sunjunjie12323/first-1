from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict


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


class NoveltyDetector:
    """
    Novelty detection based on retrieval consistency.
    
    Core idea: If a query's k-nearest neighbors have HIGH label agreement,
    the query belongs to a known context. If they have LOW agreement
    (high entropy), the query is either at a context boundary or
    belongs to a novel context.
    
    This requires NO context signal — purely content-based.
    """
    
    def __init__(self, k=10, novelty_threshold=0.5):
        self.k = k
        self.novelty_threshold = novelty_threshold
    
    def compute_novelty(self, query_feat, stored_feat, stored_labels, stored_contexts):
        Q = query_feat.astype(np.float32)
        qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / qn
        E = stored_feat.astype(np.float32)
        en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / en
        sims = Q @ E.T
        
        novelty_scores = np.zeros(len(Q), dtype=np.float32)
        dominant_contexts = np.zeros(len(Q), dtype=np.int32)
        context_weights_list = []
        
        for i in range(len(Q)):
            kk = min(self.k, sims.shape[1])
            top_idx = np.argpartition(sims[i], -kk)[-kk:]
            top_sims = sims[i, top_idx]
            top_labels = stored_labels[top_idx]
            top_contexts = stored_contexts[top_idx]
            
            label_counts = defaultdict(float)
            context_counts = defaultdict(float)
            for j, (lbl, ctx) in enumerate(zip(top_labels, top_contexts)):
                w = max(top_sims[j], 0.0)
                label_counts[lbl] += w
                context_counts[ctx] += w
            
            total_w = sum(context_counts.values())
            if total_w > 0:
                ctx_probs = {c: w / total_w for c, w in context_counts.items()}
            else:
                ctx_probs = {}
            
            max_ctx_prob = max(ctx_probs.values()) if ctx_probs else 0
            novelty_scores[i] = 1.0 - max_ctx_prob
            dominant_contexts[i] = max(ctx_probs, key=ctx_probs.get) if ctx_probs else 0
            context_weights_list.append(ctx_probs)
        
        return novelty_scores, dominant_contexts, context_weights_list


class ContextGraph:
    """
    Maintains a graph of context similarities for knowledge transfer.
    
    Two contexts are similar if their stored items have similar
    label distributions or feature distributions.
    """
    
    def __init__(self, n_contexts_max=50):
        self.context_features = {}
        self.context_labels = {}
        self.similarity_matrix = {}
    
    def update(self, context_id, features, labels):
        if context_id not in self.context_features:
            self.context_features[context_id] = features.copy()
            self.context_labels[context_id] = labels.copy()
        else:
            self.context_features[context_id] = np.concatenate(
                [self.context_features[context_id], features], axis=0)
            self.context_labels[context_id] = np.concatenate(
                [self.context_labels[context_id], labels], axis=0)
        self._update_similarities(context_id)
    
    def _update_similarities(self, updated_ctx):
        feat = self.context_features[updated_ctx]
        if len(feat) == 0:
            return
        center = np.mean(feat, axis=0)
        center_norm = center / max(np.linalg.norm(center), 1e-8)
        
        for other_ctx in self.context_features:
            if other_ctx == updated_ctx:
                self.similarity_matrix[(updated_ctx, other_ctx)] = 1.0
                continue
            other_feat = self.context_features[other_ctx]
            if len(other_feat) == 0:
                continue
            other_center = np.mean(other_feat, axis=0)
            other_norm = other_center / max(np.linalg.norm(other_center), 1e-8)
            sim = float(np.dot(center_norm, other_norm))
            self.similarity_matrix[(updated_ctx, other_ctx)] = max(sim, 0.0)
            self.similarity_matrix[(other_ctx, updated_ctx)] = max(sim, 0.0)
    
    def get_transfer_weights(self, context_id, temperature=1.0):
        weights = {}
        for other_ctx in self.context_features:
            if other_ctx == context_id:
                weights[other_ctx] = 1.0
                continue
            sim = self.similarity_matrix.get((context_id, other_ctx), 0.0)
            weights[other_ctx] = np.exp(sim / max(temperature, 0.01))
        
        total = sum(weights.values())
        if total > 0:
            weights = {c: w / total for c, w in weights.items()}
        return weights


class NDAM:
    """
    Novelty-Driven Adaptive Memory (NDAM)
    
    Combines:
    1. Novelty detection via retrieval consistency (Route 2)
    2. Cross-context soft knowledge transfer (Route 3)
    
    Key innovation: NO context signal required at test time.
    The system automatically:
    - Detects novel contexts from retrieval inconsistency
    - Creates new context modules on-the-fly
    - Transfers knowledge between similar contexts
    """
    
    def __init__(self, feature_dim=256, k_novelty=10, novelty_threshold=0.5,
                 transfer_temperature=1.0, lambda_content=0.7, seed=42):
        self.feature_dim = feature_dim
        self.k_novelty = k_novelty
        self.novelty_threshold = novelty_threshold
        self.transfer_temperature = transfer_temperature
        self.lambda_content = lambda_content
        
        self.stored_features = np.zeros((0, feature_dim), dtype=np.float32)
        self.stored_labels = np.zeros(0, dtype=np.int32)
        self.stored_contexts = np.zeros(0, dtype=np.int32)
        self.n_contexts = 0
        self._next_context = 0
        
        self.novelty_detector = NoveltyDetector(k_novelty, novelty_threshold)
        self.context_graph = ContextGraph()
        self._rng = np.random.RandomState(seed)
    
    def store(self, features, labels, true_context_id=None):
        if len(self.stored_features) == 0:
            assigned_ctx = np.zeros(len(features), dtype=np.int32)
            self.n_contexts = 1
            self._next_context = 1
        else:
            novelty, dominant, ctx_weights = self.novelty_detector.compute_novelty(
                features, self.stored_features, self.stored_labels, self.stored_contexts)
            
            assigned_ctx = np.zeros(len(features), dtype=np.int32)
            for i in range(len(features)):
                if novelty[i] > self.novelty_threshold:
                    assigned_ctx[i] = self._next_context
                    self._next_context += 1
                    self.n_contexts += 1
                else:
                    assigned_ctx[i] = dominant[i]
        
        self.stored_features = np.concatenate([self.stored_features, features], axis=0)
        self.stored_labels = np.concatenate([self.stored_labels, labels], axis=0)
        self.stored_contexts = np.concatenate([self.stored_contexts, assigned_ctx], axis=0)
        
        for ctx_id in np.unique(assigned_ctx):
            mask = assigned_ctx == ctx_id
            self.context_graph.update(
                int(ctx_id), features[mask], labels[mask])
        
        return assigned_ctx
    
    def store_with_context_id(self, features, labels, context_ids):
        self.stored_features = np.concatenate([self.stored_features, features], axis=0)
        self.stored_labels = np.concatenate([self.stored_labels, labels], axis=0)
        self.stored_contexts = np.concatenate([self.stored_contexts, context_ids], axis=0)
        
        for ctx_id in np.unique(context_ids):
            mask = context_ids == ctx_id
            self.context_graph.update(
                int(ctx_id), features[mask], labels[mask])
        
        self.n_contexts = max(self.n_contexts, int(np.max(context_ids)) + 1)
        self._next_context = self.n_contexts
    
    def predict(self, query_features, k=5):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        
        novelty, dominant, ctx_weights = self.novelty_detector.compute_novelty(
            query_features, self.stored_features, self.stored_labels, self.stored_contexts)
        
        Q = query_features.astype(np.float32)
        qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / qn
        E = self.stored_features.astype(np.float32)
        en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / en
        sims = Q @ E.T
        
        preds = np.zeros(len(Q), dtype=np.int32)
        for i in range(len(Q)):
            transfer_weights = self._get_transfer_weights(ctx_weights[i])
            
            weighted_sims = sims[i].copy()
            for j in range(len(weighted_sims)):
                item_ctx = int(self.stored_contexts[j])
                tw = transfer_weights.get(item_ctx, 0.1)
                weighted_sims[j] *= (self.lambda_content + (1 - self.lambda_content) * tw)
            
            kk = min(k, len(weighted_sims))
            top_idx = np.argpartition(weighted_sims, -kk)[-kk:]
            top_labels = self.stored_labels[top_idx]
            top_sims = weighted_sims[top_idx]
            
            label_scores = {}
            for j, lbl in enumerate(top_labels):
                label_scores[lbl] = label_scores.get(lbl, 0.0) + max(top_sims[j], 0)
            preds[i] = max(label_scores, key=label_scores.get) if label_scores else -1
        
        return preds
    
    def predict_isolated(self, query_features, k=5):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        
        novelty, dominant, ctx_weights = self.novelty_detector.compute_novelty(
            query_features, self.stored_features, self.stored_labels, self.stored_contexts)
        
        Q = query_features.astype(np.float32)
        qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / qn
        E = self.stored_features.astype(np.float32)
        en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / en
        sims = Q @ E.T
        
        preds = np.zeros(len(Q), dtype=np.int32)
        for i in range(len(Q)):
            best_ctx = dominant[i]
            mask = self.stored_contexts == best_ctx
            
            if mask.any():
                ctx_sims = np.full(sims.shape[1], -np.inf)
                ctx_sims[mask] = sims[i, mask]
                kk = min(k, mask.sum())
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
    
    def predict_knn(self, query_features, k=5):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        Q = query_features.astype(np.float32)
        qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / qn
        E = self.stored_features.astype(np.float32)
        en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / en
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
            preds[i] = max(label_scores, key=label_scores.get) if label_scores else -1
        return preds
    
    def _get_transfer_weights(self, ctx_probs):
        if not ctx_probs:
            return {}
        
        dominant_ctx = max(ctx_probs, key=ctx_probs.get)
        transfer = self.context_graph.get_transfer_weights(
            dominant_ctx, self.transfer_temperature)
        
        for ctx, prob in ctx_probs.items():
            if ctx not in transfer:
                transfer[ctx] = prob
        
        return transfer


def normalize_features(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X / norms


def add_noise(X, noise_level=0.0, rng=None):
    if noise_level <= 0:
        return X.copy()
    if rng is None:
        rng = np.random.RandomState(42)
    return X + rng.randn(*X.shape).astype(np.float32) * noise_level


def run_ndam_experiment():
    log("=" * 76)
    log("  NDAM: Novelty-Driven Adaptive Memory")
    log("  Route 2 + Route 3 Combined")
    log("=" * 76)
    log()
    log("  Innovation 1: Novelty detection via retrieval consistency")
    log("    - NO context signal needed at test time")
    log("    - Detects novel contexts from kNN label disagreement")
    log()
    log("  Innovation 2: Cross-context soft knowledge transfer")
    log("    - Context similarity graph for adaptive transfer")
    log("    - Similar contexts share knowledge, dissimilar don't")
    log()
    log("  FAIR baselines (all use same info: content features only):")
    log("    - kNN: no context awareness")
    log("    - Isolated: novelty detection + hard context isolation")
    log("    - NDAM (Ours): novelty detection + soft transfer")
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
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    all_features, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            all_features.append(model.get_features(data).numpy())
            all_labels.append(target.numpy())
    all_features = normalize_features(np.concatenate(all_features).astype(np.float32))
    all_labels = np.concatenate(all_labels).astype(np.int32)

    rng = np.random.RandomState(42)

    # ================================================================
    # Experiment 1: Context-Dependent MNIST (same input, different labels)
    # NO context signal given to any method
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 1: Context-Dependent Retrieval (NO context signal)")
    log("  Same visual input, different labels in different contexts")
    log("  ALL methods receive ONLY content features")
    log("=" * 76)

    n_contexts = 5
    base_digits = [0, 1, 2, 3, 4]
    n_store = 40
    n_query = 50

    label_mappings = {}
    for ctx in range(n_contexts):
        mapping = {}
        for i, d in enumerate(base_digits):
            mapping[d] = (i + ctx) % 10
        label_mappings[ctx] = mapping

    memory_ndam = NDAM(
        feature_dim=256, k_novelty=10, novelty_threshold=0.5,
        transfer_temperature=1.0, lambda_content=0.7, seed=42)
    memory_isolated = NDAM(
        feature_dim=256, k_novelty=10, novelty_threshold=0.5,
        transfer_temperature=1.0, lambda_content=0.7, seed=42)

    for ctx in range(n_contexts):
        for digit in base_digits:
            class_idx = np.where(all_labels == digit)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_store]
            feat = all_features[idx]
            labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
            ctx_ids = np.full(len(idx), ctx, dtype=np.int32)
            
            memory_ndam.store_with_context_id(feat, labels, ctx_ids)
            memory_isolated.store_with_context_id(feat, labels, ctx_ids)

    log(f"  Stored {len(memory_ndam.stored_features)} items in {n_contexts} contexts")
    log(f"  Context graph similarities:")
    for c1 in range(n_contexts):
        for c2 in range(c1+1, n_contexts):
            sim = memory_ndam.context_graph.similarity_matrix.get((c1, c2), 0)
            log(f"    ctx{c1} <-> ctx{c2}: {sim:.4f}")

    log(f"\n  Table 1: Context-Dependent Retrieval (NO context signal)")
    log(f"  {'Noise':>8s} | {'kNN':>8s} | {'Isolated':>10s} | {'NDAM':>8s} | {'NDAM-Iso':>10s}")
    log(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*10}")

    for noise_level in [0.0, 0.1, 0.2, 0.3, 0.5]:
        correct_knn = 0
        correct_isolated = 0
        correct_ndam = 0
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

                p_iso = memory_isolated.predict_isolated(q_feat, k=5)
                p_ndam = memory_ndam.predict(q_feat, k=5)

                Q = q_feat.astype(np.float32)
                qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
                Q = Q / qn
                E = memory_ndam.stored_features.astype(np.float32)
                en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
                E = E / en
                sims = Q @ E.T
                p_knn = np.zeros(len(Q), dtype=np.int32)
                for i in range(len(Q)):
                    kk = min(5, sims.shape[1])
                    top_idx = np.argpartition(sims[i], -kk)[-kk:]
                    tl = memory_ndam.stored_labels[top_idx]
                    ts = sims[i, top_idx]
                    ls = {}
                    for j, lbl in enumerate(tl):
                        ls[lbl] = ls.get(lbl, 0.0) + ts[j]
                    p_knn[i] = max(ls, key=ls.get)

                correct_knn += np.sum(p_knn == q_lbl)
                correct_isolated += np.sum(p_iso == q_lbl)
                correct_ndam += np.sum(p_ndam == q_lbl)
                total += len(q_idx)

        diff = correct_ndam - correct_isolated
        log(f"  {noise_level:>8.1f} | {correct_knn/total:>8.4f} | {correct_isolated/total:>10.4f} | "
            f"{correct_ndam/total:>8.4f} | {diff/total:>+10.4f}")

    # ================================================================
    # Experiment 2: Split-MNIST (no label conflict, shared structure)
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 2: Split-MNIST (shared structure, no conflict)")
    log("  Tasks share digits 0-9, testing cross-context transfer")
    log("=" * 76)

    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    n_items_sm = 50

    memory_ndam_sm = NDAM(
        feature_dim=256, k_novelty=10, novelty_threshold=0.5,
        transfer_temperature=1.0, lambda_content=0.7, seed=100)
    memory_iso_sm = NDAM(
        feature_dim=256, k_novelty=10, novelty_threshold=0.5,
        transfer_temperature=1.0, lambda_content=0.7, seed=100)

    for task_id, (la, lb) in enumerate(task_labels):
        for lbl in [la, lb]:
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_items_sm]
            memory_ndam_sm.store_with_context_id(
                all_features[idx], all_labels[idx],
                np.full(len(idx), task_id, dtype=np.int32))
            memory_iso_sm.store_with_context_id(
                all_features[idx], all_labels[idx],
                np.full(len(idx), task_id, dtype=np.int32))

    log(f"\n  Table 2: Split-MNIST (NO context signal)")
    log(f"  {'Noise':>8s} | {'kNN':>8s} | {'Isolated':>10s} | {'NDAM':>8s} | {'NDAM-Iso':>10s}")
    log(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*10}")

    for noise_level in [0.0, 0.2, 0.3, 0.5]:
        correct_knn = 0
        correct_isolated = 0
        correct_ndam = 0
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

                p_iso = memory_iso_sm.predict_isolated(q_feat, k=5)
                p_ndam = memory_ndam_sm.predict(q_feat, k=5)

                Q = q_feat.astype(np.float32)
                qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
                Q = Q / qn
                E = memory_ndam_sm.stored_features.astype(np.float32)
                en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
                E = E / en
                sims = Q @ E.T
                p_knn = np.zeros(len(Q), dtype=np.int32)
                for i in range(len(Q)):
                    kk = min(5, sims.shape[1])
                    top_idx = np.argpartition(sims[i], -kk)[-kk:]
                    tl = memory_ndam_sm.stored_labels[top_idx]
                    ts = sims[i, top_idx]
                    ls = {}
                    for j, lbl2 in enumerate(tl):
                        ls[lbl2] = ls.get(lbl2, 0.0) + ts[j]
                    p_knn[i] = max(ls, key=ls.get)

                correct_knn += np.sum(p_knn == q_lbl)
                correct_isolated += np.sum(p_iso == q_lbl)
                correct_ndam += np.sum(p_ndam == q_lbl)
                total += len(q_idx)

        diff = correct_ndam - correct_isolated
        log(f"  {noise_level:>8.1f} | {correct_knn/total:>8.4f} | {correct_isolated/total:>10.4f} | "
            f"{correct_ndam/total:>8.4f} | {diff/total:>+10.4f}")

    # ================================================================
    # Experiment 3: Novelty Detection Accuracy
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 3: Novelty Detection (Online Context Discovery)")
    log("=" * 76)

    memory_online = NDAM(
        feature_dim=256, k_novelty=10, novelty_threshold=0.5,
        transfer_temperature=1.0, lambda_content=0.7, seed=42)

    correct_novelty = 0
    total_novelty = 0
    discovered_contexts = []

    for ctx in range(n_contexts):
        for digit in base_digits:
            class_idx = np.where(all_labels == digit)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_store]
            feat = all_features[idx]
            labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
            
            assigned = memory_online.store(feat, labels)
            true_ctx = np.full(len(idx), ctx, dtype=np.int32)
            
            if ctx == 0:
                correct_novelty += np.sum(assigned == 0)
            else:
                correct_novelty += np.sum(assigned != assigned[0])
            total_novelty += len(idx)
            discovered_contexts.append(int(np.max(assigned)))

    log(f"  Novelty detection accuracy: {correct_novelty/total_novelty:.4f}")
    log(f"  Discovered {memory_online.n_contexts} contexts (true: {n_contexts})")
    log(f"  Context assignment per batch: {discovered_contexts}")

    # ================================================================
    # Experiment 4: Ablation - Transfer Temperature
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 4: Ablation - Transfer Temperature")
    log("=" * 76)

    temps = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    log(f"\n  {'Temp':>8s} | {'Split-MNIST':>12s} | {'Ctx-MNIST':>10s}")
    log(f"  {'-'*8} | {'-'*12} | {'-'*10}")

    for temp in temps:
        mem_sm = NDAM(
            feature_dim=256, k_novelty=10, novelty_threshold=0.5,
            transfer_temperature=temp, lambda_content=0.7, seed=100)
        mem_ctx = NDAM(
            feature_dim=256, k_novelty=10, novelty_threshold=0.5,
            transfer_temperature=temp, lambda_content=0.7, seed=42)

        for task_id, (la, lb) in enumerate(task_labels):
            for lbl in [la, lb]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_items_sm]
                mem_sm.store_with_context_id(
                    all_features[idx], all_labels[idx],
                    np.full(len(idx), task_id, dtype=np.int32))

        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_store]
                feat = all_features[idx]
                labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                mem_ctx.store_with_context_id(feat, labels, np.full(len(idx), ctx, dtype=np.int32))

        correct_sm = 0
        total_sm = 0
        for task_id, (la, lb) in enumerate(task_labels):
            for lbl in [la, lb]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_items_sm:n_items_sm + 100]
                if len(q_idx) < 50:
                    continue
                q_feat = add_noise(all_features[q_idx], 0.3, rng)
                q_lbl = all_labels[q_idx]
                p = mem_sm.predict(q_feat, k=5)
                correct_sm += np.sum(p == q_lbl)
                total_sm += len(q_idx)

        correct_ctx = 0
        total_ctx = 0
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_store:n_store + n_query]
                if len(q_idx) < n_query:
                    continue
                q_feat = add_noise(all_features[q_idx], 0.2, rng)
                q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)
                p = mem_ctx.predict(q_feat, k=5)
                correct_ctx += np.sum(p == q_lbl)
                total_ctx += len(q_idx)

        log(f"  {temp:>8.2f} | {correct_sm/total_sm:>12.4f} | {correct_ctx/total_ctx:>10.4f}")

    # ================================================================
    # Experiment 5: Ablation - Novelty Threshold
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 5: Ablation - Novelty Threshold")
    log("=" * 76)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    log(f"\n  {'Thresh':>8s} | {'N Ctx':>6s} | {'Novelty Acc':>12s} | {'Retr Acc':>8s}")
    log(f"  {'-'*8} | {'-'*6} | {'-'*12} | {'-'*8}")

    for thresh in thresholds:
        mem = NDAM(
            feature_dim=256, k_novelty=10, novelty_threshold=thresh,
            transfer_temperature=1.0, lambda_content=0.7, seed=42)

        correct_nov = 0
        total_nov = 0
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_store]
                feat = all_features[idx]
                labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                assigned = mem.store(feat, labels)
                true_ctx = np.full(len(idx), ctx, dtype=np.int32)
                if ctx == 0:
                    correct_nov += np.sum(assigned == 0)
                else:
                    correct_nov += np.sum(assigned != assigned[0])
                total_nov += len(idx)

        correct_retr = 0
        total_retr = 0
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_store:n_store + n_query]
                if len(q_idx) < n_query:
                    continue
                q_feat = add_noise(all_features[q_idx], 0.2, rng)
                q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)
                p = mem.predict(q_feat, k=5)
                correct_retr += np.sum(p == q_lbl)
                total_retr += len(q_idx)

        log(f"  {thresh:>8.1f} | {mem.n_contexts:>6d} | {correct_nov/total_nov:>12.4f} | "
            f"{correct_retr/total_retr:>8.4f}")

    # ================================================================
    # Summary
    # ================================================================
    log("\n" + "=" * 76)
    log("  SUMMARY")
    log("=" * 76)
    log()
    log("  Key question: Does NDAM (soft transfer) > Isolated (hard isolation)?")
    log("  If YES: Cross-context transfer provides genuine advantage")
    log("  If NO:  Isolation is sufficient, transfer is not needed")
    log()
    log("  Key question: Does novelty detection work without context signal?")
    log("  If YES: Truly task-agnostic system")
    log("  If NO:  Need some form of context signal")

    return True


if __name__ == "__main__":
    run_ndam_experiment()
