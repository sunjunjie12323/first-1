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


class EnvironmentSimulator:
    def __init__(self, n_contexts, env_dim=16, signal_noise=0.1, seed=42):
        self.n_contexts = n_contexts
        self.env_dim = env_dim
        self.signal_noise = signal_noise
        self.rng = np.random.RandomState(seed)
        self.context_signals = self.rng.randn(n_contexts, env_dim).astype(np.float32)
        norms = np.maximum(np.linalg.norm(self.context_signals, axis=1, keepdims=True), 1e-8)
        self.context_signals /= norms

    def get_signal(self, context_id):
        signal = self.context_signals[context_id % self.n_contexts].copy()
        if self.signal_noise > 0:
            signal += self.rng.randn(self.env_dim).astype(np.float32) * self.signal_noise
        return signal

    def get_signals_batch(self, context_ids):
        signals = self.context_signals[context_ids % self.n_contexts].copy()
        if self.signal_noise > 0:
            signals += self.rng.randn(*signals.shape).astype(np.float32) * self.signal_noise
        return signals


class AdaptiveContextMemory:
    """
    Adaptive Context Memory with Environment-Guided Novelty Detection
    and Cross-Context Soft Transfer.
    
    Architecture:
    - LEC pathway: content features (what)
    - MEC pathway: environmental signals (where/when)  
    - DG: novelty detection from env signal + retrieval consistency
    - CA3: soft retrieval with adaptive context weighting
    
    Key innovations:
    1. Env signal provides context hypothesis, retrieval consistency
       validates/corrects it → robust to env signal errors
    2. Cross-context soft transfer via context similarity graph
    3. Adaptive confidence: when env signal is uncertain, rely more
       on content; when content is ambiguous, rely more on env signal
    """
    
    def __init__(self, feature_dim=256, env_dim=16, novelty_threshold=0.5,
                 max_contexts=20, seed=42):
        self.feature_dim = feature_dim
        self.env_dim = env_dim
        self.novelty_threshold = novelty_threshold
        self.max_contexts = max_contexts
        self.rng = np.random.RandomState(seed)
        
        self.stored_features = np.zeros((0, feature_dim), dtype=np.float32)
        self.stored_labels = np.zeros(0, dtype=np.int32)
        self.stored_contexts = np.zeros(0, dtype=np.int32)
        self.stored_env_centers = {}
        self.context_counts = defaultdict(int)
        self.n_contexts = 0
        self._next_context = 0
    
    def store_with_env(self, features, labels, env_signals):
        assigned = np.zeros(len(features), dtype=np.int32)
        
        for i in range(len(features)):
            ctx = self._assign_context(env_signals[i])
            assigned[i] = ctx
            self.context_counts[ctx] += 1
            
            if ctx not in self.stored_env_centers:
                self.stored_env_centers[ctx] = env_signals[i].copy()
            else:
                alpha = 1.0 / max(self.context_counts[ctx], 1)
                self.stored_env_centers[ctx] = (
                    (1 - alpha) * self.stored_env_centers[ctx] + alpha * env_signals[i])
        
        self.stored_features = np.concatenate([self.stored_features, features])
        self.stored_labels = np.concatenate([self.stored_labels, labels])
        self.stored_contexts = np.concatenate([self.stored_contexts, assigned])
        
        return assigned
    
    def predict_adaptive(self, query_features, env_signals, k=5):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        
        Q = query_features.astype(np.float32)
        qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / qn
        E = self.stored_features.astype(np.float32)
        en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / en
        content_sims = Q @ E.T
        
        preds = np.zeros(len(Q), dtype=np.int32)
        
        for i in range(len(Q)):
            env_ctx, env_conf = self._infer_context(env_signals[i])
            
            ctx_weights = self._compute_context_weights(
                content_sims[i], env_ctx, env_conf)
            
            weighted_sims = content_sims[i].copy()
            for j in range(len(weighted_sims)):
                item_ctx = int(self.stored_contexts[j])
                w = ctx_weights.get(item_ctx, 0.01)
                weighted_sims[j] *= w
            
            kk = min(k, len(weighted_sims))
            top_idx = np.argpartition(weighted_sims, -kk)[-kk:]
            top_labels = self.stored_labels[top_idx]
            top_sims = weighted_sims[top_idx]
            
            label_scores = {}
            for j, lbl in enumerate(top_labels):
                label_scores[lbl] = label_scores.get(lbl, 0.0) + max(top_sims[j], 0)
            preds[i] = max(label_scores, key=label_scores.get) if label_scores else -1
        
        return preds
    
    def predict_conditional_knn(self, query_features, env_signals, k=5):
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
            env_ctx, _ = self._infer_context(env_signals[i])
            mask = self.stored_contexts == env_ctx
            if mask.any():
                ctx_sims = np.full(sims.shape[1], -np.inf)
                ctx_sims[mask] = sims[i, mask]
                kk = min(k, mask.sum())
                top_idx = np.argpartition(ctx_sims, -kk)[-kk:]
                tl = self.stored_labels[top_idx]
                ts = ctx_sims[top_idx]
                ls = {}
                for j, lbl in enumerate(tl):
                    ls[lbl] = ls.get(lbl, 0.0) + max(ts[j], 0)
                preds[i] = max(ls, key=ls.get) if ls else -1
            else:
                kk = min(k, sims.shape[1])
                top_idx = np.argpartition(sims[i], -kk)[-kk:]
                tl = self.stored_labels[top_idx]
                ts = sims[i, top_idx]
                ls = {}
                for j, lbl in enumerate(tl):
                    ls[lbl] = ls.get(lbl, 0.0) + ts[j]
                preds[i] = max(ls, key=ls.get)
        return preds
    
    def predict_soft_env(self, query_features, env_signals, k=5):
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
            env_ctx, env_conf = self._infer_context(env_signals[i])
            
            env_weights = np.zeros(sims.shape[1], dtype=np.float32)
            for j in range(len(env_weights)):
                item_ctx = int(self.stored_contexts[j])
                if item_ctx == env_ctx:
                    env_weights[j] = env_conf
                else:
                    env_weights[j] = (1 - env_conf) / max(self.n_contexts - 1, 1)
            
            weighted_sims = sims[i] * (1.0 + env_weights)
            kk = min(k, len(weighted_sims))
            top_idx = np.argpartition(weighted_sims, -kk)[-kk:]
            tl = self.stored_labels[top_idx]
            ts = weighted_sims[top_idx]
            ls = {}
            for j, lbl in enumerate(tl):
                ls[lbl] = ls.get(lbl, 0.0) + ts[j]
            preds[i] = max(ls, key=ls.get)
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
            tl = self.stored_labels[top_idx]
            ts = sims[i, top_idx]
            ls = {}
            for j, lbl in enumerate(tl):
                ls[lbl] = ls.get(lbl, 0.0) + ts[j]
            preds[i] = max(ls, key=ls.get)
        return preds
    
    def _assign_context(self, env_signal):
        if len(self.stored_env_centers) == 0:
            ctx = self._next_context
            self._next_context += 1
            self.n_contexts += 1
            return ctx
        
        best_ctx = -1
        best_sim = -1
        for ctx, center in self.stored_env_centers.items():
            cn = np.linalg.norm(center)
            sn = np.linalg.norm(env_signal)
            if cn > 1e-8 and sn > 1e-8:
                sim = float(np.dot(center, env_signal) / (cn * sn))
            else:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim
                best_ctx = ctx
        
        if best_sim < self.novelty_threshold and self.n_contexts < self.max_contexts:
            ctx = self._next_context
            self._next_context += 1
            self.n_contexts += 1
            return ctx
        
        return best_ctx
    
    def _infer_context(self, env_signal):
        if len(self.stored_env_centers) == 0:
            return 0, 0.0
        
        best_ctx = -1
        best_sim = -1
        second_sim = -1
        for ctx, center in self.stored_env_centers.items():
            cn = np.linalg.norm(center)
            sn = np.linalg.norm(env_signal)
            if cn > 1e-8 and sn > 1e-8:
                sim = float(np.dot(center, env_signal) / (cn * sn))
            else:
                sim = 0.0
            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_ctx = ctx
            elif sim > second_sim:
                second_sim = sim
        
        confidence = best_sim - second_sim if second_sim > 0 else best_sim
        confidence = max(0.0, min(1.0, confidence * 2))
        
        return best_ctx, confidence
    
    def _compute_context_weights(self, content_sims, env_ctx, env_conf):
        weights = {}
        
        env_weight = 0.5 + 0.5 * env_conf
        other_weight = (1.0 - env_weight) / max(self.n_contexts - 1, 1)
        
        for ctx in range(self.n_contexts):
            if ctx == env_ctx:
                weights[ctx] = env_weight
            else:
                mask = self.stored_contexts == ctx
                if mask.any():
                    max_sim = float(np.max(content_sims[mask]))
                    transfer = max(0, max_sim - 0.3) * 0.5
                    weights[ctx] = other_weight + transfer
                else:
                    weights[ctx] = other_weight
        
        total = sum(weights.values())
        if total > 0:
            weights = {c: w / total for c, w in weights.items()}
        return weights


def normalize_features(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X / norms


def add_noise(X, noise_level=0.0, rng=None):
    if noise_level <= 0:
        return X.copy()
    if rng is None:
        rng = np.random.RandomState(42)
    return X + rng.randn(*X.shape).astype(np.float32) * noise_level


def run_acm_experiment():
    log("=" * 76)
    log("  ACM: Adaptive Context Memory")
    log("  Env-Guided Novelty Detection + Cross-Context Soft Transfer")
    log("=" * 76)
    log()
    log("  FAIR baselines (all get SAME env signal):")
    log("  - kNN: content only (ignores env signal)")
    log("  - Cond-kNN: hard filter by env-inferred context")
    log("  - Soft-Env: soft weighting by env confidence")
    log("  - ACM (Ours): env confidence + content-based transfer")
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
    # Experiment 1: Context-Dependent MNIST with Env Signal
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 1: Context-Dependent Retrieval with Env Signal")
    log("=" * 76)

    n_contexts = 5
    base_digits = [0, 1, 2, 3, 4]
    n_store = 40
    n_query = 50
    env_dim = 16

    label_mappings = {}
    for ctx in range(n_contexts):
        mapping = {}
        for i, d in enumerate(base_digits):
            mapping[d] = (i + ctx) % 10
        label_mappings[ctx] = mapping

    for env_noise in [0.1, 0.3, 0.5, 1.0]:
        log(f"\n  --- Env Noise = {env_noise} ---")
        env_sim = EnvironmentSimulator(n_contexts, env_dim, env_noise, seed=42)
        
        memory = AdaptiveContextMemory(
            feature_dim=256, env_dim=env_dim, novelty_threshold=0.5, seed=42)
        
        for ctx in range(n_contexts):
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_store]
                feat = all_features[idx]
                labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                env_signals = env_sim.get_signals_batch(np.full(len(idx), ctx))
                memory.store_with_env(feat, labels, env_signals)
        
        log(f"  Discovered {memory.n_contexts} contexts (true: {n_contexts})")
        
        log(f"  {'Q-Noise':>8s} | {'kNN':>8s} | {'Cond-kNN':>10s} | {'Soft-Env':>10s} | {'ACM':>8s} | {'ACM-Cnd':>10s} | {'ACM-Sft':>10s}")
        log(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*10} | {'-'*10}")
        
        for q_noise in [0.0, 0.2, 0.3, 0.5]:
            correct_knn = 0
            correct_cond = 0
            correct_soft = 0
            correct_acm = 0
            total = 0
            
            for ctx in range(n_contexts):
                for digit in base_digits:
                    class_idx = np.where(all_labels == digit)[0]
                    rng.shuffle(class_idx)
                    q_idx = class_idx[n_store:n_store + n_query]
                    if len(q_idx) < n_query:
                        continue
                    q_feat = add_noise(all_features[q_idx], q_noise, rng)
                    q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)
                    q_env = env_sim.get_signals_batch(np.full(len(q_idx), ctx))
                    
                    p_knn = memory.predict_knn(q_feat, k=5)
                    p_cond = memory.predict_conditional_knn(q_feat, q_env, k=5)
                    p_soft = memory.predict_soft_env(q_feat, q_env, k=5)
                    p_acm = memory.predict_adaptive(q_feat, q_env, k=5)
                    
                    correct_knn += np.sum(p_knn == q_lbl)
                    correct_cond += np.sum(p_cond == q_lbl)
                    correct_soft += np.sum(p_soft == q_lbl)
                    correct_acm += np.sum(p_acm == q_lbl)
                    total += len(q_idx)
            
            acm_cnd = (correct_acm - correct_cond) / total
            acm_sft = (correct_acm - correct_soft) / total
            log(f"  {q_noise:>8.1f} | {correct_knn/total:>8.4f} | {correct_cond/total:>10.4f} | "
                f"{correct_soft/total:>10.4f} | {correct_acm/total:>8.4f} | "
                f"{acm_cnd:>+10.4f} | {acm_sft:>+10.4f}")

    # ================================================================
    # Experiment 2: Split-MNIST
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 2: Split-MNIST with Env Signal")
    log("=" * 76)

    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    n_items_sm = 50

    for env_noise in [0.1, 0.3, 0.5]:
        log(f"\n  --- Env Noise = {env_noise} ---")
        env_sim_sm = EnvironmentSimulator(5, env_dim, env_noise, seed=100)
        memory_sm = AdaptiveContextMemory(
            feature_dim=256, env_dim=env_dim, novelty_threshold=0.5, seed=100)
        
        for task_id, (la, lb) in enumerate(task_labels):
            for lbl in [la, lb]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                idx = class_idx[:n_items_sm]
                env_signals = env_sim_sm.get_signals_batch(np.full(len(idx), task_id))
                memory_sm.store_with_env(all_features[idx], all_labels[idx], env_signals)
        
        log(f"  Discovered {memory_sm.n_contexts} contexts")
        
        log(f"  {'Q-Noise':>8s} | {'kNN':>8s} | {'Cond-kNN':>10s} | {'Soft-Env':>10s} | {'ACM':>8s} | {'ACM-Cnd':>10s}")
        log(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*10}")
        
        for q_noise in [0.0, 0.2, 0.3, 0.5]:
            correct_knn = 0
            correct_cond = 0
            correct_soft = 0
            correct_acm = 0
            total = 0
            
            for task_id, (la, lb) in enumerate(task_labels):
                for lbl in [la, lb]:
                    class_idx = np.where(all_labels == lbl)[0]
                    rng.shuffle(class_idx)
                    q_idx = class_idx[n_items_sm:n_items_sm + 100]
                    if len(q_idx) < 50:
                        continue
                    q_feat = add_noise(all_features[q_idx], q_noise, rng)
                    q_lbl = all_labels[q_idx]
                    q_env = env_sim_sm.get_signals_batch(np.full(len(q_idx), task_id))
                    
                    p_knn = memory_sm.predict_knn(q_feat, k=5)
                    p_cond = memory_sm.predict_conditional_knn(q_feat, q_env, k=5)
                    p_soft = memory_sm.predict_soft_env(q_feat, q_env, k=5)
                    p_acm = memory_sm.predict_adaptive(q_feat, q_env, k=5)
                    
                    correct_knn += np.sum(p_knn == q_lbl)
                    correct_cond += np.sum(p_cond == q_lbl)
                    correct_soft += np.sum(p_soft == q_lbl)
                    correct_acm += np.sum(p_acm == q_lbl)
                    total += len(q_idx)
            
            acm_cnd = (correct_acm - correct_cond) / total
            log(f"  {q_noise:>8.1f} | {correct_knn/total:>8.4f} | {correct_cond/total:>10.4f} | "
                f"{correct_soft/total:>10.4f} | {correct_acm/total:>8.4f} | {acm_cnd:>+10.4f}")

    # ================================================================
    # Summary
    # ================================================================
    log("\n" + "=" * 76)
    log("  SUMMARY")
    log("=" * 76)
    log()
    log("  ACM-Cnd = ACM accuracy minus Cond-kNN accuracy")
    log("  ACM-Sft = ACM accuracy minus Soft-Env accuracy")
    log()
    log("  If ACM > Cond-kNN: adaptive context weighting helps")
    log("  If ACM > Soft-Env: content-based transfer adds value")
    log("  If ACM < Cond-kNN: hard filtering is better")
    log()
    log("  The KEY differentiator of ACM vs Cond-kNN:")
    log("  When env signal is noisy, ACM reduces env weight and")
    log("  relies more on content similarity for cross-context transfer.")

    return True


if __name__ == "__main__":
    run_acm_experiment()
