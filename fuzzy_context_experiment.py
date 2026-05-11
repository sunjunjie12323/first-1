from __future__ import annotations

import sys
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


class ContextBarcodeEncoder:
    def __init__(self, max_contexts=20, barcode_dim=512, sparsity=64, seed=42):
        self.max_contexts = max_contexts
        self.barcode_dim = barcode_dim
        self.sparsity = sparsity
        self.barcodes = {}
        self._seed = seed

    def get_barcode(self, cid):
        if cid not in self.barcodes:
            rng = np.random.RandomState(self._seed + cid * 137)
            raw = rng.randn(self.barcode_dim).astype(np.float32)
            top_idx = np.argpartition(raw, -self.sparsity)[-self.sparsity:]
            bc = np.zeros(self.barcode_dim, dtype=np.float32)
            bc[top_idx] = np.maximum(raw[top_idx], 0.0)
            norm = np.linalg.norm(bc)
            if norm > 1e-8:
                bc /= norm
            self.barcodes[cid] = bc
        return self.barcodes[cid]

    def get_mixed_barcode(self, context_weights):
        result = np.zeros(self.barcode_dim, dtype=np.float32)
        for cid, weight in context_weights.items():
            result += weight * self.get_barcode(cid)
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm
        return result

    def get_barcodes_batch(self, cids):
        return np.array([self.get_barcode(int(c)) for c in cids])


class DualChannelMemory:
    def __init__(self, feature_dim=256, barcode_dim=512, sparsity=64,
                 lambda_param=0.5, seed=42):
        self.feature_dim = feature_dim
        self.barcode_dim = barcode_dim
        self.lambda_param = lambda_param
        self.barcode_encoder = ContextBarcodeEncoder(20, barcode_dim, sparsity, seed)
        self.stored_features = np.zeros((0, feature_dim), dtype=np.float32)
        self.stored_barcodes = np.zeros((0, barcode_dim), dtype=np.float32)
        self.stored_labels = np.zeros(0, dtype=np.int32)
        self.stored_contexts = np.zeros(0, dtype=np.int32)

    def store(self, features, labels, context_ids):
        barcodes = self.barcode_encoder.get_barcodes_batch(context_ids)
        self.stored_features = np.concatenate([self.stored_features, features])
        self.stored_barcodes = np.concatenate([self.stored_barcodes, barcodes])
        self.stored_labels = np.concatenate([self.stored_labels, labels])
        self.stored_contexts = np.concatenate([self.stored_contexts, context_ids])

    def predict_dual(self, query_features, query_context_ids, lam=None):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        lam = lam if lam is not None else self.lambda_param
        c = self._content_scores(query_features)
        q_bc = self.barcode_encoder.get_barcodes_batch(query_context_ids)
        b = self._barcode_scores(q_bc)
        combined = self._combine(c, b, lam)
        return self.stored_labels[np.argmax(combined, axis=1)]

    def predict_dual_mixed(self, query_features, context_weights_list, lam=None):
        if len(self.stored_features) == 0:
            return np.full(len(query_features), -1, dtype=np.int32)
        lam = lam if lam is not None else self.lambda_param
        c = self._content_scores(query_features)
        q_bc = np.array([self.barcode_encoder.get_mixed_barcode(w)
                         for w in context_weights_list])
        b = self._barcode_scores(q_bc)
        combined = self._combine(c, b, lam)
        return self.stored_labels[np.argmax(combined, axis=1)]

    def predict_conditional_knn(self, query_features, query_context_ids, k=5):
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
            mask = self.stored_contexts == query_context_ids[i]
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

    def predict_weighted_knn(self, query_features, context_weights_list, k=5):
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
            weights = np.zeros(sims.shape[1], dtype=np.float32)
            for cid, w in context_weights_list[i].items():
                mask = self.stored_contexts == cid
                weights[mask] = w
            weighted_sims = sims[i] * (1.0 + weights)
            kk = min(k, sims.shape[1])
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

    def _content_scores(self, Q):
        Q = Q.astype(np.float32)
        qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Q = Q / qn
        E = self.stored_features.astype(np.float32)
        en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        E = E / en
        return (Q @ E.T).astype(np.float32)

    def _barcode_scores(self, Q_bc):
        qn = np.maximum(np.linalg.norm(Q_bc, axis=1, keepdims=True), 1e-8)
        Q = Q_bc / qn
        B = self.stored_barcodes.astype(np.float32)
        bn = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-8)
        B = B / bn
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


def run_fuzzy_context_experiment():
    log("=" * 76)
    log("  KEY EXPERIMENT: Fuzzy/Continuous Context")
    log("  When context is NOT a discrete ID but a continuous mixture")
    log("=" * 76)
    log()
    log("  Scenario: Robot moves between rooms gradually.")
    log("  At the boundary, context is a MIXTURE of two rooms.")
    log("  - Cond-kNN: must pick ONE context (hard decision)")
    log("  - Ours: can use MIXED barcode (soft decision)")
    log("  - Weighted-kNN: uses context weights to weight kNN scores")
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

    memory = DualChannelMemory(
        feature_dim=256, barcode_dim=512, sparsity=64,
        lambda_param=0.5, seed=42)

    for ctx in range(n_contexts):
        for digit in base_digits:
            class_idx = np.where(all_labels == digit)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_store]
            feat = all_features[idx]
            labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
            memory.store(feat, labels, np.full(len(idx), ctx, dtype=np.int32))

    log(f"  Stored {len(memory.stored_features)} items in {n_contexts} contexts")

    # ================================================================
    # Experiment: Varying context fuzziness (alpha)
    # alpha=1.0: pure context A (sharp boundary)
    # alpha=0.5: 50-50 mixture of A and B (at boundary)
    # alpha=0.0: pure context B (sharp boundary)
    # ================================================================
    log("\n" + "=" * 76)
    log("  Fuzzy Context: alpha blends between true context and adjacent")
    log("  alpha=1.0: 100% true context (sharp)")
    log("  alpha=0.5: 50% true + 50% adjacent (at boundary)")
    log("  alpha=0.0: 100% adjacent (wrong context)")
    log("=" * 76)

    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    query_noise = 0.2

    log(f"\n  Table: Retrieval Accuracy vs Context Fuzziness")
    log(f"  {'alpha':>8s} | {'kNN':>8s} | {'Cond-kNN':>10s} | {'Wtd-kNN':>8s} | {'Ours':>8s} | {'Ours-Win':>10s}")
    log(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*10}")

    for alpha in alphas:
        correct_knn = 0
        correct_cond = 0
        correct_weighted = 0
        correct_ours = 0
        total = 0

        for ctx in range(n_contexts):
            adjacent_ctx = (ctx + 1) % n_contexts
            for digit in base_digits:
                class_idx = np.where(all_labels == digit)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_store:n_store + n_query]
                if len(q_idx) < n_query:
                    continue
                q_feat = add_noise(all_features[q_idx], query_noise, rng)
                q_lbl = np.full(len(q_idx), label_mappings[ctx][digit], dtype=np.int32)

                if alpha >= 0.5:
                    hard_ctx = ctx
                else:
                    hard_ctx = adjacent_ctx

                context_weights = {ctx: alpha, adjacent_ctx: 1.0 - alpha}

                p_knn = memory.predict_knn(q_feat, k=5)
                p_cond = memory.predict_conditional_knn(
                    q_feat, np.full(len(q_idx), hard_ctx), k=5)
                p_weighted = memory.predict_weighted_knn(
                    q_feat, [context_weights] * len(q_idx), k=5)
                p_ours = memory.predict_dual_mixed(
                    q_feat, [context_weights] * len(q_idx), lam=0.5)

                correct_knn += np.sum(p_knn == q_lbl)
                correct_cond += np.sum(p_cond == q_lbl)
                correct_weighted += np.sum(p_weighted == q_lbl)
                correct_ours += np.sum(p_ours == q_lbl)
                total += len(q_idx)

        ours_win = correct_ours - correct_cond
        log(f"  {alpha:>8.1f} | {correct_knn/total:>8.4f} | {correct_cond/total:>10.4f} | "
            f"{correct_weighted/total:>8.4f} | {correct_ours/total:>8.4f} | {ours_win/total:>+10.4f}")

    # ================================================================
    # Split-MNIST version
    # ================================================================
    log("\n" + "=" * 76)
    log("  Split-MNIST with Fuzzy Context")
    log("=" * 76)

    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    n_items_sm = 50

    memory_sm = DualChannelMemory(
        feature_dim=256, barcode_dim=512, sparsity=64,
        lambda_param=0.5, seed=100)

    for task_id, (la, lb) in enumerate(task_labels):
        for lbl in [la, lb]:
            class_idx = np.where(all_labels == lbl)[0]
            rng.shuffle(class_idx)
            idx = class_idx[:n_items_sm]
            memory_sm.store(all_features[idx], all_labels[idx],
                           np.full(len(idx), task_id, dtype=np.int32))

    log(f"\n  {'alpha':>8s} | {'kNN':>8s} | {'Cond-kNN':>10s} | {'Wtd-kNN':>8s} | {'Ours':>8s} | {'Ours-Win':>10s}")
    log(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*10}")

    for alpha in alphas:
        correct_knn = 0
        correct_cond = 0
        correct_weighted = 0
        correct_ours = 0
        total = 0

        for task_id, (la, lb) in enumerate(task_labels):
            adjacent = (task_id + 1) % 5
            for lbl in [la, lb]:
                class_idx = np.where(all_labels == lbl)[0]
                rng.shuffle(class_idx)
                q_idx = class_idx[n_items_sm:n_items_sm + 100]
                if len(q_idx) < 50:
                    continue
                q_feat = add_noise(all_features[q_idx], 0.3, rng)
                q_lbl = all_labels[q_idx]

                hard_ctx = task_id if alpha >= 0.5 else adjacent
                context_weights = {task_id: alpha, adjacent: 1.0 - alpha}

                p_knn = memory_sm.predict_knn(q_feat, k=5)
                p_cond = memory_sm.predict_conditional_knn(
                    q_feat, np.full(len(q_idx), hard_ctx), k=5)
                p_weighted = memory_sm.predict_weighted_knn(
                    q_feat, [context_weights] * len(q_idx), k=5)
                p_ours = memory_sm.predict_dual_mixed(
                    q_feat, [context_weights] * len(q_idx), lam=0.5)

                correct_knn += np.sum(p_knn == q_lbl)
                correct_cond += np.sum(p_cond == q_lbl)
                correct_weighted += np.sum(p_weighted == q_lbl)
                correct_ours += np.sum(p_ours == q_lbl)
                total += len(q_idx)

        ours_win = correct_ours - correct_cond
        log(f"  {alpha:>8.1f} | {correct_knn/total:>8.4f} | {correct_cond/total:>10.4f} | "
            f"{correct_weighted/total:>8.4f} | {correct_ours/total:>8.4f} | {ours_win/total:>+10.4f}")

    # ================================================================
    # Summary
    # ================================================================
    log("\n" + "=" * 76)
    log("  SUMMARY")
    log("=" * 76)
    log()
    log("  If Ours > Cond-kNN at alpha=0.5 (boundary):")
    log("    → Dual-channel provides genuine advantage for fuzzy context")
    log("    → This is the paper's core contribution")
    log()
    log("  If Ours ≈ Cond-kNN at alpha=0.5:")
    log("    → Dual-channel doesn't help even for fuzzy context")
    log("    → Need to find a different angle or accept lower tier")
    log()
    log("  If Ours < Cond-kNN at alpha=0.5:")
    log("    → Dual-channel is actively harmful")
    log("    → Fundamental rethink needed")

    return True


if __name__ == "__main__":
    run_fuzzy_context_experiment()
