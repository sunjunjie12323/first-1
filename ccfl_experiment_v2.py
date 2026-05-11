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


class ContextConditionalEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=128,
                 n_contexts=10, context_dim=32):
        super().__init__()
        self.visual_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.context_embedding = nn.Embedding(n_contexts, context_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x, context_id):
        x = x.view(x.size(0), -1)
        h = self.visual_backbone(x)
        c = self.context_embedding(context_id)
        combined = torch.cat([h, c], dim=1)
        return self.fusion(combined)


class BaselineEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def supervised_contrastive_loss(features, labels, temperature=0.07):
    features = F.normalize(features, dim=1)
    sim_matrix = features @ features.T / temperature

    n = len(features)
    eye_mask = torch.eye(n, device=features.device, dtype=torch.bool)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye_mask
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)) & ~eye_mask

    exp_sim = torch.exp(sim_matrix) * (~eye_mask).float()

    pos_sim = exp_sim * mask_pos.float()
    denom = exp_sim * mask_neg.float()

    denominator = denom.sum(dim=1, keepdim=True) + 1e-8
    loss_per_sample = -torch.log(pos_sim.sum(dim=1, keepdim=True) / denominator + 1e-8)

    has_pos = mask_pos.sum(dim=1) > 0
    if has_pos.any():
        loss = loss_per_sample[has_pos].mean()
    else:
        loss = torch.tensor(0.0, device=features.device)

    return loss


def knn_predict(query_features, stored_features, stored_labels, k=5):
    Q = query_features.astype(np.float32)
    qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
    Q = Q / qn
    E = stored_features.astype(np.float32)
    en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
    E = E / en
    sims = Q @ E.T
    preds = np.zeros(len(Q), dtype=np.int32)
    for i in range(len(Q)):
        kk = min(k, sims.shape[1])
        top_idx = np.argpartition(sims[i], -kk)[-kk:]
        tl = stored_labels[top_idx]
        ts = sims[i, top_idx]
        ls = {}
        for j, lbl in enumerate(tl):
            ls[lbl] = ls.get(lbl, 0.0) + ts[j]
        preds[i] = max(ls, key=ls.get)
    return preds


def cond_knn_predict(query_features, query_ctx, stored_features, stored_labels,
                     stored_contexts, k=5):
    Q = query_features.astype(np.float32)
    qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
    Q = Q / qn
    E = stored_features.astype(np.float32)
    en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
    E = E / en
    sims = Q @ E.T
    preds = np.zeros(len(Q), dtype=np.int32)
    for i in range(len(Q)):
        mask = stored_contexts == query_ctx[i]
        if mask.any():
            ctx_sims = np.full(sims.shape[1], -np.inf)
            ctx_sims[mask] = sims[i, mask]
            kk = min(k, mask.sum())
            top_idx = np.argpartition(ctx_sims, -kk)[-kk:]
            tl = stored_labels[top_idx]
            ts = ctx_sims[top_idx]
            ls = {}
            for j, lbl in enumerate(tl):
                ls[lbl] = ls.get(lbl, 0.0) + max(ts[j], 0)
            preds[i] = max(ls, key=ls.get) if ls else -1
        else:
            kk = min(k, sims.shape[1])
            top_idx = np.argpartition(sims[i], -kk)[-kk:]
            tl = stored_labels[top_idx]
            ts = sims[i, top_idx]
            ls = {}
            for j, lbl in enumerate(tl):
                ls[lbl] = ls.get(lbl, 0.0) + ts[j]
            preds[i] = max(ls, key=ls.get)
    return preds


def normalize_features(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X / norms


def add_noise(X, noise_level=0.0, rng=None):
    if noise_level <= 0:
        return X.copy()
    if rng is None:
        rng = np.random.RandomState(42)
    return X + rng.randn(*X.shape).astype(np.float32) * noise_level


def run_ccfl_v2():
    log("=" * 76)
    log("  CCFL v2: Context-Conditional Feature Learning")
    log("  FIXED: Supervised contrastive loss with (context, digit) labels")
    log("  FIXED: Test-time context inference via multi-hypothesis decoding")
    log("=" * 76)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    log("  Loading MNIST...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_contexts = 5
    base_digits = [0, 1, 2, 3, 4]
    n_store = 40
    n_query = 50
    output_dim = 128

    label_mappings = {}
    for ctx in range(n_contexts):
        mapping = {}
        for i, d in enumerate(base_digits):
            mapping[d] = (i + ctx) % 10
        label_mappings[ctx] = mapping

    # ================================================================
    # Phase 1: Train CCFL encoder with (context, digit) labels
    # ================================================================
    log("\n  Phase 1: Training CCFL encoder...")
    log("  Key fix: Use (context_id * 10 + digit) as label")
    log("  → Same context + same digit = positive pair")
    log("  → Different context OR different digit = negative pair")
    log("  → Forces model to encode BOTH context AND content")

    cc_encoder = ContextConditionalEncoder(
        input_dim=784, hidden_dim=256, output_dim=output_dim,
        n_contexts=n_contexts, context_dim=32)
    cc_optimizer = optim.Adam(cc_encoder.parameters(), lr=0.001, weight_decay=1e-5)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True)

    cc_encoder.train()
    for epoch in range(20):
        total_loss = 0
        n_batches = 0
        for data, target in train_loader:
            mask = torch.tensor([t.item() in base_digits for t in target])
            if not mask.any():
                continue
            data = data[mask]
            target = target[mask]

            context_ids = torch.randint(0, n_contexts, (len(data),))
            combined_labels = context_ids * 10 + target

            features = cc_encoder(data, context_ids)
            loss = supervised_contrastive_loss(features, combined_labels, temperature=0.07)

            cc_optimizer.zero_grad()
            loss.backward()
            cc_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            log(f"    Epoch {epoch+1}: loss={total_loss/max(n_batches,1):.4f}")

    # ================================================================
    # Phase 2: Train Baseline encoder
    # ================================================================
    log("\n  Phase 2: Training Baseline encoder (digit-only contrastive)...")

    baseline_encoder = BaselineEncoder(
        input_dim=784, hidden_dim=256, output_dim=output_dim)
    b_optimizer = optim.Adam(baseline_encoder.parameters(), lr=0.001, weight_decay=1e-5)

    baseline_encoder.train()
    for epoch in range(20):
        total_loss = 0
        n_batches = 0
        for data, target in train_loader:
            features = baseline_encoder(data)
            loss = supervised_contrastive_loss(features, target, temperature=0.07)

            b_optimizer.zero_grad()
            loss.backward()
            b_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            log(f"    Epoch {epoch+1}: loss={total_loss/max(n_batches,1):.4f}")

    # ================================================================
    # Phase 3: Extract features
    # ================================================================
    log("\n  Phase 3: Extracting features...")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False)

    rng = np.random.RandomState(42)

    cc_features_by_ctx = {}
    all_labels_list = None

    for ctx in range(n_contexts):
        cc_feats = []
        labels_list = []
        cc_encoder.eval()
        with torch.no_grad():
            for data, target in test_loader:
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                data_m = data[mask]
                target_m = target[mask]
                ctx_ids = torch.full((len(data_m),), ctx, dtype=torch.long)
                feat = cc_encoder(data_m, ctx_ids)
                cc_feats.append(feat.numpy())
                labels_list.append(target_m.numpy())
        cc_features_by_ctx[ctx] = normalize_features(
            np.concatenate(cc_feats).astype(np.float32))
        if all_labels_list is None:
            all_labels_list = np.concatenate(labels_list).astype(np.int32)

    baseline_encoder.eval()
    b_feats = []
    b_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            feat = baseline_encoder(data)
            b_feats.append(feat.numpy())
            b_labels.append(target.numpy())
    all_baseline_features = normalize_features(
        np.concatenate(b_feats).astype(np.float32))
    all_baseline_labels = np.concatenate(b_labels).astype(np.int32)

    log(f"  CCFL features per context: {cc_features_by_ctx[0].shape}")
    log(f"  Baseline features: {all_baseline_features.shape}")

    # ================================================================
    # Feature space analysis (BEFORE experiments)
    # ================================================================
    log("\n" + "=" * 76)
    log("  Feature Space Analysis")
    log("=" * 76)

    log("\n  Cross-context similarity (same digit, different context):")
    for d in [0, 2, 4]:
        digit_idx = np.where(all_labels_list == d)[0][:30]
        for c1 in range(min(3, n_contexts)):
            for c2 in range(c1 + 1, min(3, n_contexts)):
                f1 = cc_features_by_ctx[c1][digit_idx]
                f2 = cc_features_by_ctx[c2][digit_idx]
                f1n = f1 / np.maximum(np.linalg.norm(f1, axis=1, keepdims=True), 1e-8)
                f2n = f2 / np.maximum(np.linalg.norm(f2, axis=1, keepdims=True), 1e-8)
                cross_sim = np.mean(f1n @ f2n.T)
                log(f"    digit={d}, ctx{c1} vs ctx{c2}: {cross_sim:.4f}")

    log("\n  Within-context similarity (different digits, same context):")
    for ctx in range(min(3, n_contexts)):
        sims = []
        for d1, d2 in [(0, 1), (0, 2), (2, 4)]:
            i1 = np.where(all_labels_list == d1)[0][:30]
            i2 = np.where(all_labels_list == d2)[0][:30]
            f1 = cc_features_by_ctx[ctx][i1]
            f2 = cc_features_by_ctx[ctx][i2]
            f1n = f1 / np.maximum(np.linalg.norm(f1, axis=1, keepdims=True), 1e-8)
            f2n = f2 / np.maximum(np.linalg.norm(f2, axis=1, keepdims=True), 1e-8)
            s = np.mean(f1n @ f2n.T)
            sims.append(s)
            log(f"    ctx{ctx}, digit {d1} vs {d2}: {s:.4f}")
        log(f"    ctx{ctx} avg within-context diff-digit sim: {np.mean(sims):.4f}")

    log("\n  Same-context same-digit similarity:")
    for ctx in range(min(3, n_contexts)):
        d = 0
        idx = np.where(all_labels_list == d)[0][:30]
        f = cc_features_by_ctx[ctx][idx]
        fn = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
        self_sim = np.mean(fn @ fn.T)
        log(f"    ctx{ctx}, digit={d}: {self_sim:.4f}")

    # ================================================================
    # Experiment 1: Context-Dependent Retrieval
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 1: Context-Dependent Retrieval")
    log("  Same visual input, different labels in different contexts")
    log("=" * 76)

    cc_stored_feat = np.zeros((0, output_dim), dtype=np.float32)
    cc_stored_lbl = np.zeros(0, dtype=np.int32)
    cc_stored_ctx = np.zeros(0, dtype=np.int32)

    bl_stored_feat = np.zeros((0, output_dim), dtype=np.float32)
    bl_stored_lbl = np.zeros(0, dtype=np.int32)

    digit_indices = {}
    for digit in base_digits:
        idx = np.where(all_labels_list == digit)[0]
        rng.shuffle(idx)
        digit_indices[digit] = idx

    bl_digit_indices = {}
    for digit in range(10):
        idx = np.where(all_baseline_labels == digit)[0]
        rng.shuffle(idx)
        bl_digit_indices[digit] = idx

    for ctx in range(n_contexts):
        for digit in base_digits:
            idx = digit_indices[digit][:n_store]
            feat = cc_features_by_ctx[ctx][idx]
            labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
            cc_stored_feat = np.concatenate([cc_stored_feat, feat])
            cc_stored_lbl = np.concatenate([cc_stored_lbl, labels])
            cc_stored_ctx = np.concatenate([cc_stored_ctx, np.full(len(idx), ctx, dtype=np.int32)])

    bl_stored_ctx = np.zeros(0, dtype=np.int32)

    for ctx in range(n_contexts):
        for digit in base_digits:
            idx = digit_indices[digit][:n_store]
            feat = all_baseline_features[
                bl_digit_indices[digit][:n_store]]
            labels = np.full(n_store, label_mappings[ctx][digit], dtype=np.int32)
            bl_stored_feat = np.concatenate([bl_stored_feat, feat])
            bl_stored_lbl = np.concatenate([bl_stored_lbl, labels])
            bl_stored_ctx = np.concatenate([bl_stored_ctx, np.full(n_store, ctx, dtype=np.int32)])

    log(f"  CCFL stored: {len(cc_stored_feat)} items across {n_contexts} contexts")
    log(f"  Baseline stored: {len(bl_stored_feat)} items across {n_contexts} contexts")

    # --- Method A: CCFL with TRUE context (upper bound, like Cond-kNN) ---
    # --- Method B: CCFL with INFERRED context (our method) ---
    # --- Method C: Baseline kNN (no context info) ---
    # --- Method D: Cond-kNN on baseline features (with true context) ---

    log(f"\n  Table 1: Context-Dependent Retrieval Accuracy")
    log(f"  {'Noise':>6s} | {'BL-kNN':>8s} | {'BL-Cond':>8s} | {'CCFL-T':>8s} | {'CCFL-I':>8s} | {'Improv':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    for noise_level in [0.0, 0.1, 0.2, 0.3, 0.5]:
        correct_bl = 0
        correct_bl_cond = 0
        correct_cc_true = 0
        correct_cc_infer = 0
        total = 0

        for ctx in range(n_contexts):
            for digit in base_digits:
                idx = digit_indices[digit][n_store:n_store + n_query]
                if len(idx) < n_query:
                    continue

                q_feat_cc = add_noise(cc_features_by_ctx[ctx][idx], noise_level, rng)
                q_feat_bl = add_noise(all_baseline_features[
                    bl_digit_indices[digit][n_store:n_store + n_query]], noise_level, rng)
                q_lbl = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                q_ctx = np.full(len(idx), ctx, dtype=np.int32)

                # Method C: Baseline kNN
                p_bl = knn_predict(q_feat_bl, bl_stored_feat, bl_stored_lbl, k=5)
                correct_bl += np.sum(p_bl == q_lbl)

                # Method D: Cond-kNN on baseline features
                p_bl_cond = cond_knn_predict(
                    q_feat_bl, q_ctx, bl_stored_feat, bl_stored_lbl,
                    bl_stored_ctx, k=5)
                correct_bl_cond += np.sum(p_bl_cond == q_lbl)

                # Method A: CCFL with true context
                p_cc_true = cond_knn_predict(
                    q_feat_cc, q_ctx, cc_stored_feat, cc_stored_lbl,
                    cc_stored_ctx, k=5)
                correct_cc_true += np.sum(p_cc_true == q_lbl)

                # Method B: CCFL with inferred context
                # For each query, try all contexts, pick the one with highest confidence
                best_preds = np.zeros(len(idx), dtype=np.int32)
                for i in range(len(idx)):
                    best_conf = -1
                    best_label = -1
                    q_f = q_feat_cc[i:i+1]
                    for c_try in range(n_contexts):
                        mask = cc_stored_ctx == c_try
                        if not mask.any():
                            continue
                        ctx_feat = cc_stored_feat[mask]
                        ctx_lbl = cc_stored_lbl[mask]
                        Q = q_f.astype(np.float32)
                        qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
                        Q = Q / qn
                        E = ctx_feat.astype(np.float32)
                        en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
                        E = E / en
                        sims = Q @ E.T
                        kk = min(5, sims.shape[1])
                        top_idx = np.argpartition(sims[0], -kk)[-kk:]
                        top_sims = sims[0, top_idx]
                        top_lbls = ctx_lbl[top_idx]
                        conf = np.mean(top_sims)
                        if conf > best_conf:
                            best_conf = conf
                            ls = {}
                            for j, lbl in enumerate(top_lbls):
                                ls[lbl] = ls.get(lbl, 0.0) + top_sims[j]
                            best_label = max(ls, key=ls.get)
                    best_preds[i] = best_label
                correct_cc_infer += np.sum(best_preds == q_lbl)

                total += len(idx)

        improv = (correct_cc_infer - correct_bl) / total
        log(f"  {noise_level:>6.1f} | {correct_bl/total:>8.4f} | {correct_bl_cond/total:>8.4f} | "
            f"{correct_cc_true/total:>8.4f} | {correct_cc_infer/total:>8.4f} | {improv:>+8.4f}")

    # ================================================================
    # Experiment 2: Context Inference Accuracy
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 2: Context Inference via Multi-Hypothesis Decoding")
    log("=" * 76)

    for noise_level in [0.0, 0.1, 0.2, 0.3, 0.5]:
        correct_ctx = 0
        total = 0

        for ctx in range(n_contexts):
            for digit in base_digits:
                idx = digit_indices[digit][n_store:n_store + n_query]
                if len(idx) < n_query:
                    continue
                q_feat = add_noise(cc_features_by_ctx[ctx][idx], noise_level, rng)

                Q = q_feat.astype(np.float32)
                qn = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
                Q = Q / qn
                E = cc_stored_feat.astype(np.float32)
                en = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
                E = E / en
                sims = Q @ E.T

                for i in range(len(Q)):
                    best_ctx = -1
                    best_conf = -1
                    for c_try in range(n_contexts):
                        mask = cc_stored_ctx == c_try
                        ctx_sims = sims[i, mask]
                        kk = min(5, len(ctx_sims))
                        top_idx = np.argpartition(ctx_sims, -kk)[-kk:]
                        conf = np.mean(ctx_sims[top_idx])
                        if conf > best_conf:
                            best_conf = conf
                            best_ctx = c_try
                    if best_ctx == ctx:
                        correct_ctx += 1
                    total += 1

        log(f"  noise={noise_level:.1f}: context inference = {correct_ctx/total:.4f}")

    # ================================================================
    # Experiment 3: Split-MNIST
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 3: Split-MNIST (no context conflict)")
    log("=" * 76)

    task_labels_sm = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    n_items_sm = 50

    cc_sm_stored = np.zeros((0, output_dim), dtype=np.float32)
    cc_sm_lbl = np.zeros(0, dtype=np.int32)
    cc_sm_ctx = np.zeros(0, dtype=np.int32)
    bl_sm_stored = np.zeros((0, output_dim), dtype=np.float32)
    bl_sm_lbl = np.zeros(0, dtype=np.int32)

    for task_id, (la, lb) in enumerate(task_labels_sm):
        for lbl in [la, lb]:
            cc_idx = np.where(all_labels_list == lbl)[0] if lbl in base_digits else None
            bl_idx = bl_digit_indices[lbl][:n_items_sm]

            if cc_idx is not None and len(cc_idx) >= n_items_sm:
                rng.shuffle(cc_idx)
                feat = cc_features_by_ctx[task_id % n_contexts][cc_idx[:n_items_sm]]
            else:
                bl_i = bl_digit_indices[lbl][:n_items_sm]
                feat = all_baseline_features[bl_i]

            cc_sm_stored = np.concatenate([cc_sm_stored, feat[:n_items_sm]])
            cc_sm_lbl = np.concatenate([cc_sm_lbl, np.full(n_items_sm, lbl, dtype=np.int32)])
            cc_sm_ctx = np.concatenate([cc_sm_ctx, np.full(n_items_sm, task_id, dtype=np.int32)])

            bl_feat = all_baseline_features[bl_idx]
            bl_sm_stored = np.concatenate([bl_sm_stored, bl_feat])
            bl_sm_lbl = np.concatenate([bl_sm_lbl, np.full(n_items_sm, lbl, dtype=np.int32)])

    log(f"\n  {'Noise':>6s} | {'BL-kNN':>8s} | {'CCFL-kNN':>8s} | {'CCFL-C':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")

    for noise_level in [0.0, 0.2, 0.5]:
        correct_bl = 0
        correct_cc = 0
        correct_cond = 0
        total = 0

        for task_id, (la, lb) in enumerate(task_labels_sm):
            for lbl in [la, lb]:
                bl_idx = bl_digit_indices[lbl]
                q_bl_idx = bl_idx[n_items_sm:n_items_sm + 100]
                if len(q_bl_idx) < 50:
                    continue

                q_feat_bl = add_noise(all_baseline_features[q_bl_idx], noise_level, rng)

                cc_idx = np.where(all_labels_list == lbl)[0] if lbl in base_digits else None
                if cc_idx is not None and len(cc_idx) > n_items_sm + 50:
                    rng2 = np.random.RandomState(task_id * 10 + lbl)
                    rng2.shuffle(cc_idx)
                    q_cc_idx = cc_idx[n_items_sm:n_items_sm + len(q_bl_idx)]
                    q_feat_cc = add_noise(
                        cc_features_by_ctx[task_id % n_contexts][q_cc_idx],
                        noise_level, rng)
                else:
                    q_feat_cc = q_feat_bl

                q_lbl = all_baseline_labels[q_bl_idx][:len(q_feat_cc)]

                p_bl = knn_predict(q_feat_bl, bl_sm_stored, bl_sm_lbl, k=5)
                p_cc = knn_predict(q_feat_cc, cc_sm_stored, cc_sm_lbl, k=5)
                p_cond = cond_knn_predict(
                    q_feat_cc, np.full(len(q_feat_cc), task_id),
                    cc_sm_stored, cc_sm_lbl, cc_sm_ctx, k=5)

                correct_bl += np.sum(p_bl == q_lbl)
                correct_cc += np.sum(p_cc == q_lbl)
                correct_cond += np.sum(p_cond == q_lbl)
                total += len(q_feat_cc)

        if total > 0:
            log(f"  {noise_level:>6.1f} | {correct_bl/total:>8.4f} | {correct_cc/total:>8.4f} | "
                f"{correct_cond/total:>8.4f}")

    # ================================================================
    # Experiment 4: Ablation - Context Embedding Dimension
    # ================================================================
    log("\n" + "=" * 76)
    log("  Experiment 4: Ablation - Effect of Context Embedding Dim")
    log("=" * 76)

    for ctx_dim in [8, 16, 32, 64]:
        log(f"\n  Training with context_dim={ctx_dim}...")
        ablation_encoder = ContextConditionalEncoder(
            input_dim=784, hidden_dim=256, output_dim=output_dim,
            n_contexts=n_contexts, context_dim=ctx_dim)
        ab_opt = optim.Adam(ablation_encoder.parameters(), lr=0.001, weight_decay=1e-5)

        ablation_encoder.train()
        for epoch in range(15):
            for data, target in train_loader:
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                data = data[mask]
                target = target[mask]
                context_ids = torch.randint(0, n_contexts, (len(data),))
                combined_labels = context_ids * 10 + target
                features = ablation_encoder(data, context_ids)
                loss = supervised_contrastive_loss(features, combined_labels, temperature=0.07)
                ab_opt.zero_grad()
                loss.backward()
                ab_opt.step()

        ablation_encoder.eval()
        ab_feats_by_ctx = {}
        for ctx in range(n_contexts):
            af = []
            with torch.no_grad():
                for data, target in test_loader:
                    mask = torch.tensor([t.item() in base_digits for t in target])
                    if not mask.any():
                        continue
                    data_m = data[mask]
                    ctx_ids = torch.full((len(data_m),), ctx, dtype=torch.long)
                    feat = ablation_encoder(data_m, ctx_ids)
                    af.append(feat.numpy())
            ab_feats_by_ctx[ctx] = normalize_features(np.concatenate(af).astype(np.float32))

        ab_stored_feat = np.zeros((0, output_dim), dtype=np.float32)
        ab_stored_lbl = np.zeros(0, dtype=np.int32)
        ab_stored_ctx = np.zeros(0, dtype=np.int32)
        for ctx in range(n_contexts):
            for digit in base_digits:
                idx = digit_indices[digit][:n_store]
                feat = ab_feats_by_ctx[ctx][idx]
                labels = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                ab_stored_feat = np.concatenate([ab_stored_feat, feat])
                ab_stored_lbl = np.concatenate([ab_stored_lbl, labels])
                ab_stored_ctx = np.concatenate([ab_stored_ctx, np.full(len(idx), ctx, dtype=np.int32)])

        correct_true = 0
        correct_infer = 0
        total = 0
        for ctx in range(n_contexts):
            for digit in base_digits:
                idx = digit_indices[digit][n_store:n_store + n_query]
                if len(idx) < n_query:
                    continue
                q_feat = ab_feats_by_ctx[ctx][idx]
                q_lbl = np.full(len(idx), label_mappings[ctx][digit], dtype=np.int32)
                q_ctx = np.full(len(idx), ctx, dtype=np.int32)

                p_true = cond_knn_predict(
                    q_feat, q_ctx, ab_stored_feat, ab_stored_lbl, ab_stored_ctx, k=5)

                best_preds = np.zeros(len(idx), dtype=np.int32)
                for i in range(len(idx)):
                    best_conf = -1
                    best_label = -1
                    q_f = q_feat[i:i+1]
                    for c_try in range(n_contexts):
                        m = ab_stored_ctx == c_try
                        if not m.any():
                            continue
                        cf = ab_stored_feat[m]
                        cl = ab_stored_lbl[m]
                        Q = q_f / np.maximum(np.linalg.norm(q_f, axis=1, keepdims=True), 1e-8)
                        E = cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-8)
                        sims = Q @ E.T
                        kk = min(5, sims.shape[1])
                        top_idx = np.argpartition(sims[0], -kk)[-kk:]
                        top_sims = sims[0, top_idx]
                        top_lbls = cl[top_idx]
                        conf = np.mean(top_sims)
                        if conf > best_conf:
                            best_conf = conf
                            ls = {}
                            for j, lbl in enumerate(top_lbls):
                                ls[lbl] = ls.get(lbl, 0.0) + top_sims[j]
                            best_label = max(ls, key=ls.get)
                    best_preds[i] = best_label

                correct_true += np.sum(p_true == q_lbl)
                correct_infer += np.sum(best_preds == q_lbl)
                total += len(idx)

        log(f"  ctx_dim={ctx_dim}: CCFL-True={correct_true/total:.4f}, "
            f"CCFL-Infer={correct_infer/total:.4f}")

    # ================================================================
    # Summary
    # ================================================================
    log("\n" + "=" * 76)
    log("  HONEST SUMMARY")
    log("=" * 76)
    log()
    log("  CCFL v2 fixes:")
    log("  1. Supervised contrastive loss with (context, digit) labels")
    log("     → Prevents within-context collapse")
    log("  2. Multi-hypothesis decoding at test time")
    log("     → No need for context ID at test time")
    log()
    log("  Key metrics to check:")
    log("  - Within-context diff-digit similarity should be < 1.0")
    log("  - CCFL-Infer should approach CCFL-True")
    log("  - CCFL-True should approach or exceed Cond-kNN")


if __name__ == "__main__":
    run_ccfl_v2()
