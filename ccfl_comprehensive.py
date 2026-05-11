from __future__ import annotations

import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict


def log(msg=""):
    print(msg, flush=True)


# ============================================================
# Models
# ============================================================

class CCFLEncoder(nn.Module):
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

    def forward(self, x, context_id):
        x = x.view(x.size(0), -1)
        h = self.visual_backbone(x)
        c = self.context_embedding(context_id)
        return self.fusion(torch.cat([h, c], dim=1))


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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ============================================================
# Loss functions
# ============================================================

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
        return loss_per_sample[has_pos].mean()
    return torch.tensor(0.0, device=features.device)


# ============================================================
# kNN utilities
# ============================================================

def normalize_features(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X / norms


def knn_predict(query, stored, labels, k=5):
    Q = normalize_features(query.astype(np.float32))
    E = normalize_features(stored.astype(np.float32))
    sims = Q @ E.T
    preds = np.zeros(len(Q), dtype=np.int32)
    for i in range(len(Q)):
        kk = min(k, sims.shape[1])
        top_idx = np.argpartition(sims[i], -kk)[-kk:]
        tl = labels[top_idx]
        ts = sims[i, top_idx]
        ls = {}
        for j, lbl in enumerate(tl):
            ls[lbl] = ls.get(lbl, 0.0) + ts[j]
        preds[i] = max(ls, key=ls.get)
    return preds


def cond_knn_predict(query, q_ctx, stored, labels, s_ctx, k=5):
    Q = normalize_features(query.astype(np.float32))
    E = normalize_features(stored.astype(np.float32))
    sims = Q @ E.T
    preds = np.zeros(len(Q), dtype=np.int32)
    for i in range(len(Q)):
        mask = s_ctx == q_ctx[i]
        if mask.any():
            cs = np.full(sims.shape[1], -np.inf)
            cs[mask] = sims[i, mask]
            kk = min(k, mask.sum())
            top_idx = np.argpartition(cs, -kk)[-kk:]
            tl = labels[top_idx]
            ts = cs[top_idx]
            ls = {}
            for j, lbl in enumerate(tl):
                ls[lbl] = ls.get(lbl, 0.0) + max(ts[j], 0)
            preds[i] = max(ls, key=ls.get) if ls else -1
        else:
            kk = min(k, sims.shape[1])
            top_idx = np.argpartition(sims[i], -kk)[-kk:]
            tl = labels[top_idx]
            ts = sims[i, top_idx]
            ls = {}
            for j, lbl in enumerate(tl):
                ls[lbl] = ls.get(lbl, 0.0) + ts[j]
            preds[i] = max(ls, key=ls.get)
    return preds


def multi_hypothesis_predict(query_feat, stored_feat, stored_lbl, stored_ctx,
                             n_contexts, k=5):
    preds = np.zeros(len(query_feat), dtype=np.int32)
    inferred_ctx = np.zeros(len(query_feat), dtype=np.int32)
    confidences = np.zeros(len(query_feat), dtype=np.float32)
    for i in range(len(query_feat)):
        best_conf = -np.inf
        best_label = -1
        best_ctx = -1
        q_f = query_feat[i:i+1]
        Q = normalize_features(q_f.astype(np.float32))
        for c_try in range(n_contexts):
            m = stored_ctx == c_try
            if not m.any():
                continue
            E = normalize_features(stored_feat[m].astype(np.float32))
            cl = stored_lbl[m]
            sims = Q @ E.T
            kk = min(k, sims.shape[1])
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
                best_ctx = c_try
        preds[i] = best_label
        inferred_ctx[i] = best_ctx
        confidences[i] = best_conf
    return preds, inferred_ctx, confidences


# ============================================================
# EWC Implementation
# ============================================================

class EWC:
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)
        self.priors = {n: p.data.clone() for n, p in self.params.items()}

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * len(data)
        n_samples = len(dataloader.dataset)
        return {n: f / n_samples for n, f in fisher.items()}

    def penalty(self):
        loss = 0
        for n, p in self.params.items():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.priors[n]).pow(2)).sum()
        return loss


# ============================================================
# Experiment 1: Context-Dependent MNIST (main result)
# ============================================================

def experiment_context_dependent_mnist():
    log("=" * 76)
    log("  Experiment 1: Context-Dependent MNIST")
    log("=" * 76)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_contexts = 5
    base_digits = [0, 1, 2, 3, 4]
    n_store = 40
    n_query = 50
    output_dim = 128

    label_mappings = {}
    for ctx in range(n_contexts):
        label_mappings[ctx] = {d: (i + ctx) % 10 for i, d in enumerate(base_digits)}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Train CCFL
    log("  Training CCFL encoder...")
    cc_encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                              n_contexts=n_contexts, context_dim=32)
    cc_opt = optim.Adam(cc_encoder.parameters(), lr=0.001, weight_decay=1e-5)
    cc_encoder.train()
    for epoch in range(20):
        for data, target in train_loader:
            mask = torch.tensor([t.item() in base_digits for t in target])
            if not mask.any():
                continue
            data, target = data[mask], target[mask]
            ctx_ids = torch.randint(0, n_contexts, (len(data),))
            combined = ctx_ids * 10 + target
            feat = cc_encoder(data, ctx_ids)
            loss = supervised_contrastive_loss(feat, combined, 0.07)
            cc_opt.zero_grad()
            loss.backward()
            cc_opt.step()

    # Train Baseline
    log("  Training Baseline encoder...")
    bl_encoder = BaselineEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim)
    bl_opt = optim.Adam(bl_encoder.parameters(), lr=0.001, weight_decay=1e-5)
    bl_encoder.train()
    for epoch in range(20):
        for data, target in train_loader:
            feat = bl_encoder(data)
            loss = supervised_contrastive_loss(feat, target, 0.07)
            bl_opt.zero_grad()
            loss.backward()
            bl_opt.step()

    # Extract features
    rng = np.random.RandomState(42)
    cc_feats = {}
    all_labels = None
    cc_encoder.eval()
    for ctx in range(n_contexts):
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                dm, tm = data[mask], target[mask]
                cid = torch.full((len(dm),), ctx, dtype=torch.long)
                fs.append(cc_encoder(dm, cid).numpy())
                ls.append(tm.numpy())
        cc_feats[ctx] = normalize_features(np.concatenate(fs).astype(np.float32))
        if all_labels is None:
            all_labels = np.concatenate(ls).astype(np.int32)

    bl_encoder.eval()
    bfs, bls = [], []
    with torch.no_grad():
        for data, target in test_loader:
            bfs.append(bl_encoder(data).numpy())
            bls.append(target.numpy())
    bl_all = normalize_features(np.concatenate(bfs).astype(np.float32))
    bl_labels = np.concatenate(bls).astype(np.int32)

    digit_idx = {d: np.where(all_labels == d)[0] for d in base_digits}
    for d in base_digits:
        rng.shuffle(digit_idx[d])
    bl_digit_idx = {d: np.where(bl_labels == d)[0] for d in range(10)}
    for d in range(10):
        rng.shuffle(bl_digit_idx[d])

    # Build stored sets
    cc_stored, cc_slbl, cc_sctx = [], [], []
    bl_stored, bl_slbl, bl_sctx = [], [], []
    for ctx in range(n_contexts):
        for d in base_digits:
            idx = digit_idx[d][:n_store]
            cc_stored.append(cc_feats[ctx][idx])
            cc_slbl.append(np.full(n_store, label_mappings[ctx][d], dtype=np.int32))
            cc_sctx.append(np.full(n_store, ctx, dtype=np.int32))
            bidx = bl_digit_idx[d][:n_store]
            bl_stored.append(bl_all[bidx])
            bl_slbl.append(np.full(n_store, label_mappings[ctx][d], dtype=np.int32))
            bl_sctx.append(np.full(n_store, ctx, dtype=np.int32))

    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)
    bl_stored = np.concatenate(bl_stored)
    bl_slbl = np.concatenate(bl_slbl)
    bl_sctx = np.concatenate(bl_sctx)

    # Evaluate
    log(f"\n  {'Noise':>6s} | {'BL-kNN':>8s} | {'BL-Cond':>8s} | {'CCFL-T':>8s} | {'CCFL-I':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    results = {}
    for noise in [0.0, 0.1, 0.2, 0.3, 0.5]:
        c_bl, c_bc, c_ct, c_ci, total = 0, 0, 0, 0, 0
        for ctx in range(n_contexts):
            for d in base_digits:
                idx = digit_idx[d][n_store:n_store+n_query]
                if len(idx) < n_query:
                    continue
                qcc = cc_feats[ctx][idx]
                qbl = bl_all[bl_digit_idx[d][n_store:n_store+n_query]]
                qlbl = np.full(n_query, label_mappings[ctx][d], dtype=np.int32)
                qctx = np.full(n_query, ctx, dtype=np.int32)

                p_bl = knn_predict(qbl, bl_stored, bl_slbl, k=5)
                p_bc = cond_knn_predict(qbl, qctx, bl_stored, bl_slbl, bl_sctx, k=5)
                p_ct = cond_knn_predict(qcc, qctx, cc_stored, cc_slbl, cc_sctx, k=5)
                p_ci, _, _ = multi_hypothesis_predict(qcc, cc_stored, cc_slbl, cc_sctx, n_contexts, k=5)

                c_bl += (p_bl == qlbl).sum()
                c_bc += (p_bc == qlbl).sum()
                c_ct += (p_ct == qlbl).sum()
                c_ci += (p_ci == qlbl).sum()
                total += n_query

        r = {'BL-kNN': c_bl/total, 'BL-Cond': c_bc/total,
             'CCFL-T': c_ct/total, 'CCFL-I': c_ci/total}
        results[noise] = r
        log(f"  {noise:>6.1f} | {r['BL-kNN']:>8.4f} | {r['BL-Cond']:>8.4f} | "
            f"{r['CCFL-T']:>8.4f} | {r['CCFL-I']:>8.4f}")

    return results


# ============================================================
# Experiment 2: Permuted-MNIST
# ============================================================

def experiment_permuted_mnist():
    log("\n" + "=" * 76)
    log("  Experiment 2: Permuted-MNIST (Standard CL Benchmark)")
    log("=" * 76)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_tasks = 5
    output_dim = 128
    rng = np.random.RandomState(42)
    permutations = [rng.permutation(784) for _ in range(n_tasks)]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Train CCFL on all tasks jointly
    log("  Training CCFL encoder on all tasks...")
    cc_encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                              n_contexts=n_tasks, context_dim=32)
    cc_opt = optim.Adam(cc_encoder.parameters(), lr=0.001, weight_decay=1e-5)
    cc_encoder.train()
    for epoch in range(15):
        for data, target in train_loader:
            task_ids = torch.randint(0, n_tasks, (len(data),))
            perm_data = data.clone()
            for t in range(n_tasks):
                mask = task_ids == t
                if mask.any():
                    flat = data[mask].view(mask.sum(), -1)
                    perm_data[mask] = flat[:, permutations[t]].view(mask.sum(), 1, 28, 28)
            combined = task_ids * 10 + target
            feat = cc_encoder(perm_data, task_ids)
            loss = supervised_contrastive_loss(feat, combined, 0.07)
            cc_opt.zero_grad()
            loss.backward()
            cc_opt.step()

    # Train Baseline
    log("  Training Baseline encoder...")
    bl_encoder = BaselineEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim)
    bl_opt = optim.Adam(bl_encoder.parameters(), lr=0.001, weight_decay=1e-5)
    bl_encoder.train()
    for epoch in range(15):
        for data, target in train_loader:
            feat = bl_encoder(data)
            loss = supervised_contrastive_loss(feat, target, 0.07)
            bl_opt.zero_grad()
            loss.backward()
            bl_opt.step()

    # Sequential EWC
    log("  Training EWC sequentially...")
    ewc_model = MLPClassifier(input_dim=784, hidden_dim=256, output_dim=10)
    ewc_list = []
    for task_id in range(n_tasks):
        perm = permutations[task_id]
        task_data = [(d.view(-1)[perm].view(1, 28, 28), t) for d, t in train_dataset]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)
        opt = optim.Adam(ewc_model.parameters(), lr=0.001)
        for epoch in range(5):
            for data, target in task_loader:
                opt.zero_grad()
                loss = F.cross_entropy(ewc_model(data), target)
                if ewc_list:
                    for ewc in ewc_list:
                        loss += 100 * ewc.penalty()
                loss.backward()
                opt.step()
        ewc_list.append(EWC(ewc_model, task_loader))

    # Extract features and evaluate
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    n_store = 50
    n_query = 100

    cc_stored, cc_slbl, cc_sctx = [], [], []
    bl_stored, bl_slbl = [], []
    ewc_correct = 0
    ewc_total = 0

    for task_id in range(n_tasks):
        perm = permutations[task_id]
        cc_fs, bl_fs = [], []
        all_data, all_target = [], []

        cc_encoder.eval()
        bl_encoder.eval()
        with torch.no_grad():
            for data, target in test_loader:
                pd = data.clone()
                flat = data.view(len(data), -1)
                pd = flat[:, perm].view(len(data), 1, 28, 28)
                cid = torch.full((len(data),), task_id, dtype=torch.long)
                cc_fs.append(cc_encoder(pd, cid).numpy())
                bl_fs.append(bl_encoder(pd).numpy())
                all_data.append(pd)
                all_target.append(target)

        cc_f = normalize_features(np.concatenate(cc_fs).astype(np.float32))
        bl_f = normalize_features(np.concatenate(bl_fs).astype(np.float32))
        targets = np.concatenate(all_target).astype(np.int32)

        for lbl in range(10):
            idx = np.where(targets == lbl)[0]
            rng.shuffle(idx)
            s_idx = idx[:n_store]
            cc_stored.append(cc_f[s_idx])
            cc_slbl.append(np.full(n_store, lbl, dtype=np.int32))
            cc_sctx.append(np.full(n_store, task_id, dtype=np.int32))
            bl_stored.append(bl_f[s_idx])
            bl_slbl.append(np.full(n_store, lbl, dtype=np.int32))

        # EWC accuracy
        all_d = torch.cat(all_data)
        all_t = torch.cat(all_target)
        with torch.no_grad():
            pred = ewc_model(all_d).argmax(dim=1)
            ewc_correct += (pred == all_t).sum().item()
            ewc_total += len(all_t)

    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)
    bl_stored = np.concatenate(bl_stored)
    bl_slbl = np.concatenate(bl_slbl)

    # Evaluate per task
    log(f"\n  {'Task':>6s} | {'BL-kNN':>8s} | {'CCFL-T':>8s} | {'CCFL-I':>8s} | {'EWC':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    avg_bl, avg_ct, avg_ci, avg_ewc = 0, 0, 0, 0

    for task_id in range(n_tasks):
        perm = permutations[task_id]
        cc_fs, bl_fs = [], []
        all_data, all_target = [], []

        cc_encoder.eval()
        bl_encoder.eval()
        with torch.no_grad():
            for data, target in test_loader:
                flat = data.view(len(data), -1)
                pd = flat[:, perm].view(len(data), 1, 28, 28)
                cid = torch.full((len(data),), task_id, dtype=torch.long)
                cc_fs.append(cc_encoder(pd, cid).numpy())
                bl_fs.append(bl_encoder(pd).numpy())
                all_data.append(pd)
                all_target.append(target)

        cc_f = normalize_features(np.concatenate(cc_fs).astype(np.float32))
        bl_f = normalize_features(np.concatenate(bl_fs).astype(np.float32))
        targets = np.concatenate(all_target).astype(np.int32)

        q_idx = np.arange(len(targets))
        rng.shuffle(q_idx)
        q_idx = q_idx[:n_query]

        p_bl = knn_predict(bl_f[q_idx], bl_stored, bl_slbl, k=5)
        p_ct = cond_knn_predict(cc_f[q_idx], np.full(n_query, task_id),
                                 cc_stored, cc_slbl, cc_sctx, k=5)
        p_ci, _, _ = multi_hypothesis_predict(cc_f[q_idx], cc_stored, cc_slbl,
                                               cc_sctx, n_tasks, k=5)

        acc_bl = (p_bl == targets[q_idx]).mean()
        acc_ct = (p_ct == targets[q_idx]).mean()
        acc_ci = (p_ci == targets[q_idx]).mean()

        # EWC per task
        all_d = torch.cat(all_data)
        all_t = torch.cat(all_target)
        with torch.no_grad():
            pred = ewc_model(all_d).argmax(dim=1)
            acc_ewc = (pred == all_t).float().mean().item()

        avg_bl += acc_bl
        avg_ct += acc_ct
        avg_ci += acc_ci
        avg_ewc += acc_ewc

        log(f"  T{task_id:>4d} | {acc_bl:>8.4f} | {acc_ct:>8.4f} | {acc_ci:>8.4f} | {acc_ewc:>8.4f}")

    n = n_tasks
    log(f"  {'Avg':>6s} | {avg_bl/n:>8.4f} | {avg_ct/n:>8.4f} | {avg_ci/n:>8.4f} | {avg_ewc/n:>8.4f}")

    return {'BL-kNN': avg_bl/n, 'CCFL-T': avg_ct/n, 'CCFL-I': avg_ci/n, 'EWC': avg_ewc/n}


# ============================================================
# Experiment 3: Novelty Detection (Unknown Context)
# ============================================================

def experiment_novelty_detection():
    log("\n" + "=" * 76)
    log("  Experiment 3: Novelty Detection for Unknown Contexts")
    log("=" * 76)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_known = 4
    n_unknown = 1
    n_contexts_total = n_known + n_unknown
    base_digits = [0, 1, 2, 3, 4]
    n_store = 40
    n_query = 50
    output_dim = 128

    label_mappings = {}
    for ctx in range(n_contexts_total):
        label_mappings[ctx] = {d: (i + ctx) % 10 for i, d in enumerate(base_digits)}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Train CCFL on KNOWN contexts only
    log(f"  Training CCFL on {n_known} known contexts...")
    cc_encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                              n_contexts=n_contexts_total, context_dim=32)
    cc_opt = optim.Adam(cc_encoder.parameters(), lr=0.001, weight_decay=1e-5)
    cc_encoder.train()
    for epoch in range(20):
        for data, target in train_loader:
            mask = torch.tensor([t.item() in base_digits for t in target])
            if not mask.any():
                continue
            data, target = data[mask], target[mask]
            ctx_ids = torch.randint(0, n_known, (len(data),))
            combined = ctx_ids * 10 + target
            feat = cc_encoder(data, ctx_ids)
            loss = supervised_contrastive_loss(feat, combined, 0.07)
            cc_opt.zero_grad()
            loss.backward()
            cc_opt.step()

    # Extract features
    rng = np.random.RandomState(42)
    cc_feats = {}
    all_labels = None
    cc_encoder.eval()
    for ctx in range(n_contexts_total):
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                dm, tm = data[mask], target[mask]
                cid = torch.full((len(dm),), ctx, dtype=torch.long)
                fs.append(cc_encoder(dm, cid).numpy())
                ls.append(tm.numpy())
        cc_feats[ctx] = normalize_features(np.concatenate(fs).astype(np.float32))
        if all_labels is None:
            all_labels = np.concatenate(ls).astype(np.int32)

    digit_idx = {d: np.where(all_labels == d)[0] for d in base_digits}
    for d in base_digits:
        rng.shuffle(digit_idx[d])

    # Build stored set (known contexts only)
    cc_stored, cc_slbl, cc_sctx = [], [], []
    for ctx in range(n_known):
        for d in base_digits:
            idx = digit_idx[d][:n_store]
            cc_stored.append(cc_feats[ctx][idx])
            cc_slbl.append(np.full(n_store, label_mappings[ctx][d], dtype=np.int32))
            cc_sctx.append(np.full(n_store, ctx, dtype=np.int32))
    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)

    # Novelty detection: multiple approaches
    log("\n  Novelty Detection Approach 1: Max Similarity to Top-k Neighbors")
    log("  (For each known context, compute max similarity; take the best)")

    known_max_sims = []
    unknown_max_sims = []

    for ctx in range(n_known):
        for d in base_digits:
            idx = digit_idx[d][n_store:n_store+n_query]
            if len(idx) < n_query:
                continue
            q_feat = cc_feats[ctx][idx]
            Q = normalize_features(q_feat.astype(np.float32))
            E = normalize_features(cc_stored.astype(np.float32))
            sims = Q @ E.T
            for i in range(len(Q)):
                known_max_sims.append(np.max(sims[i]))

    for d in base_digits:
        idx = digit_idx[d][n_store:n_store+n_query]
        if len(idx) < n_query:
            continue
        q_feat = cc_feats[n_unknown][idx]
        Q = normalize_features(q_feat.astype(np.float32))
        E = normalize_features(cc_stored.astype(np.float32))
        sims = Q @ E.T
        for i in range(len(Q)):
            unknown_max_sims.append(np.max(sims[i]))

    known_max_sims = np.array(known_max_sims)
    unknown_max_sims = np.array(unknown_max_sims)

    log(f"  Known context max-sim:   mean={known_max_sims.mean():.4f}, std={known_max_sims.std():.4f}")
    log(f"  Unknown context max-sim: mean={unknown_max_sims.mean():.4f}, std={unknown_max_sims.std():.4f}")

    # AUROC for max-sim novelty detection
    labels_nd = np.concatenate([np.ones(len(known_max_sims)), np.zeros(len(unknown_max_sims))])
    scores_nd = np.concatenate([known_max_sims, unknown_max_sims])
    sorted_idx = np.argsort(-scores_nd)
    sorted_labels = labels_nd[sorted_idx]
    tpr = np.cumsum(sorted_labels) / max(sorted_labels.sum(), 1)
    fpr = np.cumsum(1 - sorted_labels) / max((1 - sorted_labels).sum(), 1)
    auroc_maxsim = np.trapezoid(tpr, fpr)
    log(f"  Max-sim AUROC: {auroc_maxsim:.4f}")

    log("\n  Novelty Detection Approach 2: Cross-Context Label Consistency")
    log("  (Predict label under each known context; if predictions disagree → novel)")

    known_consistency = []
    unknown_consistency = []

    for ctx in range(n_known):
        for d in base_digits:
            idx = digit_idx[d][n_store:n_store+n_query]
            if len(idx) < n_query:
                continue
            q_feat = cc_feats[ctx][idx]
            for i in range(len(q_feat)):
                q_f = q_feat[i:i+1]
                preds_per_ctx = []
                for c_try in range(n_known):
                    m = cc_sctx == c_try
                    if not m.any():
                        continue
                    E = normalize_features(cc_stored[m].astype(np.float32))
                    cl = cc_slbl[m]
                    Q = normalize_features(q_f.astype(np.float32))
                    sims = Q @ E.T
                    kk = min(5, sims.shape[1])
                    top_idx = np.argpartition(sims[0], -kk)[-kk:]
                    top_lbls = cl[top_idx]
                    ls = {}
                    for lbl in top_lbls:
                        ls[lbl] = ls.get(lbl, 0) + 1
                    preds_per_ctx.append(max(ls, key=ls.get))
                consistency = len(set(preds_per_ctx)) == 1
                known_consistency.append(1.0 if consistency else 0.0)

    for d in base_digits:
        idx = digit_idx[d][n_store:n_store+n_query]
        if len(idx) < n_query:
            continue
        q_feat = cc_feats[n_unknown][idx]
        for i in range(len(q_feat)):
            q_f = q_feat[i:i+1]
            preds_per_ctx = []
            for c_try in range(n_known):
                m = cc_sctx == c_try
                if not m.any():
                    continue
                E = normalize_features(cc_stored[m].astype(np.float32))
                cl = cc_slbl[m]
                Q = normalize_features(q_f.astype(np.float32))
                sims = Q @ E.T
                kk = min(5, sims.shape[1])
                top_idx = np.argpartition(sims[0], -kk)[-kk:]
                top_lbls = cl[top_idx]
                ls = {}
                for lbl in top_lbls:
                    ls[lbl] = ls.get(lbl, 0) + 1
                preds_per_ctx.append(max(ls, key=ls.get))
            consistency = len(set(preds_per_ctx)) == 1
            unknown_consistency.append(1.0 if consistency else 0.0)

    known_consistency = np.array(known_consistency)
    unknown_consistency = np.array(unknown_consistency)

    log(f"  Known context consistency:   {known_consistency.mean():.4f}")
    log(f"  Unknown context consistency: {unknown_consistency.mean():.4f}")

    # AUROC for consistency-based novelty detection (inconsistent = novel)
    labels_c = np.concatenate([np.ones(len(known_consistency)), np.zeros(len(unknown_consistency))])
    scores_c = np.concatenate([known_consistency, unknown_consistency])
    sorted_idx_c = np.argsort(-scores_c)
    sorted_labels_c = labels_c[sorted_idx_c]
    tpr_c = np.cumsum(sorted_labels_c) / max(sorted_labels_c.sum(), 1)
    fpr_c = np.cumsum(1 - sorted_labels_c) / max((1 - sorted_labels_c).sum(), 1)
    auroc_consist = np.trapezoid(tpr_c, fpr_c)
    log(f"  Consistency AUROC: {auroc_consist:.4f}")

    auroc = max(auroc_maxsim, auroc_consist)

    # Handle unknown context: fallback to global kNN
    log("\n  Handling Unknown Contexts:")
    log("  Strategy: If confidence < threshold, use global kNN instead of context-filtered")

    for threshold in [0.4, 0.5, 0.6]:
        correct_adaptive = 0
        correct_blind = 0
        total = 0

        for ctx in range(n_contexts_total):
            for d in base_digits:
                idx = digit_idx[d][n_store:n_store+n_query]
                if len(idx) < n_query:
                    continue
                q_feat = cc_feats[ctx][idx]
                qlbl = np.full(n_query, label_mappings[ctx][d], dtype=np.int32)

                preds, _, confs = multi_hypothesis_predict(
                    q_feat, cc_stored, cc_slbl, cc_sctx, n_known, k=5)

                # Adaptive: low confidence → global kNN
                adaptive_preds = preds.copy()
                low_conf = confs < threshold
                if low_conf.any():
                    global_preds = knn_predict(q_feat[low_conf], cc_stored, cc_slbl, k=5)
                    adaptive_preds[low_conf] = global_preds

                correct_adaptive += (adaptive_preds == qlbl).sum()
                correct_blind += (preds == qlbl).sum()
                total += n_query

        log(f"  threshold={threshold:.1f}: adaptive={correct_adaptive/total:.4f}, "
            f"blind={correct_blind/total:.4f}")

    return auroc


# ============================================================
# Experiment 4: Split-MNIST with CL baselines
# ============================================================

def experiment_split_mnist():
    log("\n" + "=" * 76)
    log("  Experiment 4: Split-MNIST with CL Baselines")
    log("=" * 76)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    n_tasks = len(tasks)
    output_dim = 128

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # CCFL: train on all tasks jointly
    log("  Training CCFL encoder...")
    cc_encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                              n_contexts=n_tasks, context_dim=32)
    cc_opt = optim.Adam(cc_encoder.parameters(), lr=0.001, weight_decay=1e-5)
    cc_encoder.train()
    for epoch in range(15):
        for data, target in train_loader:
            task_ids = torch.zeros(len(data), dtype=torch.long)
            for t_id, (la, lb) in enumerate(tasks):
                mask = (target == la) | (target == lb)
                task_ids[mask] = t_id
            combined = task_ids * 10 + target
            feat = cc_encoder(data, task_ids)
            loss = supervised_contrastive_loss(feat, combined, 0.07)
            cc_opt.zero_grad()
            loss.backward()
            cc_opt.step()

    # Baseline encoder
    log("  Training Baseline encoder...")
    bl_encoder = BaselineEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim)
    bl_opt = optim.Adam(bl_encoder.parameters(), lr=0.001, weight_decay=1e-5)
    bl_encoder.train()
    for epoch in range(15):
        for data, target in train_loader:
            feat = bl_encoder(data)
            loss = supervised_contrastive_loss(feat, target, 0.07)
            bl_opt.zero_grad()
            loss.backward()
            bl_opt.step()

    # EWC sequential
    log("  Training EWC sequentially...")
    ewc_model = MLPClassifier(input_dim=784, hidden_dim=256, output_dim=10)
    ewc_list = []
    for t_id, (la, lb) in enumerate(tasks):
        task_data = [(d, t) for d, t in train_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)
        opt = optim.Adam(ewc_model.parameters(), lr=0.001)
        for epoch in range(5):
            for data, target in task_loader:
                opt.zero_grad()
                loss = F.cross_entropy(ewc_model(data), target)
                for ewc in ewc_list:
                    loss += 100 * ewc.penalty()
                loss.backward()
                opt.step()
        ewc_list.append(EWC(ewc_model, task_loader))

    # Evaluate
    rng = np.random.RandomState(42)
    n_store = 50
    n_query = 200

    cc_stored, cc_slbl, cc_sctx = [], [], []
    bl_stored, bl_slbl = [], []

    for t_id, (la, lb) in enumerate(tasks):
        cc_fs, bl_fs = [], []
        cc_encoder.eval()
        bl_encoder.eval()
        with torch.no_grad():
            for data, target in test_loader:
                mask = (target == la) | (target == lb)
                if not mask.any():
                    continue
                dm = data[mask]
                tm = target[mask]
                cid = torch.full((dm.size(0),), t_id, dtype=torch.long)
                cc_fs.append(cc_encoder(dm, cid).numpy())
                bl_fs.append(bl_encoder(dm).numpy())

        cc_f = normalize_features(np.concatenate(cc_fs).astype(np.float32))
        bl_f = normalize_features(np.concatenate(bl_fs).astype(np.float32))
        t_labels = np.concatenate([tm.numpy() for _, target in test_loader
                                    for tm in [target[(target == la) | (target == lb)]]])
        # simpler: just use the test data we already collected
        all_t = []
        for data, target in test_loader:
            m = (target == la) | (target == lb)
            all_t.append(target[m].numpy())
        t_labels = np.concatenate(all_t).astype(np.int32)

        for lbl in [la, lb]:
            idx = np.where(t_labels == lbl)[0]
            rng.shuffle(idx)
            s_idx = idx[:n_store]
            cc_stored.append(cc_f[s_idx])
            cc_slbl.append(np.full(n_store, lbl, dtype=np.int32))
            cc_sctx.append(np.full(n_store, t_id, dtype=np.int32))
            bl_stored.append(bl_f[s_idx])
            bl_slbl.append(np.full(n_store, lbl, dtype=np.int32))

    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)
    bl_stored = np.concatenate(bl_stored)
    bl_slbl = np.concatenate(bl_slbl)

    log(f"\n  {'Task':>6s} | {'BL-kNN':>8s} | {'CCFL-T':>8s} | {'CCFL-I':>8s} | {'EWC':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    avg_bl, avg_ct, avg_ci, avg_ewc = 0, 0, 0, 0

    for t_id, (la, lb) in enumerate(tasks):
        cc_fs, bl_fs = [], []
        cc_encoder.eval()
        bl_encoder.eval()
        with torch.no_grad():
            for data, target in test_loader:
                mask = (target == la) | (target == lb)
                if not mask.any():
                    continue
                dm = data[mask]
                cid = torch.full((dm.size(0),), t_id, dtype=torch.long)
                cc_fs.append(cc_encoder(dm, cid).numpy())
                bl_fs.append(bl_encoder(dm).numpy())

        cc_f = normalize_features(np.concatenate(cc_fs).astype(np.float32))
        bl_f = normalize_features(np.concatenate(bl_fs).astype(np.float32))
        all_t = []
        for data, target in test_loader:
            m = (target == la) | (target == lb)
            all_t.append(target[m].numpy())
        t_labels = np.concatenate(all_t).astype(np.int32)

        q_idx = np.arange(len(t_labels))
        rng.shuffle(q_idx)
        q_idx = q_idx[:n_query]

        p_bl = knn_predict(bl_f[q_idx], bl_stored, bl_slbl, k=5)
        p_ct = cond_knn_predict(cc_f[q_idx], np.full(len(q_idx), t_id),
                                 cc_stored, cc_slbl, cc_sctx, k=5)
        p_ci, _, _ = multi_hypothesis_predict(cc_f[q_idx], cc_stored, cc_slbl,
                                               cc_sctx, n_tasks, k=5)

        acc_bl = (p_bl == t_labels[q_idx]).mean()
        acc_ct = (p_ct == t_labels[q_idx]).mean()
        acc_ci = (p_ci == t_labels[q_idx]).mean()

        # EWC
        task_test = [(d, t) for d, t in test_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_test, batch_size=256, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in task_loader:
                pred = ewc_model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += len(target)
        acc_ewc = correct / max(total, 1)

        avg_bl += acc_bl
        avg_ct += acc_ct
        avg_ci += acc_ci
        avg_ewc += acc_ewc

        log(f"  T{t_id:>4d} | {acc_bl:>8.4f} | {acc_ct:>8.4f} | {acc_ci:>8.4f} | {acc_ewc:>8.4f}")

    n = n_tasks
    log(f"  {'Avg':>6s} | {avg_bl/n:>8.4f} | {avg_ct/n:>8.4f} | {avg_ci/n:>8.4f} | {avg_ewc/n:>8.4f}")

    return {'BL-kNN': avg_bl/n, 'CCFL-T': avg_ct/n, 'CCFL-I': avg_ci/n, 'EWC': avg_ewc/n}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    log("=" * 76)
    log("  CCFL Comprehensive Experiments for IJCNN")
    log("=" * 76)

    r1 = experiment_context_dependent_mnist()
    r2 = experiment_permuted_mnist()
    r3 = experiment_novelty_detection()
    r4 = experiment_split_mnist()

    log("\n" + "=" * 76)
    log("  COMPREHENSIVE RESULTS SUMMARY")
    log("=" * 76)

    log("\n  Exp 1: Context-Dependent MNIST (noise=0)")
    log(f"    BL-kNN: {r1[0.0]['BL-kNN']:.4f}, BL-Cond: {r1[0.0]['BL-Cond']:.4f}, "
        f"CCFL-T: {r1[0.0]['CCFL-T']:.4f}, CCFL-I: {r1[0.0]['CCFL-I']:.4f}")

    log("\n  Exp 2: Permuted-MNIST")
    for k, v in r2.items():
        log(f"    {k}: {v:.4f}")

    log(f"\n  Exp 3: Novelty Detection AUROC = {r3:.4f}")

    log("\n  Exp 4: Split-MNIST")
    for k, v in r4.items():
        log(f"    {k}: {v:.4f}")
