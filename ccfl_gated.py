from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict


def log(msg=""):
    print(msg, flush=True)


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
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x, context_id):
        x = x.view(x.size(0), -1)
        h = self.visual_backbone(x)
        c = self.context_embedding(context_id)
        alpha = self.gate(h)
        h_ctx = self.fusion(torch.cat([h, c], dim=1))
        return alpha * h_ctx + (1 - alpha) * h[:, :self.output_dim]

    def add_context(self, n_new=1):
        old_emb = self.context_embedding.weight.data
        n_old = old_emb.size(0)
        ctx_dim = old_emb.size(1)
        new_emb = nn.Embedding(n_old + n_new, ctx_dim)
        with torch.no_grad():
            new_emb.weight[:n_old] = old_emb
            nn.init.xavier_uniform_(new_emb.weight[n_old:])
        self.context_embedding = new_emb


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
    return preds, inferred_ctx


class ReplayBuffer:
    def __init__(self, max_per_class=50):
        self.buffer_data = []
        self.buffer_target = []
        self.buffer_ctx = []
        self.max_per_class = max_per_class
        self.class_counts = defaultdict(int)

    def add_batch(self, data, target, ctx_id):
        for i in range(len(data)):
            t_val = int(target[i].item()) if hasattr(target[i], 'item') else int(target[i])
            c_val = int(ctx_id[i].item()) if hasattr(ctx_id[i], 'item') else int(ctx_id)
            key = (t_val, c_val)
            if self.class_counts[key] < self.max_per_class:
                self.buffer_data.append(data[i].cpu().clone())
                self.buffer_target.append(t_val)
                self.buffer_ctx.append(c_val)
                self.class_counts[key] += 1

    def sample(self, n, device='cpu'):
        if len(self.buffer_data) == 0:
            return None, None, None
        idx = np.random.choice(len(self.buffer_data), min(n, len(self.buffer_data)), replace=False)
        data = torch.stack([self.buffer_data[i] for i in idx]).to(device)
        target = torch.tensor([self.buffer_target[i] for i in idx], device=device)
        ctx = torch.tensor([self.buffer_ctx[i] for i in idx], device=device)
        return data, target, ctx

    def __len__(self):
        return len(self.buffer_data)


class EWC:
    def __init__(self, model, dataloader):
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.priors = {n: p.data.clone() for n, p in self.params.items()}
        model.eval()
        for data, target in dataloader:
            model.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            for n, p in self.params.items():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data.pow(2) * len(data)
        ns = max(len(dataloader.dataset), 1)
        self.fisher = {n: f / ns for n, f in self.fisher.items()}

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.priors[n]).pow(2)).sum()
        return loss


# ============================================================
# Experiment 1: Split-MNIST with Gated CCFL
# ============================================================

def experiment_split_mnist_gated():
    log("=" * 76)
    log("  Experiment 1: Split-MNIST with Gated CCFL")
    log("  Innovation: Gated context conditioning (alpha * f(x,c) + (1-alpha) * h(x))")
    log("  When contexts don't conflict, gate learns alpha->0 (use content only)")
    log("  When contexts conflict, gate learns alpha->1 (use context conditioning)")
    log("=" * 76)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_tasks = 5
    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    output_dim = 128
    n_store = 50
    n_query = 200
    rng = np.random.RandomState(42)

    # Gated CCFL with Replay
    log("  Training Gated CCFL with Replay...")
    encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                           n_contexts=1, context_dim=32)
    replay = ReplayBuffer(max_per_class=50)

    for t_id, (la, lb) in enumerate(tasks):
        log(f"    Task {t_id}: digits {la},{lb}")
        if t_id > 0:
            encoder.add_context(1)

        task_data = [(d, t) for d, t in train_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)

        opt = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-5)
        encoder.train()
        for epoch in range(10):
            for data, target in task_loader:
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long)
                combined = ctx_ids * 10 + target
                feat = encoder(data, ctx_ids)
                loss = supervised_contrastive_loss(feat, combined, 0.07)
                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(256)
                    r_combined = r_ctx * 10 + r_target
                    r_feat = encoder(r_data, r_ctx)
                    loss = loss + supervised_contrastive_loss(r_feat, r_combined, 0.07)
                opt.zero_grad()
                loss.backward()
                opt.step()

        count = 0
        for data, target in task_loader:
            ctx_ids = torch.full((len(data),), t_id, dtype=torch.long)
            replay.add_batch(data, target, ctx_ids)
            count += len(data)
            if count >= 200:
                break

    # Measure gate values
    log("\n  Gate values (alpha) per task:")
    encoder.eval()
    for t_id, (la, lb) in enumerate(tasks):
        task_data = [(d, t) for d, t in test_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=False)
        with torch.no_grad():
            alphas = []
            for data, target in task_loader:
                x = data.view(data.size(0), -1)
                h = encoder.visual_backbone(x)
                alpha = encoder.gate(h)
                alphas.append(alpha.mean().item())
        log(f"    Task {t_id}: mean alpha = {np.mean(alphas):.4f}")

    # EWC
    log("  Training EWC...")
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
                    loss += 5000 * ewc.penalty(ewc_model)
                loss.backward()
                opt.step()
        ewc_list.append(EWC(ewc_model, task_loader))

    # ER
    log("  Training ER...")
    er_model = MLPClassifier(input_dim=784, hidden_dim=256, output_dim=10)
    er_buf_d, er_buf_t = [], []
    for t_id, (la, lb) in enumerate(tasks):
        task_data = [(d, t) for d, t in train_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)
        opt = optim.Adam(er_model.parameters(), lr=0.001)
        for epoch in range(5):
            for data, target in task_loader:
                opt.zero_grad()
                loss = F.cross_entropy(er_model(data), target)
                if er_buf_d:
                    idx = np.random.choice(len(er_buf_d), min(256, len(er_buf_d)), replace=False)
                    r_d = torch.stack([er_buf_d[i] for i in idx])
                    r_t = torch.tensor([er_buf_t[i] for i in idx])
                    loss += F.cross_entropy(er_model(r_d), r_t)
                loss.backward()
                opt.step()
        for d, t in task_data[:200]:
            er_buf_d.append(d)
            er_buf_t.append(t)

    # Evaluate
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    cc_stored, cc_slbl, cc_sctx = [], [], []
    for t_id, (la, lb) in enumerate(tasks):
        encoder.eval()
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                mask = (target == la) | (target == lb)
                if not mask.any():
                    continue
                dm = data[mask]
                cid = torch.full((dm.size(0),), t_id, dtype=torch.long)
                fs.append(encoder(dm, cid).detach().numpy())
                ls.append(target[mask].numpy())
        feats = normalize_features(np.concatenate(fs).astype(np.float32))
        labels = np.concatenate(ls).astype(np.int32)
        for lbl in [la, lb]:
            idx = np.where(labels == lbl)[0]
            rng.shuffle(idx)
            s_idx = idx[:n_store]
            cc_stored.append(feats[s_idx])
            cc_slbl.append(np.full(n_store, lbl, dtype=np.int32))
            cc_sctx.append(np.full(n_store, t_id, dtype=np.int32))
    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)

    log(f"\n  {'Task':>6s} | {'ER':>8s} | {'EWC':>8s} | {'CCFL-G':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")

    avg_er, avg_ewc, avg_ccfl = 0, 0, 0

    for t_id, (la, lb) in enumerate(tasks):
        encoder.eval()
        sf = []
        with torch.no_grad():
            for data, target in test_loader:
                mask = (target == la) | (target == lb)
                if not mask.any():
                    continue
                dm = data[mask]
                cid = torch.full((dm.size(0),), t_id, dtype=torch.long)
                sf.append(encoder(dm, cid).detach().numpy())
        s_feat = normalize_features(np.concatenate(sf).astype(np.float32))
        all_t = []
        for data, target in test_loader:
            m = (target == la) | (target == lb)
            all_t.append(target[m].numpy())
        t_labels = np.concatenate(all_t).astype(np.int32)

        q_idx = np.arange(len(t_labels))
        rng.shuffle(q_idx)
        q_idx = q_idx[:n_query]

        p_ccfl, _ = multi_hypothesis_predict(s_feat[q_idx], cc_stored, cc_slbl, cc_sctx, n_tasks, k=5)
        acc_ccfl = (p_ccfl == t_labels[q_idx]).mean()

        task_test = [(d, t) for d, t in test_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_test, batch_size=256, shuffle=False)
        correct_er, correct_ewc, total = 0, 0, 0
        with torch.no_grad():
            for data, target in task_loader:
                pred_er = er_model(data).argmax(dim=1)
                pred_ewc = ewc_model(data).argmax(dim=1)
                correct_er += (pred_er == target).sum().item()
                correct_ewc += (pred_ewc == target).sum().item()
                total += len(target)
        acc_er = correct_er / max(total, 1)
        acc_ewc = correct_ewc / max(total, 1)

        avg_er += acc_er
        avg_ewc += acc_ewc
        avg_ccfl += acc_ccfl
        log(f"  T{t_id:>4d} | {acc_er:>8.4f} | {acc_ewc:>8.4f} | {acc_ccfl:>8.4f}")

    n = n_tasks
    log(f"  {'Avg':>6s} | {avg_er/n:>8.4f} | {avg_ewc/n:>8.4f} | {avg_ccfl/n:>8.4f}")

    return {'ER': avg_er/n, 'EWC': avg_ewc/n, 'CCFL-Gated': avg_ccfl/n}


# ============================================================
# Experiment 2: Permuted-MNIST with Gated CCFL
# ============================================================

def experiment_permuted_mnist_gated():
    log("\n" + "=" * 76)
    log("  Experiment 2: Permuted-MNIST with Gated CCFL")
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
    n_store = 50
    n_query = 200

    # Gated CCFL with Replay
    log("  Training Gated CCFL with Replay...")
    encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                           n_contexts=1, context_dim=32)
    replay = ReplayBuffer(max_per_class=50)

    for t_id in range(n_tasks):
        log(f"    Task {t_id}")
        if t_id > 0:
            encoder.add_context(1)

        perm = permutations[t_id]
        task_data = [(d.view(-1)[perm].view(1, 28, 28), t) for d, t in train_dataset]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)

        opt = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-5)
        encoder.train()
        for epoch in range(10):
            for data, target in task_loader:
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long)
                combined = ctx_ids * 10 + target
                feat = encoder(data, ctx_ids)
                loss = supervised_contrastive_loss(feat, combined, 0.07)
                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(256)
                    r_combined = r_ctx * 10 + r_target
                    r_feat = encoder(r_data, r_ctx)
                    loss = loss + supervised_contrastive_loss(r_feat, r_combined, 0.07)
                opt.zero_grad()
                loss.backward()
                opt.step()

        count = 0
        for data, target in task_loader:
            ctx_ids = torch.full((len(data),), t_id, dtype=torch.long)
            replay.add_batch(data, target, ctx_ids)
            count += len(data)
            if count >= 200:
                break

    # Measure gate values
    log("\n  Gate values (alpha) per task:")
    encoder.eval()
    for t_id in range(n_tasks):
        perm = permutations[t_id]
        with torch.no_grad():
            alphas = []
            for data, _ in torch.utils.data.DataLoader(test_dataset, batch_size=256):
                pd = data.view(len(data), -1)[:, perm].view(len(data), 1, 28, 28)
                x = pd.view(pd.size(0), -1)
                h = encoder.visual_backbone(x)
                alpha = encoder.gate(h)
                alphas.append(alpha.mean().item())
                break
        log(f"    Task {t_id}: mean alpha = {np.mean(alphas):.4f}")

    # EWC
    log("  Training EWC...")
    ewc_model = MLPClassifier(input_dim=784, hidden_dim=256, output_dim=10)
    ewc_list = []
    for t_id in range(n_tasks):
        perm = permutations[t_id]
        task_data = [(d.view(-1)[perm].view(1, 28, 28), t) for d, t in train_dataset]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)
        opt = optim.Adam(ewc_model.parameters(), lr=0.001)
        for epoch in range(5):
            for data, target in task_loader:
                opt.zero_grad()
                loss = F.cross_entropy(ewc_model(data), target)
                for ewc in ewc_list:
                    loss += 5000 * ewc.penalty(ewc_model)
                loss.backward()
                opt.step()
        ewc_list.append(EWC(ewc_model, task_loader))

    # Evaluate
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    cc_stored, cc_slbl, cc_sctx = [], [], []
    for t_id in range(n_tasks):
        perm = permutations[t_id]
        encoder.eval()
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                pd = data.view(len(data), -1)[:, perm].view(len(data), 1, 28, 28)
                cid = torch.full((len(data),), t_id, dtype=torch.long)
                fs.append(encoder(pd, cid).detach().numpy())
                ls.append(target.numpy())
        feats = normalize_features(np.concatenate(fs).astype(np.float32))
        labels = np.concatenate(ls).astype(np.int32)
        for lbl in range(10):
            idx = np.where(labels == lbl)[0]
            rng.shuffle(idx)
            s_idx = idx[:n_store]
            cc_stored.append(feats[s_idx])
            cc_slbl.append(np.full(n_store, lbl, dtype=np.int32))
            cc_sctx.append(np.full(n_store, t_id, dtype=np.int32))
    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)

    log(f"\n  {'Task':>6s} | {'EWC':>8s} | {'CCFL-G':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8}")

    avg_ewc, avg_ccfl = 0, 0

    for t_id in range(n_tasks):
        perm = permutations[t_id]
        encoder.eval()
        sf = []
        with torch.no_grad():
            for data, target in test_loader:
                pd = data.view(len(data), -1)[:, perm].view(len(data), 1, 28, 28)
                cid = torch.full((len(data),), t_id, dtype=torch.long)
                sf.append(encoder(pd, cid).detach().numpy())
        s_feat = normalize_features(np.concatenate(sf).astype(np.float32))
        targets = np.concatenate([t.numpy() for _, t in test_loader]).astype(np.int32)

        q_idx = np.arange(len(targets))
        rng.shuffle(q_idx)
        q_idx = q_idx[:n_query]

        p_ccfl, _ = multi_hypothesis_predict(s_feat[q_idx], cc_stored, cc_slbl, cc_sctx, n_tasks, k=5)
        acc_ccfl = (p_ccfl == targets[q_idx]).mean()

        task_test = [(d.view(-1)[perm].view(1, 28, 28), t) for d, t in test_dataset]
        task_loader = torch.utils.data.DataLoader(task_test, batch_size=256, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in task_loader:
                pred = ewc_model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += len(target)
        acc_ewc = correct / max(total, 1)

        avg_ewc += acc_ewc
        avg_ccfl += acc_ccfl
        log(f"  T{t_id:>4d} | {acc_ewc:>8.4f} | {acc_ccfl:>8.4f}")

    n = n_tasks
    log(f"  {'Avg':>6s} | {avg_ewc/n:>8.4f} | {avg_ccfl/n:>8.4f}")

    return {'EWC': avg_ewc/n, 'CCFL-Gated': avg_ccfl/n}


# ============================================================
# Experiment 3: Context-Dependent MNIST with Gated CCFL
# ============================================================

def experiment_context_dependent_gated():
    log("\n" + "=" * 76)
    log("  Experiment 3: Context-Dependent MNIST with Gated CCFL")
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

    rng = np.random.RandomState(42)

    # Sequential Gated CCFL with Replay
    log("  Training Gated CCFL with Replay...")
    encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                           n_contexts=1, context_dim=32)
    replay = ReplayBuffer(max_per_class=30)

    for ctx in range(n_contexts):
        log(f"    Context {ctx}")
        if ctx > 0:
            encoder.add_context(1)

        opt = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-5)
        encoder.train()
        for epoch in range(10):
            for data, target in torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True):
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                dm, tm = data[mask], target[mask]
                ctx_ids = torch.full((len(dm),), ctx, dtype=torch.long)
                combined = ctx_ids * 10 + tm
                feat = encoder(dm, ctx_ids)
                loss = supervised_contrastive_loss(feat, combined, 0.07)
                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(256)
                    r_combined = r_ctx * 10 + r_target
                    r_feat = encoder(r_data, r_ctx)
                    loss = loss + supervised_contrastive_loss(r_feat, r_combined, 0.07)
                opt.zero_grad()
                loss.backward()
                opt.step()

        count = 0
        for data, target in torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True):
            mask = torch.tensor([t.item() in base_digits for t in target])
            if not mask.any():
                continue
            dm, tm = data[mask], target[mask]
            ctx_ids = torch.full((len(dm),), ctx, dtype=torch.long)
            replay.add_batch(dm, tm, ctx_ids)
            count += len(dm)
            if count >= 100:
                break

    # Measure gate values
    log("\n  Gate values (alpha) per context:")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    encoder.eval()
    for ctx in range(n_contexts):
        with torch.no_grad():
            alphas = []
            for data, target in test_loader:
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                dm = data[mask]
                ctx_ids = torch.full((len(dm),), ctx, dtype=torch.long)
                x = dm.view(dm.size(0), -1)
                h = encoder.visual_backbone(x)
                alpha = encoder.gate(h)
                alphas.append(alpha.mean().item())
                break
        log(f"    Context {ctx}: mean alpha = {np.mean(alphas):.4f}")

    # Extract features
    cc_feats = {}
    all_labels = None
    encoder.eval()
    for ctx in range(n_contexts):
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                dm, tm = data[mask], target[mask]
                cid = torch.full((len(dm),), ctx, dtype=torch.long)
                fs.append(encoder(dm, cid).detach().numpy())
                ls.append(tm.numpy())
        cc_feats[ctx] = normalize_features(np.concatenate(fs).astype(np.float32))
        if all_labels is None:
            all_labels = np.concatenate(ls).astype(np.int32)

    digit_idx = {d: np.where(all_labels == d)[0] for d in base_digits}
    for d in base_digits:
        rng.shuffle(digit_idx[d])

    cc_stored, cc_slbl, cc_sctx = [], [], []
    for ctx in range(n_contexts):
        for d in base_digits:
            idx = digit_idx[d][:n_store]
            cc_stored.append(cc_feats[ctx][idx])
            cc_slbl.append(np.full(n_store, label_mappings[ctx][d], dtype=np.int32))
            cc_sctx.append(np.full(n_store, ctx, dtype=np.int32))
    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)

    log(f"\n  {'Noise':>6s} | {'CCFL-T':>8s} | {'CCFL-I':>8s} | {'CtxInf':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")

    for noise in [0.0, 0.2, 0.5]:
        c_ct, c_ci, c_ctx, total = 0, 0, 0, 0
        for ctx in range(n_contexts):
            for d in base_digits:
                idx = digit_idx[d][n_store:n_store+n_query]
                if len(idx) < n_query:
                    continue
                qcc = cc_feats[ctx][idx] + rng.randn(n_query, output_dim).astype(np.float32) * noise
                qlbl = np.full(n_query, label_mappings[ctx][d], dtype=np.int32)
                qctx = np.full(n_query, ctx, dtype=np.int32)

                p_ct = cond_knn_predict(qcc, qctx, cc_stored, cc_slbl, cc_sctx, k=5)
                p_ci, inf_ctx = multi_hypothesis_predict(qcc, cc_stored, cc_slbl, cc_sctx, n_contexts, k=5)

                c_ct += (p_ct == qlbl).sum()
                c_ci += (p_ci == qlbl).sum()
                c_ctx += (inf_ctx == qctx).sum()
                total += n_query

        log(f"  {noise:>6.1f} | {c_ct/total:>8.4f} | {c_ci/total:>8.4f} | {c_ctx/total:>8.4f}")

    return True


if __name__ == "__main__":
    log("=" * 76)
    log("  Gated CCFL Experiments")
    log("  Innovation: Adaptive gate learns when context conditioning is needed")
    log("  f(x,c) = alpha(h(x)) * g(h(x)||e(c)) + (1-alpha(h(x))) * h(x)")
    log("=" * 76)

    r1 = experiment_split_mnist_gated()
    r2 = experiment_permuted_mnist_gated()
    r3 = experiment_context_dependent_gated()

    log("\n" + "=" * 76)
    log("  RESULTS SUMMARY")
    log("=" * 76)

    log("\n  Split-MNIST:")
    for k, v in r1.items():
        log(f"    {k}: {v:.4f}")

    log("\n  Permuted-MNIST:")
    for k, v in r2.items():
        log(f"    {k}: {v:.4f}")

    log("\n  Context-Dependent MNIST: (see table above)")
