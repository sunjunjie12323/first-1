from __future__ import annotations

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
            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * len(data)
        n_samples = max(len(dataloader.dataset), 1)
        return {n: f / n_samples for n, f in fisher.items()}

    def penalty(self):
        loss = 0
        for n, p in self.params.items():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.priors[n]).pow(2)).sum()
        return loss


class ReplayBuffer:
    def __init__(self, max_per_class=50):
        self.buffer_data = []
        self.buffer_target = []
        self.buffer_ctx = []
        self.max_per_class = max_per_class
        self.class_counts = defaultdict(int)

    def add(self, data, target, ctx_id):
        if data.dim() == 3:
            data = data.unsqueeze(0)
        if isinstance(target, (int, float)):
            target = torch.tensor([target])
        elif target.dim() == 0:
            target = target.unsqueeze(0)
        if isinstance(ctx_id, (int, float)):
            ctx_id = torch.tensor([ctx_id])
        elif ctx_id.dim() == 0:
            ctx_id = ctx_id.unsqueeze(0)
        for i in range(len(data)):
            key = (int(target[i].item() if target[i].dim() == 0 else target[i]), int(ctx_id[i].item() if ctx_id[i].dim() == 0 else ctx_id[i]))
            if self.class_counts[key] < self.max_per_class:
                self.buffer_data.append(data[i].clone())
                self.buffer_target.append(target[i].clone())
                self.buffer_ctx.append(ctx_id[i].clone() if ctx_id[i].dim() == 0 else ctx_id[i])
                self.class_counts[key] += 1

    def sample(self, n):
        if len(self.buffer_data) == 0:
            return None, None, None
        idx = np.random.choice(len(self.buffer_data), min(n, len(self.buffer_data)), replace=False)
        data = torch.stack([self.buffer_data[i] for i in idx])
        target = torch.tensor([self.buffer_target[i] for i in idx])
        ctx = torch.tensor([self.buffer_ctx[i] for i in idx])
        return data, target, ctx

    def __len__(self):
        return len(self.buffer_data)


def experiment_sequential_ccfl_replay():
    log("=" * 76)
    log("  Experiment: Sequential CCFL with Replay Buffer")
    log("  Key fix: Do NOT freeze backbone, use replay to prevent forgetting")
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
    replay_size = 256

    rng = np.random.RandomState(42)

    # --- Sequential CCFL with Replay ---
    log("\n  [1/4] Training Sequential CCFL with Replay...")
    seq_encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                               n_contexts=1, context_dim=32)
    replay = ReplayBuffer(max_per_class=50)

    for t_id, (la, lb) in enumerate(tasks):
        log(f"    Learning task {t_id}: digits {la},{lb}")
        if t_id > 0:
            seq_encoder.add_context(1)

        task_data = [(d, t) for d, t in train_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)

        opt = optim.Adam(seq_encoder.parameters(), lr=0.001, weight_decay=1e-5)
        seq_encoder.train()
        for epoch in range(10):
            for data, target in task_loader:
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long)
                combined = ctx_ids * 10 + target
                feat = seq_encoder(data, ctx_ids)

                loss = supervised_contrastive_loss(feat, combined, 0.07)

                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(replay_size)
                    r_feat = seq_encoder(r_data, r_ctx)
                    r_combined = r_ctx * 10 + r_target
                    loss_replay = supervised_contrastive_loss(r_feat, r_combined, 0.07)
                    loss = loss + loss_replay

                opt.zero_grad()
                loss.backward()
                opt.step()

        count = 0
        for d, t in task_loader:
            for i in range(len(d)):
                if count >= 200:
                    break
                replay.add(d[i], torch.tensor(t[i].item()), t_id)
                count += 1
            if count >= 200:
                break

    # --- EWC ---
    log("  [2/4] Training EWC...")
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
                    loss += 5000 * ewc.penalty()
                loss.backward()
                opt.step()
        ewc_list.append(EWC(ewc_model, task_loader))

    # --- ER (Experience Replay baseline) ---
    log("  [3/4] Training ER baseline...")
    er_model = MLPClassifier(input_dim=784, hidden_dim=256, output_dim=10)
    er_buffer_data = []
    er_buffer_target = []
    for t_id, (la, lb) in enumerate(tasks):
        task_data = [(d, t) for d, t in train_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)
        opt = optim.Adam(er_model.parameters(), lr=0.001)
        for epoch in range(5):
            for data, target in task_loader:
                opt.zero_grad()
                loss = F.cross_entropy(er_model(data), target)
                if er_buffer_data:
                    idx = np.random.choice(len(er_buffer_data), min(256, len(er_buffer_data)), replace=False)
                    r_d = torch.stack([er_buffer_data[i] for i in idx])
                    r_t = torch.tensor([er_buffer_target[i] for i in idx])
                    loss += F.cross_entropy(er_model(r_d), r_t)
                loss.backward()
                opt.step()
        for d, t in task_data[:200]:
            er_buffer_data.append(d)
            er_buffer_target.append(t)

    # --- Fine-tuning ---
    log("  [4/4] Training Fine-tuning baseline...")
    ft_model = MLPClassifier(input_dim=784, hidden_dim=256, output_dim=10)
    for t_id, (la, lb) in enumerate(tasks):
        task_data = [(d, t) for d, t in train_dataset if t in (la, lb)]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)
        opt = optim.Adam(ft_model.parameters(), lr=0.001)
        for epoch in range(5):
            for data, target in task_loader:
                opt.zero_grad()
                loss = F.cross_entropy(ft_model(data), target)
                loss.backward()
                opt.step()

    # --- Evaluate ---
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    log(f"\n  {'Task':>6s} | {'FineTune':>8s} | {'EWC':>8s} | {'ER':>8s} | {'CCFL-R':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    avg_ft, avg_ewc, avg_er, avg_ccfl = 0, 0, 0, 0

    # Build CCFL stored set
    cc_stored, cc_slbl, cc_sctx = [], [], []
    for t_id, (la, lb) in enumerate(tasks):
        seq_encoder.eval()
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                mask = (target == la) | (target == lb)
                if not mask.any():
                    continue
                dm = data[mask]
                cid = torch.full((dm.size(0),), t_id, dtype=torch.long)
                fs.append(seq_encoder(dm, cid).numpy())
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

    for t_id, (la, lb) in enumerate(tasks):
        seq_encoder.eval()
        sf = []
        with torch.no_grad():
            for data, target in test_loader:
                mask = (target == la) | (target == lb)
                if not mask.any():
                    continue
                dm = data[mask]
                cid = torch.full((dm.size(0),), t_id, dtype=torch.long)
                sf.append(seq_encoder(dm, cid).numpy())
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
        correct_ft, correct_ewc, correct_er, total = 0, 0, 0, 0
        with torch.no_grad():
            for data, target in task_loader:
                pred_ft = ft_model(data).argmax(dim=1)
                pred_ewc = ewc_model(data).argmax(dim=1)
                pred_er = er_model(data).argmax(dim=1)
                correct_ft += (pred_ft == target).sum().item()
                correct_ewc += (pred_ewc == target).sum().item()
                correct_er += (pred_er == target).sum().item()
                total += len(target)
        acc_ft = correct_ft / max(total, 1)
        acc_ewc = correct_ewc / max(total, 1)
        acc_er = correct_er / max(total, 1)

        avg_ft += acc_ft
        avg_ewc += acc_ewc
        avg_er += acc_er
        avg_ccfl += acc_ccfl

        log(f"  T{t_id:>4d} | {acc_ft:>8.4f} | {acc_ewc:>8.4f} | {acc_er:>8.4f} | {acc_ccfl:>8.4f}")

    n = n_tasks
    log(f"  {'Avg':>6s} | {avg_ft/n:>8.4f} | {avg_ewc/n:>8.4f} | {avg_er/n:>8.4f} | {avg_ccfl/n:>8.4f}")

    return {'FineTune': avg_ft/n, 'EWC': avg_ewc/n, 'ER': avg_er/n, 'CCFL-Replay': avg_ccfl/n}


def experiment_permuted_mnist_replay():
    log("\n" + "=" * 76)
    log("  Experiment: Permuted-MNIST Sequential with Replay")
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

    # Sequential CCFL with Replay
    log("  Training Sequential CCFL with Replay...")
    seq_encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                               n_contexts=1, context_dim=32)
    replay = ReplayBuffer(max_per_class=50)

    for t_id in range(n_tasks):
        log(f"    Learning task {t_id}")
        if t_id > 0:
            seq_encoder.add_context(1)

        perm = permutations[t_id]
        task_data = [(d.view(-1)[perm].view(1, 28, 28), t) for d, t in train_dataset]
        task_loader = torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True)

        opt = optim.Adam(seq_encoder.parameters(), lr=0.001, weight_decay=1e-5)
        seq_encoder.train()
        for epoch in range(10):
            for data, target in task_loader:
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long)
                combined = ctx_ids * 10 + target
                feat = seq_encoder(data, ctx_ids)
                loss = supervised_contrastive_loss(feat, combined, 0.07)
                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(256)
                    r_feat = seq_encoder(r_data, r_ctx)
                    r_combined = r_ctx * 10 + r_target
                    loss = loss + supervised_contrastive_loss(r_feat, r_combined, 0.07)
                opt.zero_grad()
                loss.backward()
                opt.step()

        count = 0
        for d, t in torch.utils.data.DataLoader(task_data, batch_size=256, shuffle=True):
            for i in range(len(d)):
                if count >= 200:
                    break
                replay.add(d[i], torch.tensor(t[i].item()), t_id)
                count += 1
            if count >= 200:
                break
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
                    loss += 5000 * ewc.penalty()
                loss.backward()
                opt.step()
        ewc_list.append(EWC(ewc_model, task_loader))

    # Evaluate
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    cc_stored, cc_slbl, cc_sctx = [], [], []
    for t_id in range(n_tasks):
        perm = permutations[t_id]
        seq_encoder.eval()
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                pd = data.view(len(data), -1)[:, perm].view(len(data), 1, 28, 28)
                cid = torch.full((len(data),), t_id, dtype=torch.long)
                fs.append(seq_encoder(pd, cid).numpy())
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

    log(f"\n  {'Task':>6s} | {'EWC':>8s} | {'CCFL-R':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8}")

    avg_ewc, avg_ccfl = 0, 0

    for t_id in range(n_tasks):
        perm = permutations[t_id]
        seq_encoder.eval()
        sf = []
        with torch.no_grad():
            for data, target in test_loader:
                pd = data.view(len(data), -1)[:, perm].view(len(data), 1, 28, 28)
                cid = torch.full((len(data),), t_id, dtype=torch.long)
                sf.append(seq_encoder(pd, cid).numpy())
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

    return {'EWC': avg_ewc/n, 'CCFL-Replay': avg_ccfl/n}


def experiment_context_dependent_replay():
    log("\n" + "=" * 76)
    log("  Experiment: Context-Dependent MNIST (Sequential with Replay)")
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

    # Sequential CCFL: learn one context at a time
    log("  Training Sequential CCFL with Replay...")
    seq_encoder = CCFLEncoder(input_dim=784, hidden_dim=256, output_dim=output_dim,
                               n_contexts=1, context_dim=32)
    replay = ReplayBuffer(max_per_class=30)

    for ctx in range(n_contexts):
        log(f"    Learning context {ctx}")
        if ctx > 0:
            seq_encoder.add_context(1)

        opt = optim.Adam(seq_encoder.parameters(), lr=0.001, weight_decay=1e-5)
        seq_encoder.train()
        for epoch in range(10):
            for data, target in torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True):
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                dm, tm = data[mask], target[mask]
                ctx_ids = torch.full((len(dm),), ctx, dtype=torch.long)
                mapped_labels = torch.tensor([label_mappings[ctx][t.item()] for t in tm])
                combined = ctx_ids * 10 + tm
                feat = seq_encoder(dm, ctx_ids)
                loss = supervised_contrastive_loss(feat, combined, 0.07)

                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(256)
                    r_combined = r_ctx * 10 + r_target
                    r_feat = seq_encoder(r_data, r_ctx)
                    loss = loss + supervised_contrastive_loss(r_feat, r_combined, 0.07)

                opt.zero_grad()
                loss.backward()
                opt.step()

        for data, target in torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True):
            mask = torch.tensor([t.item() in base_digits for t in target])
            if not mask.any():
                continue
            dm, tm = data[mask], target[mask]
            for i in range(min(100, len(dm))):
                replay.add(dm[i], torch.tensor(tm[i].item()), ctx)
            break

    # Extract features
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    cc_feats = {}
    all_labels = None
    seq_encoder.eval()
    for ctx in range(n_contexts):
        fs, ls = [], []
        with torch.no_grad():
            for data, target in test_loader:
                mask = torch.tensor([t.item() in base_digits for t in target])
                if not mask.any():
                    continue
                dm, tm = data[mask], target[mask]
                cid = torch.full((len(dm),), ctx, dtype=torch.long)
                fs.append(seq_encoder(dm, cid).numpy())
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
    log("  CCFL Sequential Learning with Replay Buffer")
    log("  Fix: Do NOT freeze backbone, use replay to prevent forgetting")
    log("=" * 76)

    r1 = experiment_sequential_ccfl_replay()
    r2 = experiment_permuted_mnist_replay()
    r3 = experiment_context_dependent_replay()

    log("\n" + "=" * 76)
    log("  RESULTS SUMMARY")
    log("=" * 76)

    log("\n  Split-MNIST Sequential:")
    for k, v in r1.items():
        log(f"    {k}: {v:.4f}")

    log("\n  Permuted-MNIST Sequential:")
    for k, v in r2.items():
        log(f"    {k}: {v:.4f}")

    log("\n  Context-Dependent MNIST Sequential: (see table above)")
