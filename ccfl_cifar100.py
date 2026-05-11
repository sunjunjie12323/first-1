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


# ============================================================
# CNN Backbone for CIFAR-100
# ============================================================

class SmallCNN(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class CCFL_CNN(nn.Module):
    def __init__(self, output_dim=128, n_contexts=10, context_dim=32):
        super().__init__()
        self.backbone = SmallCNN(output_dim=256)
        self.context_embedding = nn.Embedding(n_contexts, context_dim)
        self.fusion = nn.Sequential(
            nn.Linear(256 + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x, context_id):
        h = self.backbone.features(x)
        h = h.view(h.size(0), -1)
        h = self.backbone.fc[:1](h)
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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=256, output_dim=100):
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


class CNNClassifier(nn.Module):
    def __init__(self, output_dim=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)


# ============================================================
# Loss & Utilities
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
    def __init__(self, max_per_class=30):
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


# ============================================================
# Experiment: CIFAR-100 Split (10 tasks x 10 classes)
# ============================================================

def experiment_cifar100():
    log("=" * 76)
    log("  Experiment: CIFAR-100 Split-Task Sequential Learning")
    log("  10 tasks, each with 10 classes")
    log("=" * 76)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)

    n_tasks = 10
    classes_per_task = 10
    output_dim = 128
    n_store = 20
    n_query = 50

    task_classes = []
    all_classes = list(range(100))
    rng = np.random.RandomState(42)
    rng.shuffle(all_classes)
    for t in range(n_tasks):
        task_classes.append(all_classes[t * classes_per_task:(t + 1) * classes_per_task])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  Device: {device}")

    # --- Sequential CCFL with Replay ---
    log("\n  [1/3] Training Sequential CCFL with Replay...")
    cc_encoder = CCFL_CNN(output_dim=output_dim, n_contexts=1, context_dim=32).to(device)
    replay = ReplayBuffer(max_per_class=10)

    for t_id in range(n_tasks):
        log(f"    Task {t_id}: classes {task_classes[t_id][:3]}...")
        if t_id > 0:
            cc_encoder.add_context(1)

        task_idx = [i for i, (_, t) in enumerate(train_dataset) if t in task_classes[t_id]]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_idx)
        task_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=task_sampler)

        opt = optim.Adam(cc_encoder.parameters(), lr=0.001, weight_decay=1e-5)
        cc_encoder.train()
        for epoch in range(5):
            for data, target in task_loader:
                data, target = data.to(device), target.to(device)
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long, device=device)
                combined = ctx_ids * 100 + target
                feat = cc_encoder(data, ctx_ids)
                loss = supervised_contrastive_loss(feat, combined, 0.07)

                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(128, device)
                    r_combined = r_ctx * 100 + r_target
                    r_feat = cc_encoder(r_data, r_ctx)
                    loss = loss + supervised_contrastive_loss(r_feat, r_combined, 0.07)

                opt.zero_grad()
                loss.backward()
                opt.step()

        count = 0
        for data, target in task_loader:
            data, target = data.to(device), target.to(device)
            ctx_ids = torch.full((len(data),), t_id, dtype=torch.long, device=device)
            replay.add_batch(data, target, ctx_ids)
            count += len(data)
            if count >= 500:
                break

    # --- ER Baseline ---
    log("  [2/3] Training ER baseline...")
    er_model = CNNClassifier(output_dim=100).to(device)
    er_buffer_data = []
    er_buffer_target = []

    for t_id in range(n_tasks):
        task_idx = [i for i, (_, t) in enumerate(train_dataset) if t in task_classes[t_id]]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_idx)
        task_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=task_sampler)

        opt = optim.Adam(er_model.parameters(), lr=0.001)
        er_model.train()
        for epoch in range(5):
            for data, target in task_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(er_model(data), target)
                if er_buffer_data:
                    idx = np.random.choice(len(er_buffer_data), min(128, len(er_buffer_data)), replace=False)
                    r_d = torch.stack([er_buffer_data[i] for i in idx]).to(device)
                    r_t = torch.tensor([er_buffer_target[i] for i in idx], device=device)
                    loss += F.cross_entropy(er_model(r_d), r_t)
                loss.backward()
                opt.step()

        count = 0
        for data, target in task_loader:
            for i in range(len(data)):
                if count >= 500:
                    break
                er_buffer_data.append(data[i].cpu().clone())
                er_buffer_target.append(target[i].item())
                count += 1
            if count >= 500:
                break

    # --- EWC Baseline ---
    log("  [3/3] Training EWC baseline...")
    ewc_model = CNNClassifier(output_dim=100).to(device)
    ewc_list = []

    for t_id in range(n_tasks):
        task_idx = [i for i, (_, t) in enumerate(train_dataset) if t in task_classes[t_id]]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_idx)
        task_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=task_sampler)

        opt = optim.Adam(ewc_model.parameters(), lr=0.001)
        ewc_model.train()
        for epoch in range(5):
            for data, target in task_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(ewc_model(data), target)
                for ewc in ewc_list:
                    loss += 5000 * ewc.penalty()
                loss.backward()
                opt.step()

        fisher = {}
        params = {n: p for n, p in ewc_model.named_parameters() if p.requires_grad}
        for n, p in params.items():
            fisher[n] = torch.zeros_like(p)
        ewc_model.eval()
        for data, target in task_loader:
            ewc_model.zero_grad()
            loss = F.cross_entropy(ewc_model(data.to(device)), target.to(device))
            loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * len(data)
        ewc_list.append({
            'fisher': {n: f / max(len(task_idx), 1) for n, f in fisher.items()},
            'priors': {n: p.data.clone() for n, p in params.items()},
            'penalty': lambda self_ref=ewc_list: sum(
                (self_ref[-1]['fisher'][n] * (p - self_ref[-1]['priors'][n]).pow(2)).sum()
                for n, p in ewc_model.named_parameters() if n in self_ref[-1]['fisher']
            ) if len(self_ref) > 0 else torch.tensor(0.0)
        })

    # --- Evaluate ---
    log("\n  Evaluating all methods...")

    # Build CCFL stored set
    cc_stored, cc_slbl, cc_sctx = [], [], []
    for t_id in range(n_tasks):
        cc_encoder.eval()
        task_test_idx = [i for i, (_, t) in enumerate(test_dataset) if t in task_classes[t_id]]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_test_idx)
        task_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler=task_sampler)

        fs, ls = [], []
        with torch.no_grad():
            for data, target in task_loader:
                data = data.to(device)
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long, device=device)
                fs.append(cc_encoder(data, ctx_ids).cpu().numpy())
                ls.append(target.numpy())
        feats = normalize_features(np.concatenate(fs).astype(np.float32))
        labels = np.concatenate(ls).astype(np.int32)

        for lbl in task_classes[t_id]:
            idx = np.where(labels == lbl)[0]
            rng.shuffle(idx)
            s_idx = idx[:n_store]
            if len(s_idx) > 0:
                cc_stored.append(feats[s_idx])
                cc_slbl.append(np.full(len(s_idx), lbl, dtype=np.int32))
                cc_sctx.append(np.full(len(s_idx), t_id, dtype=np.int32))
    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)

    log(f"\n  {'Task':>6s} | {'ER':>8s} | {'EWC':>8s} | {'CCFL-T':>8s} | {'CCFL-I':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    avg_er, avg_ewc, avg_ct, avg_ci = 0, 0, 0, 0

    for t_id in range(n_tasks):
        cc_encoder.eval()
        task_test_idx = [i for i, (_, t) in enumerate(test_dataset) if t in task_classes[t_id]]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_test_idx)
        task_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler=task_sampler)

        sf = []
        all_targets = []
        with torch.no_grad():
            for data, target in task_loader:
                data = data.to(device)
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long, device=device)
                sf.append(cc_encoder(data, ctx_ids).cpu().numpy())
                all_targets.append(target.numpy())
        s_feat = normalize_features(np.concatenate(sf).astype(np.float32))
        t_labels = np.concatenate(all_targets).astype(np.int32)

        q_idx = np.arange(len(t_labels))
        rng.shuffle(q_idx)
        q_idx = q_idx[:n_query]

        p_ct = cond_knn_predict(s_feat[q_idx], np.full(n_query, t_id), cc_stored, cc_slbl, cc_sctx, k=5)
        p_ci, _ = multi_hypothesis_predict(s_feat[q_idx], cc_stored, cc_slbl, cc_sctx, n_tasks, k=5)
        acc_ct = (p_ct == t_labels[q_idx]).mean()
        acc_ci = (p_ci == t_labels[q_idx]).mean()

        # ER & EWC
        correct_er, correct_ewc, total = 0, 0, 0
        er_model.eval()
        ewc_model.eval()
        with torch.no_grad():
            for data, target in task_loader:
                data = data.to(device)
                pred_er = er_model(data).argmax(dim=1).cpu()
                pred_ewc = ewc_model(data).argmax(dim=1).cpu()
                correct_er += (pred_er == target).sum().item()
                correct_ewc += (pred_ewc == target).sum().item()
                total += len(target)
        acc_er = correct_er / max(total, 1)
        acc_ewc = correct_ewc / max(total, 1)

        avg_er += acc_er
        avg_ewc += acc_ewc
        avg_ct += acc_ct
        avg_ci += acc_ci

        log(f"  T{t_id:>4d} | {acc_er:>8.4f} | {acc_ewc:>8.4f} | {acc_ct:>8.4f} | {acc_ci:>8.4f}")

    n = n_tasks
    log(f"  {'Avg':>6s} | {avg_er/n:>8.4f} | {avg_ewc/n:>8.4f} | {avg_ct/n:>8.4f} | {avg_ci/n:>8.4f}")

    return {'ER': avg_er/n, 'EWC': avg_ewc/n, 'CCFL-T': avg_ct/n, 'CCFL-I': avg_ci/n}


if __name__ == "__main__":
    log("=" * 76)
    log("  CCFL CIFAR-100 Experiment")
    log("=" * 76)

    results = experiment_cifar100()

    log("\n" + "=" * 76)
    log("  CIFAR-100 RESULTS SUMMARY")
    log("=" * 76)
    for k, v in results.items():
        log(f"    {k}: {v:.4f}")
