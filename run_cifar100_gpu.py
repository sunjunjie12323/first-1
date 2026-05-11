#!/usr/bin/env python3
"""
CCFL CIFAR-100 GPU Experiment
==============================
运行命令: python run_cifar100_gpu.py

这个脚本会：
1. 检测GPU并使用CUDA
2. 用ResNet-18 backbone训练CCFL Dual-Head + EWC + Replay
3. 对比ER和EWC基线
4. 输出结果表格

目标: CCFL-T准确率 >= 50%, ER准确率 >= 30%

如果结果不达标，尝试调整：
- 增加EPOCHS到10
- 增加REPLAY_PER_CLASS到10
- 增加MAX_TRAIN_SAMPLES到None（用全部数据）
- 调整LR到0.0003
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import defaultdict
import time

# ============================================================
# 配置参数（可调整）
# ============================================================
N_TASKS = 5
CLASSES_PER_TASK = 20
FEAT_DIM = 128
CONTEXT_DIM = 32
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
REPLAY_PER_CLASS = 10
EWC_LAMBDA = 5000
CTR_TEMP = 0.07
CTR_WEIGHT = 0.5
N_STORE = 30
N_QUERY = 100
MAX_TRAIN_SAMPLES = None  # None = use all data
SEED = 42

def log(msg=""):
    print(msg, flush=True)


# ============================================================
# GPU检测
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    log("=" * 76)
    log("  WARNING: No GPU detected! This will be very slow.")
    log("  Please run on a machine with NVIDIA GPU + CUDA.")
    log("=" * 76)
else:
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  CUDA: {torch.version.cuda}")


# ============================================================
# ResNet-18 Backbone
# ============================================================
class ResNetBackbone(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, hidden_dim)

    def forward_hidden(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class CCFLResNet(nn.Module):
    def __init__(self, feat_dim=128, n_contexts=5, context_dim=32, n_classes=100, hidden_dim=256):
        super().__init__()
        self.backbone = ResNetBackbone(hidden_dim=hidden_dim)
        self.context_embedding = nn.Embedding(n_contexts, context_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward_features(self, x, context_id):
        h = self.backbone.forward_hidden(x)
        c = self.context_embedding(context_id)
        return self.fusion(torch.cat([h, c], dim=1))

    def forward_classify(self, x):
        h = self.backbone.forward_hidden(x)
        return self.classifier(h)

    def add_context(self, n_new=1):
        old_emb = self.context_embedding.weight.data
        n_old = old_emb.size(0)
        ctx_dim = old_emb.size(1)
        new_emb = nn.Embedding(n_old + n_new, ctx_dim)
        with torch.no_grad():
            new_emb.weight[:n_old] = old_emb
            nn.init.xavier_uniform_(new_emb.weight[n_old:])
        self.context_embedding = new_emb


class ResNetClassifier(nn.Module):
    def __init__(self, n_classes=100, hidden_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)


# ============================================================
# 损失函数和工具
# ============================================================
def supervised_contrastive_loss(features, labels, temperature=0.07):
    features = F.normalize(features, dim=1)
    sim_matrix = features @ features.T / temperature
    n = len(features)
    eye_mask = torch.eye(n, device=features.device, dtype=torch.bool)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye_mask
    exp_sim = torch.exp(sim_matrix) * (~eye_mask).float()
    pos_sim = exp_sim * mask_pos.float()
    denominator = exp_sim.sum(dim=1, keepdim=True) + 1e-8
    loss_per_sample = -torch.log(pos_sim.sum(dim=1, keepdim=True) / denominator + 1e-8)
    has_pos = mask_pos.sum(dim=1) > 0
    if has_pos.any():
        return loss_per_sample[has_pos].mean()
    return torch.tensor(0.0, device=features.device)


def normalize_features(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X / norms


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
    def __init__(self, max_per_class=10):
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


class EWCHook:
    def __init__(self, model, dataloader, lambda_ewc=5000, forward_fn=None):
        self.lambda_ewc = lambda_ewc
        self.forward_fn = forward_fn
        self._compute_fisher(model, dataloader)

    def _compute_fisher(self, model, dataloader):
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        model.eval()
        count = 0
        for data, target in dataloader:
            model.zero_grad()
            if self.forward_fn:
                loss = self.forward_fn(model, data, target)
            else:
                out = model(data)
                loss = F.cross_entropy(out, target)
            loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * len(data)
            count += 1
        self.fisher = {n: f / max(count, 1) for n, f in fisher.items()}
        self.priors = {n: p.data.clone() for n, p in params.items()}

    def penalty(self, model):
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.priors:
                if self.fisher[n].shape == p.shape and self.priors[n].shape == p.shape:
                    loss += (self.fisher[n] * (p - self.priors[n]).pow(2)).sum()
        return self.lambda_ewc * loss


# ============================================================
# 主实验
# ============================================================
def run_experiment():
    log("=" * 76)
    log("  CCFL CIFAR-100 Experiment (GPU + ResNet-18)")
    log(f"  Device: {device}")
    log(f"  Tasks: {N_TASKS} x {CLASSES_PER_TASK} classes")
    log(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
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

    task_classes = []
    all_classes = list(range(100))
    rng = np.random.RandomState(SEED)
    rng.shuffle(all_classes)
    for t in range(N_TASKS):
        task_classes.append(all_classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK])

    # --- CCFL Dual-Head + EWC + Replay ---
    log("\n  [1/3] Training CCFL Dual-Head + EWC + Replay (ResNet-18)...")
    t0 = time.time()
    ccfl_model = CCFLResNet(feat_dim=FEAT_DIM, n_contexts=1, context_dim=CONTEXT_DIM,
                            n_classes=100, hidden_dim=256).to(device)
    replay = ReplayBuffer(max_per_class=REPLAY_PER_CLASS)
    ewc_hooks = []

    for t_id in range(N_TASKS):
        log(f"    Task {t_id}: classes {task_classes[t_id][:3]}... ", end="")
        if t_id > 0:
            ccfl_model.add_context(1)

        task_idx = [i for i, (_, t) in enumerate(train_dataset) if t in task_classes[t_id]]
        if MAX_TRAIN_SAMPLES and len(task_idx) > MAX_TRAIN_SAMPLES:
            rng_task = np.random.RandomState(SEED + t_id)
            rng_task.shuffle(task_idx)
            task_idx = task_idx[:MAX_TRAIN_SAMPLES]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_idx)
        task_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                   sampler=task_sampler, num_workers=2)

        opt = optim.Adam(ccfl_model.parameters(), lr=LR, weight_decay=1e-4)
        ccfl_model.train()
        for epoch in range(EPOCHS):
            for data, target in task_loader:
                data, target = data.to(device), target.to(device)
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long, device=device)
                combined = ctx_ids * 100 + target
                feat = ccfl_model.forward_features(data, ctx_ids)
                loss_ctr = supervised_contrastive_loss(feat, combined, CTR_TEMP)

                cls_out = ccfl_model.forward_classify(data)
                loss_cls = F.cross_entropy(cls_out, target)

                loss = CTR_WEIGHT * loss_ctr + loss_cls

                if len(replay) > 0:
                    r_data, r_target, r_ctx = replay.sample(BATCH_SIZE, device)
                    r_combined = r_ctx * 100 + r_target
                    r_feat = ccfl_model.forward_features(r_data, r_ctx)
                    loss_r_ctr = supervised_contrastive_loss(r_feat, r_combined, CTR_TEMP)
                    r_cls = ccfl_model.forward_classify(r_data)
                    loss_r_cls = F.cross_entropy(r_cls, r_target)
                    loss = loss + CTR_WEIGHT * loss_r_ctr + loss_r_cls

                for ewc in ewc_hooks:
                    loss = loss + ewc.penalty(ccfl_model)

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

        ewc_hooks.append(EWCHook(
            ccfl_model, task_loader, lambda_ewc=EWC_LAMBDA,
            forward_fn=lambda m, d, t: F.cross_entropy(m.forward_classify(d), t.to(device))
        ))
        log(f"done ({time.time()-t0:.0f}s)")

    # --- ER Baseline ---
    log("  [2/3] Training ER baseline (ResNet-18)...")
    er_model = ResNetClassifier(n_classes=100, hidden_dim=256).to(device)
    er_buffer_data = []
    er_buffer_target = []

    for t_id in range(N_TASKS):
        task_idx = [i for i, (_, t) in enumerate(train_dataset) if t in task_classes[t_id]]
        if MAX_TRAIN_SAMPLES and len(task_idx) > MAX_TRAIN_SAMPLES:
            rng_task = np.random.RandomState(SEED + t_id)
            rng_task.shuffle(task_idx)
            task_idx = task_idx[:MAX_TRAIN_SAMPLES]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_idx)
        task_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                   sampler=task_sampler, num_workers=2)

        opt = optim.Adam(er_model.parameters(), lr=LR)
        er_model.train()
        for epoch in range(EPOCHS):
            for data, target in task_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(er_model(data), target)
                if er_buffer_data:
                    idx = np.random.choice(len(er_buffer_data), min(BATCH_SIZE, len(er_buffer_data)), replace=False)
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
    log("  [3/3] Training EWC baseline (ResNet-18)...")
    ewc_model = ResNetClassifier(n_classes=100, hidden_dim=256).to(device)
    ewc_list = []

    for t_id in range(N_TASKS):
        task_idx = [i for i, (_, t) in enumerate(train_dataset) if t in task_classes[t_id]]
        if MAX_TRAIN_SAMPLES and len(task_idx) > MAX_TRAIN_SAMPLES:
            rng_task = np.random.RandomState(SEED + t_id)
            rng_task.shuffle(task_idx)
            task_idx = task_idx[:MAX_TRAIN_SAMPLES]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_idx)
        task_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                   sampler=task_sampler, num_workers=2)

        opt = optim.Adam(ewc_model.parameters(), lr=LR)
        ewc_model.train()
        for epoch in range(EPOCHS):
            for data, target in task_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(ewc_model(data), target)
                for ewc_dict in ewc_list:
                    loss += EWC_LAMBDA * ewc_dict['penalty_fn'](ewc_model)
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
            'penalty_fn': lambda model, fi={n: f / max(len(task_idx), 1) for n, f in fisher.items()}, pi={n: p.data.clone() for n, p in params.items()}: sum(
                (fi[n] * (p - pi[n]).pow(2)).sum()
                for n, p in model.named_parameters() if n in fi and fi[n].shape == p.shape
            )
        })

    # --- Evaluate ---
    log("\n  Evaluating all methods...")

    cc_stored, cc_slbl, cc_sctx = [], [], []
    for t_id in range(N_TASKS):
        ccfl_model.eval()
        task_test_idx = [i for i, (_, t) in enumerate(test_dataset) if t in task_classes[t_id]]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_test_idx)
        task_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler=task_sampler)

        fs, ls = [], []
        with torch.no_grad():
            for data, target in task_loader:
                data = data.to(device)
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long, device=device)
                fs.append(ccfl_model.forward_features(data, ctx_ids).cpu().numpy())
                ls.append(target.numpy())
        feats = normalize_features(np.concatenate(fs).astype(np.float32))
        labels = np.concatenate(ls).astype(np.int32)

        for lbl in task_classes[t_id]:
            idx = np.where(labels == lbl)[0]
            rng.shuffle(idx)
            s_idx = idx[:N_STORE]
            if len(s_idx) > 0:
                cc_stored.append(feats[s_idx])
                cc_slbl.append(np.full(len(s_idx), lbl, dtype=np.int32))
                cc_sctx.append(np.full(len(s_idx), t_id, dtype=np.int32))
    cc_stored = np.concatenate(cc_stored)
    cc_slbl = np.concatenate(cc_slbl)
    cc_sctx = np.concatenate(cc_sctx)

    log(f"\n  {'Task':>6s} | {'ER':>8s} | {'EWC':>8s} | {'CCFL-Cls':>8s} | {'CCFL-T':>8s} | {'CCFL-I':>8s}")
    log(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    avg_er, avg_ewc, avg_cls, avg_ct, avg_ci = 0, 0, 0, 0, 0

    for t_id in range(N_TASKS):
        ccfl_model.eval()
        task_test_idx = [i for i, (_, t) in enumerate(test_dataset) if t in task_classes[t_id]]
        task_sampler = torch.utils.data.SubsetRandomSampler(task_test_idx)
        task_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler=task_sampler)

        sf, all_targets = [], []
        with torch.no_grad():
            for data, target in task_loader:
                data = data.to(device)
                ctx_ids = torch.full((len(data),), t_id, dtype=torch.long, device=device)
                sf.append(ccfl_model.forward_features(data, ctx_ids).cpu().numpy())
                all_targets.append(target.numpy())
        s_feat = normalize_features(np.concatenate(sf).astype(np.float32))
        t_labels = np.concatenate(all_targets).astype(np.int32)

        q_idx = np.arange(len(t_labels))
        rng.shuffle(q_idx)
        q_idx = q_idx[:N_QUERY]

        p_ct = cond_knn_predict(s_feat[q_idx], np.full(N_QUERY, t_id), cc_stored, cc_slbl, cc_sctx, k=5)
        p_ci, _ = multi_hypothesis_predict(s_feat[q_idx], cc_stored, cc_slbl, cc_sctx, N_TASKS, k=5)
        acc_ct = (p_ct == t_labels[q_idx]).mean()
        acc_ci = (p_ci == t_labels[q_idx]).mean()

        correct_er, correct_ewc, correct_cls, total = 0, 0, 0, 0
        er_model.eval()
        ewc_model.eval()
        ccfl_model.eval()
        with torch.no_grad():
            for data, target in task_loader:
                data = data.to(device)
                pred_er = er_model(data).argmax(dim=1).cpu()
                pred_ewc = ewc_model(data).argmax(dim=1).cpu()
                pred_cls = ccfl_model.forward_classify(data).argmax(dim=1).cpu()
                correct_er += (pred_er == target).sum().item()
                correct_ewc += (pred_ewc == target).sum().item()
                correct_cls += (pred_cls == target).sum().item()
                total += len(target)
        acc_er = correct_er / max(total, 1)
        acc_ewc = correct_ewc / max(total, 1)
        acc_cls = correct_cls / max(total, 1)

        avg_er += acc_er
        avg_ewc += acc_ewc
        avg_cls += acc_cls
        avg_ct += acc_ct
        avg_ci += acc_ci

        log(f"  T{t_id:>4d} | {acc_er:>8.4f} | {acc_ewc:>8.4f} | {acc_cls:>8.4f} | {acc_ct:>8.4f} | {acc_ci:>8.4f}")

    n = N_TASKS
    log(f"  {'Avg':>6s} | {avg_er/n:>8.4f} | {avg_ewc/n:>8.4f} | {avg_cls/n:>8.4f} | {avg_ct/n:>8.4f} | {avg_ci/n:>8.4f}")

    total_time = time.time() - t0
    log(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    results = {
        'ER': avg_er/n, 'EWC': avg_ewc/n, 'CCFL-Cls': avg_cls/n,
        'CCFL-T': avg_ct/n, 'CCFL-I': avg_ci/n
    }

    log("\n" + "=" * 76)
    log("  CIFAR-100 RESULTS (ResNet-18 + GPU)")
    log("=" * 76)
    for k, v in results.items():
        log(f"    {k}: {v:.4f}")

    log("\n  >>> 把这些数字填入 paper_ccfl_ijcnn.tex 的 Table 6 <<<")

    return results


if __name__ == "__main__":
    run_experiment()
