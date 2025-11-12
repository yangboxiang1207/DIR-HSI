# main.py
# -*- coding: utf-8 -*-
import os
import sys
import math
import argparse
import random
from copy import deepcopy
from typing import Dict, Any, List

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------ sys.path ------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ---- backbone 不改动 ----
import trans

# ---- 项目模块 ----
from models.projection_head import ProjectionHead
from losses.contrastive import PairInfoNCE
from utils.noise_provider import NoiseProvider

# ------------------ 全局噪声轮换计数器（跨 epoch 连续） ------------------
_global_noise_counter = 0


# ======================= 数据集 =======================
class PaviaUDataset(torch.utils.data.Dataset):
    """
    dataset/PaviaU/{PaviaU.mat,PaviaU_gt.mat} -> 中心像素分类 patch（干净视图已按训练 μ/σ 标准化）
    退化视图在训练循环内用同一 μ/σ 标准化。
    """
    def __init__(self, root: str, split: str,
                 patch_size: int = 13, train_ratio: float = 0.2, seed: int = 42,
                 train_mu: np.ndarray = None, train_sigma: np.ndarray = None):
        assert split in ("train", "val")
        self.root = root
        self.ps = patch_size
        self.split = split

        mat_img = sio.loadmat(os.path.join(root, "PaviaU.mat"))
        mat_gt  = sio.loadmat(os.path.join(root, "PaviaU_gt.mat"))

        img = next((mat_img[k] for k in mat_img if not k.startswith("__")), None)
        gt  = next((mat_gt[k]  for k in mat_gt  if not k.startswith("__")), None)
        if img is None or gt is None:
            raise RuntimeError("未在 PaviaU.mat / PaviaU_gt.mat 中找到有效矩阵变量。")

        if img.ndim == 2:
            img = img[:, :, None]
        elif img.ndim == 3 and img.shape[0] < 16 and img.shape[-1] > 16:
            img = np.transpose(img, (1, 2, 0))
        self.img_raw = img.astype(np.float32)
        self.gt  = gt.astype(np.int64)

        H, W, C = self.img_raw.shape
        self.in_chans = C

        coords = np.argwhere(self.gt > 0)
        rows_all, cols_all = coords[:, 0], coords[:, 1]
        labels_all = self.gt[rows_all, cols_all] - 1
        self.num_classes = int(labels_all.max() + 1)

        rng = np.random.RandomState(seed)
        idx_all = np.arange(len(rows_all))
        rng.shuffle(idx_all)
        cut = int(len(idx_all) * train_ratio)
        idx_train = idx_all[:cut]
        idx_val   = idx_all[cut:]

        sel = idx_train if split == "train" else idx_val
        self.rows = rows_all[sel]
        self.cols = cols_all[sel]
        self.labels = labels_all[sel]

        if split == "train":
            flat = self.img_raw.reshape(-1, C)
            self.mu = flat.mean(0).astype(np.float32)
            self.sigma = (flat.std(0) + 1e-8).astype(np.float32)
        else:
            assert train_mu is not None and train_sigma is not None, "val split 需要 train 的 μ/σ"
            self.mu = train_mu.astype(np.float32)
            self.sigma = train_sigma.astype(np.float32)

        self.img = (self.img_raw - self.mu) / self.sigma

    def __len__(self): return len(self.labels)

    def _get_patch(self, r: int, c: int) -> np.ndarray:
        H, W, C = self.img.shape
        ps = self.ps; half = ps // 2
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - half, c + half + 1
        rr = np.clip(np.arange(r0, r1), 0, H - 1)
        cc = np.clip(np.arange(c0, c1), 0, W - 1)
        patch = self.img[rr][:, cc]
        patch = np.transpose(patch, (2, 0, 1))
        return patch.astype(np.float32)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = int(self.rows[i]); c = int(self.cols[i]); y = int(self.labels[i])
        x = self._get_patch(r, c)
        return {
            "image": torch.from_numpy(x),
            "label": torch.tensor(y).long(),
            "row": torch.tensor(r).long(),
            "col": torch.tensor(c).long(),
            "index": torch.tensor(i).long(),
        }


# ======================= 指标 =======================
def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1
    return cm

def _scores_from_cm(cm: np.ndarray) -> Dict[str, float]:
    n = cm.sum()
    po = np.trace(cm) / max(1, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_cls = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), 0.0)
    aa = float(np.mean(per_cls))
    row_m = cm.sum(1); col_m = cm.sum(0)
    pe = float(np.dot(row_m, col_m)) / max(1, n * n)
    kappa = (po - pe) / (1 - pe + 1e-12)
    return {"OA": po * 100.0, "AA": aa * 100.0, "Kappa": kappa * 100.0}

@torch.no_grad()
def evaluate_clean(backbone: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    backbone.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss = 0.0; ys, ps = [], []; n = 0
    for batch in loader:
        x = batch["image"].to(device).float()
        y = batch["label"].to(device).long()
        logits, _ = backbone(x)
        loss = ce(logits, y)
        pred = logits.argmax(1)
        bs = x.size(0); n += bs
        tot_loss += loss.item() * bs
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys, 0) if ys else np.array([], dtype=np.int64)
    ps = np.concatenate(ps, 0) if ps else np.array([], dtype=np.int64)
    K = int(logits.size(1))
    cm = _confusion_matrix(ys, ps, K)
    scores = _scores_from_cm(cm)
    return {"loss": tot_loss / max(1, n), "OA": scores["OA"], "AA": scores["AA"], "Kappa": scores["Kappa"]}


# ======================= SupCon（监督式对比） =======================
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss（Khosla et al., 2020）"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        z: (2B, d)  = concat([zc, zd], dim=0)，已 L2 标准化
        y: (B,)     只传 clean 的标签，退化共享同标签
        """
        z = F.normalize(z, dim=1)
        B = y.size(0)
        zc, zd = z[:B], z[B:]
        z_all = torch.cat([zc, zd], dim=0)   # (2B,d)
        y_all = torch.cat([y, y], dim=0)     # (2B,)

        sim = torch.div(torch.matmul(z_all, z_all.t()), self.t)
        sim = sim - sim.max(dim=1, keepdim=True).values   # 数值稳定

        mask_pos = torch.eq(y_all[:, None], y_all[None, :]).float().to(z.device)
        self_mask = torch.eye(2 * B, device=z.device)
        mask_pos = mask_pos * (1 - self_mask)

        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask_pos.sum(1)
        loss = -(mask_pos * log_prob).sum(1) / (pos_count + 1e-12)
        return loss.mean()


# ======================= 噪声采样（含先验权重） =======================
def build_weighted_noises(noise_types: List[str], prior_str: str) -> List[str]:
    """
    prior_str 形如: "poisson:4,thick_fog:4"
    返回展开后的类型列表（用于 round-robin / mixed）
    """
    weights = {n: 1.0 for n in noise_types}
    if prior_str:
        for kv in prior_str.split(","):
            kv = kv.strip()
            if not kv: continue
            name, w = kv.split(":")
            name = name.strip(); w = float(w)
            if name in weights:
                weights[name] = max(w, 1e-6)
    expanded = []
    for n in noise_types:
        expanded += [n] * int(round(weights[n]))
    expanded = expanded if expanded else noise_types[:]  # 保底
    return expanded

def _build_balanced_noise_batch(noise_types: List[str], batch_size: int, start_idx: int,
                                mode: str = "uniform") -> List[str]:
    """
    - uniform: 整个 batch 同一种噪声（round-robin）
    - mixed:   在 batch 内均匀铺开所有噪声
    """
    M = len(noise_types)
    if mode == "uniform":
        t = noise_types[start_idx % M]
        return [t] * batch_size
    else:
        counts = [batch_size // M] * M
        for i in range(batch_size % M):
            counts[(start_idx + i) % M] += 1
        arr = []
        for i, ct in enumerate(counts):
            arr += [noise_types[(start_idx + i) % M]] * ct
        return arr


# ======================= 关键稳态：冻结 BN 运行统计 =======================
def freeze_bn_running_stats(model: nn.Module):
    """只冻结 BN 的 running_mean/var，保留仿射参数可训练。"""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.training = False


# ======================= 训练（CE + SupCon/Pair + EMA一致性 + Center） =======================
def ema_update(ema_model: nn.Module, model: nn.Module, m: float = 0.999):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(m).add_(p.data, alpha=1 - m)

def train_one_epoch(backbone: nn.Module,
                    ema_backbone: nn.Module,
                    proj_head: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    noise_provider: NoiseProvider,
                    con_loss_fn_pair: PairInfoNCE,
                    con_loss_fn_sup: SupConLoss,
                    mu: torch.Tensor, sigma: torch.Tensor,
                    num_classes: int,
                    # 权重与策略
                    lambda_contrast: float,
                    lambda_consistency: float,
                    lambda_center: float,
                    contrast_type: str,
                    noise_list: List[str],
                    batch_noise_mode: str,
                    grad_clip: float,
                    epoch_idx: int,
                    ema_momentum: float,
                    ramp_epochs: int) -> Dict[str, float]:
    backbone.train()
    freeze_bn_running_stats(backbone)  # 关键：训练中不更新 BN 运行统计

    ce = nn.CrossEntropyLoss()
    meter = {"loss": 0.0, "ce": 0.0, "con": 0.0, "cons": 0.0, "center": 0.0, "OA": 0.0}
    n = 0

    # ramp-up 系数
    r = min(1.0, max(0.0, epoch_idx / max(1, ramp_epochs)))
    lam_con = lambda_contrast * r
    lam_cons = lambda_consistency * r

    global _global_noise_counter

    for batch in loader:
        x_clean = batch["image"].to(device).float()
        y = batch["label"].to(device).long()
        rows = batch["row"].cpu().numpy()
        cols = batch["col"].cpu().numpy()
        B = x_clean.size(0)

        # 均衡噪声（带先验权重的列表，跨 epoch 连续轮换）
        types = _build_balanced_noise_batch(noise_list, B, start_idx=_global_noise_counter,
                                            mode=batch_noise_mode)
        _global_noise_counter += 1

        # 同位裁退化 & 标准化
        x_deg_np = noise_provider.get_patches_batch(rows, cols, types, ps=x_clean.size(-1))
        x_deg = torch.from_numpy(x_deg_np).to(device=device, dtype=x_clean.dtype)
        x_deg = (x_deg - mu.view(1, -1, 1, 1)) / sigma.view(1, -1, 1, 1)

        optimizer.zero_grad(set_to_none=True)

        # 干净：CE + 特征
        logits_clean, feat_clean = backbone(x_clean)
        loss_ce = ce(logits_clean, y)

        # 退化：特征 + logits
        logits_deg, feat_deg = backbone(x_deg)

        # 对比损失（SupCon 默认）
        zc = proj_head(feat_clean)
        zd = proj_head(feat_deg)
        if contrast_type == "supcon":
            z_all = torch.cat([zc, zd], dim=0)
            loss_con = con_loss_fn_sup(z_all, y)
        else:
            loss_con = con_loss_fn_pair(zc, zd)

        # 一致性（EMA Teacher，使用 teacher(clean) 作为目标）
        with torch.no_grad():
            ema_backbone.eval()
            logits_teacher_clean, _ = ema_backbone(x_clean)
            p_t_clean = F.softmax(logits_teacher_clean, dim=1)
        p_s_deg_log   = F.log_softmax(logits_deg, dim=1)
        p_s_clean_log = F.log_softmax(logits_clean, dim=1)
        loss_cons = F.kl_div(p_s_deg_log,   p_t_clean, reduction='batchmean') + \
                    F.kl_div(p_s_clean_log, p_t_clean, reduction='batchmean')

        # Center Loss（batch 内类中心，用 clean 特征做锚）
        with torch.no_grad():
            centers = torch.zeros(num_classes, feat_clean.size(1), device=device)
            counts  = torch.zeros(num_classes, device=device)
            for k in range(num_classes):
                idx = (y == k)
                if idx.any():
                    centers[k] = feat_clean[idx].mean(0)
                    counts[k]  = idx.float().sum()
        center_c = centers[y]
        center_d = centers[y]
        loss_center = 0.5 * (
            (feat_clean - center_c).pow(2).sum(1).mean() +
            (feat_deg   - center_d).pow(2).sum(1).mean()
        )

        # 总损失（含 ramp-up）
        loss = loss_ce + lam_con * loss_con + lam_cons * loss_cons + lambda_center * loss_center
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(proj_head.parameters()), grad_clip)
        optimizer.step()

        # EMA 更新
        ema_update(ema_backbone, backbone, m=ema_momentum)

        with torch.no_grad():
            pred = logits_clean.argmax(1)
            oa = (pred == y).float().mean().item() * 100.0
            bs = x_clean.size(0); n += bs
            meter["loss"]   += loss.item() * bs
            meter["ce"]     += loss_ce.item() * bs
            meter["con"]    += loss_con.item() * bs
            meter["cons"]   += loss_cons.item() * bs
            meter["center"] += loss_center.item() * bs
            meter["OA"]     += oa * bs

    for k in meter:
        meter[k] /= max(1, n)
    return meter


# ======================= 构建 & 运行 =======================
def build_dataloaders(data_root: str, patch_size: int, batch_size: int, workers: int,
                      seed: int, train_ratio: float):
    ds_train_tmp = PaviaUDataset(os.path.join(data_root, "PaviaU"),
                                 split="train", patch_size=patch_size,
                                 train_ratio=train_ratio, seed=seed)
    train_mu = ds_train_tmp.mu.copy()
    train_sigma = ds_train_tmp.sigma.copy()

    ds_train = PaviaUDataset(os.path.join(data_root, "PaviaU"),
                             split="train", patch_size=patch_size,
                             train_ratio=train_ratio, seed=seed,
                             train_mu=train_mu, train_sigma=train_sigma)
    ds_val   = PaviaUDataset(os.path.join(data_root, "PaviaU"),
                             split="val", patch_size=patch_size,
                             train_ratio=train_ratio, seed=seed,
                             train_mu=train_mu, train_sigma=train_sigma)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    loader_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return ds_train, ds_val, loader_train, loader_val


def build_backbone_from_trans(num_classes: int, in_chans: int, patch_size: int, device: torch.device) -> nn.Module:
    params = {
        "data": {
            "num_classes": num_classes,
            "patch_size": patch_size,
            "serve_patch_size": patch_size,
            "spectral_size": in_chans,
        },
        "net": {
            "model_type": 0,
            "depth": 2,
            "heads": 8,
            "mlp_dim": 64,
            "kernal": 3,
            "padding": 1,
            "dropout": 0.0,
            "dim": 64,
            "mask_pct": 50,
            "use_mask": False,
        }
    }
    model = trans.TransFormerNet(params).to(device)
    return model


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def main():
    parser = argparse.ArgumentParser("PaviaU: CE + SupCon + EMA Consistency + Center + Weighted Noises (Stabilized)")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--noise_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="model")

    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # 划分（默认 0.2 与 baseline 对齐）
    parser.add_argument("--train_ratio", type=float, default=0.2)

    # 对比学习
    parser.add_argument("--enable_contrast", action="store_true")
    parser.add_argument("--contrast_type", type=str, default="supcon", choices=["supcon", "pair"])
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda_contrast", type=float, default=0.1)

    # 一致性（EMA Teacher）
    parser.add_argument("--lambda_consistency", type=float, default=0.7)
    parser.add_argument("--ema_momentum", type=float, default=0.999)

    # Center Loss
    parser.add_argument("--lambda_center", type=float, default=5e-4)

    # 噪声采样策略
    parser.add_argument("--noise_schedule", type=str, default="roundrobin", choices=["roundrobin", "random"])
    parser.add_argument("--batch_noise_mode", type=str, default="uniform", choices=["uniform", "mixed"])
    parser.add_argument("--noise_prior", type=str, default="",
                        help="按 'name:weight,name2:weight2' 指定噪声采样权重，如 'poisson:4,thick_fog:4'")

    # 稳定训练的关键调度
    parser.add_argument("--ramp_epochs", type=int, default=10, help="对比/一致性 ramp-up 轮数")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="学习率 warmup 轮数")

    args = parser.parse_args()

    # 确定性/稳定性
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据
    ds_train, ds_val, loader_train, loader_val = build_dataloaders(
        args.data_root, args.patch_size, args.batch_size, args.workers, args.seed, args.train_ratio
    )
    num_classes = ds_train.num_classes; in_chans = ds_train.in_chans

    # backbone + 探测维度
    backbone = build_backbone_from_trans(num_classes, in_chans, args.patch_size, device)
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, in_chans, args.patch_size, args.patch_size, device=device)
        logits, feat = backbone(dummy)
        assert logits.dim() == 2 and logits.size(1) == num_classes
        feat_dim = int(feat.size(1))

    # EMA Teacher
    ema_backbone = deepcopy(backbone).to(device)
    for p in ema_backbone.parameters(): p.requires_grad = False

    # 头与优化器
    proj_head = ProjectionHead(in_dim=feat_dim, proj_dim=args.proj_dim).to(device)
    optimizer = optim.AdamW(list(backbone.parameters()) + list(proj_head.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    # 损失
    con_loss_fn_pair = PairInfoNCE(temperature=args.temperature)
    con_loss_fn_sup  = SupConLoss(temperature=args.temperature)

    # 噪声
    noise_root = args.noise_root or os.path.join(args.data_root, "noise_Pavia")
    noise_provider = NoiseProvider(noise_root=noise_root, patch_size=args.patch_size)
    base_noise_types = noise_provider.noise_types
    weighted_noise_types = build_weighted_noises(base_noise_types, args.noise_prior)

    # μ/σ
    mu_t = torch.from_numpy(ds_train.mu).to(device).float()
    sigma_t = torch.from_numpy(ds_train.sigma).to(device).float()

    # 打印
    print(f"[device] {device}")
    print(f"[data] in_chans={in_chans}, num_classes={num_classes}, patch_size={args.patch_size}, train_ratio={args.train_ratio}")
    print(f"[data] train_samples={len(ds_train)}, val_samples={len(ds_val)}")
    print(f"[noise] base={base_noise_types}  prior={args.noise_prior or 'uniform'}  expanded_len={len(weighted_noise_types)}")
    print(f"[contrast] enable={args.enable_contrast}, type={args.contrast_type}, "
          f"lambda={args.lambda_contrast}, temp={args.temperature}, proj_dim={args.proj_dim}")
    print(f"[consistency] lambda_consistency={args.lambda_consistency}, ema_momentum={args.ema_momentum}")
    print(f"[center] lambda_center={args.lambda_center}")
    print(f"[ramp] ramp_epochs={args.ramp_epochs} | [warmup] warmup_epochs={args.warmup_epochs}")
    print(f"[noise_schedule] schedule={args.noise_schedule}, batch_mode={args.batch_noise_mode}")

    best_OA = 0.0
    best_path = os.path.join(args.save_dir, "paviau_contrast_best.pt")

    base_lr = args.lr

    for epoch in range(1, args.epochs + 1):
        # ---- 学习率：Warmup + Cosine ----
        if epoch <= args.warmup_epochs:
            lr_now = base_lr * (epoch / max(1, args.warmup_epochs))
        else:
            t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            lr_now = base_lr * 0.5 * (1 + math.cos(math.pi * t))
        set_optimizer_lr(optimizer, lr_now)

        if args.enable_contrast:
            train_log = train_one_epoch(
                backbone=backbone,
                ema_backbone=ema_backbone,
                proj_head=proj_head,
                loader=loader_train,
                optimizer=optimizer,
                device=device,
                noise_provider=noise_provider,
                con_loss_fn_pair=con_loss_fn_pair,
                con_loss_fn_sup=con_loss_fn_sup,
                mu=mu_t, sigma=sigma_t,
                num_classes=num_classes,
                lambda_contrast=args.lambda_contrast,
                lambda_consistency=args.lambda_consistency,
                lambda_center=args.lambda_center,
                contrast_type=args.contrast_type,
                noise_list=(weighted_noise_types if args.noise_schedule == "roundrobin" else base_noise_types),
                batch_noise_mode=args.batch_noise_mode,
                grad_clip=1.0,
                epoch_idx=epoch,
                ema_momentum=args.ema_momentum,
                ramp_epochs=args.ramp_epochs
            )
        else:
            # baseline：仅 CE（同样冻结 BN，且保持 EMA 跟随）
            backbone.train()
            freeze_bn_running_stats(backbone)
            ce = nn.CrossEntropyLoss()
            meter = {"loss": 0.0, "OA": 0.0}; n = 0
            for batch in loader_train:
                x = batch["image"].to(device).float()
                y = batch["label"].to(device).long()
                optimizer.zero_grad(set_to_none=True)
                logits, _ = backbone(x)
                loss = ce(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
                optimizer.step()
                ema_update(ema_backbone, backbone, m=args.ema_momentum)
                with torch.no_grad():
                    pred = logits.argmax(1)
                    oa = (pred == y).float().mean().item() * 100.0
                    bs = x.size(0); n += bs
                    meter["loss"] += loss.item() * bs
                    meter["OA"]   += oa * bs
            train_log = {"loss": meter["loss"]/max(1,n), "ce": meter["loss"]/max(1,n),
                         "con": 0.0, "cons": 0.0, "center": 0.0, "OA": meter["OA"]/max(1,n)}

        # ---- 关键：验证使用 EMA Teacher，曲线稳定 ----
        val_log = evaluate_clean(ema_backbone, loader_val, device)

        print(f"[Epoch {epoch:03d}] lr={lr_now:.6f} | "
              f"train_loss={train_log['loss']:.4f} ce={train_log['ce']:.4f} con={train_log['con']:.4f} "
              f"cons={train_log['cons']:.4f} center={train_log['center']:.4f} train_OA={train_log['OA']:.2f}% | "
              f"val_loss={val_log['loss']:.4f} val_OA={val_log['OA']:.2f}% val_AA={val_log['AA']:.2f}% val_Kappa={val_log['Kappa']:.2f}%")

        if val_log["OA"] > best_OA:
            best_OA = val_log["OA"]
            torch.save({
                "backbone": backbone.state_dict(),
                "ema_backbone": ema_backbone.state_dict(),
                "proj_head": proj_head.state_dict(),
                "num_classes": num_classes,
                "feat_dim": feat_dim,
                "proj_dim": args.proj_dim,
                "args": vars(args),
                "mu": ds_train.mu, "sigma": ds_train.sigma,
            }, best_path)
            print(f"  >> New best on val (EMA): OA={best_OA:.2f}% (saved to {best_path})")


if __name__ == "__main__":
    main()
