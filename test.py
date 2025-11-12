# test.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from typing import Dict, Any, List

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 让项目子目录可被 import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in ("models", "losses", "utils"):
    _p = os.path.join(_THIS_DIR, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import trans  # 你的 backbone 定义
from utils.noise_provider import NoiseProvider


# ======================= PaviaU 读取与划分（与 main.py 严格一致） =======================
class PaviaUSplit(Dataset):
    """
    只负责提供 (row, col, label) 的索引数据（用于构建 clean / degraded 的评测流）
    - clean 评测用 PaviaU 标准化过的 patch（用 train μ/σ）
    - degraded 评测用 noise_provider 同位裁 patch，再按 train μ/σ 标准化
    """
    def __init__(self,
                 root: str,
                 split: str,
                 patch_size: int,
                 train_ratio: float,
                 seed: int,
                 train_mu: np.ndarray = None,
                 train_sigma: np.ndarray = None):
        assert split in ("train", "val")
        self.root = root
        self.ps = patch_size
        self.split = split

        mat_img = sio.loadmat(os.path.join(root, "PaviaU.mat"))
        mat_gt  = sio.loadmat(os.path.join(root, "PaviaU_gt.mat"))

        # 自动探测变量
        img = next((mat_img[k] for k in mat_img.keys() if not k.startswith("__")), None)
        gt  = next((mat_gt[k]  for k in mat_gt.keys()  if not k.startswith("__")), None)
        if img is None or gt is None:
            raise RuntimeError("未在 PaviaU.mat / PaviaU_gt.mat 中找到有效矩阵变量。")

        # 统一为 (H,W,C)
        if img.ndim == 2:
            img = img[:, :, None]
        elif img.ndim == 3 and img.shape[0] < 16 and img.shape[-1] > 16:
            img = np.transpose(img, (1, 2, 0))
        self.img_raw = img.astype(np.float32)  # 清洁影像（仅 clean 评测需要）
        self.gt  = gt.astype(np.int64)

        H, W, C = self.img_raw.shape
        self.in_chans = C

        # 划分
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

        # μ/σ：由训练阶段给出（从 checkpoint 里读取）
        assert train_mu is not None and train_sigma is not None, "需要传入训练阶段的 μ/σ"
        self.mu = train_mu.astype(np.float32)
        self.sigma = train_sigma.astype(np.float32)

        # 准备 clean 评测的标准化全图
        flat = self.img_raw.reshape(-1, C)  # 不用于统计，仅用于说明
        self.img = (self.img_raw - self.mu) / self.sigma

    def __len__(self):
        return len(self.labels)

    def _get_clean_patch(self, r: int, c: int) -> np.ndarray:
        H, W, C = self.img.shape
        ps = self.ps
        half = ps // 2
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - half, c + half + 1
        rr = np.clip(np.arange(r0, r1), 0, H - 1)
        cc = np.clip(np.arange(c0, c1), 0, W - 1)
        patch = self.img[rr][:, cc]            # (ps, ps, C) 已标准化
        patch = np.transpose(patch, (2, 0, 1)) # (C, ps, ps)
        return patch.astype(np.float32)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = int(self.rows[i]); c = int(self.cols[i])
        y = int(self.labels[i])
        clean_patch = self._get_clean_patch(r, c)
        return {
            "clean": torch.from_numpy(clean_patch),  # (C,ps,ps)
            "label": torch.tensor(y).long(),
            "row": torch.tensor(r).long(),
            "col": torch.tensor(c).long(),
            "index": torch.tensor(i).long(),
        }


# ============================== 指标（OA/AA/Kappa） ==============================
def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def _scores_from_cm(cm: np.ndarray) -> Dict[str, float]:
    n = cm.sum()
    po = np.trace(cm) / max(1, n)  # OA
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), 0.0)
    aa = float(np.mean(per_class_acc))
    row_m = cm.sum(1)
    col_m = cm.sum(0)
    pe = float(np.dot(row_m, col_m)) / max(1, n * n)
    kappa = (po - pe) / (1 - pe + 1e-12)
    return {"OA": float(po * 100.0), "AA": float(aa * 100.0), "Kappa": float(kappa * 100.0)}

@torch.no_grad()
def eval_clean(backbone: nn.Module,
               loader: DataLoader,
               device: torch.device) -> Dict[str, float]:
    backbone.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss = 0.0
    ys, ps = [], []
    n = 0
    for batch in loader:
        x = batch["clean"].to(device).float()
        y = batch["label"].to(device).long()
        logits, _feat = backbone(x)
        loss = ce(logits, y)
        pred = logits.argmax(dim=1)
        bs = x.size(0)
        n += bs
        tot_loss += loss.item() * bs
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys, axis=0)
    ps = np.concatenate(ps, axis=0)
    num_classes = int(logits.size(1))
    cm = _confusion_matrix(ys, ps, num_classes)
    scores = _scores_from_cm(cm)
    return {"loss": tot_loss / max(1, n), **scores}

@torch.no_grad()
def eval_degraded(backbone: nn.Module,
                  loader: DataLoader,
                  device: torch.device,
                  noise_provider: NoiseProvider,
                  noise_type: str,
                  mu: torch.Tensor,
                  sigma: torch.Tensor) -> Dict[str, float]:
    """
    在给定噪声类型上评测退化数据：
    - 使用 loader 提供的 (row,col,label) 批
    - 从 noise_provider 同位裁 patch
    - 用训练 μ/σ 标准化，再送 backbone
    """
    backbone.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss = 0.0
    ys, ps = [], []
    n = 0

    for batch in loader:
        rows = batch["row"].cpu().numpy()
        cols = batch["col"].cpu().numpy()
        y = batch["label"].to(device).long()

        # 取退化 patch（原值），再按训练 μ/σ 标准化
        x_deg_np = noise_provider.get_patches_batch(rows, cols, [noise_type]*len(rows), ps=loader.dataset.ps)
        x_deg = torch.from_numpy(x_deg_np).to(device=device, dtype=torch.float32)  # (B,C,ps,ps)
        x_deg = (x_deg - mu.view(1, -1, 1, 1)) / sigma.view(1, -1, 1, 1)

        logits, _feat = backbone(x_deg)
        loss = ce(logits, y)
        pred = logits.argmax(dim=1)

        bs = x_deg.size(0)
        n += bs
        tot_loss += loss.item() * bs
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    ys = np.concatenate(ys, axis=0)
    ps = np.concatenate(ps, axis=0)
    num_classes = int(logits.size(1))
    cm = _confusion_matrix(ys, ps, num_classes)
    scores = _scores_from_cm(cm)
    return {"loss": tot_loss / max(1, n), **scores}


# ============================== 构建 backbone 并加载权重 ==============================
def build_backbone_from_trans(num_classes: int,
                              in_chans: int,
                              patch_size: int,
                              device: torch.device,
                              ckpt_args: dict = None) -> nn.Module:
    """
    与 main.py 一致构造；若 ckpt_args 提供可用的 net/data 超参，进行覆盖以保持一致性。
    """
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
    # 用 ckpt 里的 args 尽可能对齐
    if ckpt_args is not None:
        # 如果训练时改动过这些参数，尽量覆盖
        net = params["net"]
        for k in ["depth", "heads", "mlp_dim", "kernal", "padding", "dropout", "dim", "mask_pct"]:
            if k in ckpt_args:
                net[k] = ckpt_args[k]
        if "patch_size" in ckpt_args:
            params["data"]["patch_size"] = ckpt_args["patch_size"]
            params["data"]["serve_patch_size"] = ckpt_args["patch_size"]

    model = trans.TransFormerNet(params).to(device)
    return model


# ===================================== 主流程 =====================================
def main():
    ap = argparse.ArgumentParser("Evaluate on degraded PaviaU")
    ap.add_argument("--ckpt", type=str, required=True, help="训练保存的权重路径（.pt）")
    ap.add_argument("--data_root", type=str, default="dataset", help="包含 PaviaU/ 与 noise_Pavia/ 的目录")
    ap.add_argument("--noise_root", type=str, default=None, help="默认使用 {data_root}/noise_Pavia")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)

    # 测试划分与 patch
    ap.add_argument("--patch_size", type=int, default=None, help="测试 patch 大小（默认从 ckpt.args 读取）")
    ap.add_argument("--seed", type=int, default=None, help="划分随机种子（默认从 ckpt.args 读取）")
    ap.add_argument("--train_ratio", type=float, default=None, help="训练集比例（默认从 ckpt.args 读取）")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"], help="评测哪个 split（默认 val）")

    # 退化评测
    ap.add_argument("--noises", type=str, default="all",
                    help="要评测的噪声类型，逗号分隔；或 'all' 评测全部可用噪声")
    ap.add_argument("--eval_clean", action="store_true", help="同时评测干净数据作为对照")

    args = ap.parse_args()

    # 加载 checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    num_classes = int(ckpt.get("num_classes"))
    ckpt_args = ckpt.get("args", {}) or {}

    # 从 ckpt.args 取默认参数（若命令行未指定）
    patch_size = args.patch_size if args.patch_size is not None else int(ckpt_args.get("patch_size", 13))
    seed = args.seed if args.seed is not None else int(ckpt_args.get("seed", 42))
    train_ratio = args.train_ratio if args.train_ratio is not None else float(ckpt_args.get("train_ratio", 0.2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 数据与 split
    pavia_root = os.path.join(args.data_root, "PaviaU")
    # μ/σ（训练时保存）
    mu = np.array(ckpt["mu"], dtype=np.float32)
    sigma = np.array(ckpt["sigma"], dtype=np.float32)

    # 构建 split（只产生 rows/cols/labels 与 clean patch）
    ds_split = PaviaUSplit(pavia_root, split=args.split,
                           patch_size=patch_size, train_ratio=train_ratio, seed=seed,
                           train_mu=mu, train_sigma=sigma)
    in_chans = ds_split.in_chans
    loader = DataLoader(ds_split, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # 重建 backbone 并加载权重
    backbone = build_backbone_from_trans(num_classes=num_classes,
                                         in_chans=in_chans,
                                         patch_size=patch_size,
                                         device=device,
                                         ckpt_args=ckpt_args)
    missing, unexpected = backbone.load_state_dict(ckpt["backbone"], strict=False)
    if missing:
        print("[warn] missing keys in backbone:", missing)
    if unexpected:
        print("[warn] unexpected keys in backbone:", unexpected)

    backbone = backbone.to(device)
    backbone.eval()

    # 打印关键信息
    print(f"[device] {device}")
    print(f"[ckpt] loaded: {args.ckpt}")
    print(f"[data] split={args.split}, train_ratio={train_ratio}, seed={seed}, patch_size={patch_size}")
    print(f"[data] samples={len(ds_split)}, in_chans={in_chans}, num_classes={ds_split.num_classes}")

    # 可选：评测干净数据
    if args.eval_clean:
        clean_log = eval_clean(backbone, loader, device)
        print(f"[CLEAN]  loss={clean_log['loss']:.4f}  OA={clean_log['OA']:.2f}%  AA={clean_log['AA']:.2f}%  Kappa={clean_log['Kappa']:.2f}%")

    # 退化评测
    noise_root = args.noise_root or os.path.join(args.data_root, "noise_Pavia")
    noise_provider = NoiseProvider(noise_root=noise_root, patch_size=patch_size)

    # 解析要评测的噪声列表
    if args.noises.lower() == "all":
        noise_list = noise_provider.noise_types
    else:
        noise_list = [s.strip() for s in args.noises.split(",") if s.strip()]
        # 简单校验：
        unknown = [n for n in noise_list if n not in noise_provider.noise_types]
        if unknown:
            print(f"[warn] 未在 {noise_root} 找到这些噪声文件名（去掉扩展名）: {unknown}")
            print(f"       可用噪声类型：{noise_provider.noise_types}")
            noise_list = [n for n in noise_list if n in noise_provider.noise_types]

    mu_t = torch.from_numpy(mu).to(device).float()
    sigma_t = torch.from_numpy(sigma).to(device).float()

    # 逐噪声评测
    logs: List[Dict[str, Any]] = []
    for ntype in noise_list:
        res = eval_degraded(backbone, loader, device, noise_provider, ntype, mu_t, sigma_t)
        logs.append((ntype, res))
        print(f"[DEG:{ntype:<12}] loss={res['loss']:.4f}  OA={res['OA']:.2f}%  AA={res['AA']:.2f}%  Kappa={res['Kappa']:.2f}%")

    # 简单汇总（平均）
    if logs:
        oa = np.mean([x[1]["OA"] for x in logs])
        aa = np.mean([x[1]["AA"] for x in logs])
        kappa = np.mean([x[1]["Kappa"] for x in logs])
        print(f"[DEG:AVG]        OA={oa:.2f}%  AA={aa:.2f}%  Kappa={kappa:.2f}%")

if __name__ == "__main__":
    import torch
    main()
