# models/projection_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    两层MLP投影头：Linear -> BN -> ReLU -> Linear
    仅训练期用于对比损失；评估时可不调用。
    """
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        hidden_dim = max(256, in_dim)  # 稍微宽一些更稳定
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, dim=-1, eps=1e-8)  # L2归一化，便于cosine相似度
        return z
