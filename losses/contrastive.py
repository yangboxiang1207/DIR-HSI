# losses/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PairInfoNCE(nn.Module):
    """
    批内InfoNCE（对称）：正样为 (zc[i], zd[i])，负样为其它样本及其退化视图。
    输入:
        zc: (B, D)  来自干净样本的投影向量 (L2 norm)
        zd: (B, D)  来自对应退化样本的投影向量 (L2 norm)
    返回:
        标量 loss
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    @torch.no_grad()
    def _cos_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (N, D), b: (M, D) -> (N, M)
        return a @ b.t()

    def forward(self, zc: torch.Tensor, zd: torch.Tensor) -> torch.Tensor:
        assert zc.dim() == 2 and zd.dim() == 2 and zc.shape == zd.shape
        B, D = zc.shape

        # 构造相似度矩阵
        logits_c2d = self._cos_sim(zc, zd) / self.tau  # clean作为query
        logits_d2c = self._cos_sim(zd, zc) / self.tau  # deg作为query

        labels = torch.arange(B, device=zc.device, dtype=torch.long)

        loss_c2d = F.cross_entropy(logits_c2d, labels)
        loss_d2c = F.cross_entropy(logits_d2c, labels)

        return 0.5 * (loss_c2d + loss_d2c)
