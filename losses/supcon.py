# losses/supcon.py
import torch
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        z: (B, d) L2-normalized embeddings of *all views* in the batch
           这里我们组织为：前半段是“干净 zc”，后半段是“退化 zd”，总共 2B
        y: (B,) labels（只传干净的标签即可，退化视图同 index 的标签与其对应）
        """
        z = F.normalize(z, dim=1)
        B = y.size(0)
        zc, zd = z[:B], z[B:]                  # 对齐视图
        z_all = torch.cat([zc, zd], dim=0)     # 2B, d
        y_all = torch.cat([y, y], dim=0)       # 2B

        sim = torch.div(torch.matmul(z_all, z_all.t()), self.t)  # 2B x 2B
        mask = torch.eq(y_all.unsqueeze(1), y_all.unsqueeze(0)).float()  # 正样本掩码（同类）
        logits = sim - torch.max(sim, dim=1, keepdim=True).values  # 数值稳定

        # 排除自身
        self_mask = torch.eye(2*B, device=z.device)
        mask = mask * (1 - self_mask)

        # 对每个锚点 i：正样本集合 P(i)；负样本为其余
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 仅在正样本上取均值
        pos_count = mask.sum(1)
        loss = -(mask * log_prob).sum(1) / (pos_count + 1e-12)
        return loss.mean()
