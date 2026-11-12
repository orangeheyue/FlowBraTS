import torch
from torch import nn

from src.data.protocol import validate_mask


class ModalityRegionGraph(nn.Module):
    """缺失感知的模态—区域依赖图；区域顺序为 NCR/NET、ET、ED。"""

    def __init__(self, channels=256, hidden=64):
        super().__init__()
        # 行顺序严格对应 [FLAIR,T1CE,T1,T2]。
        prior = torch.tensor(
            [[0.0, 0.0, 2.0], [2.0, 3.0, 0.0],
             [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=torch.float32,
        )
        self.register_buffer("prior", prior)
        self.dynamic = nn.Sequential(
            nn.Linear(channels + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 12),
        )

    def forward(self, latent, availability_mask):
        """Desc: 估计当前病例的图和缺失区域证据需求。

        Args:
            latent: `[B,C,D,H,W]` 观测融合特征。
            availability_mask: `[B,4]`。
        Output:
            adjacency `[B,4,3]` 与 region_need `[B,3]`。
        """
        mask = validate_mask(availability_mask, latent.shape[0]).to(latent.device)
        pooled = latent.mean(dim=(2, 3, 4))
        dynamic = self.dynamic(torch.cat((pooled, mask.float()), dim=1)).view(-1, 4, 3)
        adjacency = torch.softmax(self.prior[None] + dynamic, dim=1)
        missing = (~mask).float().unsqueeze(-1)
        region_need = (adjacency * missing).sum(dim=1)
        return adjacency, region_need

    @staticmethod
    def sparse_loss(adjacency):
        """Desc: 以归一化熵鼓励图连接具有选择性。"""
        entropy = -(adjacency.clamp_min(1e-8) * adjacency.clamp_min(1e-8).log()).sum(dim=1)
        return entropy.mean()
