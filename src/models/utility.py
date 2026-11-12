import torch
from torch import nn


class CounterfactualUtility(nn.Module):
    """预测生成残差对 WT、TC、ET 三个评估区域的收益与不确定性。"""

    def __init__(self, channels=256, hidden=64, temperature=0.5,
                 threshold=0.0, uncertainty_penalty=1.0):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.uncertainty_penalty = uncertainty_penalty
        self.estimator = nn.Sequential(
            nn.Linear(channels * 2 + 12 + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        self.channel_region = nn.Parameter(torch.zeros(channels, 3))

    def forward(self, observed, residual, adjacency, mask):
        """Desc: 估计区域收益并生成保守门控。

        Args:
            observed/residual: `[B,C,D,H,W]`。
            adjacency: `[B,4,3]`。
            mask: `[B,4]`。
        Output:
            utility/std/gate `[B,3]` 和 channel_gate `[B,C,1,1,1]`。
        """
        pooled = torch.cat((observed.mean((2, 3, 4)), residual.mean((2, 3, 4))), dim=1)
        values = self.estimator(torch.cat((pooled, adjacency.flatten(1), mask.float()), dim=1))
        utility, log_variance = values.chunk(2, dim=1)
        std = (0.5 * log_variance.clamp(-8.0, 8.0)).exp()
        score = utility - self.uncertainty_penalty * std - self.threshold
        region_gate = torch.sigmoid(score / self.temperature)
        assignment = torch.softmax(self.channel_region, dim=1)
        channel_gate = region_gate @ assignment.transpose(0, 1)
        return utility, std, region_gate, channel_gate[:, :, None, None, None]
