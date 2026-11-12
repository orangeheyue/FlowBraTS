import math

import torch
from torch import nn


def timestep_embedding(timestep, channels):
    """Desc: 构造标准正余弦时间嵌入。"""
    half = channels // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequency = torch.exp(-scale * torch.arange(half, device=timestep.device))
    angle = timestep[:, None] * frequency[None]
    embedding = torch.cat((angle.sin(), angle.cos()), dim=1)
    if channels % 2:
        embedding = torch.cat((embedding, embedding.new_zeros(len(timestep), 1)), dim=1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, channels, condition_channels):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_channels, channels * 2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, condition):
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.conv1(self.act(self.norm1(x)))
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        return x + self.conv2(self.act(hidden))


class VelocityNet(nn.Module):
    """用于低分辨率 3D latent 的轻量速度场。"""

    def __init__(self, channels=256, hidden=64, condition_channels=64):
        super().__init__()
        self.condition_channels = condition_channels
        self.condition = nn.Sequential(
            nn.Linear(condition_channels + 12 + 3 + 4, condition_channels),
            nn.SiLU(),
            nn.Linear(condition_channels, condition_channels),
        )
        self.input = nn.Conv3d(channels * 2, hidden, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, condition_channels),
            ResidualBlock(hidden, condition_channels),
        ])
        self.output = nn.Conv3d(hidden, channels, 1)

    def forward(self, residual_t, timestep, observed, adjacency, region_need, mask):
        time = timestep_embedding(timestep, self.condition_channels)
        graph = adjacency.flatten(1)
        condition = self.condition(torch.cat((time, graph, region_need, mask.float()), dim=1))
        hidden = self.input(torch.cat((residual_t, observed), dim=1))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)
