"""Interchangeable lightweight velocity fields for the FlowBraTS v2.1 ablation."""
import torch
from torch import nn

from .velocity import ResidualBlock, timestep_embedding


class ConditionedVelocity(nn.Module):
    def __init__(self, channels=64, hidden=64, condition_channels=64):
        super().__init__()
        self.condition_channels = condition_channels
        self.condition = nn.Sequential(
            nn.Linear(condition_channels + 12 + 3 + 4, condition_channels), nn.SiLU(),
            nn.Linear(condition_channels, condition_channels))
        self.input = nn.Conv3d(channels * 2, hidden, 1)
        self.output = nn.Conv3d(hidden, channels, 1)

    def context(self, timestep, adjacency, region_need, mask):
        time = timestep_embedding(timestep, self.condition_channels)
        return self.condition(torch.cat(
            (time, adjacency.flatten(1), region_need, mask.float()), dim=1))

    def prepare(self, residual_t, observed):
        return self.input(torch.cat((residual_t, observed), dim=1))


class MLPVelocityNet(ConditionedVelocity):
    """Pointwise channel MLP with cheap local depthwise mixing."""
    def __init__(self, channels=64, hidden=64, condition_channels=64):
        super().__init__(channels, hidden, condition_channels)
        self.film = nn.Linear(condition_channels, hidden * 2)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv3d(hidden, hidden * 8, 1), nn.SiLU(),
            nn.Conv3d(hidden * 8, hidden, 1),
            nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU())
            for _ in range(6)])

    def forward(self, residual_t, timestep, observed, adjacency, region_need, mask):
        condition = self.context(timestep, adjacency, region_need, mask)
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.prepare(residual_t, observed)
        hidden = hidden * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class Transformer3DVelocityNet(ConditionedVelocity):
    """Two-layer global Transformer over the 5x5x5 bottleneck tokens."""
    def __init__(self, channels=64, hidden=64, condition_channels=64):
        super().__init__(channels, hidden, condition_channels)
        self.condition_projection = nn.Linear(condition_channels, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=0.,
                                           activation='gelu', batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 8, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, residual_t, timestep, observed, adjacency, region_need, mask):
        condition = self.condition_projection(
            self.context(timestep, adjacency, region_need, mask))
        hidden = self.prepare(residual_t, observed)
        shape = hidden.shape
        tokens = hidden.flatten(2).transpose(1, 2) + condition[:, None]
        tokens = self.final_norm(self.transformer(tokens))
        hidden = tokens.transpose(1, 2).reshape(shape)
        return self.output(hidden)


class SelectiveAxisBlock(nn.Module):
    """Input-selective diagonal SSM scanned in both directions of D/H/W."""
    def __init__(self, channels, condition_channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.select = nn.Conv1d(channels, channels * 3, 1)
        self.condition = nn.Linear(condition_channels, channels * 2)
        self.mix = nn.Conv3d(channels * 6, channels, 1)
        self.ffn = nn.Sequential(nn.Conv3d(channels, channels * 2, 1), nn.SiLU(),
                                 nn.Conv3d(channels * 2, channels, 1))

    def scan(self, sequence):
        a, b, c = self.select(sequence).chunk(3, dim=1)
        a = .5 + .49 * a.sigmoid()
        product = a.cumprod(-1)
        state = product * (b.tanh() / product.clamp_min(1e-6)).cumsum(-1)
        return c.sigmoid() * state

    def axis(self, x, axis):
        moved = x.movedim(axis, -1)
        shape = moved.shape
        sequence = moved.reshape(-1, shape[1], shape[-1])
        forward = self.scan(sequence)
        backward = self.scan(sequence.flip(-1)).flip(-1)
        return forward.reshape(shape).movedim(-1, axis), backward.reshape(shape).movedim(-1, axis)

    def forward(self, x, condition):
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm(x) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        directions = []
        for axis in (2, 3, 4):
            directions.extend(self.axis(hidden, axis))
        x = x + self.mix(torch.cat(directions, dim=1))
        return x + self.ffn(x)


class Mamba3DVelocityNet(ConditionedVelocity):
    """Pure-PyTorch 3D Mamba-style field with six directional selective scans."""
    def __init__(self, channels=64, hidden=64, condition_channels=64):
        super().__init__(channels, hidden, condition_channels)
        self.blocks = nn.ModuleList([SelectiveAxisBlock(hidden, condition_channels)
                                     for _ in range(6)])

    def forward(self, residual_t, timestep, observed, adjacency, region_need, mask):
        condition = self.context(timestep, adjacency, region_need, mask)
        hidden = self.prepare(residual_t, observed)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


class UNet3DVelocityNet(ConditionedVelocity):
    """One-downsample 3D U-Net suitable for a 5x5x5 latent grid."""
    def __init__(self, channels=64, hidden=64, condition_channels=64):
        width = 48
        super().__init__(channels, width, condition_channels)
        self.enc = ResidualBlock(width, condition_channels)
        self.down = nn.Conv3d(width, width, 3, stride=2, padding=1)
        self.bottleneck = ResidualBlock(width, condition_channels)
        self.up = nn.ConvTranspose3d(width, width, 3, stride=2, padding=1)
        self.fuse = nn.Conv3d(width * 2, width, 1)
        self.dec = ResidualBlock(width, condition_channels)

    def forward(self, residual_t, timestep, observed, adjacency, region_need, mask):
        condition = self.context(timestep, adjacency, region_need, mask)
        skip = self.enc(self.prepare(residual_t, observed), condition)
        low = self.bottleneck(self.down(skip), condition)
        up = self.up(low, output_size=skip.shape)
        hidden = self.dec(self.fuse(torch.cat((up, skip), dim=1)), condition)
        return self.output(hidden)


def build_velocity(kind, channels=64, hidden=64, condition_channels=64):
    choices = {'mlp': MLPVelocityNet, 'transformer3d': Transformer3DVelocityNet,
               'mamba3d': Mamba3DVelocityNet, 'unet3d': UNet3DVelocityNet}
    if kind == 'residual':
        from .velocity import VelocityNet
        return VelocityNet(channels, hidden, condition_channels)
    if kind not in choices:
        raise ValueError(f'Unknown velocity field {kind!r}; expected residual or {tuple(choices)}')
    return choices[kind](channels, hidden, condition_channels)
