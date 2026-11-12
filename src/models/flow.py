import torch
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from torch import nn

from .velocity import VelocityNet


class ConditionalResidualFlow(nn.Module):
    """从条件残差先验到完整任务残差的 Rectified Flow。"""

    def __init__(self, channels=256, hidden=64, flow_channels=64, velocity_type='residual'):
        super().__init__()
        self.flow_channels = flow_channels
        self.to_flow = nn.Conv3d(channels, flow_channels, 1, bias=False)
        self.to_latent = nn.Conv3d(flow_channels, channels, 1, bias=False)
        self.prior_mean = nn.Sequential(
            nn.Conv3d(flow_channels, hidden, 1), nn.SiLU(),
            nn.Conv3d(hidden, flow_channels, 1)
        )
        self.prior_condition = nn.Linear(12 + 3 + 4, flow_channels)
        self.prior_log_sigma = nn.Sequential(
            nn.Linear(flow_channels + 12 + 3 + 4, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        if velocity_type == 'residual':
            self.velocity = VelocityNet(flow_channels, hidden)
        else:
            from .velocity_variants import build_velocity
            self.velocity = build_velocity(velocity_type, flow_channels, hidden)
        self.velocity_type = velocity_type
        self.path = CondOTProbPath()

    def source(self, observed, adjacency, region_need, mask, noise=None):
        """Desc: 构造条件初始化 `mu(c)+sigma(c)epsilon`。"""
        if noise is None:
            noise = torch.randn_like(observed)
        graph_condition = torch.cat((adjacency.flatten(1), region_need, mask.float()), dim=1)
        condition_map = self.prior_condition(graph_condition)[:, :, None, None, None]
        mean = self.prior_mean(observed + condition_map)
        pooled = observed.mean(dim=(2, 3, 4))
        sigma = self.prior_log_sigma(
            torch.cat((pooled, graph_condition), dim=1)
        ).clamp(-4.0, 1.0).exp()
        return mean + sigma[:, :, None, None, None] * noise

    def loss(self, observed, teacher, adjacency, region_need, mask):
        """Desc: 计算条件 OT 直线路径的 Flow Matching 损失。"""
        observed_flow = self.to_flow(observed)
        teacher_flow = self.to_flow(teacher.detach())
        target = teacher_flow - observed_flow
        source = self.source(observed_flow, adjacency, region_need, mask)
        timestep = torch.rand(observed.shape[0], device=observed.device)
        sample = self.path.sample(x_0=source, x_1=target, t=timestep)
        prediction = self.velocity(
            sample.x_t, timestep, observed_flow, adjacency, region_need, mask
        )
        return (prediction - sample.dx_t).square().mean()

    def sample(self, observed, adjacency, region_need, mask, nfe=2, noise=None,
               method="euler", enable_grad=False, deterministic=False):
        """Desc: 使用 flow_matching 官方 ODESolver 生成任务残差。"""
        if nfe < 1:
            raise ValueError("nfe must be positive")
        supported = {"euler": "euler", "heun": "heun3", "rk4": "rk4"}
        if method not in supported:
            raise ValueError("solver must be euler, heun or rk4")
        observed_flow = self.to_flow(observed)
        if deterministic:
            noise = torch.zeros_like(observed_flow)
        source = self.source(observed_flow, adjacency, region_need, mask, noise)

        def velocity_model(x, t, **_extras):
            if t.ndim == 0:
                t = t.expand(x.shape[0])
            return self.velocity(x, t, observed_flow, adjacency, region_need, mask)

        solver = ODESolver(velocity_model=velocity_model)
        residual_flow = solver.sample(
            x_init=source,
            step_size=1.0 / nfe,
            method=supported[method],
            time_grid=observed.new_tensor([0.0, 1.0]),
            enable_grad=enable_grad,
        )
        return self.to_latent(residual_flow)
