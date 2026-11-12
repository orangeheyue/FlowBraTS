import torch
from torch import nn

from src.data.protocol import validate_mask
from .flow import ConditionalResidualFlow
from .graph import ModalityRegionGraph
from .utility import CounterfactualUtility


class FlowBraTS(nn.Module):
    """M2FTrans + 动态依赖图 + 潜空间残差 Flow + 反事实效用门控。"""

    MODES = ("baseline", "ungated", "infer", "flow_train", "utility_train", "joint_train")

    def __init__(self, backbone, channels=256, flow_channels=64, hidden=64,
                 nfe=2, solver="euler",
                 gate_temperature=0.5, utility_threshold=0.0,
                 uncertainty_penalty=1.0):
        super().__init__()
        self.backbone = backbone
        self.graph = ModalityRegionGraph(channels, hidden)
        self.flow = ConditionalResidualFlow(channels, hidden, flow_channels)
        self.utility = CounterfactualUtility(
            channels, hidden, gate_temperature, utility_threshold, uncertainty_penalty
        )
        self.nfe = nfe
        self.solver = solver

    def extract_teacher(self, full_image):
        """Desc: 训练期提取全模态教师表征；推理代码不调用此接口。"""
        mask = torch.ones(full_image.shape[0], 4, dtype=torch.bool, device=full_image.device)
        with torch.no_grad():
            return self.backbone.encode(full_image, mask)["latent"].detach()

    def forward(self, image, availability_mask, mode="infer", teacher_latent=None):
        """Desc: 执行缺失模态分割或对应训练路径。

        Args:
            image: `[B,4,D,H,W]`；在 backbone 内编码前清零。
            availability_mask: `[B,4]`。
            mode: baseline/ungated/infer 或三个训练模式。
            teacher_latent: 仅 Flow 训练允许的全模态目标。
        Output:
            含 logits、latent、图、残差、门控和 FM 损失的字典。
        """
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        mask = validate_mask(availability_mask, image.shape[0]).to(image.device)
        if teacher_latent is not None and mode not in ("flow_train", "joint_train"):
            raise ValueError("teacher latent is forbidden outside Flow training")

        encoded = self.backbone.encode(image, mask)
        observed = encoded["latent"]
        adjacency, region_need = self.graph(observed, mask)
        zero = observed.sum() * 0.0

        if mode in ("flow_train", "joint_train"):
            if teacher_latent is None:
                raise ValueError("teacher_latent is required for Flow training")
            loss_fm = self.flow.loss(
                observed, teacher_latent, adjacency, region_need, mask
            )
        else:
            loss_fm = zero

        if mode == "flow_train":
            return {
                "loss_fm": loss_fm, "observed": observed,
                "adjacency": adjacency, "region_need": region_need,
            }

        full = mask.all(dim=1)
        residual = torch.zeros_like(observed)
        if mode != "baseline" and (~full).any():
            # 默认零噪声确保分割评估可重复；训练 FM 仍使用随机源分布。
            residual = self.flow.sample(
                observed, adjacency, region_need, mask, self.nfe,
                method=self.solver,
                enable_grad=mode == "joint_train",
                deterministic=True,
            )
            residual = residual * (~full)[:, None, None, None, None]

        estimate, std, region_gate, channel_gate = self.utility(
            observed, residual, adjacency, mask
        )
        if mode == "baseline":
            final = observed
        elif mode == "ungated":
            final = observed + residual
        else:
            final = observed + channel_gate * residual

        logits = self.backbone.decode(final, encoded["context"], mask)
        output = {
            "logits": logits,
            "baseline_logits": encoded["logits"].detach(),
            "observed": observed,
            "adjacency": adjacency,
            "region_need": region_need,
            "residual": residual,
            "utility": estimate,
            "utility_std": std,
            "region_gate": region_gate,
            "channel_gate": channel_gate,
            "loss_fm": loss_fm,
        }
        if mode in ("utility_train", "joint_train"):
            # This decoder supplies only detached utility targets, never a
            # segmentation training gradient. Avoid retaining its activation graph.
            with torch.no_grad():
                output["ungated_logits"] = self.backbone.decode(
                    observed + residual.detach(), encoded["context"], mask
                )
        return output
