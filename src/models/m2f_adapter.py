import sys
from pathlib import Path

import torch
from torch import nn

from src.data.protocol import apply_mask, validate_mask


class M2FTransAdapter(nn.Module):
    """直接复用 M2FTrans_v1，并暴露其 256×5×5×5 融合瓶颈。"""

    def __init__(self, source_root, checkpoint):
        """Desc: 构建并严格加载 M2FTrans_v1 预训练模型。

        Args:
            source_root: M2FTrans_v1 源码目录。
            checkpoint: 指定的 BraTS2020 预训练权重。
        Output:
            可分离 encode/decode 的适配器。
        """
        super().__init__()
        source_root = str(Path(source_root).resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from models.fusiontrans import Model  # pylint: disable=import-outside-toplevel

        self.backbone = Model(num_cls=4)
        self.load_report = (self._load_strict(checkpoint) if checkpoint is not None else {
            "checkpoint": None, "initialization": "random", "missing_keys": [], "unexpected_keys": []})
        # 只移除最终 softmax，使新增损失可直接使用 logits；其他结构和参数不变。
        self.backbone.decoder_fusion.softmax = nn.Identity()
        self.backbone.is_training = False

    def _load_strict(self, checkpoint):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        result = self.backbone.load_state_dict(state, strict=True)
        return {
            "checkpoint": str(path),
            "loaded_keys": len(state),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    def encode(self, image, availability_mask):
        """Desc: 先清零缺失模态，再运行原 M2FTrans 编码/融合并截取 decoder 输入。

        Args:
            image: `[B,4,80,80,80]`。
            availability_mask: `[B,4]`。
        Output:
            字典，含融合 latent、原解码上下文和基线 logits。
        """
        mask = validate_mask(availability_mask, image.shape[0]).to(image.device)
        observed = apply_mask(image, mask)
        captured = {}

        def capture(_module, inputs):
            captured["decoder_inputs"] = inputs

        handle = self.backbone.decoder_fusion.register_forward_pre_hook(capture)
        try:
            logits = self.backbone(observed, mask)
        finally:
            handle.remove()
        context = captured.get("decoder_inputs")
        if context is None or len(context) != 7:
            raise RuntimeError("failed to capture M2FTrans decoder inputs")
        return {"latent": context[5], "context": context, "logits": logits}

    def decode(self, latent, context, availability_mask):
        """Desc: 复用原 M2FTrans Decoder，仅替换融合 bottleneck。"""
        mask = validate_mask(availability_mask, latent.shape[0]).to(latent.device)
        args = (*context[:5], latent, mask)
        logits, _ = self.backbone.decoder_fusion(*args)
        return logits
