import torch
from torch import nn
from torch.nn import functional as functional

from src.data.protocol import apply_mask


class TinyM2FAdapter(nn.Module):
    """仅供单测使用的 M2FTrans 接口替身，不进入训练或实验。"""

    def __init__(self, channels=8):
        super().__init__()
        self.encoder = nn.Conv3d(4, channels, 1)
        self.decoder = nn.Conv3d(channels, 4, 1)

    def encode(self, image, mask):
        observed = apply_mask(image, mask)
        latent = functional.adaptive_avg_pool3d(self.encoder(observed), 2)
        return {"latent": latent, "context": image.shape[2:],
                "logits": self.decode(latent, image.shape[2:], mask)}

    def decode(self, latent, context, _mask):
        logits = self.decoder(latent)
        return functional.interpolate(logits, size=context, mode="trilinear", align_corners=False)
