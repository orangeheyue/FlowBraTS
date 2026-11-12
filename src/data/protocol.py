from itertools import product

import torch


MODALITIES = ("flair", "t1ce", "t1", "t2")
MASKS = tuple(bits for bits in product((False, True), repeat=4) if any(bits))


def validate_mask(mask, batch_size=None):
    """Desc: 校验非空四模态可用性掩码。

    Args:
        mask: `[B,4]`，True 表示模态可用。
        batch_size: 可选的期望 batch 大小。
    Output:
        bool 类型的 `[B,4]` 掩码。
    """
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2 or mask.shape[1] != 4:
        raise ValueError("availability mask must have shape [B, 4]")
    if batch_size is not None and mask.shape[0] != batch_size:
        raise ValueError("mask batch size does not match image batch size")
    if (~mask.any(dim=1)).any():
        raise ValueError("all-zero modality mask is forbidden")
    return mask


def apply_mask(image, mask):
    """Desc: 在任何 Student 编码前显式清零缺失模态。

    Args:
        image: `[B,4,D,H,W]` MRI。
        mask: `[B,4]` 可用性掩码。
    Output:
        与 image 同形状的观测输入。
    """
    mask = validate_mask(mask, image.shape[0]).to(image.device)
    return image * mask[:, :, None, None, None].to(image.dtype)


def sample_masks(batch_size, device=None, generator=None):
    """Desc: 从 15 种非空组合中逐样本均匀采样。"""
    choices = torch.tensor(MASKS, dtype=torch.bool, device=device)
    index = torch.randint(len(MASKS), (batch_size,), device=device, generator=generator)
    return choices[index]


def mask_name(mask):
    """Desc: 生成稳定、可读的模态组合名称。"""
    bits = validate_mask(mask, 1)[0].tolist()
    return "+".join(name.upper() for name, used in zip(MODALITIES, bits) if used)
