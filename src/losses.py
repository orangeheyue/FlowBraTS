import torch
from torch.nn import functional as functional


def internal_labels(target):
    """Desc: 将 BraTS 原始 ET=4 显式映射为内部 ET=3。"""
    target = target.long().clone()
    target[target == 4] = 3
    valid = torch.tensor([0, 1, 2, 3], device=target.device)
    if not torch.isin(target.unique(), valid).all():
        raise ValueError("target contains labels outside {0,1,2,3,4}")
    return target


def region_probabilities(logits):
    """Desc: 从四类互斥 logits 构造按 `[WT,TC,ET]` 排列的概率。"""
    probability = logits.softmax(dim=1)
    wt = probability[:, 1] + probability[:, 2] + probability[:, 3]
    tc = probability[:, 1] + probability[:, 3]
    et = probability[:, 3]
    return torch.stack((wt, tc, et), dim=1)


def region_targets(target):
    """Desc: 从内部标签构造按 `[WT,TC,ET]` 排列的二值目标。"""
    target = internal_labels(target)
    wt = target != 0
    tc = (target == 1) | (target == 3)
    et = target == 3
    return torch.stack((wt, tc, et), dim=1).float()


def regional_loss(logits, target, reduction="mean", region_weights=None):
    """Desc: 返回每病例、每区域的 Dice+BCE 损失。"""
    prediction = region_probabilities(logits)
    truth = region_targets(target)
    dims = tuple(range(2, prediction.ndim))
    intersection = (prediction * truth).sum(dims)
    dice = 1.0 - (2.0 * intersection + 1e-5) / (
        prediction.sum(dims) + truth.sum(dims) + 1e-5
    )
    # WT/TC 是多个 softmax 类概率之和，不能等价改写为单个 BCEWithLogits；
    # PyTorch 又禁止在 autocast 中直接执行概率形式 BCE，因此局部使用 FP32。
    with torch.amp.autocast(prediction.device.type, enabled=False):
        safe_prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
        bce = functional.binary_cross_entropy(
            safe_prediction, truth.float(), reduction="none"
        ).mean(dims)
    loss = dice + bce
    if reduction == "none":
        return loss
    if reduction == "mean":
        if region_weights is None:
            return loss.mean()
        weights = torch.as_tensor(region_weights, device=loss.device, dtype=loss.dtype)
        if weights.shape != (3,) or (weights <= 0).any():
            raise ValueError("region_weights must contain three positive values")
        return ((loss * weights).sum(dim=1) / weights.sum()).mean()
    raise ValueError("reduction must be none or mean")


def et_tversky_loss(logits, target, alpha=0.7, beta=0.3):
    """ET Tversky；较大的 alpha 重点抑制小病灶和无 ET 块的假阳性。"""
    probability = region_probabilities(logits)[:, 2].float()
    truth = region_targets(target)[:, 2].float()
    dims = tuple(range(1, probability.ndim))
    tp = (probability * truth).sum(dims)
    fp = (probability * (1.0 - truth)).sum(dims)
    fn = ((1.0 - probability) * truth).sum(dims)
    present = truth.sum(dims) > 0
    if not present.any():
        return probability.sum() * 0.0
    return (1.0 - (tp + 1e-5) / (tp + alpha * fp + beta * fn + 1e-5))[present].mean()


def empty_et_false_positive_loss(logits, target, topk_fraction=0.002):
    """惩罚无 ET 训练块中置信度最高的零散预测，而不被大量背景稀释。"""
    probability = region_probabilities(logits)[:, 2].float()
    truth = region_targets(target)[:, 2]
    empty = truth.flatten(1).sum(dim=1) == 0
    if not empty.any():
        return probability.sum() * 0.0
    flattened = probability[empty].flatten(1)
    k = max(1, min(flattened.shape[1], round(flattened.shape[1] * topk_fraction)))
    return flattened.topk(k, dim=1).values.mean()


def segmentation_loss(logits, target, ce_weight=0.2, region_weights=None,
                      ce_class_weights=None, et_tversky_weight=0.0,
                      et_fp_weight=0.0, et_fp_topk_fraction=0.002,
                      et_tversky_alpha=0.7, et_tversky_beta=0.3):
    """BraTS 区域损失；可配置 ET 专项召回与假阳性约束。"""
    target = internal_labels(target)
    class_weights = None
    if ce_class_weights is not None:
        class_weights = torch.as_tensor(
            ce_class_weights, device=logits.device, dtype=logits.float().dtype
        )
        if class_weights.shape != (4,) or (class_weights <= 0).any():
            raise ValueError("ce_class_weights must contain four positive values")
    loss = regional_loss(logits, target, region_weights=region_weights)
    loss = loss + ce_weight * functional.cross_entropy(logits, target, weight=class_weights)
    if et_tversky_weight:
        loss = loss + et_tversky_weight * et_tversky_loss(
            logits, target, alpha=et_tversky_alpha, beta=et_tversky_beta)
    if et_fp_weight:
        loss = loss + et_fp_weight * empty_et_false_positive_loss(
            logits, target, et_fp_topk_fraction
        )
    return loss


def utility_target(baseline_logits, generated_logits, target):
    """Desc: 配对干预效用，正值表示生成降低区域损失。"""
    with torch.no_grad():
        return regional_loss(baseline_logits, target, "none") - regional_loss(
            generated_logits, target, "none"
        )


def utility_loss(estimate, std, target):
    """Desc: 异方差效用回归损失。"""
    variance = std.square().clamp_min(1e-6)
    return (0.5 * (estimate - target).square() / variance + variance.log()).mean()


def safety_loss(final_logits, baseline_logits, target, margin=0.0):
    """Desc: 惩罚生成后区域损失高于无生成分支。"""
    final = regional_loss(final_logits, target, "none")
    baseline = regional_loss(baseline_logits, target, "none").detach()
    return functional.relu(final - baseline + margin).mean()
