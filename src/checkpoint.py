from pathlib import Path

import torch


INNOVATION_PREFIXES = ("graph.", "flow.", "utility.", "refinement.", "et_head.")


def innovation_state(model):
    """Desc: 只保存新增模块，避免重复写入 695MB 的冻结基线。"""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(INNOVATION_PREFIXES)
    }


def full_state(model):
    """Desc: 保存端到端训练所需的完整模型参数。"""
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def load_innovations(model, checkpoint):
    """Desc: 严格加载新增模块，任何键缺失或形状不符均报错。"""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = payload["model"]
    expected = set(innovation_state(model))
    actual = set(state)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(f"innovation checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    current = model.state_dict()
    wrong_shape = [key for key in state if current[key].shape != state[key].shape]
    if wrong_shape:
        raise RuntimeError(f"innovation checkpoint shape mismatch: {wrong_shape}")
    current.update(state)
    model.load_state_dict(current, strict=True)
    return payload


def load_checkpoint(model, checkpoint):
    """Desc: 自动加载旧版增量 checkpoint 或端到端完整 checkpoint。"""
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    checkpoint_type = payload.get("checkpoint_type")
    if checkpoint_type is None:
        checkpoint_type = (
            "innovations"
            if state and all(key.startswith(INNOVATION_PREFIXES) for key in state)
            else "full"
        )
    if checkpoint_type == "innovations":
        return load_innovations(model, path)
    if checkpoint_type != "full":
        raise RuntimeError(f"unknown checkpoint type: {checkpoint_type}")
    model.load_state_dict(state, strict=True)
    return payload



def initialize_from_v1(model, checkpoint):
    """Explicit weight transfer only; new v21 modules keep their initialization."""
    payload=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
    if payload.get("smoke_only"):raise ValueError("Smoke checkpoints cannot initialize formal v21 runs")
    source=payload["model"];current=model.state_dict()
    unexpected=set(source)-set(current)
    missing=set(current)-set(source)
    allowed_missing={key for key in current if key.startswith(("refinement.","et_head.")) or key=="flow.task_basis"}
    # v1 innovations-only checkpoints may additionally omit the pretrained backbone.
    if payload.get("checkpoint_type")=="innovations":
        allowed_missing.update(key for key in current if key.startswith("backbone."))
    if unexpected or missing-allowed_missing:
        raise RuntimeError(f"Invalid v1 transfer: unexpected={sorted(unexpected)}, missing={sorted(missing-allowed_missing)}")
    wrong=[key for key,value in source.items() if current[key].shape!=value.shape]
    if wrong:raise RuntimeError(f"Shape mismatch in transferred weights: {wrong}")
    current.update(source);model.load_state_dict(current,strict=True)
    return {"source":str(checkpoint),"loaded_keys":len(source),"new_keys":sorted(missing)}
