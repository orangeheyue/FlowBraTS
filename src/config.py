import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.m2f_adapter import M2FTransAdapter
from src.models.network import FlowBraTS
from src.models.network_v21 import FlowBraTSV21
from src.models.m2f_adapter_v21 import M2FTransV21Adapter


ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/default.yaml"):
    """Desc: 读取单一 YAML 配置源。"""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    config=yaml.safe_load(path.read_text(encoding="utf-8"))
    parent=config.pop('_base_',None)
    if parent:
        base=load_config(parent)
        def merge(left,right):
            for key,value in right.items():
                if isinstance(value,dict) and isinstance(left.get(key),dict):merge(left[key],value)
                else:left[key]=value
        merge(base,config)
        return base
    return config


def seed_everything(seed):
    """Desc: 固定 Python、NumPy、PyTorch 随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(config):
    """Desc: 构建严格加载预训练权重的完整 FlowBraTS。"""
    version=config["model"].get("version","v1")
    if version not in ("v1","v21"):raise ValueError(f"Unknown model version: {version}")
    adapter=M2FTransV21Adapter if version=="v21" else M2FTransAdapter
    backbone = adapter(
        config["m2ftrans"]["root"], config["m2ftrans"]["checkpoint"]
    )
    model = config["model"]
    extra=({"refinement_hidden":model.get("refinement_hidden",32),
            "task_scale":model.get("task_scale",.25),
            "et_head":model.get("et_head",False),
            "velocity_type":model.get("velocity_type","residual"),
            "disable_dmg":model.get("disable_dmg",False),
            "disable_rfm":model.get("disable_rfm",False),
            "disable_cfg":model.get("disable_cfg",False),
            "task_class_weights":model.get("task_class_weights",[1.,1.,1.,1.])}
           if version=="v21" else {})
    network=FlowBraTSV21 if version=="v21" else FlowBraTS
    return network(
        backbone,
        channels=model["latent_channels"],
        flow_channels=model["flow_channels"],
        hidden=model["hidden_channels"],
        nfe=model["nfe"],
        solver=model["solver"],
        gate_temperature=model["gate_temperature"],
        utility_threshold=model["utility_threshold"],
        uncertainty_penalty=model["uncertainty_penalty"],
        **extra,
    )
