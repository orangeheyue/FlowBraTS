"""Shared protocol and data helpers for missing-modality visualisation."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOW_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "preproced_dataset"

# Preprocessed BraTS channel order used by RFNet/M2FTrans and all local models.
CHANNELS = ("FLAIR", "T1ce", "T1", "T2")
DISPLAY_CHANNELS = (("T1", 2), ("T2", 3), ("FLAIR", 0), ("T1ce", 1))

# Matches the five rows in the supplied reference figure. True means available.
SCENARIOS = (
    ("all_present", "All modalities", (True, True, True, True)),
    ("t1_missing", "T1 missing", (True, True, False, True)),
    ("t1_t2_missing", "T1 + T2 missing", (True, True, False, False)),
    ("t1_t2_flair_missing", "T1 + T2 + FLAIR missing", (False, True, False, False)),
    ("t1_flair_t1ce_missing", "T1 + FLAIR + T1ce missing", (False, False, False, True)),
)

MODEL_LABELS = {
    "rfnet": "RFNet",
    "m2ftrans": "M²FTrans",
    "diffunet": "DiffUNet",
    "flowbrats": "FlowBraTS",
}

CHECKPOINTS = {
    2018: {
        "rfnet": PROJECT_ROOT / "pretrained_weights/RFNet/model_last_brats2018_split1.pth",
        "m2ftrans": PROJECT_ROOT / "pretrained_weights/M2FTrans/m2ftransv1_brats2018_1_80.pth",
        "diffunet": PROJECT_ROOT / "baselines/DiffUNet/runs/brats2018_split1/best.pt",
        "flowbrats": FLOW_ROOT / "outputs/finetune_brats2018/finetune/best.pt",
    },
    2020: {
        "rfnet": PROJECT_ROOT / "pretrained_weights/RFNet/model_last_brast2020.pth",
        "m2ftrans": PROJECT_ROOT / "pretrained_weights/M2FTrans/m2ftransv1_brats2020_80.pth",
        "diffunet": PROJECT_ROOT / "baselines/DiffUNet/runs/brats2020/best.pt",
        "flowbrats": FLOW_ROOT / "outputs/finetune_brats2020/finetune/best.pt",
    },
}

FLOW_CONFIGS = {
    2018: FLOW_ROOT / "configs/finetune_brats2018.yaml",
    2020: FLOW_ROOT / "configs/finetune_brats2020.yaml",
}


def data_dir(year: int) -> Path:
    return DATA_ROOT / f"BraTS{year}"


def split_file(year: int) -> Path:
    suffix = "1" if year == 2018 else ""
    return (DATA_ROOT / "Common_Dataset_Split/BraTS" /
            f"BRATS{year}_Training_none_npy/test{suffix}.txt")


def read_test_ids(year: int) -> list[str]:
    path = split_file(year)
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = sorted(path.read_text(encoding="utf-8").split())
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case IDs in {path}")
    return ids


def case_paths(year: int, case_id: str) -> tuple[Path, Path]:
    root = data_dir(year)
    return root / "vol" / f"{case_id}_vol.npy", root / "seg" / f"{case_id}_seg.npy"


def load_case(year: int, case_id: str, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    volume_path, seg_path = case_paths(year, case_id)
    for path in (volume_path, seg_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    mode = "r" if mmap else None
    volume = np.load(volume_path, mmap_mode=mode)
    seg = np.load(seg_path, mmap_mode=mode)
    if volume.shape[:3] != seg.shape or volume.shape[-1] != 4:
        raise ValueError(f"Unexpected shapes for {case_id}: volume={volume.shape}, seg={seg.shape}")
    return volume, seg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_seed(year: int, case_id: str, scenario: str, model: str, base: int = 1037) -> int:
    digest = hashlib.sha256(f"{year}|{case_id}|{scenario}|{model}|{base}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def window_starts(length: int, patch: int = 80, stride: int = 40) -> list[int]:
    """Reproduce the original M2FTrans 80^3/50%-overlap window positions."""
    if length < patch:
        raise ValueError(f"Dimension {length} is smaller than patch {patch}")
    count = int(np.ceil((length - patch) / stride))
    # Do not deduplicate: this intentionally matches the reference evaluator.
    return [index * stride for index in range(count)] + [length - patch]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prediction_path(cache_root: Path, year: int, model: str, case_id: str, scenario: str) -> Path:
    return cache_root / f"BraTS{year}" / model / case_id / f"{scenario}.npy"
