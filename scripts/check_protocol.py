import argparse
import json
from pathlib import Path

import numpy as np

from src.config import load_config


EXPECTED_COUNTS = {"train": 219, "val": 50, "test": 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--full-scan", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = Path(config["data"]["root"])
    split_dir = Path(config["data"]["split_dir"])

    splits = {}
    for split, expected in EXPECTED_COUNTS.items():
        names = [line.strip() for line in (split_dir / f"{split}.txt").read_text().splitlines()
                 if line.strip()]
        if len(names) != expected or len(names) != len(set(names)):
            raise RuntimeError(f"invalid {split} split: expected {expected}, got {len(names)}")
        splits[split] = names
    if set(splits["train"]) & set(splits["val"]):
        raise RuntimeError("train and val overlap")
    if set(splits["train"]) & set(splits["test"]):
        raise RuntimeError("train and test overlap")
    if set(splits["val"]) & set(splits["test"]):
        raise RuntimeError("val and test overlap")

    names = [name for split in ("train", "val", "test") for name in splits[split]]
    checked = names if args.full_scan else names[:1]
    shapes = []
    for name in checked:
        image_path = data_root / "vol" / f"{name}_vol.npy"
        label_path = data_root / "seg" / f"{name}_seg.npy"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing image/label pair for {name}")
        image = np.load(image_path, mmap_mode="r")
        label = np.load(label_path, mmap_mode="r")
        if image.ndim != 4 or image.shape[-1] != 4 or image.shape[:3] != label.shape:
            raise RuntimeError(f"invalid array shape for {name}: {image.shape}, {label.shape}")
        values = set(np.unique(label).tolist())
        if not values <= {0, 1, 2, 3}:
            raise RuntimeError(f"invalid internal labels for {name}: {sorted(values)}")
        if not np.isfinite(image).all():
            raise RuntimeError(f"non-finite MRI value in {name}")
        shapes.append(image.shape)

    report = {
        "passed": True,
        "counts": {key: len(value) for key, value in splits.items()},
        "modal_order": config["data"]["modal_order"],
        "checked_cases": len(checked),
        "minimum_shape": np.min(np.asarray(shapes), axis=0).tolist(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
