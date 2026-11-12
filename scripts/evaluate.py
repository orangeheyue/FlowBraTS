import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.config import build_model, load_config, seed_everything
from src.data.m2f_data import build_dataset
from src.data.protocol import MASKS, mask_name


class M2FEvalWrapper(nn.Module):
    """将 FlowBraTS 输出适配为 M2FTrans 原评估器要求的概率张量。"""

    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode
        self.is_training = False

    def forward(self, image, mask):
        output = self.model(image, mask, mode=self.mode)
        return output["logits"].softmax(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--mode", choices=("baseline", "ungated", "infer"), default="infer")
    parser.add_argument("--output", default="outputs/evaluation.csv")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode != "baseline" and not args.checkpoint:
        raise ValueError("generated evaluation requires a trained checkpoint")
    if args.split == "test" and not args.confirm_test:
        raise ValueError("test split requires --confirm-test after validation-based model selection")
    seed_everything(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("the reused M2FTrans evaluator requires CUDA")
    source_root = str(Path(config["m2ftrans"]["root"]).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from predict import test_dice_hd95_softmax  # pylint: disable=import-outside-toplevel

    model = build_model(config).cuda().eval()
    if args.checkpoint:
        payload = load_checkpoint(model, args.checkpoint)
        if payload.get("smoke_only"):
            raise RuntimeError("smoke checkpoint cannot be used for formal evaluation")
    wrapper = nn.DataParallel(M2FEvalWrapper(model, args.mode)).cuda().eval()
    dataset = build_dataset(
        source_root, config["data"]["root"], config["data"]["split_dir"], args.split
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    detail_dir = output.with_name(f"{output.stem}_per_mask")
    detail_dir.mkdir(parents=True, exist_ok=True)
    for bits in MASKS:
        name = mask_name(bits)
        detail_output = detail_dir / f"{name.replace('+', '_')}.csv"
        dice, hd95 = test_dice_hd95_softmax(
            # 原 predict.py 通过这个历史字符串选择 BraTS 四类评估分支。
            loader, wrapper, dataname="/home/sjj/MMMSeg/BraTS/BRATS2020",
            feature_mask=list(bits), mask_name=name, csv_name=str(detail_output),
        )
        rows.append((name, *np.asarray(dice).tolist(), *np.asarray(hd95).tolist()))
    average = np.asarray([row[1:] for row in rows], dtype=float).mean(axis=0)
    rows.append(("AVERAGE", *average.tolist()))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mask", "WT_Dice", "TC_Dice", "ET_Dice", "ETPro_Dice",
                         "WT_HD95", "TC_HD95", "ET_HD95", "ETPro_HD95"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
