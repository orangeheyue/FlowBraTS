"""Run one model/year in an isolated process and cache selected axial predictions."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import (CHECKPOINTS, FLOW_ROOT, SCENARIOS, atomic_json, load_case,
                    prediction_path, seed_everything, stable_seed, window_starts)


class ProbabilityAdapter(nn.Module):
    """Unify the four model APIs to P(class | patch, availability)."""

    def __init__(self, network: nn.Module, kind: str):
        super().__init__()
        self.network = network
        self.kind = kind

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.kind in {"rfnet", "m2ftrans"}:
            return self.network(image, mask)
        if self.kind == "diffunet":
            image = image * mask[:, :, None, None, None].to(image.dtype)
            shape = image.shape[2:]
            pads = [(-dim) % 32 for dim in shape]
            padded = F.pad(image, (0, pads[2], 0, pads[1], 0, pads[0]))
            logits = self.network(padded, ddim=True)
            return logits[:, :, :shape[0], :shape[1], :shape[2]].softmax(dim=1)
        if self.kind == "flowbrats":
            return self.network(image, mask, mode="infer")["logits"].softmax(dim=1)
        raise ValueError(self.kind)


def _clean_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def _insert_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_network(model_name: str, year: int, checkpoint: Path) -> tuple[nn.Module, dict]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if model_name == "rfnet":
        root = CHECKPOINTS[year]["rfnet"].parents[2] / "baselines/RFNet"
        _insert_path(root)
        model = importlib.import_module("models").Model(num_cls=4)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = _clean_state(payload["state_dict"])
        model.load_state_dict(state, strict=True)
        model.is_training = False
        return model, {"epoch": payload.get("epoch"), "loaded_keys": len(state)}

    if model_name == "m2ftrans":
        root = CHECKPOINTS[year]["m2ftrans"].parents[2] / "baselines/M2FTrans/M2FTrans_v1"
        _insert_path(root)
        module = importlib.import_module("models.fusiontrans")
        model = module.Model(num_cls=4)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = _clean_state(payload["state_dict"])
        model.load_state_dict(state, strict=True)
        model.is_training = False
        return model, {"epoch": payload.get("epoch"), "loaded_keys": len(state)}

    if model_name == "diffunet":
        root = CHECKPOINTS[year]["diffunet"].parents[2]
        _insert_path(root)
        DiffUNet = importlib.import_module("diffunet.diffunet_model").DiffUNet
        model = DiffUNet(4, 4)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = _clean_state(payload["state_dict"])
        model.load_state_dict(state, strict=True)
        return model, {
            "epoch": payload.get("epoch"),
            "best_validation": payload.get("best"),
            "selection": payload.get("selection"),
            "loaded_keys": len(state),
        }

    if model_name == "flowbrats":
        _insert_path(FLOW_ROOT)
        build_model = importlib.import_module("src.config").build_model
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("smoke_only"):
            raise RuntimeError(f"Refusing smoke-only FlowBraTS checkpoint: {checkpoint}")
        config = payload["config"]
        model = build_model(config)
        model.load_state_dict(payload["model"], strict=True)
        return model, {
            "epoch": payload.get("epoch"),
            "stage": payload.get("stage"),
            "checkpoint_type": payload.get("checkpoint_type"),
            "best_validation": payload.get("best_validation"),
            "solver": config["model"].get("solver"),
            "nfe": config["model"].get("nfe"),
            "version": config["model"].get("version"),
            "velocity_type": config["model"].get("velocity_type", "residual"),
            "loaded_keys": len(payload["model"]),
        }
    raise ValueError(model_name)


def infer_axial_plane(
    adapter: nn.Module,
    volume: np.ndarray,
    z_target: int,
    availability: tuple[bool, bool, bool, bool],
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    """Compute exactly the selected plane of the reference sliding-window average."""
    height, width, depth, channels = volume.shape
    if channels != 4 or not 0 <= z_target < depth:
        raise ValueError((volume.shape, z_target))
    h_starts = window_starts(height)
    w_starts = window_starts(width)
    z_starts = [z for z in window_starts(depth) if z <= z_target < z + 80]
    probabilities = torch.zeros((4, height, width), dtype=torch.float32, device=device)
    weights = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    mask = torch.tensor([availability], dtype=torch.bool, device=device)
    autocast_enabled = amp and device.type == "cuda"
    with torch.inference_mode():
        for h in h_starts:
            for w in w_starts:
                for z in z_starts:
                    patch = np.array(volume[h:h + 80, w:w + 80, z:z + 80],
                                     dtype=np.float32, copy=True)
                    image = torch.from_numpy(patch.transpose(3, 0, 1, 2))[None].to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=autocast_enabled):
                        output = adapter(image, mask)
                    if output.shape != (1, 4, 80, 80, 80):
                        raise RuntimeError(f"Unexpected model output: {tuple(output.shape)}")
                    if not torch.isfinite(output).all():
                        raise FloatingPointError("Non-finite model output")
                    local_z = z_target - z
                    probabilities[:, h:h + 80, w:w + 80] += output[0, :, :, :, local_z].float()
                    weights[:, h:h + 80, w:w + 80] += 1
    if torch.any(weights == 0):
        raise RuntimeError("Sliding-window plane has uncovered pixels")
    return (probabilities / weights).argmax(dim=0).to(torch.uint8).cpu().numpy()


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["rfnet", "m2ftrans", "diffunet", "flowbrats"])
    parser.add_argument("--year", required=True, type=int, choices=[2018, 2020])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true",
                        help="Optional faster inference; disabled by default for protocol fidelity")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["datasets"][str(args.year)]["cases"]
    checkpoint = CHECKPOINTS[args.year][args.model]
    device = torch.device(args.device)

    seed_everything(1037)
    started = time.time()
    network, checkpoint_info = load_network(args.model, args.year, checkpoint)
    adapter = ProbabilityAdapter(network, args.model).to(device).eval()
    print(f"Loaded {args.model} BraTS{args.year}: {checkpoint}", flush=True)

    completed = 0
    for case in cases:
        case_id, z_target = case["case_id"], int(case["slice"])
        volume, _ = load_case(args.year, case_id, mmap=True)
        expected_shape = tuple(volume.shape[:2])
        for scenario, label, availability in SCENARIOS:
            destination = prediction_path(args.cache_root, args.year, args.model, case_id, scenario)
            if destination.is_file() and not args.overwrite:
                cached = np.load(destination, mmap_mode="r")
                if cached.shape == expected_shape and cached.dtype == np.uint8 and cached.max() <= 3:
                    print(f"reuse {case_id} z={z_target} {label}", flush=True)
                    completed += 1
                    continue
                destination.rename(destination.with_suffix(destination.suffix + ".invalid"))
            seed = stable_seed(args.year, case_id, scenario, args.model)
            seed_everything(seed)
            before = time.time()
            prediction = infer_axial_plane(adapter, volume, z_target, availability, device, args.amp)
            _save_array(destination, prediction)
            completed += 1
            print(f"done {completed}/{len(cases) * len(SCENARIOS)} {case_id} z={z_target} "
                  f"{label} ({time.time() - before:.1f}s)", flush=True)

    metadata = {
        "model": args.model,
        "year": args.year,
        "checkpoint": str(checkpoint),
        "checkpoint_info": checkpoint_info,
        "protocol": "80^3 windows, stride 40; exact axial plane of probability-averaged volume",
        "amp": args.amp,
        "cases": [item["case_id"] for item in cases],
        "scenarios": [item[0] for item in SCENARIOS],
        "elapsed_seconds": time.time() - started,
        "complete": True,
    }
    atomic_json(args.cache_root / f"BraTS{args.year}" / args.model / "metadata.json", metadata)


if __name__ == "__main__":
    main()
