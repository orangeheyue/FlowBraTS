"""End-to-end, resumable missing-modality visualisation pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import CHECKPOINTS, MODEL_LABELS
from render import render_manifest
from select_cases import make_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2020], choices=[2018, 2020])
    parser.add_argument("--models", nargs="+", default=list(MODEL_LABELS), choices=list(MODEL_LABELS))
    parser.add_argument("--gpu", type=int, default=1,
                        help="Physical GPU index; defaults to GPU1 as requested")
    parser.add_argument("--manifest", type=Path, default=here / "selection.json")
    parser.add_argument("--cache-root", type=Path, default=here / "cache")
    parser.add_argument("--output-root", type=Path, default=here / "outputs")
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    manifest = make_manifest(args.years, args.manifest, args.refresh_selection)
    print(f"Selection manifest: {args.manifest}")
    for year in args.years:
        selected = [(item["case_id"], item["slice"])
                    for item in manifest["datasets"][str(year)]["cases"]]
        print(f"BraTS{year}: {selected}")
    if args.select_only:
        return

    if not args.render_only:
        missing = [str(CHECKPOINTS[year][model]) for year in args.years for model in args.models
                   if not CHECKPOINTS[year][model].is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for year in args.years:
            for model in args.models:
                command = [
                    sys.executable, str(here / "infer_worker.py"),
                    "--model", model,
                    "--year", str(year),
                    "--manifest", str(args.manifest),
                    "--cache-root", str(args.cache_root),
                    "--device", "cuda:0",
                ]
                if args.amp:
                    command.append("--amp")
                if args.overwrite_predictions:
                    command.append("--overwrite")
                print(f"\n=== BraTS{year} / {MODEL_LABELS[model]} on physical GPU{args.gpu} ===",
                      flush=True)
                subprocess.run(command, check=True, env=environment, cwd=here)

    outputs = render_manifest(args.manifest, args.years, args.models, args.cache_root,
                              args.output_root, args.dpi)
    print(f"Completed: {len(outputs) // 2} figures (PNG + PDF) in {args.output_root}")


if __name__ == "__main__":
    main()
