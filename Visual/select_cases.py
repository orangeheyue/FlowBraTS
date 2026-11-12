"""Select five representative test cases using GT only (never model predictions)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import SCENARIOS, atomic_json, case_paths, load_case, read_test_ids, split_file


def _slice_statistics(seg: np.ndarray) -> tuple[int, dict[str, int]]:
    # Labels: 1=NCR/NET, 2=ED, 3=ET. Weight small regions so all colours remain visible.
    wt = (seg > 0).sum(axis=(0, 1))
    tc = np.logical_or(seg == 1, seg == 3).sum(axis=(0, 1))
    et = (seg == 3).sum(axis=(0, 1))
    valid = np.flatnonzero((wt > 0) & (tc >= 10) & (et >= 10))
    score = wt + tc + 2 * et
    z = int(valid[np.argmax(score[valid])]) if valid.size else int(np.argmax(score))
    return z, {"WT": int(wt[z]), "TC": int(tc[z]), "ET": int(et[z])}


def select_cases(year: int, count: int = 5) -> list[dict]:
    candidates = []
    for case_id in read_test_ids(year):
        _, seg = load_case(year, case_id, mmap=True)
        z, areas = _slice_statistics(seg)
        candidates.append({"case_id": case_id, "slice": z, "areas": areas})

    # Stratify by the maximum-slice WT area and use quintile centres. Cases with all
    # three tumour labels visible are preferred; fallback retains deterministic order.
    eligible = [item for item in candidates if min(item["areas"].values()) >= 10]
    if len(eligible) < count:
        eligible = candidates
    eligible.sort(key=lambda item: (item["areas"]["WT"], item["case_id"]))
    ranks = np.linspace(0.1, 0.9, count)
    indices = [int(round(q * (len(eligible) - 1))) for q in ranks]
    selected = []
    used = set()
    for index in indices:
        order = sorted(range(len(eligible)), key=lambda other: (abs(other - index), other))
        choice = next(other for other in order if other not in used)
        used.add(choice)
        selected.append(eligible[choice])
    return selected


def make_manifest(years: list[int], output: Path, refresh: bool = False) -> dict:
    if output.exists() and not refresh:
        import json
        manifest = json.loads(output.read_text(encoding="utf-8"))
        if all(str(year) in manifest.get("datasets", {}) for year in years):
            return manifest
    manifest = {
        "selection_policy": (
            "GT-only deterministic quintile sampling by maximum-slice WT area; "
            "slice maximises WT + TC + 2*ET and requires TC/ET visibility when possible"
        ),
        "scenarios": [
            {"key": key, "label": label, "availability_flair_t1ce_t1_t2": list(mask)}
            for key, label, mask in SCENARIOS
        ],
        "datasets": {},
    }
    for year in years:
        cases = select_cases(year)
        manifest["datasets"][str(year)] = {
            "split_file": str(split_file(year)),
            "cases": cases,
        }
        for item in cases:
            volume, seg = case_paths(year, item["case_id"])
            item["volume"] = str(volume)
            item["segmentation"] = str(seg)
    atomic_json(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2020], choices=[2018, 2020])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("selection.json"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    manifest = make_manifest(args.years, args.output, args.refresh)
    for year in args.years:
        print(f"BraTS{year}")
        for item in manifest["datasets"][str(year)]["cases"]:
            print(f"  {item['case_id']} slice={item['slice']} areas={item['areas']}")


if __name__ == "__main__":
    main()
