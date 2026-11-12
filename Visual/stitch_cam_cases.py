#!/usr/bin/env python3
"""Stitch the fixed BraTS2018-151 and BraTS2020-097 CAM panels."""

from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "cam_outputs" / "figures"
OUTPUT = HERE / "cam_outputs" / "paired_cases"

CASES = {
    "BraTS2018_151": (
        FIGURES / "BraTS2018",
        "Brats18_TCIA02_151_1_slice_87_{target}_CAM.png",
    ),
    "BraTS2020_097": (
        FIGURES / "BraTS2020",
        "HG_BraTS20_Training_097_slice_67_{target}_CAM.png",
    ),
}

TARGETS = {
    "ET": "ET",
    "NCR": "NCR_NET",
    "ED": "ED",
}


def stitch(target_label: str, source_target: str) -> tuple[Path, Path]:
    sources = [directory / pattern.format(target=source_target) for directory, pattern in CASES.values()]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CAM source(s): " + ", ".join(map(str, missing)))

    images = [Image.open(path).convert("RGB") for path in sources]
    if len({image.size for image in images}) != 1:
        raise ValueError(f"CAM panels must have identical sizes: {[image.size for image in images]}")

    width, height = images[0].size
    gap = 20
    canvas = Image.new("RGB", (2 * width + gap, height), "white")
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (width + gap, 0))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"CAM_{target_label}_BraTS2018_151_BraTS2020_097"
    png_path = OUTPUT / f"{stem}.png"
    pdf_path = OUTPUT / f"{stem}.pdf"
    canvas.save(png_path, dpi=(240, 240), optimize=True)
    canvas.save(pdf_path, "PDF", resolution=240.0)

    for image in images:
        image.close()
    canvas.close()
    return png_path, pdf_path


def main() -> None:
    for label, source_target in TARGETS.items():
        png_path, pdf_path = stitch(label, source_target)
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
