"""Render cached predictions in the supplied layout using Pillow only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import DISPLAY_CHANNELS, MODEL_LABELS, SCENARIOS, load_case, prediction_path


LABEL_COLOURS = {
    1: np.asarray((0.92, 0.23, 0.21), dtype=np.float32),  # NCR / NET
    2: np.asarray((0.05, 0.78, 0.28), dtype=np.float32),  # ED
    3: np.asarray((1.00, 0.86, 0.10), dtype=np.float32),  # ET
}
BACKGROUND = (251, 249, 251)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def display_crop(volume: np.ndarray, z_target: int, padding: int = 4) -> tuple[slice, slice]:
    brain = np.any(np.abs(np.asarray(volume[:, :, z_target, :])) > 1e-8, axis=-1)
    coordinates = np.argwhere(brain)
    if not coordinates.size:
        return slice(0, volume.shape[0]), slice(0, volume.shape[1])
    low = np.maximum(coordinates.min(axis=0) - padding, 0)
    high = np.minimum(coordinates.max(axis=0) + padding + 1, volume.shape[:2])
    return slice(int(low[0]), int(high[0])), slice(int(low[1]), int(high[1]))


def normalise_mri(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    foreground = image[np.abs(image) > 1e-8]
    if foreground.size < 2:
        return np.zeros_like(image)
    low, high = np.percentile(foreground, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def orient(image: np.ndarray) -> np.ndarray:
    return np.rot90(image)


def segmentation_overlay(background: np.ndarray, labels: np.ndarray, alpha: float = 0.72) -> np.ndarray:
    gray = normalise_mri(background)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for label, colour in LABEL_COLOURS.items():
        region = labels == label
        rgb[region] = (1 - alpha) * rgb[region] + alpha * colour
    return np.asarray(np.clip(rgb * 255, 0, 255), dtype=np.uint8)


def _fit_panel(array: np.ndarray, size: tuple[int, int], grayscale: bool = False) -> Image.Image:
    if grayscale:
        pixels = np.asarray(np.clip(normalise_mri(array) * 255, 0, 255), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="L").convert("RGB")
    else:
        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "black")
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def _centred_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
                  font: ImageFont.ImageFont, fill=(20, 20, 20), spacing: int = 4) -> None:
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                      colour=(58, 98, 151), width: int = 3, dash: int = 12) -> None:
    left, top, right, bottom = box
    for start in range(left, right, dash * 2):
        draw.line((start, top, min(start + dash, right), top), fill=colour, width=width)
        draw.line((start, bottom, min(start + dash, right), bottom), fill=colour, width=width)
    for start in range(top, bottom, dash * 2):
        draw.line((left, start, left, min(start + dash, bottom)), fill=colour, width=width)
        draw.line((right, start, right, min(start + dash, bottom)), fill=colour, width=width)


def render_case(
    year: int,
    case: dict,
    models: list[str],
    cache_root: Path,
    output_root: Path,
    dpi: int = 240,
) -> tuple[Path, Path]:
    case_id, z_target = case["case_id"], int(case["slice"])
    volume, segmentation = load_case(year, case_id, mmap=True)
    row_crop, col_crop = display_crop(volume, z_target)
    flair = orient(np.asarray(volume[row_crop, col_crop, z_target, 0]))
    gt = orient(np.asarray(segmentation[row_crop, col_crop, z_target], dtype=np.uint8))

    columns = [name for name, _ in DISPLAY_CHANNELS] + [MODEL_LABELS[name] for name in models] + ["GT"]
    panel_w, panel_h, gap = 290, 270, 6
    margin_x, title_h, header_h, footer_h = 18, 72, 42, 112
    grid_w = len(columns) * panel_w + (len(columns) - 1) * gap
    grid_h = len(SCENARIOS) * panel_h + (len(SCENARIOS) - 1) * gap
    canvas = Image.new("RGB", (grid_w + 2 * margin_x, title_h + header_h + grid_h + footer_h), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font, header_font, missing_font = _font(27), _font(20, True), _font(20, True)
    note_font, legend_font = _font(15), _font(18)

    _centred_text(draw, (0, 0, canvas.width, title_h),
                  f"BraTS{year}  |  {case_id}  |  axial array slice {z_target}", title_font)
    for column, label in enumerate(columns):
        x = margin_x + column * (panel_w + gap)
        _centred_text(draw, (x, title_h, x + panel_w, title_h + header_h), label, header_font)

    grid_top = title_h + header_h
    for row, (scenario, _scenario_label, availability) in enumerate(SCENARIOS):
        y = grid_top + row * (panel_h + gap)
        for column, (modality, channel) in enumerate(DISPLAY_CHANNELS):
            x = margin_x + column * (panel_w + gap)
            if availability[channel]:
                image = orient(np.asarray(volume[row_crop, col_crop, z_target, channel]))
                panel = _fit_panel(image, (panel_w, panel_h), grayscale=True)
                canvas.paste(panel, (x, y))
            else:
                draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), fill=BACKGROUND)
                _dashed_rectangle(draw, (x + 1, y + 1, x + panel_w - 2, y + panel_h - 2))
                _centred_text(draw, (x, y, x + panel_w, y + panel_h),
                              f"{modality}\nmissing", missing_font)

        for offset, model in enumerate(models, start=len(DISPLAY_CHANNELS)):
            path = prediction_path(cache_root, year, model, case_id, scenario)
            if not path.is_file():
                raise FileNotFoundError(f"Missing cached prediction: {path}")
            prediction = np.load(path)
            if prediction.shape != tuple(volume.shape[:2]):
                raise ValueError(f"Invalid prediction shape {path}: {prediction.shape}")
            prediction = orient(prediction[row_crop, col_crop])
            panel = _fit_panel(segmentation_overlay(flair, prediction), (panel_w, panel_h))
            x = margin_x + offset * (panel_w + gap)
            canvas.paste(panel, (x, y))

        panel = _fit_panel(segmentation_overlay(flair, gt), (panel_w, panel_h))
        x = margin_x + (len(columns) - 1) * (panel_w + gap)
        canvas.paste(panel, (x, y))

    legend_y = grid_top + grid_h + 22
    legend_items = ((2, "ED"), (3, "ET"), (1, "NCR / non-enhancing core"))
    item_widths = (110, 110, 330)
    cursor = (canvas.width - sum(item_widths)) // 2
    for (label, text), item_width in zip(legend_items, item_widths):
        colour = tuple(np.asarray(LABEL_COLOURS[label] * 255, dtype=np.uint8))
        draw.rectangle((cursor, legend_y, cursor + 32, legend_y + 22), fill=colour)
        draw.text((cursor + 43, legend_y - 1), text, font=legend_font, fill=(40, 40, 40))
        cursor += item_width
    note = "Raw predictions; shared FLAIR background is for display only, including rows where FLAIR is missing."
    _centred_text(draw, (0, legend_y + 37, canvas.width, canvas.height - 4), note,
                  note_font, fill=(85, 85, 85))

    folder = output_root / f"BraTS{year}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id}_slice_{z_target}"
    png_path, pdf_path = folder / f"{stem}.png", folder / f"{stem}.pdf"
    canvas.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    canvas.save(pdf_path, format="PDF", resolution=float(dpi))
    return png_path, pdf_path


def render_manifest(
    manifest_path: Path,
    years: list[int],
    models: list[str],
    cache_root: Path,
    output_root: Path,
    dpi: int,
) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = []
    for year in years:
        for case in manifest["datasets"][str(year)]["cases"]:
            png, pdf = render_case(year, case, models, cache_root, output_root, dpi)
            outputs.extend((png, pdf))
            print(f"rendered {png}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=here / "selection.json")
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2020], choices=[2018, 2020])
    parser.add_argument("--models", nargs="+", default=list(MODEL_LABELS), choices=list(MODEL_LABELS))
    parser.add_argument("--cache-root", type=Path, default=here / "cache")
    parser.add_argument("--output-root", type=Path, default=here / "outputs")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    render_manifest(args.manifest, args.years, args.models, args.cache_root,
                    args.output_root, args.dpi)


if __name__ == "__main__":
    main()
