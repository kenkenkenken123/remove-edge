"""Detect tissue content and crop dark borders from TIFF images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile


@dataclass(frozen=True)
class CropResult:
    """Bounding box and metadata from edge removal."""

    y0: int
    y1: int
    x0: int
    x1: int
    original_shape: tuple[int, int]
    threshold: float
    downsample: int
    halo_threshold: float | None = None
    tight_y0: int | None = None
    tight_y1: int | None = None
    tight_x0: int | None = None
    tight_x1: int | None = None

    @property
    def cropped_shape(self) -> tuple[int, int]:
        if self.tight_y0 is not None and self.tight_y1 is not None:
            return self.tight_y1 - self.tight_y0, self.tight_x1 - self.tight_x0
        return self.y1 - self.y0, self.x1 - self.x0


@dataclass(frozen=True)
class OutputPaths:
    """Default paths for all generated files."""

    mask_tif: Path
    mask_png: Path
    final_tif: Path
    final_png: Path
    preview_jpg: Path


@dataclass(frozen=True)
class ProcessResult:
    """Crop metadata and paths to saved output files."""

    bounds: CropResult
    paths: OutputPaths


def _load_raw(path: Path) -> np.ndarray:
    """Load a TIFF without bit-depth conversion."""
    with tifffile.TiffFile(path) as tif:
        if tif.pages[0].is_memmappable:
            return np.array(tifffile.memmap(path))
        return tif.asarray()


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert raw TIFF data to 8-bit grayscale (0-255)."""
    if image.dtype == np.uint8:
        return image.copy()

    data = image.astype(np.float32)
    if data.max() <= 0:
        return np.zeros(image.shape, dtype=np.uint8)

    # Slides with background haze rarely have true zero; use a black-point offset.
    lo = 0.0 if (data == 0).mean() > 0.05 else float(np.percentile(data, 10))
    hi = float(np.percentile(data, 99.5))
    if hi <= lo:
        hi = float(data.max())
    if hi <= lo:
        hi = lo + 1.0

    return np.clip((data - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    """Load a TIFF and convert it to 8-bit for processing."""
    return convert_to_uint8(_load_raw(Path(path)))


def _to_float32(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float32:
        return image
    return image.astype(np.float32)


def _auto_threshold(small: np.ndarray) -> float:
    """Estimate a tissue/background split from downsampled intensities."""
    flat = small.ravel()
    background = float(np.percentile(flat, 20))
    tissue = float(np.percentile(flat, 80))
    return background + (tissue - background) * 0.35


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, x1, y1) bounding box of all foreground pixels."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None

    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))
    return x0, y0, x1, y1


def _content_mask(small: np.ndarray, thresh: float) -> np.ndarray:
    mask = (small > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def detect_content_bounds(
    image: np.ndarray,
    *,
    threshold: float | None = None,
    downsample: int = 8,
    margin: int = 48,
) -> CropResult:
    """
    Find the bounding box of tissue content, excluding dark borders and edge glow.

    Uses all detected tissue pixels on a downsampled copy (including separate
    fragments), then maps coordinates back to full resolution.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")

    height, width = image.shape
    downsample = max(1, downsample)
    small_h = max(1, height // downsample)
    small_w = max(1, width // downsample)

    small = cv2.resize(
        _to_float32(image),
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )

    thresh = _auto_threshold(small) if threshold is None else threshold

    mask = _content_mask(small, thresh)
    bbox = _mask_bbox(mask)

    if bbox is None:
        return CropResult(0, height, 0, width, (height, width), thresh, downsample)

    x0_small, y0_small, x1_small, y1_small = bbox
    y0 = max(0, y0_small * downsample - margin)
    y1 = min(height, y1_small * downsample + margin)
    x0 = max(0, x0_small * downsample - margin)
    x1 = min(width, x1_small * downsample + margin)

    return CropResult(y0, y1, x0, x1, (height, width), thresh, downsample)


def _auto_halo_threshold(crop: np.ndarray, detection_threshold: float) -> float:
    """
    Estimate the intensity cutoff that separates faint edge glow from tissue.

    Kept conservative so dim tissue at the edges is not removed.
    """
    sample = crop[::4, ::4].astype(np.float32)
    content = sample[sample > detection_threshold * 0.75]
    if content.size < 100:
        return detection_threshold * 1.05

    low = float(np.percentile(content, 10))
    mid = float(np.percentile(content, 40))
    cutoff = low + (mid - low) * 0.15
    return max(detection_threshold * 1.02, cutoff)


def _tight_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return y0, y1, x0, x1 bounding box of nonzero pixels."""
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)
    if not rows.any() or not cols.any():
        height, width = image.shape
        return 0, height, 0, width

    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))
    return y0, y1, x0, x1


def _clean_faint_border(
    crop: np.ndarray,
    detection_threshold: float,
    halo_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Zero out faint border pixels inside the cropped region."""
    cutoff = (
        halo_threshold
        if halo_threshold is not None
        else _auto_halo_threshold(crop, detection_threshold)
    )
    cleaned = crop.copy()
    cleaned[cleaned < cutoff] = 0
    return cleaned, cutoff


def _save_image(path: Path, image: np.ndarray) -> None:
    """Save an 8-bit grayscale image as PNG or TIFF."""
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image for output, got {image.dtype}")

    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".tif", ".tiff"}:
        tifffile.imwrite(path, image, compression=None)
        return
    if suffix in {".png", ".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image)
        return

    raise ValueError(f"Unsupported output format: {path.suffix}")


def _default_output_paths(input_path: Path, output_dir: Path | None = None) -> OutputPaths:
    """Build default paths for mask, final TIFF, final PNG, and preview."""
    out_dir = output_dir or Path("output")
    stem = input_path.stem
    return OutputPaths(
        mask_tif=out_dir / f"{stem}_masked.tif",
        mask_png=out_dir / f"{stem}_masked.png",
        final_tif=out_dir / f"{stem}_masked_final.tif",
        final_png=out_dir / f"{stem}_masked_final.png",
        preview_jpg=out_dir / f"{stem}_preview.jpg",
    )


def _build_mask(final_image: np.ndarray) -> np.ndarray:
    """Binary tissue mask aligned with the final output (255 = tissue, 0 = background)."""
    return ((final_image > 0).astype(np.uint8) * 255)


def remove_edge(
    input_path: str | Path,
    *,
    output_dir: Path | None = None,
    mask_path: Path | None = None,
    mask_png_path: Path | None = None,
    final_tif_path: Path | None = None,
    final_png_path: Path | None = None,
    threshold: float | None = None,
    downsample: int = 8,
    margin: int = 48,
    clean_border: bool = True,
    halo_threshold: float | None = None,
    tight_crop: bool = True,
) -> tuple[ProcessResult, np.ndarray]:
    """
    Crop dark borders from a TIFF and remove faint edge glow.

    Saves four outputs plus returns the final image array:
    - masked TIFF / PNG: binary tissue mask (255 = tissue)
    - masked final TIFF / PNG: cropped tissue on black background
    """
    input_path = Path(input_path)
    defaults = _default_output_paths(input_path, output_dir)
    paths = OutputPaths(
        mask_tif=mask_path or defaults.mask_tif,
        mask_png=mask_png_path or defaults.mask_png,
        final_tif=final_tif_path or defaults.final_tif,
        final_png=final_png_path or defaults.final_png,
        preview_jpg=defaults.preview_jpg,
    )
    for path in (paths.mask_tif, paths.mask_png, paths.final_tif, paths.final_png):
        path.parent.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    bounds = detect_content_bounds(
        image,
        threshold=threshold,
        downsample=downsample,
        margin=margin,
    )

    cropped = image[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1].copy()

    applied_halo: float | None = None
    tight_y0 = tight_y1 = tight_x0 = tight_x1 = None

    if clean_border:
        cropped, applied_halo = _clean_faint_border(
            cropped, bounds.threshold, halo_threshold
        )

    if tight_crop:
        ty0, ty1, tx0, tx1 = _tight_bbox(cropped)
        cropped = cropped[ty0:ty1, tx0:tx1]
        tight_y0 = bounds.y0 + ty0
        tight_y1 = bounds.y0 + ty1
        tight_x0 = bounds.x0 + tx0
        tight_x1 = bounds.x0 + tx1

    crop_result = CropResult(
        bounds.y0,
        bounds.y1,
        bounds.x0,
        bounds.x1,
        bounds.original_shape,
        bounds.threshold,
        bounds.downsample,
        applied_halo,
        tight_y0,
        tight_y1,
        tight_x0,
        tight_x1,
    )

    mask = _build_mask(cropped)
    _save_image(paths.mask_tif, mask)
    _save_image(paths.mask_png, mask)
    _save_image(paths.final_tif, cropped)
    _save_image(paths.final_png, cropped)

    return ProcessResult(crop_result, paths), cropped


def save_preview(
    input_path: str | Path,
    preview_path: str | Path,
    bounds: CropResult,
    *,
    max_size: int = 1200,
) -> None:
    """Save a JPEG preview with the detected crop rectangle drawn."""
    input_path = Path(input_path)
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    height, width = image.shape
    scale = max(height, width) / max_size
    if scale < 1:
        scale = 1

    preview_h = int(height / scale)
    preview_w = int(width / scale)
    small = cv2.resize(image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    preview = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    inv = 1 / scale
    x0 = bounds.tight_x0 if bounds.tight_x0 is not None else bounds.x0
    y0 = bounds.tight_y0 if bounds.tight_y0 is not None else bounds.y0
    x1 = bounds.tight_x1 if bounds.tight_x1 is not None else bounds.x1
    y1 = bounds.tight_y1 if bounds.tight_y1 is not None else bounds.y1

    # Bounds are slice-exclusive; convert to inclusive pixel coords for drawing.
    last_x = min(x1 - 1, width - 1)
    last_y = min(y1 - 1, height - 1)
    pt1 = (max(0, int(x0 * inv)), max(0, int(y0 * inv)))
    pt2 = (
        min(preview_w - 1, int(last_x * inv)),
        min(preview_h - 1, int(last_y * inv)),
    )
    thickness = max(3, int(3 / scale))
    x0p, y0p = pt1
    x1p, y1p = pt2
    green = (0, 255, 0)
    cv2.line(preview, (x0p, y0p), (x1p, y0p), green, thickness)  # top
    cv2.line(preview, (x0p, y1p), (x1p, y1p), green, thickness)  # bottom
    cv2.line(preview, (x0p, y0p), (x0p, y1p), green, thickness)  # left
    cv2.line(preview, (x1p, y0p), (x1p, y1p), green, thickness)  # right

    cv2.imwrite(str(preview_path), preview)
