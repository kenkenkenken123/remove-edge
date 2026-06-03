# remove-edge

Python tool for microscopy **slide TIFFs**: detect tissue vs slide background, write masks and cream-background images at **full scan resolution**, and optionally build a **Massed uframe** composite (uframe colors on tissue + cream glass) when a matching uframe JPG is present.

## Problem

Microscopy TIFF files often contain:

- **Dark outer borders** — empty scanner area around the slide
- **Faint gray halo** — low-intensity glow around tissue (not true background)
- **Background haze** — some slides have gray haze instead of pure black (e.g. `PE26EVGH_Target lesion 1b.tif`)
- **Multiple tissue fragments** — separate pieces that must all be kept

Downstream work also needs:

- A clear **tissue vs background** map (mask)
- A **normalized / model view** (uframe JPG) aligned with the slide
- A **Massed uframe** image: model colors on tissue, uniform cream on glass — matching files like `Massed_uframe_*.tif`

This tool automates detection, mask export, background fill, and uframe compositing.

---

## Inputs

| File | Location | Required | Role |
|------|----------|----------|------|
| Slide TIFF | `input/` or path you pass | **Yes** | Grayscale scan (often 16-bit); drives tissue detection |
| Uframe JPG | `input/` | For Massed / combined only | Color model view; name must match `uframe_*_<slide_stem>.jpg` |

Example pair:

```
input/PE21HKSH_Target lesion 1G.tif
input/uframe_20250519_HKSH_Training_norm_train_mod2_i512d32n3ep100kimg600bs2_PE21HKSH_Target lesion 1G.jpg
```

If uframe is missing, masks and `masked_final` still run; Massed TIFF and `_combined.jpg` fail for that slide.

---

## Big picture

```
  slide.tif
      │
      ├─► detect tissue (downsampled, then full-res mask)
      │
      ├─► masked.png / .tif     (stencil: white=glass, black=tissue)
      ├─► masked_final          (cream glass, black empty tissue)
      └─► preview.jpg           (full scan + green detection box)

  uframe.jpg + masked.png
      │
      ├─► Massed_uframe_....tif   (full resolution)
      └─► <slide>_combined.jpg    (same picture, shrunk for viewing)
```

All raster outputs except the preview and `_combined.jpg` keep the **same width × height as the input TIFF**.

---

## Concepts

### Slide background color

Cream RGB **`(249, 245, 242)`** — constant `BACKGROUND_RGB` in code. Used in `masked_final` and in white (background) regions of Massed / combined.

### `masked` — where is tissue?

Binary map at **full resolution**. Saved values are inverted for viewing:

| Pixel in `masked.png` | Value | Meaning |
|------------------------|-------|---------|
| **White** | 255 | Slide background (glass, borders) |
| **Black** | 0 | Tissue |

Think of it as a **stencil**: black shape = tissue, white = not tissue. Massed compositing reads this file.

### `masked_final` — cream slide, empty tissue

Same regions as `masked`, but as a color image:

| Region (from mask) | Color |
|--------------------|--------|
| White in `masked` (background) | Cream `(249, 245, 242)` |
| Black in `masked` (tissue) | **Black** — no tissue pixels, no uframe |

Useful when you want “clean glass” only, with tissue punched out.

### Massed uframe / `_combined.jpg` — uframe on tissue, cream on glass

Built from **`masked.png` + uframe JPG**, not from the slide TIFF pixels directly.

**Per pixel (not a blend):**

```
Start with:  copy entire uframe image

If masked is BLACK  (tissue):     keep uframe color
If masked is WHITE (background): replace with cream (249, 245, 242)
```

There is **no** 50/50 mix. It is **keep** or **replace** by mask.

| Output | Resolution | Notes |
|--------|------------|--------|
| `Massed_uframe_<suffix>.tif` | Full (same as slide/uframe) | Name derived from uframe: `uframe_foo.jpg` → `Massed_uframe_foo.tif` |
| `<slide_stem>_combined.jpg` | Preview (max side 4096 px) | **Same compositing** as the Massed TIFF, only smaller |

This recipe matches reference `Massed_uframe_*.tif` files in this project (uframe on tissue, flat cream on glass).

### Compare the three image types

| | Tissue area | Background area |
|---|-------------|-----------------|
| **masked** | Black | White |
| **masked_final** | Black (empty) | Cream |
| **Massed / combined** | **Uframe colors** | Cream |

Same mask semantics; different fill in the tissue zone.

---

## Method (slide pipeline)

Every slide follows the same detection pipeline. Outputs are **not** cropped to a smaller file — they stay on the full scan canvas.

```
┌─────────────────┐
│  Input TIFF     │  16-bit grayscale (typical)
└────────┬────────┘
         │  Step 1: Convert to 8-bit
         ▼
┌─────────────────┐
│  8-bit image    │  0–255
└────────┬────────┘
         │  Step 2: Tissue mask at full resolution
         │          (detection on downsampled copy)
         ▼
┌─────────────────┐
│  tissue_mask    │  internal: 255=tissue, 0=background
└────────┬────────┘
         │  Step 3: Optional halo / tight box (preview only)
         ▼
┌─────────────────┐
│  Save masked,   │  masked inverted for display;
│  masked_final,  │  masked_final = cream + black tissue
│  preview        │
└─────────────────┘
```

Halo removal and tight crop run on a **crop region** to refine the **green rectangle** in `preview.jpg`. The saved **`masked` / `masked_final`** use the full-resolution threshold mask on the whole image.

### Step 1 — Load & convert to 8-bit

| Input | Processing | Output |
|-------|------------|--------|
| 16-bit TIFF (0–65535) | Percentile stretch to 0–255 | 8-bit grayscale |

**Black-point rule:**

- If **>5%** of pixels are true zero → black point = 0
- Otherwise (hazy slides) → black point = **10th percentile** (subtract background haze)

**White point:** 99.5th percentile (avoids hot-pixel clipping)

All thresholds use the **0–255** range after this step.

### Step 2 — Detect tissue (full-resolution mask)

Done on a **downsampled copy** (8× by default) for speed, then resized with nearest-neighbor to full slide size.

1. **Auto-threshold** — separates tissue from background:
   - Background reference: 25th percentile of pixel values
   - Tissue reference: 85th percentile
   - Threshold = background + 45% × (tissue − background)

2. **Binary mask** — pixels above threshold = tissue (internal `tissue_mask`: 255 = tissue, 0 = background)

3. **Morphology** — close + open with elliptical kernel (fills gaps, removes noise)

4. **Bounding box** — rectangle around **all** tissue fragments (used for preview and halo region)

5. **Margin** — add 48 px padding (configurable with `--margin`)

Saved `masked` files use **`255 - tissue_mask`** so **white = background, black = tissue** (see [Concepts](#concepts)).

### Step 3 — Halo & tight crop (preview metadata)

Inside the loose bounding box, optional **halo removal** zeros faint edge glow. Optional **tight crop** finds a smaller box around nonzero pixels after halo.

These steps update **preview** crop lines (`tight_*` in `CropResult`). They do **not** change the saved binary mask values.

| Threshold | Flag | Purpose |
|-----------|------|---------|
| Detection | `--threshold` | Tissue vs background for **mask** |
| Halo | `--halo-threshold` | Faint glow removal inside crop region (preview / metadata) |

Disable halo with `--no-clean-border`. Disable tight box with `--no-tight-crop`.

### Step 4 — Massed uframe (after mask exists)

If `find_uframe_jpg()` locates a matching uframe in `input/`, `output/`, or the slide directory:

1. Load `masked.png` and uframe (same pixel size expected).
2. `_build_massed_uframe()`: copy uframe → paint cream where `mask > 0`.
3. Write full TIFF + downscaled `_combined.jpg`.

---

## Output files

Example stem: `PE21HKSH_Target lesion 1G`

| File | Format | Purpose |
|------|--------|---------|
| `<name>_masked.tif` | 8-bit TIFF | Binary mask — **white = background, black = tissue** |
| `<name>_masked.png` | 8-bit PNG | Same mask, easy to view |
| `<name>_masked_final.tif` | 8-bit RGB TIFF | Cream background; **black empty tissue**; full resolution |
| `<name>_masked_final.png` | 8-bit RGB PNG | Same as final TIFF |
| `<name>_preview.jpg` | JPEG | Full scan with green crop box (downscaled if huge) |
| `Massed_uframe_<uframe-suffix>.tif` | 8-bit RGB TIFF | Uframe on tissue + cream on glass; full resolution |
| `<name>_combined.jpg` | JPEG | Same as Massed TIFF, preview size (max side 4096) |

---

## Recommended workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Put slide + uframe in input/
#    uframe_*_<slide_stem>.jpg  must match the TIFF stem

# 3. Preview first — check the green crop box
python -m remove_edge "input/PE21HKSH_Target lesion 1G.tif" --preview-only

# 4. Process one slide (all outputs above)
python -m remove_edge "input/PE21HKSH_Target lesion 1G.tif"

# Or batch
python -m remove_edge input/

# 5. Tune if needed
python -m remove_edge "input/PE21HKSH_Target lesion 1G.tif" --threshold 30
```

Check:

- `masked.png` — tissue black, glass white
- `masked_final.png` — cream around black tissue holes
- `combined.jpg` / `Massed_uframe_*.tif` — colored tissue from uframe, cream elsewhere

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Preview only
python -m remove_edge "PE21HKSH_Target lesion 1G.tif" --preview-only

# Process one file
python -m remove_edge "PE21HKSH_Target lesion 1G.tif"

# Process all TIFFs in a folder
python -m remove_edge input/

# Custom output directory
python -m remove_edge input/ --output-dir output/run1
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | TIFF file **or folder** of `.tif` / `.tiff` files |
| `--output-dir` | `output/` | Directory for all output files |
| `--masked` | `output/<name>_masked.tif` | Binary mask TIFF path |
| `--masked-png` | `output/<name>_masked.png` | Binary mask PNG path |
| `--final-tif` | `output/<name>_masked_final.tif` | Masked final TIFF path |
| `--final-png` | `output/<name>_masked_final.png` | Masked final PNG path |
| `--threshold` | auto | Tissue detection threshold (0–255) |
| `--halo-threshold` | auto | Faint-border cutoff inside crop region (0–255) |
| `--margin` | 48 | Extra pixels around detected tissue |
| `--downsample` | 8 | Downsample factor for detection speed |
| `--no-clean-border` | off | Skip halo removal on crop region |
| `--no-tight-crop` | off | Skip tight crop box for preview |
| `--preview` | `output/<name>_preview.jpg` | Preview JPEG path |
| `--preview-only` | off | Preview only, no mask/final/Massed outputs |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Tissue cut off at edges | Threshold too high | Lower `--threshold` (e.g. `--threshold 25`) |
| Too much gray background kept | Threshold too low | Raise `--threshold` (e.g. `--threshold 55`) |
| Cream where tissue should be (Massed) | Mask wrong (white/black flipped or bad threshold) | Fix `masked.png` first; re-run |
| Massed / combined missing | No matching uframe JPG | Add `uframe_*_<slide_stem>.jpg` under `input/` |
| Faint gray halo in preview box | Halo cutoff too low | Raise `--halo-threshold` or adjust preview expectations |
| Dim tissue removed in preview region | Halo cutoff too high | Lower `--halo-threshold` or `--no-clean-border` |
| Murky gray whole image | Hazy slide background | Often handled by 8-bit black-point rule; re-run with latest code |
| Missing small fragments | Margin too small | Increase `--margin` (e.g. `--margin 100`) |

### Slide types seen in this project

| File | Background | Typical auto threshold |
|------|------------|------------------------|
| `PE21HKSH_Target lesion 1G.tif` | Black | ~20–55 |
| `PE26EVGH_Target lesion 2B.tif` | Black | ~20–60 |
| `PE26EVGH_Target lesion 1b.tif` | Gray haze | ~40–45 |

---

## Python API

```python
from pathlib import Path
from remove_edge import load_image, detect_content_bounds, remove_edge
from remove_edge.core import save_combined_jpg

# Load as 8-bit
image = load_image("PE21HKSH_Target lesion 1G.tif")

# Preview detection only
bounds = detect_content_bounds(image)
print(bounds.cropped_shape, bounds.threshold)

# Masks + masked_final (full resolution)
result, final_rgb = remove_edge("PE21HKSH_Target lesion 1G.tif")
print(result.paths.mask_png)
print(result.paths.final_png)

# Massed TIFF + combined preview (needs masked.png + uframe in input/)
combined = save_combined_jpg(
    "PE21HKSH_Target lesion 1G.tif",
    "output",
    search_dirs=(Path("input"), Path("output")),
)
print(combined.massed_tif)
print(combined.preview_jpg)
```

---

## Project structure

```
remove-edge/
├── .venv/
├── requirements.txt
├── README.md
├── input/               # slide TIFFs + uframe_*_<stem>.jpg
├── output/              # masks, masked_final, Massed, combined, preview
├── remove_edge/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   └── core.py          # detection, masks, Massed compositing
└── ...
```

Run as module:

```bash
python -m remove_edge input/
```

---

## Dependencies

- **numpy** — array operations
- **opencv-python-headless** — resize, morphology, JPEG I/O
- **tifffile** — read/write TIFF (16-bit in, 8-bit out)
- **pillow** — PNG I/O (large images; decompression limit disabled in code)
- **scikit-image** — (available for future extensions)

---

## Bit depth

- **Input:** 16-bit TIFF → converted to 8-bit on load for detection
- **Masks:** 8-bit grayscale
- **masked_final / Massed:** 8-bit RGB
- **Output dimensions:** same as input TIFF (except preview and `_combined.jpg`)

16-bit precision is not preserved by design — the goal is masks, cream-background views, and uframe composites for analysis and training pipelines that expect 8-bit RGB.
