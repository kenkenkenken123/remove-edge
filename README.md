# remove-edge

Python tool to clean microscopy TIFF scans: remove dark borders, faint edge glow, and crop to tissue — producing analysis-ready **8-bit PNG** output.

## Problem

Microscopy TIFF files often contain:

- **Dark outer borders** — empty scanner area around the slide
- **Faint gray halo** — low-intensity glow around tissue (not true background)
- **Background haze** — some slides have gray haze instead of pure black (e.g. `PE26EVGH_Target lesion 1b.tif`)
- **Multiple tissue fragments** — separate pieces that must all be kept

This tool detects tissue, crops to it, and produces a clean black-background image for downstream analysis.

---

## Method (pipeline pattern)

Every file follows the same **5-step pipeline**:

```
┌─────────────────┐
│  Input TIFF     │  16-bit grayscale (typical)
│  (microscopy)   │
└────────┬────────┘
         │  Step 1: Convert to 8-bit
         ▼
┌─────────────────┐
│  8-bit image    │  0–255, all further steps use this
│  (0–255)        │
└────────┬────────┘
         │  Step 2: Detect tissue & crop borders
         ▼
┌─────────────────┐
│  Cropped region │  Green box in preview
└────────┬────────┘
         │  Step 3: Remove faint halo
         ▼
┌─────────────────┐
│  Clean image    │  Sub-threshold pixels → black
└────────┬────────┘
         │  Step 4: Tight crop
         ▼
┌─────────────────┐
│  Final PNG/TIFF │  8-bit, tissue only, black background
└─────────────────┘
```

### Step 1 — Load & convert to 8-bit

| Input | Processing | Output |
|-------|------------|--------|
| 16-bit TIFF (0–65535) | Percentile stretch to 0–255 | 8-bit grayscale |

**Black-point rule:**
- If **>5%** of pixels are true zero → black point = 0
- Otherwise (hazy slides) → black point = **10th percentile** (subtract background haze)

**White point:** 99.5th percentile (avoids hot-pixel clipping)

All thresholds in this tool use the **0–255** range after this step.

### Step 2 — Detect tissue & crop outer borders

Done on a **downsampled copy** (8× by default) for speed, then mapped back to full resolution.

1. **Auto-threshold** — separates tissue from background:
   - Background reference: 25th percentile of pixel values
   - Tissue reference: 85th percentile
   - Threshold = background + 45% × (tissue − background)

2. **Binary mask** — pixels above threshold = tissue

3. **Morphology** — close + open with elliptical kernel (fills gaps, removes noise)

4. **Bounding box** — rectangle around **all** tissue fragments (not just the largest)

5. **Margin** — add 48 px padding (configurable with `--margin`)

### Step 3 — Remove faint border (halo)

Inside the crop, pixels below the **halo cutoff** are set to **0** (pure black).

- Auto halo cutoff: estimated from tissue pixel distribution (conservative — keeps dim edges)
- Override with `--halo-threshold`

**Two thresholds — different jobs:**

| Threshold | Flag | Purpose |
|-----------|------|---------|
| Detection | `--threshold` | Where to **crop** (tissue vs background) |
| Halo | `--halo-threshold` | What to **zero out** inside the crop (glow vs tissue) |

### Step 4 — Tight crop

After halo removal, crop again to the bounding box of remaining nonzero pixels. Removes empty black margins inside the first crop.

Disable with `--no-tight-crop`.

### Step 5 — Save

Five files are written on every run:

| File | Format | Purpose |
|------|--------|---------|
| `<name>_masked.tif` | 8-bit TIFF | Binary tissue mask (255 = tissue, 0 = background) |
| `<name>_masked.png` | 8-bit PNG | Same binary mask, easy to view |
| `<name>_masked_final.tif` | 8-bit TIFF | Final cropped tissue on black background |
| `<name>_masked_final.png` | 8-bit PNG | Same as final TIFF, easy to view |
| `<name>_preview.jpg` | JPEG | Full scan with green crop box |

---

## Recommended workflow

Always follow this pattern:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Preview first — check the green crop box
python -m remove_edge "your_file.tif" --preview-only

# 3. Process if the preview looks correct
python -m remove_edge "your_file.tif"

# Or process every TIFF in a folder
python -m remove_edge input/

# 4. Tune if needed (see troubleshooting below)
python -m remove_edge "your_file.tif" --threshold 30 --halo-threshold 70
```

### Output files

| File | Purpose |
|------|---------|
| `output/<name>_masked.tif` | **Binary mask** — white = tissue, black = background (TIFF) |
| `output/<name>_masked.png` | **Binary mask** — same as above (PNG) |
| `output/<name>_masked_final.tif` | **Final TIFF** — cropped tissue on black (for analysis) |
| `output/<name>_masked_final.png` | **Final PNG** — same as TIFF, easy to view |
| `output/<name>_preview.jpg` | **Preview** — full scan with green crop box |

The preview shows where the tool will crop. The PNG is the cleaned, cropped result.

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

# Process one file (writes all 5 output files)
python -m remove_edge "PE21HKSH_Target lesion 1G.tif"

# Process all TIFFs in a folder
python -m remove_edge input/

# Custom output directory (single file or batch)
python -m remove_edge input/ --output-dir output/run1
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | TIFF file **or folder** of `.tif` / `.tiff` files |
| `--output-dir` | `output/` | Directory for all output files |
| `--masked` | `output/<name>_masked.tif` | Binary mask TIFF path |
| `--masked-png` | `output/<name>_masked.png` | Binary mask PNG path |
| `--final-tif` | `output/<name>_masked_final.tif` | Final TIFF path |
| `--final-png` | `output/<name>_masked_final.png` | Final PNG path |
| `--threshold` | auto | Tissue detection threshold (0–255) |
| `--halo-threshold` | auto | Faint-border removal cutoff (0–255) |
| `--margin` | 48 | Extra pixels around detected tissue |
| `--downsample` | 8 | Downsample factor for detection speed |
| `--no-clean-border` | off | Skip halo removal |
| `--no-tight-crop` | off | Keep black margins after cleaning |
| `--preview` | `output/<name>_preview.jpg` | Preview JPEG path |
| `--preview-only` | off | Preview only, no output file |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Tissue cut off at edges | Threshold too high | Lower `--threshold` (e.g. `--threshold 25`) |
| Too much gray background kept | Threshold too low | Raise `--threshold` (e.g. `--threshold 55`) |
| Faint gray halo remains | Halo cutoff too low | Raise `--halo-threshold` (e.g. `--halo-threshold 90`) |
| Dim tissue removed | Halo cutoff too high | Lower `--halo-threshold` or use `--no-clean-border` |
| Murky gray whole image | Hazy slide background | Fixed automatically; re-run with latest code |
| Missing small fragments | Margin too small | Increase `--margin` (e.g. `--margin 100`) |
| Crop too tight | Margin too small / tight crop | Increase `--margin` or use `--no-tight-crop` |

### Slide types seen in this project

| File | Background | Typical auto threshold |
|------|------------|------------------------|
| `PE21HKSH_Target lesion 1G.tif` | Black | ~20–55 |
| `PE26EVGH_Target lesion 2B.tif` | Black | ~20–60 |
| `PE26EVGH_Target lesion 1b.tif` | Gray haze | ~40–45 |

---

## Python API

```python
from remove_edge import load_image, detect_content_bounds, remove_edge

# Load as 8-bit
image = load_image("PE26EVGH_Target lesion 1b.tif")

# Preview detection only
bounds = detect_content_bounds(image)
print(bounds.cropped_shape, bounds.threshold)

# Full pipeline
result, _ = remove_edge("PE26EVGH_Target lesion 1b.tif")
print(result.paths.mask_tif)
print(result.paths.mask_png)
print(result.paths.final_tif)
print(result.paths.final_png)
```

---

## Project structure

```
remove-edge/
├── .venv/
├── requirements.txt
├── README.md
├── remove_edge/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   └── core.py          # pipeline logic
├── output/              # generated PNG + preview
└── *.tif                # input files
```

Run as module:

```bash
python -m remove_edge <input.tif>
```

---

## Dependencies

- **numpy** — array operations
- **opencv-python-headless** — resize, morphology, save images
- **tifffile** — read TIFF (16-bit, memory-mapped when supported)
- **pillow** — image I/O support
- **scikit-image** — (available for future extensions)

---

## Bit depth

This tool always works in **8-bit** (0–255):

- Input: 16-bit TIFF → converted to 8-bit on load
- Processing: 8-bit
- Output: 8-bit PNG or TIFF

16-bit precision is not preserved by design — the goal is clean, cropped images for visual analysis and downstream tools that expect 8-bit input.
