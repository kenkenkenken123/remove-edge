"""CLI entry point: python -m remove_edge input.tif  OR  python -m remove_edge input/"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from remove_edge.core import detect_content_bounds, load_image, remove_edge, save_preview


def collect_tif_paths(path: Path) -> list[Path]:
    """Return TIFF paths from a single file or all TIFFs in a folder."""
    if path.is_file():
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Not a TIFF file: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input not found: {path}")

    files = [
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    return sorted(files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove dark borders and edge glow from TIFF microscopy images.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input TIFF file or folder containing .tif files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--masked",
        type=Path,
        default=None,
        help="Masked TIFF path (single file only; default: output/<stem>_masked.tif)",
    )
    parser.add_argument(
        "--masked-png",
        type=Path,
        default=None,
        help="Masked PNG path (single file only; default: output/<stem>_masked.png)",
    )
    parser.add_argument(
        "--final-tif",
        type=Path,
        default=None,
        help="Final TIFF path (single file only; default: output/<stem>_masked_final.tif)",
    )
    parser.add_argument(
        "--final-png",
        type=Path,
        default=None,
        help="Final PNG path (single file only; default: output/<stem>_masked_final.png)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Tissue detection threshold in 0-255 (auto-estimated when omitted)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=8,
        help="Downsample factor for detection (default: 8)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=48,
        help="Extra pixels around detected content (default: 48; negative = tighter crop)",
    )
    parser.add_argument(
        "--no-clean-border",
        action="store_true",
        help="Keep the faint gray halo around tissue (skip border cleaning)",
    )
    parser.add_argument(
        "--no-tight-crop",
        action="store_true",
        help="Keep black margins around tissue after border cleaning",
    )
    parser.add_argument(
        "--halo-threshold",
        type=float,
        default=None,
        help="Manual cutoff for faint-border removal, 0-255 (auto-estimated by default)",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Preview JPEG path (single file only; default: output/<stem>_preview.jpg)",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only generate previews, do not write output files",
    )
    return parser


def _check_single_file_options(args: argparse.Namespace, n_files: int) -> None:
    if n_files <= 1:
        return
    for name in ("masked", "masked_png", "final_tif", "final_png", "preview"):
        if getattr(args, name) is not None:
            print(
                f"Warning: --{name.replace('_', '-')} ignored when processing a folder.",
                file=sys.stderr,
            )


def process_file(args: argparse.Namespace, input_path: Path) -> bool:
    """Process one TIFF. Returns True on success."""
    preview = args.preview or args.output_dir / f"{input_path.stem}_preview.jpg"

    try:
        if args.preview_only:
            image = load_image(input_path)
            bounds = detect_content_bounds(
                image,
                threshold=args.threshold,
                downsample=args.downsample,
                margin=args.margin,
            )
            save_preview(input_path, preview, bounds)
            print(f"[{input_path.name}] Preview saved: {preview}")
            print(f"  crop y=[{bounds.y0},{bounds.y1}] x=[{bounds.x0},{bounds.x1}]")
            print(f"  size {bounds.original_shape} -> {bounds.cropped_shape}")
            print(f"  threshold {bounds.threshold:.1f}")
            return True

        result, _ = remove_edge(
            input_path,
            output_dir=args.output_dir,
            mask_path=args.masked,
            mask_png_path=args.masked_png,
            final_tif_path=args.final_tif,
            final_png_path=args.final_png,
            threshold=args.threshold,
            downsample=args.downsample,
            margin=args.margin,
            clean_border=not args.no_clean_border,
            halo_threshold=args.halo_threshold,
            tight_crop=not args.no_tight_crop,
        )
        bounds = result.bounds
        paths = result.paths
        save_preview(input_path, preview, bounds)

        print(f"[{input_path.name}]")
        print(f"  Masked TIFF:       {paths.mask_tif}")
        print(f"  Masked PNG:        {paths.mask_png}")
        print(f"  Masked final TIFF: {paths.final_tif}")
        print(f"  Masked final PNG:  {paths.final_png}")
        print(f"  Preview:           {preview}")
        print(f"  {bounds.original_shape[1]}x{bounds.original_shape[0]} -> {bounds.cropped_shape[1]}x{bounds.cropped_shape[0]}")
        print(f"  threshold {bounds.threshold:.1f}", end="")
        if bounds.halo_threshold is not None:
            print(f", halo {bounds.halo_threshold:.1f}", end="")
        print()
        return True

    except Exception as exc:
        print(f"[{input_path.name}] FAILED: {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        tif_files = collect_tif_paths(args.input)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if not tif_files:
        parser.error(f"No TIFF files found in: {args.input}")

    _check_single_file_options(args, len(tif_files))

    if len(tif_files) > 1:
        print(f"Processing {len(tif_files)} TIFF file(s) from {args.input}")
        print(f"Output directory: {args.output_dir}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok = sum(process_file(args, path) for path in tif_files)
    failed = len(tif_files) - ok

    if len(tif_files) > 1:
        print(f"\nDone: {ok} succeeded, {failed} failed, {len(tif_files)} total.")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
