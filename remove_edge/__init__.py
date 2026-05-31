"""Remove dark borders and edge glow from TIFF microscopy images."""

from remove_edge.core import (
    CropResult,
    OutputPaths,
    ProcessResult,
    convert_to_uint8,
    detect_content_bounds,
    load_image,
    remove_edge,
)

__all__ = [
    "CropResult",
    "OutputPaths",
    "ProcessResult",
    "convert_to_uint8",
    "detect_content_bounds",
    "load_image",
    "remove_edge",
]
