"""Image utility functions for the pipeline."""
import base64
from pathlib import Path
from typing import Optional


def encode_image_base64(image_path: str | Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str | Path) -> str:
    """Determine MIME type from file extension."""
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }
    return mime_map.get(suffix, "image/png")


def resize_image_if_needed(
    image_path: str | Path,
    max_dimension: int = 2048,
    output_path: Optional[str | Path] = None,
) -> str:
    """Resize image if any dimension exceeds max_dimension. Returns path to (possibly resized) image."""
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size

    if w <= max_dimension and h <= max_dimension:
        return str(image_path)

    scale = max_dimension / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.LANCZOS)

    if output_path is None:
        p = Path(image_path)
        output_path = p.parent / f"{p.stem}_resized{p.suffix}"

    img.save(str(output_path))
    return str(output_path)
