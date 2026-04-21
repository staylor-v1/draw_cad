"""Vision analysis tool using the inference abstraction layer."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageFilter

from src.inference.base import BaseVisionClient, LLMResponse
from src.schemas.geometry import (
    Dimension,
    DrawingMetadata,
    DrawingTextRepresentation,
    Feature,
    GDTRemark,
    GeometryData,
)
from src.tools.ocr import extract_text_regions
from src.utils.image_utils import encode_image_base64, get_image_mime_type
from src.utils.file_utils import load_prompt
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM response text, handling markdown code blocks."""
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("failed_to_extract_json", text_length=len(text))
    return {}


def analyze_drawing(
    image_path: str | Path,
    vision_client: BaseVisionClient,
    model: str,
    prompt_path: str = "prompts/vision_prompt.md",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> GeometryData:
    """Analyze an engineering drawing using the vision model."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = load_prompt(prompt_path)
    image_b64 = encode_image_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    logger.info("analyzing_drawing", image_path=str(image_path), model=model)

    response: LLMResponse = vision_client.analyze_image(
        image_base64=image_b64,
        prompt=prompt,
        model=model,
        mime_type=mime_type,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_json = extract_json_from_response(response.content)
    if not raw_json:
        logger.warning("vision_empty_json", raw_content=response.content[:500])
        return GeometryData(raw_response=response.content)

    geometry = GeometryData.from_vision_dict(raw_json)
    geometry.raw_response = response.content
    logger.info(
        "vision_analysis_complete",
        views=len(geometry.views),
        dimensions=len(geometry.dimensions),
        features=len(geometry.features),
    )
    return geometry


def analyze_drawing_mock(image_path: str | Path) -> GeometryData:
    """Mock vision analysis for testing without a live model backend."""
    logger.info("mock_vision_analysis", image_path=str(image_path))
    return GeometryData(
        views=["Top", "Front", "Right"],
        dimensions=[
            Dimension(label="Length", value=100.0, unit="mm"),
            Dimension(label="Width", value=50.0, unit="mm"),
            Dimension(label="Thickness", value=10.0, unit="mm"),
            Dimension(label="Hole Diameter", value=5.0, unit="mm", count=1),
        ],
        features=[
            Feature(type="Base Plate", description="Rectangular plate 100x50x10mm"),
            Feature(type="Through Hole", description="1x 5mm hole centered on the top face"),
        ],
        notes="Standard tolerance +/- 0.1mm",
    )


def extract_drawing_text_representation(
    image_path: str | Path,
    masked_output_path: Optional[str | Path] = None,
) -> DrawingTextRepresentation:
    """Extract drawing text/markup and vectorized part geometry from a drawing image.

    The function:
    1. runs OCR and classifies metadata, GD&T, dimensions, and notes,
    2. masks borders/title block + text areas to focus on part linework,
    3. vectorizes linework to SVG using contour tracing + polyline simplification.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    text_regions = extract_text_regions(image_path)
    metadata = _extract_title_block_metadata(text_regions)
    gdt_markups = [
        GDTRemark(text=r.text, bbox=r.bbox, confidence=r.confidence)
        for r in text_regions
        if r.category == "gdt"
    ]
    notes = [r.text for r in text_regions if r.category == "note"]
    dimensions = [r.text for r in text_regions if r.category == "dimension"]

    masked_img = _mask_non_part_regions(image_path, text_regions)
    if masked_output_path is not None:
        masked_output_path = Path(masked_output_path)
        masked_img.save(masked_output_path)
        masked_path_str: Optional[str] = str(masked_output_path)
    else:
        masked_path_str = None

    svg = _vectorize_masked_image_to_svg(masked_img)
    return DrawingTextRepresentation(
        metadata=metadata,
        gdt_markups=gdt_markups,
        notes=notes,
        dimensions=dimensions,
        svg=svg,
        masked_image_path=masked_path_str,
    )


def _extract_title_block_metadata(text_regions: list[Any]) -> DrawingMetadata:
    """Heuristically parse title-block metadata from OCR regions."""
    raw_fields: dict[str, str] = {}
    patterns = {
        "title": [r"\btitle\b"],
        "author": [r"\bauthor\b", r"\bdrawn\s*by\b"],
        "drawing_number": [r"\bdrawing\s*(no|number|#)\b", r"\bpart\s*(no|number|#)\b"],
        "revision": [r"\brev(ision)?\b"],
        "scale": [r"\bscale\b"],
        "date": [r"\bdate\b"],
    }
    lowered = [(r.text, r.text.strip().lower()) for r in text_regions]
    for raw, text in lowered:
        for field_name, field_patterns in patterns.items():
            if any(re.search(p, text) for p in field_patterns):
                value = raw.split(":", 1)[1].strip() if ":" in raw else raw.strip()
                raw_fields[field_name] = value

    return DrawingMetadata(
        title=raw_fields.get("title"),
        author=raw_fields.get("author"),
        drawing_number=raw_fields.get("drawing_number"),
        revision=raw_fields.get("revision"),
        scale=raw_fields.get("scale"),
        date=raw_fields.get("date"),
        raw_fields=raw_fields,
    )


def _mask_non_part_regions(image_path: Path, text_regions: list[Any]) -> Image.Image:
    """Mask border and OCR text regions so only part linework remains."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape

    # Remove border and title-block-like areas near page edges.
    margin = max(6, int(min(h, w) * 0.02))
    arr[:margin, :] = 255
    arr[h - margin:, :] = 255
    arr[:, :margin] = 255
    arr[:, w - margin:] = 255

    # Remove likely title block zone at lower-right.
    tb_h = int(h * 0.22)
    tb_w = int(w * 0.30)
    arr[h - tb_h:, w - tb_w:] = 255

    # Remove OCR text boxes.
    pad = 4
    for region in text_regions:
        if not region.bbox or len(region.bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in region.bbox]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        arr[y1:y2, x1:x2] = 255

    cleaned = Image.fromarray(arr).filter(ImageFilter.MedianFilter(size=3))
    return cleaned


def _vectorize_masked_image_to_svg(image: Image.Image) -> str:
    """Convert masked drawing linework to SVG polyline paths."""
    arr = np.array(image, dtype=np.uint8)
    # Binary linework map after masking/cleanup.
    feature = arr < 160

    segments = _trace_polylines(feature)
    simplified = [
        _rdp(poly[:: max(1, len(poly) // 800)], epsilon=1.8)
        for poly in segments
        if len(poly) >= 3
    ]

    w, h = image.size
    paths: list[str] = []
    for poly in simplified:
        if len(poly) < 2:
            continue
        d = [f"M {poly[0][0]:.2f} {poly[0][1]:.2f}"]
        for pt in poly[1:]:
            d.append(f"L {pt[0]:.2f} {pt[1]:.2f}")
        paths.append(f'<path d="{" ".join(d)}" fill="none" stroke="black" stroke-width="1"/>')

    body = "\n  ".join(paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">\n'
        f"  {body}\n"
        "</svg>"
    )


def _trace_polylines(feature_map: np.ndarray) -> list[list[tuple[float, float]]]:
    """Very small connected-component tracer returning coordinate chains."""
    visited = np.zeros_like(feature_map, dtype=bool)
    h, w = feature_map.shape
    polylines: list[list[tuple[float, float]]] = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not feature_map[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            points: list[tuple[float, float]] = []
            while stack:
                cx, cy = stack.pop()
                points.append((float(cx), float(cy)))
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and feature_map[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(points) >= 10:
                points.sort(key=lambda p: (p[0], p[1]))
                polylines.append(points)
    return polylines


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification (line/curve fitting primitive)."""
    if len(points) < 3:
        return points

    start = np.array(points[0], dtype=float)
    end = np.array(points[-1], dtype=float)
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0:
        dists = [np.linalg.norm(np.array(p) - start) for p in points]
    else:
        lx, ly = line[0], line[1]
        dists = [
            abs((start[0] - p[0]) * ly - (start[1] - p[1]) * lx) / norm
            for p in points
        ]

    idx = int(np.argmax(dists))
    dmax = dists[idx]
    if dmax > epsilon and 0 < idx < len(points) - 1:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]
