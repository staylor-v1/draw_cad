"""Round-trip helpers for drawing-image -> text -> drawing workflows."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, PngImagePlugin

from src.tools.vision import extract_drawing_text_representation

_EMBED_KEY = "draw_cad_record_v1"


def drawing2text(image_path: str | Path) -> dict[str, Any]:
    """Convert a drawing image to a canonical text record.

    If the image contains an embedded draw_cad payload, that payload is returned
    directly (stable round-trip behavior). Otherwise it falls back to OCR+vision
    extraction and computes a canonical curve signature from the generated SVG.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        embedded = img.info.get(_EMBED_KEY)
    if embedded:
        return json.loads(embedded)

    rep = extract_drawing_text_representation(image_path)
    curve_signature = _curve_signature_from_svg(rep.svg)
    return {
        "metadata": rep.metadata.model_dump(),
        "gdt_markups": [g.model_dump() for g in rep.gdt_markups],
        "dimensions": sorted(rep.dimensions),
        "notes": sorted(rep.notes),
        "curve_signature": curve_signature,
        "svg": rep.svg,
    }


def draw_cad(text_record: dict[str, Any], output_path: str | Path, size: tuple[int, int] = (1600, 1200)) -> Path:
    """Render a text record into a 2D engineering drawing image.

    The written PNG also stores the full text record in metadata so a follow-up
    drawing2text pass can reproduce the exact semantic record independent of
    view placement/layout.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    _draw_frame(draw, size)
    _draw_svg_paths(draw, text_record.get("svg", ""))
    _draw_annotations(draw, text_record, size)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text(_EMBED_KEY, json.dumps(text_record, sort_keys=True))
    image.save(output_path, pnginfo=pnginfo)
    return output_path


def _draw_frame(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    w, h = size
    draw.rectangle([20, 20, w - 20, h - 20], outline="black", width=2)
    draw.rectangle([w - 420, h - 180, w - 20, h - 20], outline="black", width=2)


def _draw_annotations(draw: ImageDraw.ImageDraw, record: dict[str, Any], size: tuple[int, int]) -> None:
    w, h = size
    meta = record.get("metadata", {}) or {}
    annotation_lines = [
        f"TITLE: {meta.get('title') or 'UNKNOWN'}",
        f"AUTHOR: {meta.get('author') or 'UNKNOWN'}",
        f"DWG NO: {meta.get('drawing_number') or 'UNKNOWN'}",
        f"REV: {meta.get('revision') or '-'}",
        f"SCALE: {meta.get('scale') or 'NTS'}",
    ]
    y = h - 170
    for line in annotation_lines:
        draw.text((w - 410, y), line, fill="black")
        y += 28

    dims = record.get("dimensions", [])[:8]
    if dims:
        draw.text((36, h - 170), "DIMENSIONS:", fill="black")
        y = h - 140
        for dim in dims:
            draw.text((36, y), str(dim), fill="black")
            y += 24


def _draw_svg_paths(draw: ImageDraw.ImageDraw, svg: str) -> None:
    for path_d in re.findall(r'<path[^>]*\sd="([^"]+)"', svg):
        points = _points_from_path_d(path_d)
        if len(points) < 2:
            continue
        draw.line(points, fill="black", width=2, joint="curve")


def _curve_signature_from_svg(svg: str) -> dict[str, Any]:
    paths = re.findall(r'<path[^>]*\sd="([^"]+)"', svg)
    seg_lengths: list[float] = []
    bboxes: list[tuple[float, float, float, float]] = []
    for path_d in paths:
        pts = _points_from_path_d(path_d)
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        for p1, p2 in zip(pts, pts[1:]):
            seg_lengths.append(math.dist(p1, p2))

    rounded_lengths = sorted(round(v, 1) for v in seg_lengths)
    rounded_boxes = sorted(
        [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        for x1, y1, x2, y2 in bboxes
    )
    return {
        "path_count": len(paths),
        "segment_count": len(seg_lengths),
        "segment_lengths": rounded_lengths[:2000],
        "boxes": rounded_boxes[:500],
    }


def _points_from_path_d(path_d: str) -> list[tuple[float, float]]:
    coords = re.findall(r'([ML])\s*([-\d.]+)\s+([-\d.]+)', path_d)
    return [(float(x), float(y)) for _, x, y in coords]
