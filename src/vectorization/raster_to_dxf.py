"""Convert scanned drawing rasters into simple DXF line evidence.

This module is deliberately CAD-oriented rather than art-vector-oriented. It
extracts straight-line primitives, boxed annotation frames, and review overlays
that downstream segmentation tools can inspect without re-reading pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Iterable

import cv2
import ezdxf
import numpy as np
from PIL import Image, ImageDraw

from src.segmentation.title_block import NormalizedBox


@dataclass(frozen=True)
class VectorSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle: float
    orientation: str
    source: str = "hough"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["x1"] = round(self.x1, 3)
        data["y1"] = round(self.y1, 3)
        data["x2"] = round(self.x2, 3)
        data["y2"] = round(self.y2, 3)
        data["length"] = round(self.length, 3)
        data["angle"] = round(self.angle, 3)
        return data


@dataclass(frozen=True)
class VectorRectangle:
    crop: NormalizedBox
    confidence: float
    kind: str
    line_rows: int
    line_cols: int
    source: str = "vector_rectangles_v1"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["crop"] = self.crop.to_dict()
        return data


@dataclass(frozen=True)
class VectorizationResult:
    image_path: str
    width: int
    height: int
    segments: tuple[VectorSegment, ...]
    rectangles: tuple[VectorRectangle, ...]
    threshold_method: str

    def to_dict(self) -> dict:
        return {
            "image": self.image_path,
            "size": {"width": self.width, "height": self.height},
            "thresholdMethod": self.threshold_method,
            "counts": {
                "segments": len(self.segments),
                "horizontal": sum(1 for segment in self.segments if segment.orientation == "horizontal"),
                "vertical": sum(1 for segment in self.segments if segment.orientation == "vertical"),
                "diagonal": sum(1 for segment in self.segments if segment.orientation == "diagonal"),
                "rectangles": len(self.rectangles),
            },
            "segments": [segment.to_dict() for segment in self.segments],
            "rectangles": [rectangle.to_dict() for rectangle in self.rectangles],
        }


def raster_to_vector(image_path: str | Path, *, max_dimension: int = 1800) -> VectorizationResult:
    """Return line primitives and frame rectangles for a raster drawing image."""

    path = Path(image_path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")

    height, width = gray.shape
    scaled, scale = _scale_for_processing(gray, max_dimension=max_dimension)
    threshold = _drawing_threshold(scaled)
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    segments = _hough_segments(cleaned, scale=scale)
    segments = _merge_axis_aligned_segments(segments, width=width, height=height)
    rectangles = _dedupe_rectangles(
        [
            *_detect_rectangles(cleaned, scale=scale, width=width, height=height),
            *_detect_rectangles_from_segments(segments, width=width, height=height),
        ]
    )

    return VectorizationResult(
        image_path=str(path),
        width=width,
        height=height,
        segments=tuple(segments),
        rectangles=tuple(rectangles),
        threshold_method="otsu_binary_inverse_morph_close",
    )


def write_dxf(result: VectorizationResult, output_path: str | Path) -> None:
    """Write extracted vector primitives to DXF LINE/LWPOLYLINE entities."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new("R2010")
    for layer, color in (("horizontal", 3), ("vertical", 5), ("diagonal", 1), ("rectangles", 2)):
        doc.layers.add(layer, color=color)
    msp = doc.modelspace()
    for segment in result.segments:
        msp.add_line(
            (segment.x1, result.height - segment.y1),
            (segment.x2, result.height - segment.y2),
            dxfattribs={"layer": segment.orientation},
        )
    for rectangle in result.rectangles:
        left, top, right, bottom = rectangle.crop.to_pixels(result.width, result.height)
        points = [
            (left, result.height - top),
            (right, result.height - top),
            (right, result.height - bottom),
            (left, result.height - bottom),
        ]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "rectangles"})
    doc.saveas(output)


def draw_vector_overlay(image_path: str | Path, result: VectorizationResult, max_size: tuple[int, int] = (1200, 850)) -> Image.Image:
    """Draw a compact visual review of extracted lines and rectangles."""

    image = Image.open(image_path).convert("RGB")
    preview = image.copy()
    draw = ImageDraw.Draw(preview, "RGBA")
    colors = {
        "horizontal": (26, 131, 84, 165),
        "vertical": (44, 96, 170, 165),
        "diagonal": (196, 78, 48, 135),
    }
    long_segments = sorted(result.segments, key=lambda item: item.length, reverse=True)[:900]
    for segment in long_segments:
        draw.line((segment.x1, segment.y1, segment.x2, segment.y2), fill=colors[segment.orientation], width=2)
    for rectangle in result.rectangles:
        left, top, right, bottom = rectangle.crop.to_pixels(result.width, result.height)
        draw.rectangle((left, top, right, bottom), outline=(211, 151, 31, 230), width=3)
    preview.thumbnail(max_size)
    return preview


def _scale_for_processing(gray: np.ndarray, *, max_dimension: int) -> tuple[np.ndarray, float]:
    height, width = gray.shape
    largest = max(width, height)
    if largest <= max_dimension:
        return gray, 1.0
    scale = max_dimension / largest
    resized = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _drawing_threshold(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    return threshold


def _hough_segments(binary: np.ndarray, *, scale: float) -> list[VectorSegment]:
    height, width = binary.shape
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    min_line = max(14, int(min(width, height) * 0.025))
    max_gap = max(3, int(min(width, height) * 0.008))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, int(min(width, height) * 0.045)),
        minLineLength=min_line,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []

    inverse_scale = 1.0 / scale
    segments = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        ox1, oy1, ox2, oy2 = [float(value) * inverse_scale for value in (x1, y1, x2, y2)]
        segments.append(_make_segment(ox1, oy1, ox2, oy2))
    return segments


def _make_segment(x1: float, y1: float, x2: float, y2: float, *, source: str = "hough") -> VectorSegment:
    if x2 < x1 or (abs(x2 - x1) < 1e-6 and y2 < y1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx = x2 - x1
    dy = y2 - y1
    length = hypot(dx, dy)
    angle = degrees(atan2(dy, dx)) if length else 0.0
    abs_angle = abs(angle)
    if abs_angle <= 8 or abs_angle >= 172:
        orientation = "horizontal"
        y = round((y1 + y2) / 2.0, 2)
        return VectorSegment(x1, y, x2, y, length, 0.0, orientation, source)
    if 82 <= abs_angle <= 98:
        orientation = "vertical"
        x = round((x1 + x2) / 2.0, 2)
        y_low, y_high = sorted((y1, y2))
        return VectorSegment(x, y_low, x, y_high, length, 90.0, orientation, source)
    return VectorSegment(x1, y1, x2, y2, length, angle, "diagonal", source)


def _merge_axis_aligned_segments(segments: Iterable[VectorSegment], *, width: int, height: int) -> list[VectorSegment]:
    result: list[VectorSegment] = []
    by_orientation = {"horizontal": [], "vertical": []}
    for segment in segments:
        if segment.orientation in by_orientation:
            by_orientation[segment.orientation].append(segment)
        elif segment.length >= max(8, min(width, height) * 0.015):
            result.append(segment)

    tolerance = max(2.0, min(width, height) * 0.003)
    gap = max(5.0, min(width, height) * 0.012)
    min_length = max(10.0, min(width, height) * 0.018)
    for orientation, items in by_orientation.items():
        coord_key = (lambda item: item.y1) if orientation == "horizontal" else (lambda item: item.x1)
        items = sorted(items, key=lambda item: (coord_key(item), item.x1 if orientation == "horizontal" else item.y1))
        groups: list[list[VectorSegment]] = []
        for item in items:
            if not groups or abs(coord_key(item) - coord_key(groups[-1][-1])) > tolerance:
                groups.append([item])
            else:
                groups[-1].append(item)
        for group in groups:
            if orientation == "horizontal":
                y = float(np.median([item.y1 for item in group]))
                intervals = sorted((min(item.x1, item.x2), max(item.x1, item.x2)) for item in group)
                result.extend(_merge_intervals_to_segments(intervals, fixed=y, orientation=orientation, gap=gap, min_length=min_length))
            else:
                x = float(np.median([item.x1 for item in group]))
                intervals = sorted((min(item.y1, item.y2), max(item.y1, item.y2)) for item in group)
                result.extend(_merge_intervals_to_segments(intervals, fixed=x, orientation=orientation, gap=gap, min_length=min_length))
    return sorted(result, key=lambda item: (item.orientation, item.y1, item.x1, -item.length))


def _merge_intervals_to_segments(
    intervals: list[tuple[float, float]],
    *,
    fixed: float,
    orientation: str,
    gap: float,
    min_length: float,
) -> list[VectorSegment]:
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start - end <= gap:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))

    segments = []
    for start, end in merged:
        if end - start < min_length:
            continue
        if orientation == "horizontal":
            segments.append(_make_segment(start, fixed, end, fixed, source="merged_hough"))
        else:
            segments.append(_make_segment(fixed, start, fixed, end, source="merged_hough"))
    return segments


def _detect_rectangles(binary: np.ndarray, *, scale: float, width: int, height: int) -> list[VectorRectangle]:
    proc_h, proc_w = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, proc_w // 80), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, proc_h // 90))),
    )
    ruled = cv2.bitwise_or(horizontal, vertical)
    closed = cv2.morphologyEx(ruled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles: list[VectorRectangle] = []
    inverse_scale = 1.0 / scale
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ow = w * inverse_scale
        oh = h * inverse_scale
        if not _rectangle_size_ok(ow, oh, width, height):
            continue
        roi_h = horizontal[y : y + h, x : x + w]
        roi_v = vertical[y : y + h, x : x + w]
        line_rows = int((roi_h.sum(axis=1) > 255 * w * 0.22).sum())
        line_cols = int((roi_v.sum(axis=0) > 255 * h * 0.22).sum())
        if line_rows < 2 or line_cols < 2:
            continue
        ox = x * inverse_scale
        oy = y * inverse_scale
        pad = max(2, int(min(width, height) * 0.003))
        crop = NormalizedBox(
            (ox - pad) / width,
            (oy - pad) / height,
            (ow + 2 * pad) / width,
            (oh + 2 * pad) / height,
        ).clipped()
        kind = "feature_control_frame" if line_cols >= 3 and ow / max(oh, 1.0) >= 2.0 else "boxed_annotation"
        confidence = min(0.95, 0.28 + min(line_rows / 3, 1) * 0.23 + min(line_cols / 5, 1) * 0.25 + min((ow / max(oh, 1) - 1) / 4, 1) * 0.19)
        rectangles.append(VectorRectangle(crop=crop, confidence=round(confidence, 3), kind=kind, line_rows=line_rows, line_cols=line_cols))
    return rectangles


def _detect_rectangles_from_segments(
    segments: Iterable[VectorSegment],
    *,
    width: int,
    height: int,
) -> list[VectorRectangle]:
    horizontal = [item for item in segments if item.orientation == "horizontal"]
    vertical = [item for item in segments if item.orientation == "vertical"]
    rectangles: list[VectorRectangle] = []
    y_min = max(8.0, height * 0.014)
    y_max = min(height * 0.16, max(80.0, height * 0.13))
    side_tolerance = max(5.0, width * 0.01)
    vertical_tolerance = max(5.0, height * 0.014)

    for top_index, top in enumerate(horizontal):
        top_x0, top_x1 = sorted((top.x1, top.x2))
        if top_x1 - top_x0 < max(18.0, width * 0.026):
            continue
        for bottom in horizontal[top_index + 1 :]:
            y0, y1 = sorted((top.y1, bottom.y1))
            box_h = y1 - y0
            if box_h < y_min or box_h > y_max:
                continue
            bottom_x0, bottom_x1 = sorted((bottom.x1, bottom.x2))
            overlap = min(top_x1, bottom_x1) - max(top_x0, bottom_x0)
            if overlap < min(top_x1 - top_x0, bottom_x1 - bottom_x0) * 0.48:
                continue
            box_x0 = min(top_x0, bottom_x0)
            box_x1 = max(top_x1, bottom_x1)
            box_w = box_x1 - box_x0
            if not _rectangle_size_ok(box_w, box_h, width, height):
                continue

            supporting_verticals = []
            for vert in vertical:
                vx = vert.x1
                vy0, vy1 = sorted((vert.y1, vert.y2))
                if vx < box_x0 - side_tolerance or vx > box_x1 + side_tolerance:
                    continue
                if vy0 > y0 + vertical_tolerance or vy1 < y1 - vertical_tolerance:
                    continue
                supporting_verticals.append(vert)
            distinct_xs = _distinct_values([item.x1 for item in supporting_verticals], tolerance=side_tolerance)
            if len(distinct_xs) < 2:
                continue

            left_x = min(distinct_xs)
            right_x = max(distinct_xs)
            if right_x - left_x >= box_w * 0.52:
                box_x0 = min(box_x0, left_x)
                box_x1 = max(box_x1, right_x)
                box_w = box_x1 - box_x0
            pad = max(2, int(min(width, height) * 0.003))
            crop = NormalizedBox(
                (box_x0 - pad) / width,
                (y0 - pad) / height,
                (box_w + 2 * pad) / width,
                (box_h + 2 * pad) / height,
            ).clipped()
            line_cols = len(distinct_xs)
            kind = "feature_control_frame" if line_cols >= 3 and box_w / max(box_h, 1.0) >= 2.0 else "boxed_annotation"
            confidence = min(
                0.92,
                0.33
                + min(line_cols / 5.0, 1.0) * 0.27
                + min((box_w / max(box_h, 1.0) - 0.8) / 4.0, 1.0) * 0.2
                + min(overlap / max(box_w, 1.0), 1.0) * 0.12,
            )
            rectangles.append(
                VectorRectangle(
                    crop=crop,
                    confidence=round(confidence, 3),
                    kind=kind,
                    line_rows=2,
                    line_cols=line_cols,
                    source="merged_segment_rectangles_v1",
                )
            )
    return rectangles


def _rectangle_size_ok(w: float, h: float, image_w: int, image_h: int) -> bool:
    if h < max(9, image_h * 0.014) or h > image_h * 0.16:
        return False
    if w < max(18, image_w * 0.026) or w > image_w * 0.42:
        return False
    return w / max(h, 1) >= 0.75


def _dedupe_rectangles(rectangles: list[VectorRectangle]) -> tuple[VectorRectangle, ...]:
    kept: list[VectorRectangle] = []
    for rectangle in sorted(rectangles, key=lambda item: item.confidence, reverse=True):
        if any(_box_iou(rectangle.crop, other.crop) > 0.52 for other in kept):
            continue
        kept.append(rectangle)
    return tuple(sorted(kept, key=lambda item: (item.crop.y, item.crop.x)))


def _distinct_values(values: Iterable[float], *, tolerance: float) -> list[float]:
    distinct: list[float] = []
    for value in sorted(values):
        if not distinct or abs(value - distinct[-1]) > tolerance:
            distinct.append(value)
    return distinct


def _box_iou(a: NormalizedBox, b: NormalizedBox) -> float:
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x + b.w, b.y + b.h
    ix = max(0.0, min(ax1, bx1) - max(a.x, b.x))
    iy = max(0.0, min(ay1, by1) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-6)
