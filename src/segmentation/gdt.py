"""GD&T and annotation callout detection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.segmentation.title_block import NormalizedBox


@dataclass(frozen=True)
class GdtCallout:
    id: str
    crop: NormalizedBox
    confidence: float
    kind: str
    symbol: str
    value: str
    line_rows: int
    line_cols: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["crop"] = self.crop.to_dict()
        data["notes"] = list(self.notes)
        return data


def detect_gdt_callouts(image_path: str | Path) -> list[dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    return [callout.to_dict() for callout in detect_gdt_callouts_from_array(image)]


def detect_gdt_callouts_from_array(gray: np.ndarray) -> list[GdtCallout]:
    """Detect ruled GD&T feature-control-frame-like callouts.

    This is a first-pass teacher detector. It favors high recall for framed
    annotations because missed callouts contaminate projection reconstruction.
    """

    height, width = gray.shape
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 55), 1)),
    )
    vertical = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height // 55))),
    )
    lines = cv2.bitwise_or(horizontal, vertical)
    closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[GdtCallout] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not _candidate_size_ok(w, h, width, height):
            continue
        roi_lines = lines[y : y + h, x : x + w]
        roi_horizontal = horizontal[y : y + h, x : x + w]
        roi_vertical = vertical[y : y + h, x : x + w]
        line_rows = int((roi_horizontal.sum(axis=1) > 255 * w * 0.25).sum())
        line_cols = int((roi_vertical.sum(axis=0) > 255 * h * 0.25).sum())
        if line_rows < 2 or line_cols < 2:
            continue
        kind = "feature_control_frame" if line_cols >= 3 else "boxed_annotation_candidate"
        score = _score_callout(w, h, line_rows, line_cols, float(roi_lines.mean() / 255.0), kind)
        if score < 0.35:
            continue
        pad = max(2, int(min(width, height) * 0.004))
        box = NormalizedBox(
            max(0, x - pad) / width,
            max(0, y - pad) / height,
            min(width, w + 2 * pad) / width,
            min(height, h + 2 * pad) / height,
        ).clipped()
        candidates.append(
            GdtCallout(
                id="",
                crop=box,
                confidence=round(score, 3),
                kind=kind,
                symbol="unclassified",
                value="",
                line_rows=line_rows,
                line_cols=line_cols,
                notes=(
                    "ruled callout detected by line morphology",
                    "symbol classification is pending YOLO/Florence/Gemma review",
                ),
            )
        )

    merged = _dedupe_callouts(candidates)
    ordered = sorted(merged, key=lambda item: (item.crop.y, item.crop.x))
    return [
        GdtCallout(
            id=f"gdt-{index + 1}",
            crop=item.crop,
            confidence=item.confidence,
            kind=item.kind,
            symbol=item.symbol,
            value=item.value,
            line_rows=item.line_rows,
            line_cols=item.line_cols,
            notes=item.notes,
        )
        for index, item in enumerate(ordered)
    ]


def make_gdt_masked_projection(image_path: str | Path, projection: dict, callouts: list[dict]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    projection_box = NormalizedBox(**projection["crop"])
    left, top, right, bottom = projection_box.to_pixels(width, height)
    crop = image.crop((left, top, right, bottom))
    crop_width, crop_height = crop.size

    mask_color = (255, 255, 255)
    pixels = crop.load()
    for callout in callouts:
        callout_box = NormalizedBox(**callout["crop"])
        cx0, cy0, cx1, cy1 = callout_box.to_pixels(width, height)
        ix0 = max(left, cx0) - left
        iy0 = max(top, cy0) - top
        ix1 = min(right, cx1) - left
        iy1 = min(bottom, cy1) - top
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        for y in range(max(0, iy0), min(crop_height, iy1)):
            for x in range(max(0, ix0), min(crop_width, ix1)):
                pixels[x, y] = mask_color
    return crop


def _candidate_size_ok(w: int, h: int, image_w: int, image_h: int) -> bool:
    if h < max(10, image_h * 0.018) or h > image_h * 0.12:
        return False
    if w < max(25, image_w * 0.035) or w > image_w * 0.36:
        return False
    return w / max(h, 1) >= 1.5


def _score_callout(w: int, h: int, line_rows: int, line_cols: int, density: float, kind: str) -> float:
    aspect_score = min(max((w / max(h, 1) - 1.4) / 3.5, 0.0), 1.0)
    row_score = min(line_rows / 3.0, 1.0)
    col_score = min(line_cols / 5.0, 1.0)
    density_score = 1.0 if 0.05 <= density <= 0.35 else 0.55
    kind_bonus = 0.08 if kind == "feature_control_frame" else 0.0
    return min(0.96, 0.2 + aspect_score * 0.22 + row_score * 0.22 + col_score * 0.22 + density_score * 0.14 + kind_bonus)


def _dedupe_callouts(callouts: list[GdtCallout]) -> list[GdtCallout]:
    kept: list[GdtCallout] = []
    for candidate in sorted(callouts, key=lambda item: item.confidence, reverse=True):
        if any(_box_iou(candidate.crop, other.crop) > 0.45 for other in kept):
            continue
        kept.append(candidate)
    return kept


def _box_iou(a: NormalizedBox, b: NormalizedBox) -> float:
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x + b.w, b.y + b.h
    ix = max(0.0, min(ax1, bx1) - max(a.x, b.x))
    iy = max(0.0, min(ay1, by1) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-6)
