"""Annotation masking helpers for projection cleanup."""

from __future__ import annotations

from pathlib import Path

import cv2

from src.segmentation.title_block import NormalizedBox
from src.vectorization.raster_to_dxf import VectorSegment, raster_to_vector


PROJECTION_EDGE_FRACTION = 0.11


def build_annotation_masks(
    image_path: str | Path,
    annotations: list[dict],
    protected_regions: list[dict] | None = None,
) -> list[dict]:
    """Build rectangular masks for annotation crops and nearby vector linework."""

    protected_boxes = [NormalizedBox(**item["crop"]) for item in protected_regions or [] if item.get("crop")]
    masks = _annotation_crop_masks(annotations, protected_boxes)
    try:
        vector = raster_to_vector(image_path)
    except (OSError, ValueError, cv2.error):
        return masks

    annotation_targets = [
        {
            "box": NormalizedBox(**item["crop"]),
            "source": item.get("source") or "detected_annotation",
            "label": item.get("label") or item.get("kind") or item.get("id") or "annotation",
        }
        for item in annotations
        if item.get("crop") and not _box_center_in_any_region(NormalizedBox(**item["crop"]), protected_boxes, item)
    ]
    segment_masks = _segment_masks_near_annotations(
        vector.segments,
        annotation_targets,
        protected_boxes,
        vector.width,
        vector.height,
    )
    seen = {_crop_key(item["crop"]) for item in masks}
    for mask in segment_masks:
        key = _crop_key(mask["crop"])
        if key in seen:
            continue
        masks.append(mask)
        seen.add(key)
    return masks


def _annotation_crop_masks(annotations: list[dict], protected_boxes: list[NormalizedBox]) -> list[dict]:
    masks = []
    for item in annotations:
        if not item.get("crop"):
            continue
        box = NormalizedBox(**item["crop"]).clipped()
        if _box_center_in_any_region(box, protected_boxes, item):
            continue
        masks.append(
            {
                "label": item.get("label") or item.get("kind") or item.get("id") or "annotation",
                "crop": box.to_dict(),
                "source": item.get("source") or "annotation_crop",
            }
        )
    return masks


def _segment_masks_near_annotations(
    segments: tuple[VectorSegment, ...],
    annotation_targets: list[dict],
    protected_boxes: list[NormalizedBox],
    width: int,
    height: int,
) -> list[dict]:
    if not annotation_targets:
        return []
    max_len = max(width, height) * 0.36
    min_len = min(width, height) * 0.02
    masks = []
    for index, segment in enumerate(segments):
        if segment.length < min_len or segment.length > max_len:
            continue
        if segment.orientation == "diagonal" and segment.length > max_len * 0.82:
            continue
        touched_targets = [
            target for target in annotation_targets if _segment_touches_box(segment, target["box"], width, height)
        ]
        if not touched_targets:
            continue
        touches_teacher = any(target["source"] == "teacher_fixture" for target in touched_targets)
        midpoint_in_protected = _segment_midpoint_in_any_box(segment, protected_boxes, width, height)
        if (
            segment.orientation in {"horizontal", "vertical"}
            and midpoint_in_protected
            and not (touches_teacher and _segment_near_protected_edge(segment, protected_boxes, width, height))
        ):
            continue
        crop = _segment_to_box(segment, width, height)
        masks.append(
            {
                "label": f"vector annotation line {index + 1}",
                "crop": crop.to_dict(),
                "source": "vector_annotation_line",
                "orientation": segment.orientation,
            }
        )
    return masks


def _box_center_in_any_region(box: NormalizedBox, regions: list[NormalizedBox], item: dict) -> bool:
    if item.get("source") == "teacher_fixture":
        return False
    center_x = box.x + box.w / 2.0
    center_y = box.y + box.h / 2.0
    return any(_point_in_box(center_x, center_y, region) for region in regions)


def _segment_touches_box(segment: VectorSegment, box: NormalizedBox, width: int, height: int) -> bool:
    margin_x = max(8, int(width * 0.018)) / width
    margin_y = max(8, int(height * 0.018)) / height
    expanded = NormalizedBox(box.x - margin_x, box.y - margin_y, box.w + 2 * margin_x, box.h + 2 * margin_y).clipped()
    sx0 = min(segment.x1, segment.x2) / width
    sx1 = max(segment.x1, segment.x2) / width
    sy0 = min(segment.y1, segment.y2) / height
    sy1 = max(segment.y1, segment.y2) / height
    endpoint_hit = (
        _point_in_box(segment.x1 / width, segment.y1 / height, expanded)
        or _point_in_box(segment.x2 / width, segment.y2 / height, expanded)
    )
    if segment.orientation in {"horizontal", "vertical"}:
        return endpoint_hit
    overlap_hit = sx1 >= expanded.x and sx0 <= expanded.x + expanded.w and sy1 >= expanded.y and sy0 <= expanded.y + expanded.h
    return endpoint_hit or overlap_hit


def _segment_midpoint_in_any_box(
    segment: VectorSegment,
    boxes: list[NormalizedBox],
    width: int,
    height: int,
) -> bool:
    midpoint_x = ((segment.x1 + segment.x2) / 2.0) / width
    midpoint_y = ((segment.y1 + segment.y2) / 2.0) / height
    return any(_point_in_box(midpoint_x, midpoint_y, box) for box in boxes)


def _segment_near_protected_edge(
    segment: VectorSegment,
    boxes: list[NormalizedBox],
    width: int,
    height: int,
) -> bool:
    for box in boxes:
        if segment.orientation == "vertical":
            x = segment.x1 / width
            near_left = abs(x - box.x) <= max(0.012, box.w * PROJECTION_EDGE_FRACTION)
            near_right = abs(x - (box.x + box.w)) <= max(0.012, box.w * PROJECTION_EDGE_FRACTION)
            if near_left or near_right:
                return True
        if segment.orientation == "horizontal":
            y = segment.y1 / height
            near_top = abs(y - box.y) <= max(0.012, box.h * PROJECTION_EDGE_FRACTION)
            near_bottom = abs(y - (box.y + box.h)) <= max(0.012, box.h * PROJECTION_EDGE_FRACTION)
            if near_top or near_bottom:
                return True
    return False


def _segment_to_box(segment: VectorSegment, width: int, height: int) -> NormalizedBox:
    pad = max(3, int(min(width, height) * 0.006))
    x0 = min(segment.x1, segment.x2) - pad
    x1 = max(segment.x1, segment.x2) + pad
    y0 = min(segment.y1, segment.y2) - pad
    y1 = max(segment.y1, segment.y2) + pad
    if segment.orientation == "horizontal":
        y0 -= pad
        y1 += pad
    elif segment.orientation == "vertical":
        x0 -= pad
        x1 += pad
    return NormalizedBox(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height).clipped()


def _point_in_box(x: float, y: float, box: NormalizedBox) -> bool:
    return box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h


def _crop_key(crop: dict) -> tuple[int, int, int, int]:
    return tuple(int(round(crop[key] * 1000)) for key in ("x", "y", "w", "h"))
