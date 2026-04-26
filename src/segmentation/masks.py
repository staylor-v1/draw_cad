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

    protected_targets = [
        {
            "box": NormalizedBox(**item["crop"]),
            "axis": item.get("axis", "unassigned"),
            "label": item.get("label", ""),
            "segmentation_mode": item.get("segmentationMode", ""),
            "confidence": item.get("confidence", 0.0),
        }
        for item in protected_regions or []
        if item.get("crop") and _is_reliable_projection_target(item)
    ]
    protected_boxes = [target["box"] for target in protected_targets]
    try:
        vector = raster_to_vector(image_path)
    except (OSError, ValueError, cv2.error):
        return _annotation_crop_masks(annotations, protected_boxes, ())

    thread_zones = _detect_thread_like_zones(annotations, vector.segments, vector.width, vector.height)
    masks = _annotation_crop_masks(annotations, protected_boxes, thread_zones)
    annotation_targets = [
        {
            "box": NormalizedBox(**item["crop"]),
            "source": item.get("source") or "detected_annotation",
            "label": item.get("label") or item.get("kind") or item.get("id") or "annotation",
        }
        for item in annotations
        if item.get("crop")
        and not _box_center_in_any_region(NormalizedBox(**item["crop"]), protected_boxes, item)
        and not _box_overlaps_any(NormalizedBox(**item["crop"]), thread_zones, min_fraction=0.45)
    ]
    segment_masks = _segment_masks_near_annotations(
        vector.segments,
        annotation_targets,
        protected_targets,
        thread_zones,
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


def _annotation_crop_masks(
    annotations: list[dict],
    protected_boxes: list[NormalizedBox],
    thread_zones: tuple[NormalizedBox, ...],
) -> list[dict]:
    masks = []
    for item in annotations:
        if not item.get("crop"):
            continue
        box = NormalizedBox(**item["crop"]).clipped()
        if _box_center_in_any_region(box, protected_boxes, item):
            continue
        if item.get("source") != "teacher_fixture" and _box_overlaps_any(box, thread_zones, min_fraction=0.45):
            continue
        if _should_skip_full_annotation_crop(item, box, protected_boxes):
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
    protected_targets: list[dict],
    thread_zones: tuple[NormalizedBox, ...],
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
        if _segment_midpoint_in_any_box(segment, list(thread_zones), width, height):
            continue
        touched_targets = [
            target for target in annotation_targets if _segment_touches_box(segment, target["box"], width, height)
        ]
        if not touched_targets:
            continue
        touches_teacher = any(target["source"] == "teacher_fixture" for target in touched_targets)
        protected_boxes = [target["box"] for target in protected_targets]
        midpoint_in_protected = _segment_midpoint_in_any_box(segment, protected_boxes, width, height)
        if (
            segment.orientation in {"horizontal", "vertical"}
            and midpoint_in_protected
            and not (touches_teacher and _segment_near_protected_edge(segment, protected_boxes, width, height))
        ):
            continue
        crop = _segment_to_box(segment, width, height)
        if _mask_intrudes_into_narrow_projection(crop, protected_targets):
            continue
        masks.append(
            {
                "label": f"vector annotation line {index + 1}",
                "crop": crop.to_dict(),
                "shape": "line",
                "line": _segment_to_line(segment, width, height),
                "source": "vector_annotation_line",
                "orientation": segment.orientation,
            }
        )
    return masks


def _detect_thread_like_zones(
    annotations: list[dict],
    segments: tuple[VectorSegment, ...],
    width: int,
    height: int,
) -> tuple[NormalizedBox, ...]:
    zones: list[NormalizedBox] = []
    for item in annotations:
        if item.get("source") == "teacher_fixture" or not item.get("crop"):
            continue
        box = NormalizedBox(**item["crop"]).clipped()
        if _thread_cluster_score(box, segments, width, height) < 0.85:
            continue
        zones.append(_expand_box_xy(box, 0.01, 0.045).clipped())
    return _merge_zones(zones)


def _thread_cluster_score(
    box: NormalizedBox,
    segments: tuple[VectorSegment, ...],
    width: int,
    height: int,
) -> float:
    in_box = [
        segment
        for segment in segments
        if segment.orientation in {"diagonal", "horizontal"}
        and _segment_midpoint_in_box(segment, box, width, height)
        and min(width, height) * 0.012 <= segment.length <= max(width, height) * 0.24
    ]
    if len(in_box) < 8:
        return 0.0

    diagonal = [segment for segment in in_box if segment.orientation == "diagonal"]
    horizontal = [segment for segment in in_box if segment.orientation == "horizontal"]
    diagonal_cluster = _largest_angle_cluster(diagonal)
    horizontal_rows = _distinct_segment_coordinates(horizontal, axis="y", width=width, height=height)

    score = 0.0
    if diagonal_cluster >= 3:
        score += 0.6
    if horizontal_rows >= 5:
        score += 0.5
    if len(in_box) >= 14:
        score += 0.25
    if box.w >= 0.09 and box.h >= 0.05:
        score += 0.15
    return score


def _largest_angle_cluster(segments: list[VectorSegment], tolerance: float = 12.0) -> int:
    if not segments:
        return 0
    angles = sorted(segment.angle for segment in segments)
    largest = 1
    for index, angle in enumerate(angles):
        largest = max(largest, sum(1 for other in angles[index:] if abs(other - angle) <= tolerance))
    return largest


def _distinct_segment_coordinates(
    segments: list[VectorSegment],
    *,
    axis: str,
    width: int,
    height: int,
) -> int:
    if not segments:
        return 0
    tolerance = max(0.006, 4 / max(width, height))
    values = []
    for segment in segments:
        if axis == "y":
            values.append(((segment.y1 + segment.y2) / 2.0) / height)
        else:
            values.append(((segment.x1 + segment.x2) / 2.0) / width)
    groups: list[float] = []
    for value in sorted(values):
        if not groups or abs(value - groups[-1]) > tolerance:
            groups.append(value)
    return len(groups)


def _segment_midpoint_in_box(segment: VectorSegment, box: NormalizedBox, width: int, height: int) -> bool:
    midpoint_x = ((segment.x1 + segment.x2) / 2.0) / width
    midpoint_y = ((segment.y1 + segment.y2) / 2.0) / height
    return _point_in_box(midpoint_x, midpoint_y, box)


def _expand_box(box: NormalizedBox, pad: float) -> NormalizedBox:
    return NormalizedBox(box.x - pad, box.y - pad, box.w + 2 * pad, box.h + 2 * pad)


def _expand_box_xy(box: NormalizedBox, pad_x: float, pad_y: float) -> NormalizedBox:
    return NormalizedBox(box.x - pad_x, box.y - pad_y, box.w + 2 * pad_x, box.h + 2 * pad_y)


def _merge_zones(zones: list[NormalizedBox]) -> tuple[NormalizedBox, ...]:
    merged: list[NormalizedBox] = []
    for zone in zones:
        current = zone
        kept = []
        for other in merged:
            if _box_overlap_fraction(current, other) > 0.05:
                current = NormalizedBox(
                    min(current.x, other.x),
                    min(current.y, other.y),
                    max(current.x + current.w, other.x + other.w) - min(current.x, other.x),
                    max(current.y + current.h, other.y + other.h) - min(current.y, other.y),
                ).clipped()
            else:
                kept.append(other)
        kept.append(current)
        merged = kept
    return tuple(merged)


def _box_overlaps_any(box: NormalizedBox, zones: tuple[NormalizedBox, ...], *, min_fraction: float) -> bool:
    return any(_box_overlap_fraction(box, zone) >= min_fraction for zone in zones)


def _box_overlap_fraction(a: NormalizedBox, b: NormalizedBox) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    if ix <= 0 or iy <= 0:
        return 0.0
    return (ix * iy) / max(min(a.w * a.h, b.w * b.h), 1e-6)


def _should_skip_full_annotation_crop(item: dict, box: NormalizedBox, protected_boxes: list[NormalizedBox]) -> bool:
    """Avoid filled masks for broad teacher crops that pass through part geometry."""

    if item.get("source") != "teacher_fixture":
        return False
    if item.get("kind") != "section_marker":
        return False
    if box.h < 0.18 and box.w < 0.18:
        return False
    return any(_box_overlap_fraction(box, protected_box) >= 0.45 for protected_box in protected_boxes)


def _mask_intrudes_into_narrow_projection(mask: NormalizedBox, protected_targets: list[dict]) -> bool:
    for target in protected_targets:
        box = target["box"]
        if target.get("axis") != "side" and box.w > 0.11:
            continue
        ix = max(0.0, min(mask.x + mask.w, box.x + box.w) - max(mask.x, box.x))
        iy = max(0.0, min(mask.y + mask.h, box.y + box.h) - max(mask.y, box.y))
        if ix <= 0 or iy <= 0:
            continue
        relative_y_end = (min(mask.y + mask.h, box.y + box.h) - box.y) / max(box.h, 1e-6)
        width_fraction = ix / max(box.w, 1e-6)
        if relative_y_end > 0.02 and width_fraction > 0.10:
            return True
    return False


def _is_reliable_projection_target(item: dict) -> bool:
    """Return True when a projection crop is reliable enough to suppress masks.

    Broad, unassigned detector candidates often include annotation fields. Using
    them as protected regions prevents obvious callouts from being masked.
    """

    axis = item.get("axis", "unassigned")
    if axis and axis != "unassigned":
        return True
    if item.get("segmentationMode") == "teacher_fixture":
        return True
    return float(item.get("confidence") or 0.0) >= 0.75


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


def _segment_to_line(segment: VectorSegment, width: int, height: int) -> dict:
    stroke = max(3, int(min(width, height) * 0.004))
    return {
        "x1": segment.x1 / width,
        "y1": segment.y1 / height,
        "x2": segment.x2 / width,
        "y2": segment.y2 / height,
        "width": stroke / min(width, height),
    }


def _point_in_box(x: float, y: float, box: NormalizedBox) -> bool:
    return box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h


def _crop_key(crop: dict) -> tuple[int, int, int, int]:
    return tuple(int(round(crop[key] * 1000)) for key in ("x", "y", "w", "h"))
