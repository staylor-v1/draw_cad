"""Evaluate GD&T mark masks plus grounded leader/dimension line masks.

This is a harness for the hybrid operating model:

1. use a box detector or teacher fixture to identify the GD&T/dimension mark;
2. use vector evidence to trace thin leader/dimension lines attached to it;
3. produce per-pixel line masks instead of broad rectangular masks.

The harness is deliberately model-agnostic. SAM2/RF-DETR/YOLO-seg can be
plugged in once we have polygon masks, while this script gives us a strong
teacher baseline and scoring report using the labelled drawings we already own.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.callouts import load_callout_fixture, load_non_callout_fixture, load_projection_fixture
from src.segmentation.title_block import NormalizedBox
from src.segmentation.vision_callouts import detect_vision_callouts
from src.vectorization.raster_to_dxf import VectorSegment, raster_to_vector


DEFAULT_IMAGES = [
    Path("training_data/gdt/simple1.webp"),
    Path("training_data/gdt/threaded_cap.jpg"),
    Path("training_data/gdt_external/rendered/nist_pmi/nist_pmi_ctc_01-1.png"),
    Path("training_data/gdt_external/rendered/nist_am/nist_am_test_artifact_engineering_drawing-1.png"),
]


@dataclass(frozen=True)
class GroundedSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    crop: dict[str, float]
    confidence: float
    orientation: str
    terminal_cue: str
    terminal: dict[str, float]
    source: str


@dataclass(frozen=True)
class GroundedCallout:
    id: str
    kind: str
    label: str
    crop: dict[str, float]
    confidence: float
    source: str
    segments: list[GroundedSegment]
    mask_pixels: int
    bounding_box_pixels: int
    line_box_pixels: int
    notes: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-pixel GD&T callout grounding masks.")
    parser.add_argument("--output-dir", default="experiments/gdt_mask_grounding")
    parser.add_argument("--image", action="append", help="Image to evaluate. Defaults to core GD&T examples plus NIST samples.")
    parser.add_argument("--max-lines-per-callout", type=int, default=3)
    parser.add_argument("--vision-conf", type=float, default=0.45)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [Path(item) for item in args.image] if args.image else [path for path in DEFAULT_IMAGES if path.exists()]

    records = []
    for image_path in image_paths:
        records.append(evaluate_image(image_path, output_dir, args.max_lines_per_callout, args.vision_conf))

    summary = {
        "approaches": approach_matrix(),
        "images": records,
        "totals": summarize(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


def evaluate_image(image_path: Path, output_dir: Path, max_lines: int, vision_conf: float) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    teacher_annotations = load_callout_fixture(image_path)
    if teacher_annotations:
        annotations = teacher_annotations
        annotation_source = "teacher_fixture"
    else:
        annotations = [
            item
            for item in detect_vision_callouts(image_path, conf=vision_conf)
            if item.get("source") == "fine_tuned_yolo" and item.get("className") != "non_callout_thread_texture"
        ]
        annotation_source = "fine_tuned_yolo"

    projections = load_projection_fixture(image_path)
    non_callouts = [NormalizedBox(**item["crop"]) for item in load_non_callout_fixture(image_path) if item.get("crop")]
    vector = raster_to_vector(image_path)
    grounded = [
        ground_annotation(
            item,
            vector.segments,
            width,
            height,
            projections,
            non_callouts,
            max_lines=max_lines,
        )
        for item in annotations
        if item.get("crop")
    ]

    overlay_path = output_dir / f"{image_path.stem}_grounding_overlay.png"
    mask_path = output_dir / f"{image_path.stem}_line_mask.png"
    draw_overlay(image, grounded, overlay_path, mask_path)

    grounded_count = sum(1 for item in grounded if item.segments)
    mask_pixels = sum(item.mask_pixels for item in grounded)
    box_pixels = sum(item.bounding_box_pixels for item in grounded)
    line_box_pixels = sum(item.line_box_pixels for item in grounded)
    return {
        "image": str(image_path),
        "annotationSource": annotation_source,
        "annotations": len(grounded),
        "grounded": grounded_count,
        "groundingRecall": round(grounded_count / max(len(grounded), 1), 3),
        "segments": sum(len(item.segments) for item in grounded),
        "markBoxPixels": box_pixels,
        "thinLineMaskPixels": mask_pixels,
        "lineBoundingBoxPixels": line_box_pixels,
        "thinMaskVsLineBox": round(mask_pixels / max(line_box_pixels, 1), 4),
        "thinMaskVsMarkBoxes": round(mask_pixels / max(box_pixels, 1), 4),
        "overlay": str(overlay_path),
        "lineMask": str(mask_path),
        "callouts": [serialize_grounded(item) for item in grounded],
    }


def ground_annotation(
    item: dict,
    segments: tuple[VectorSegment, ...],
    width: int,
    height: int,
    projections: list[dict],
    non_callouts: list[NormalizedBox],
    *,
    max_lines: int,
) -> GroundedCallout:
    box = NormalizedBox(**item["crop"]).clipped()
    target_projection = projection_for_item(item, projections)
    candidates = []
    for segment in segments:
        if not segment_length_ok(segment, width, height):
            continue
        if segment_midpoint_in_regions(segment, non_callouts, width, height):
            continue
        if not segment_touches_box(segment, box, width, height):
            continue
        score, terminal, terminal_cue, notes = score_segment(segment, box, target_projection, projections, width, height)
        if score <= 0.0:
            continue
        candidates.append((score, segment, terminal, terminal_cue, notes))

    selected: list[tuple[float, VectorSegment, dict[str, float], str, list[str]]] = []
    for candidate in sorted(candidates, key=lambda row: row[0], reverse=True):
        if len(selected) >= max_lines:
            break
        if any(segments_redundant(candidate[1], chosen[1], width, height) for chosen in selected):
            continue
        selected.append(candidate)

    grounded_segments = [
        segment_to_grounded(segment, width, height, score, terminal, terminal_cue)
        for score, segment, terminal, terminal_cue, _notes in selected
    ]
    mask = segments_to_mask(grounded_segments, width, height)
    line_box_pixels = sum(line_box_pixel_area(item.crop, width, height) for item in grounded_segments)
    notes = sorted({note for *_prefix, notes in selected for note in notes})
    if not grounded_segments:
        notes.append("no attached vector line found; candidate may need SAM/Gemma or manual point prompt")

    left, top, right, bottom = box.to_pixels(width, height)
    return GroundedCallout(
        id=item.get("id", item.get("className", "callout")),
        kind=item.get("kind", item.get("className", "callout")),
        label=item.get("label", item.get("value", item.get("className", "callout"))),
        crop=box.to_dict(),
        confidence=float(item.get("confidence", 1.0)),
        source=item.get("source", "unknown"),
        segments=grounded_segments,
        mask_pixels=int(mask.sum()),
        bounding_box_pixels=max(1, (right - left) * (bottom - top)),
        line_box_pixels=line_box_pixels,
        notes=notes,
    )


def segment_length_ok(segment: VectorSegment, width: int, height: int) -> bool:
    shortest = min(width, height)
    longest = max(width, height)
    if segment.length < max(8.0, shortest * 0.012):
        return False
    if segment.length > longest * 0.38:
        return False
    return True


def score_segment(
    segment: VectorSegment,
    box: NormalizedBox,
    target_projection: NormalizedBox | None,
    projections: list[dict],
    width: int,
    height: int,
) -> tuple[float, dict[str, float], str, list[str]]:
    endpoints = [
        {"x": segment.x1 / width, "y": segment.y1 / height},
        {"x": segment.x2 / width, "y": segment.y2 / height},
    ]
    expanded_mark = expand_box_pixels(box, width, height, px=6)
    if all(point_in_box(point, expanded_mark) for point in endpoints):
        return 0.0, endpoints[0], "none", ["segment is internal to the mark box"]
    distances = [distance_to_box(point, box) for point in endpoints]
    near_index = 0 if distances[0] <= distances[1] else 1
    far_point = endpoints[1 - near_index]
    if distances[1 - near_index] <= max(0.012, 6 / max(width, height)):
        return 0.0, far_point, "none", ["segment does not leave the mark box"]
    terminal_score = 0.0
    notes: list[str] = []

    if target_projection and point_in_expanded_box(far_point, target_projection, 0.025, 0.025):
        terminal_score += 0.45
        notes.append("terminal reaches expected projection")
    elif any(point_in_expanded_box(far_point, NormalizedBox(**item["crop"]), 0.02, 0.02) for item in projections if item.get("crop")):
        terminal_score += 0.25
        notes.append("terminal reaches a projection")

    if distances[near_index] <= max(0.018, 8 / max(width, height)):
        terminal_score += 0.25
        notes.append("segment endpoint is attached to mark box")
    else:
        terminal_score += 0.12
        notes.append("segment overlaps mark box")

    terminal_cue = "none"
    if segment.orientation == "diagonal":
        terminal_score += 0.12
        terminal_cue = "leader_or_arrow_candidate"
    if segment.orientation in {"horizontal", "vertical"} and target_projection:
        terminal_score += 0.08
        terminal_cue = "extension_or_dimension_line_candidate"

    return min(0.98, terminal_score), far_point, terminal_cue, notes


def segment_to_grounded(
    segment: VectorSegment,
    width: int,
    height: int,
    score: float,
    terminal: dict[str, float],
    terminal_cue: str,
) -> GroundedSegment:
    pad = max(2, int(min(width, height) * 0.003))
    x0 = max(0, min(segment.x1, segment.x2) - pad)
    y0 = max(0, min(segment.y1, segment.y2) - pad)
    x1 = min(width, max(segment.x1, segment.x2) + pad)
    y1 = min(height, max(segment.y1, segment.y2) + pad)
    return GroundedSegment(
        x1=round(segment.x1 / width, 5),
        y1=round(segment.y1 / height, 5),
        x2=round(segment.x2 / width, 5),
        y2=round(segment.y2 / height, 5),
        crop=NormalizedBox(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height).clipped().to_dict(),
        confidence=round(score, 3),
        orientation=segment.orientation,
        terminal_cue=terminal_cue,
        terminal={"x": round(terminal["x"], 5), "y": round(terminal["y"], 5)},
        source="vector_grounded_line_v1",
    )


def projection_for_item(item: dict, projections: list[dict]) -> NormalizedBox | None:
    view = item.get("view")
    if not view:
        return None
    for projection in projections:
        if projection.get("id") == view or projection.get("axis") == view:
            return NormalizedBox(**projection["crop"])
    return None


def segment_touches_box(segment: VectorSegment, box: NormalizedBox, width: int, height: int) -> bool:
    expanded = expand_box_pixels(box, width, height, px=14)
    points = [
        {"x": segment.x1 / width, "y": segment.y1 / height},
        {"x": segment.x2 / width, "y": segment.y2 / height},
        {"x": ((segment.x1 + segment.x2) / 2) / width, "y": ((segment.y1 + segment.y2) / 2) / height},
    ]
    if any(point_in_box(point, expanded) for point in points):
        return True
    sx0, sx1 = sorted((segment.x1 / width, segment.x2 / width))
    sy0, sy1 = sorted((segment.y1 / height, segment.y2 / height))
    return sx1 >= expanded.x and sx0 <= expanded.x + expanded.w and sy1 >= expanded.y and sy0 <= expanded.y + expanded.h


def segment_midpoint_in_regions(segment: VectorSegment, regions: list[NormalizedBox], width: int, height: int) -> bool:
    point = {"x": ((segment.x1 + segment.x2) / 2) / width, "y": ((segment.y1 + segment.y2) / 2) / height}
    return any(point_in_box(point, region) for region in regions)


def segments_redundant(a: VectorSegment, b: VectorSegment, width: int, height: int) -> bool:
    if a.orientation != b.orientation:
        return False
    ac = (((a.x1 + a.x2) / 2) / width, ((a.y1 + a.y2) / 2) / height)
    bc = (((b.x1 + b.x2) / 2) / width, ((b.y1 + b.y2) / 2) / height)
    return math.dist(ac, bc) < 0.018


def segments_to_mask(segments: list[GroundedSegment], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(3, int(min(width, height) * 0.004))
    for segment in segments:
        p1 = (int(round(segment.x1 * width)), int(round(segment.y1 * height)))
        p2 = (int(round(segment.x2 * width)), int(round(segment.y2 * height)))
        cv2.line(mask, p1, p2, 1, thickness=thickness, lineType=cv2.LINE_AA)
    return mask > 0


def line_box_pixel_area(crop: dict[str, float], width: int, height: int) -> int:
    left, top, right, bottom = NormalizedBox(**crop).to_pixels(width, height)
    return max(1, (right - left) * (bottom - top))


def draw_overlay(image: Image.Image, callouts: list[GroundedCallout], overlay_path: Path, mask_path: Path) -> None:
    width, height = image.size
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for callout in callouts:
        box = NormalizedBox(**callout.crop)
        left, top, right, bottom = box.to_pixels(width, height)
        draw.rectangle((left, top, right, bottom), outline=(255, 150, 0, 230), width=3)
        for segment in callout.segments:
            p1 = (int(round(segment.x1 * width)), int(round(segment.y1 * height)))
            p2 = (int(round(segment.x2 * width)), int(round(segment.y2 * height)))
            draw.line((p1, p2), fill=(0, 190, 255, 220), width=max(3, int(min(width, height) * 0.004)))
            terminal = (int(round(segment.terminal["x"] * width)), int(round(segment.terminal["y"] * height)))
            radius = max(4, int(min(width, height) * 0.006))
            draw.ellipse((terminal[0] - radius, terminal[1] - radius, terminal[0] + radius, terminal[1] + radius), fill=(210, 35, 210, 190))
        combined_mask |= segments_to_mask(callout.segments, width, height).astype(np.uint8)
    overlay.thumbnail((1400, 1000))
    overlay.save(overlay_path)
    Image.fromarray((combined_mask * 255).astype(np.uint8)).save(mask_path)


def distance_to_box(point: dict[str, float], box: NormalizedBox) -> float:
    dx = max(box.x - point["x"], 0.0, point["x"] - (box.x + box.w))
    dy = max(box.y - point["y"], 0.0, point["y"] - (box.y + box.h))
    return math.hypot(dx, dy)


def expand_box_pixels(box: NormalizedBox, width: int, height: int, *, px: int) -> NormalizedBox:
    return NormalizedBox(box.x - px / width, box.y - px / height, box.w + 2 * px / width, box.h + 2 * px / height).clipped()


def point_in_expanded_box(point: dict[str, float], box: NormalizedBox, pad_x: float, pad_y: float) -> bool:
    return point_in_box(point, NormalizedBox(box.x - pad_x, box.y - pad_y, box.w + 2 * pad_x, box.h + 2 * pad_y).clipped())


def point_in_box(point: dict[str, float], box: NormalizedBox) -> bool:
    return box.x <= point["x"] <= box.x + box.w and box.y <= point["y"] <= box.y + box.h


def serialize_grounded(callout: GroundedCallout) -> dict:
    data = asdict(callout)
    data["segments"] = [asdict(segment) for segment in callout.segments]
    return data


def summarize(records: list[dict]) -> dict:
    annotations = sum(record["annotations"] for record in records)
    grounded = sum(record["grounded"] for record in records)
    line_pixels = sum(record["thinLineMaskPixels"] for record in records)
    line_boxes = sum(record["lineBoundingBoxPixels"] for record in records)
    mark_boxes = sum(record["markBoxPixels"] for record in records)
    return {
        "images": len(records),
        "annotations": annotations,
        "grounded": grounded,
        "groundingRecall": round(grounded / max(annotations, 1), 3),
        "segments": sum(record["segments"] for record in records),
        "thinLineMaskPixels": line_pixels,
        "lineBoundingBoxPixels": line_boxes,
        "markBoxPixels": mark_boxes,
        "thinMaskVsLineBox": round(line_pixels / max(line_boxes, 1), 4),
        "thinMaskVsMarkBoxes": round(line_pixels / max(mark_boxes, 1), 4),
    }


def approach_matrix() -> list[dict]:
    return [
        {
            "name": "YOLO box + rectangular segment masks",
            "status": "existing_baseline",
            "fit": "Good for boxed GD&T marks; weak for leaders because a segment bounding box can cover part geometry.",
        },
        {
            "name": "YOLO box + vector-grounded thin line masks",
            "status": "implemented_in_this_harness",
            "fit": "Best immediate path: keeps box detector for marks and traces only one-pixel-like leader/dimension strokes.",
        },
        {
            "name": "YOLO11-seg fine-tuned instance masks",
            "status": "recommended_next_trainable_mask_model",
            "fit": "Good if we annotate each callout as mark+leader polygons; fast local training in the current Ultralytics stack.",
        },
        {
            "name": "SAM2 box/point prompted masks",
            "status": "recommended_annotation_assistant",
            "fit": "Good for interactive refinement from boxes/points; not class-aware and may select only the box unless prompted along the leader.",
        },
        {
            "name": "RF-DETR Segmentation",
            "status": "promising_later_candidate",
            "fit": "Strong modern instance segmentation candidate once we have mask labels; heavier integration than YOLO-seg.",
        },
        {
            "name": "Gemma4 supervised grounding pass",
            "status": "hybrid_reasoning_layer",
            "fit": "Use after geometric candidates: choose the line that makes semantic sense, e.g. a diameter callout points to a hole.",
        },
    ]


def render_report(summary: dict) -> str:
    lines = [
        "# GD&T Mask Grounding Evaluation",
        "",
        "## Approach Matrix",
        "",
        "| Approach | Status | Fit |",
        "|---|---|---|",
    ]
    for item in summary["approaches"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['fit']} |")
    totals = summary["totals"]
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Images: {totals['images']}",
            f"- Annotations: {totals['annotations']}",
            f"- Grounded callouts: {totals['grounded']} ({totals['groundingRecall']})",
            f"- Grounded line segments: {totals['segments']}",
            f"- Thin line mask / line bounding-box pixels: {totals['thinMaskVsLineBox']}",
            f"- Thin line mask / mark-box pixels: {totals['thinMaskVsMarkBoxes']}",
            "",
            "## Images",
            "",
            "| Image | Source | Annotations | Grounded | Segments | Thin/LineBox | Overlay |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for record in summary["images"]:
        lines.append(
            f"| {record['image']} | {record['annotationSource']} | {record['annotations']} | "
            f"{record['grounded']} | {record['segments']} | {record['thinMaskVsLineBox']} | {record['overlay']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A low thin/line-box ratio is good: it means the mask is stroke-like instead of a broad rectangle.",
            "Remaining misses are the cases where a leader is curved, broken by text, merged into projection geometry,",
            "or requires semantic reasoning about which feature the callout should terminate on.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
