"""Fine-tuned vision detector support for GD&T and drawing callouts."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import Image

from src.segmentation.title_block import NormalizedBox


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GDT_VISION_MODEL = (
    REPO_ROOT / "runs/detect/experiments/gdt_symbol_vision_train/runs/yolov8n_symbol/weights/best.pt"
)
LEGACY_CALLOUT_CLASS_NAME = "callout"
NON_CALLOUT_CLASS_NAMES = {"non_callout_thread_texture"}
SYMBOL_CALLOUT_CLASS_NAMES = {
    "datum_feature",
    "boxed_basic_dimension",
    "linear_dimension",
    "diameter_dimension",
    "fcf_perpendicularity",
    "fcf_position",
    "fcf_flatness",
    "fcf_profile",
    "section_marker",
    "stacked_callout",
}


def detect_vision_callouts(
    image_path: str | Path,
    *,
    model_path: str | Path | None = None,
    conf: float = 0.05,
    imgsz: int = 1024,
) -> list[dict]:
    """Detect callout candidates with the fine-tuned YOLO model.

    The detector is intentionally optional: missing model weights, missing
    Ultralytics, or runtime failures return an empty list so the deterministic
    toolchain remains usable while the vision model is being tuned.
    """

    resolved_model = _resolve_model_path(model_path)
    if resolved_model is None:
        return []

    try:
        model = _load_yolo_model(str(resolved_model))
    except Exception:
        return []

    try:
        image = Image.open(image_path)
        width, height = image.size
        result = model.predict(
            str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=os.environ.get("DRAW_CAD_GDT_VISION_DEVICE", "0"),
            verbose=False,
        )[0]
    except Exception:
        return []

    detections = []
    boxes = result.boxes or []
    names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    for index, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        crop = _xyxy_to_box(xyxy, width, height)
        is_negative = class_name in NON_CALLOUT_CLASS_NAMES
        kind = class_name if class_name != LEGACY_CALLOUT_CLASS_NAME else "vision_callout"
        source_kind = "fine_tuned_yolo_negative" if is_negative else "fine_tuned_yolo"
        detections.append(
            {
                "id": f"vision-gdt-{index + 1}",
                "label": _label_for_class(class_name, index),
                "value": "",
                "confidence": round(confidence, 3),
                "crop": crop.to_dict(),
                "symbol": _symbol_for_class(class_name),
                "kind": kind,
                "source": source_kind,
                "model": Path(resolved_model).name,
                "classId": class_id,
                "className": class_name,
                "notes": [
                    "fine-tuned YOLO symbol detector candidate",
                    "negative thread-texture detections are retained for review but excluded from callout masks",
                ],
            }
        )
    return sorted(detections, key=lambda item: (item["crop"]["y"], item["crop"]["x"]))


def usable_vision_callouts_for_masks(
    vision_callouts: list[dict],
    existing_annotations: list[dict],
    *,
    min_confidence: float = 0.72,
) -> list[dict]:
    """Return conservative vision detections that can safely join mask inputs."""

    kept = []
    existing_boxes = [NormalizedBox(**item["crop"]) for item in existing_annotations if item.get("crop")]
    for item in vision_callouts:
        if not _is_maskable_callout(item):
            continue
        if item.get("confidence", 0.0) < min_confidence:
            continue
        box = NormalizedBox(**item["crop"])
        if any(_box_iou(box, other) > 0.35 or _box_containment(box, other) > 0.65 for other in existing_boxes):
            continue
        kept.append(item)
    return kept


def _is_maskable_callout(item: dict) -> bool:
    class_name = item.get("className")
    if class_name in NON_CALLOUT_CLASS_NAMES:
        return False
    return class_name == LEGACY_CALLOUT_CLASS_NAME or class_name in SYMBOL_CALLOUT_CLASS_NAMES


def _label_for_class(class_name: str, index: int) -> str:
    if class_name in NON_CALLOUT_CLASS_NAMES:
        return f"YOLO non-callout thread texture {index + 1}"
    label = class_name.replace("fcf_", "FCF ").replace("_", " ")
    return f"YOLO {label} candidate {index + 1}"


def _symbol_for_class(class_name: str) -> str:
    if class_name == "fcf_perpendicularity":
        return "perpendicularity"
    if class_name == "fcf_position":
        return "position"
    if class_name == "fcf_flatness":
        return "flatness"
    if class_name == "fcf_profile":
        return "profile"
    return "unclassified"


def _resolve_model_path(model_path: str | Path | None) -> Path | None:
    configured = model_path or os.environ.get("DRAW_CAD_GDT_VISION_MODEL") or DEFAULT_GDT_VISION_MODEL
    path = Path(configured)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


@lru_cache(maxsize=2)
def _load_yolo_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def _xyxy_to_box(xyxy: list[float], width: int, height: int) -> NormalizedBox:
    x0, y0, x1, y1 = xyxy
    return NormalizedBox(
        x=x0 / width,
        y=y0 / height,
        w=max(0.0, x1 - x0) / width,
        h=max(0.0, y1 - y0) / height,
    ).clipped()


def _box_iou(a: NormalizedBox, b: NormalizedBox) -> float:
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x + b.w, b.y + b.h
    ix = max(0.0, min(ax1, bx1) - max(a.x, b.x))
    iy = max(0.0, min(ay1, by1) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-6)


def _box_containment(a: NormalizedBox, b: NormalizedBox) -> float:
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x + b.w, b.y + b.h
    ix = max(0.0, min(ax1, bx1) - max(a.x, b.x))
    iy = max(0.0, min(ay1, by1) - max(a.y, b.y))
    return (ix * iy) / max(a.w * a.h, 1e-6)
