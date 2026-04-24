"""Optional local drawing-evidence extractors for Gemma 4 CAD reconstruction."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image, ImageOps

from gemma4_agent.training import extract_json_object
from gemma4_agent.toolbox import encode_image_for_ollama


GEMMA_EVIDENCE_SYSTEM_PROMPT = """You extract structured evidence from mechanical engineering drawings.

Return strict JSON only. Separate physical part geometry from annotations. Do not treat
title blocks, borders, dimension arrows, GD&T frames, note text, or paper styling as CAD
geometry. Focus on features that must influence the 3D part: holes, bores, bosses, slots,
cutouts, pockets, ribs, shafts, flanges, steps, chamfers, fillets, threads, revolved
profiles, symmetry, and view relationships."""


GEMMA_EVIDENCE_USER_PROMPT = """Analyze this engineering drawing for CAD reconstruction.

Return this JSON shape:
{
  "physical_features": [],
  "view_layout": "",
  "dimensions": [],
  "gd_t": [],
  "annotation_regions": [],
  "title_block_or_sheet_regions": [],
  "reconstruction_hints": [],
  "uncertainties": []
}

Use concise strings. Include only evidence visible in the drawing. If a field is
uncertain, put that uncertainty in "uncertainties" instead of returning malformed
JSON or prose outside the JSON object."""


JSON_REPAIR_SYSTEM_PROMPT = """You repair malformed JSON.

Return one valid JSON object only. Do not add markdown, explanations, or code fences."""


@dataclass(frozen=True)
class ExtractorRuntime:
    """Runtime configuration for optional drawing-evidence extractors."""

    model: str = "gemma4:26b"
    base_url: str = "http://localhost:11434"
    florence2_model_path: Path | None = None
    yolo_obb_model_path: Path | None = None
    donut_model_path: Path | None = None
    device: str | None = None


class DrawingEvidenceExtractor(Protocol):
    name: str

    def extract(self, drawing_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        """Return JSON-safe drawing evidence."""


def build_extractors(
    names: list[str],
    runtime: ExtractorRuntime,
) -> list[DrawingEvidenceExtractor]:
    """Build local drawing-evidence extractors from CLI names."""
    extractors: list[DrawingEvidenceExtractor] = []
    for raw_name in names:
        name = raw_name.strip().lower()
        if name in {"", "none"}:
            continue
        if name == "gemma4":
            extractors.append(Gemma4EvidenceExtractor(runtime))
        elif name == "heuristic":
            extractors.append(HeuristicEvidenceExtractor())
        elif name == "florence2":
            extractors.append(Florence2EvidenceExtractor(runtime))
        elif name in {"yolo_donut", "yolo-donut", "yolo"}:
            extractors.append(YoloDonutEvidenceExtractor(runtime))
        else:
            raise ValueError(f"Unknown extractor backend: {raw_name}")
    return extractors


def run_extractors(
    *,
    extractors: list[DrawingEvidenceExtractor],
    drawing_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run all configured extractors and merge their evidence for prompt context."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for extractor in extractors:
        try:
            result = extractor.extract(drawing_path, output_path / extractor.name)
        except Exception as exc:
            result = {
                "backend": extractor.name,
                "success": False,
                "available": False,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }
        results.append(result)

    evidence = merge_drawing_evidence(results)
    evidence["results"] = results
    evidence_path = output_path / "drawing_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    evidence["evidence_path"] = str(evidence_path)
    return evidence


def merge_drawing_evidence(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create compact prompt context from extractor outputs."""
    merged: dict[str, Any] = {
        "backend_names": [str(result.get("backend")) for result in results],
        "available_backend_names": [
            str(result.get("backend"))
            for result in results
            if result.get("available", True) and result.get("success")
        ],
        "physical_features": [],
        "dimensions": [],
        "gd_t": [],
        "annotation_regions": [],
        "title_block_or_sheet_regions": [],
        "reconstruction_hints": [],
        "uncertainties": [],
        "view_layout": [],
    }
    for result in results:
        evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
        for key in (
            "physical_features",
            "dimensions",
            "gd_t",
            "annotation_regions",
            "title_block_or_sheet_regions",
            "reconstruction_hints",
            "uncertainties",
        ):
            merged[key].extend(_string_items(evidence.get(key)))
        view_layout = str(evidence.get("view_layout", "")).strip()
        if view_layout:
            merged["view_layout"].append(view_layout)
        if result.get("success") is False:
            merged["uncertainties"].append(
                f"{result.get('backend')} extractor unavailable or failed: {result.get('error')}"
            )
    for key, value in list(merged.items()):
        if isinstance(value, list):
            merged[key] = list(dict.fromkeys(value))[:40]
    return merged


class Gemma4EvidenceExtractor:
    """Use local Ollama Gemma 4 vision to produce structured drawing evidence."""

    name = "gemma4"

    def __init__(self, runtime: ExtractorRuntime):
        self.runtime = runtime

    def extract(self, drawing_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_b64, mime_type = encode_image_for_ollama(drawing_path)
        content = self._chat_content(image_b64=image_b64, use_json_format=True)
        if not content.strip():
            content = self._chat_content(
                image_b64=image_b64,
                use_json_format=False,
                user_prompt=(
                    f"{GEMMA_EVIDENCE_USER_PROMPT}\n\n"
                    "Your previous response was empty. Return the JSON object now."
                ),
            )
        parse_error = None
        try:
            evidence = _normalize_evidence(self._parse_or_repair(content))
        except Exception as exc:
            parse_error = f"{exc.__class__.__name__}: {exc}"
            evidence = _normalize_evidence(
                _fallback_evidence_from_text(content, parse_error, drawing_path=drawing_path)
            )
        if not _has_useful_evidence(evidence):
            fallback = _normalize_evidence(
                _fallback_evidence_from_text(
                    content,
                    "Gemma returned valid JSON but no structured drawing hints.",
                    drawing_path=drawing_path,
                )
            )
            evidence = _merge_single_evidence(evidence, fallback)
        result = {
            "backend": self.name,
            "success": True,
            "available": True,
            "mime_type": mime_type,
            "raw_content": content,
            "evidence": evidence,
        }
        if parse_error:
            result["parse_error"] = parse_error
        (output_path / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _chat_content(
        self,
        *,
        image_b64: str,
        use_json_format: bool,
        user_prompt: str = GEMMA_EVIDENCE_USER_PROMPT,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.runtime.model,
            "messages": [
                {"role": "system", "content": GEMMA_EVIDENCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_b64],
                },
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048},
        }
        if use_json_format:
            payload["format"] = "json"
        api_url = f"{self.runtime.base_url.rstrip('/')}/api/chat"
        with httpx.Client(timeout=600) as client:
            response = client.post(api_url, json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("message", {}).get("content", ""))

    def _parse_or_repair(self, content: str) -> dict[str, Any]:
        try:
            return extract_json_object(content)
        except Exception:
            repaired = self._repair_json(content)
            return extract_json_object(repaired)

    def _repair_json(self, content: str) -> str:
        api_url = f"{self.runtime.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.runtime.model,
            "messages": [
                {"role": "system", "content": JSON_REPAIR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Repair this malformed JSON into one valid JSON object with the same schema:\n\n"
                        f"{content}"
                    ),
                },
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 2048},
        }
        with httpx.Client(timeout=600) as client:
            response = client.post(api_url, json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "")


class HeuristicEvidenceExtractor:
    """Fast local image heuristics for drawing layout and annotation density."""

    name = "heuristic"

    def extract(self, drawing_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        path = Path(drawing_path)
        with Image.open(path) as image:
            gray = ImageOps.grayscale(image)
            width, height = gray.size
            pixels = gray.load()
            dark_count = 0
            border_dark_count = 0
            border_margin_x = max(1, int(width * 0.06))
            border_margin_y = max(1, int(height * 0.06))
            for y in range(height):
                for x in range(width):
                    if pixels[x, y] < 200:
                        dark_count += 1
                        if (
                            x < border_margin_x
                            or x >= width - border_margin_x
                            or y < border_margin_y
                            or y >= height - border_margin_y
                        ):
                            border_dark_count += 1
            dark_ratio = dark_count / max(width * height, 1)
            border_dark_ratio = border_dark_count / max(dark_count, 1)
        aspect = width / max(height, 1)
        evidence = {
            "physical_features": [],
            "dimensions": [],
            "gd_t": [],
            "annotation_regions": [
                f"dark pixel density {dark_ratio:.3f}",
                f"border/title-block candidate density {border_dark_ratio:.3f}",
            ],
            "title_block_or_sheet_regions": _sheet_region_hints(aspect, border_dark_ratio),
            "reconstruction_hints": [
                "Use this heuristic evidence only to avoid sheet/border geometry; it does not identify exact CAD features.",
                f"source raster size {width}x{height} px, aspect ratio {aspect:.3f}",
            ],
            "uncertainties": [
                "Heuristic extractor cannot distinguish holes, slots, GD&T, or dimensions without a VLM/detector backend."
            ],
        }
        result = {
            "backend": self.name,
            "success": True,
            "available": True,
            "evidence": evidence,
        }
        (output_path / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


class Florence2EvidenceExtractor:
    """Experimental local Florence-2 extractor for fine-tuned engineering drawing models."""

    name = "florence2"

    def __init__(self, runtime: ExtractorRuntime):
        self.runtime = runtime

    def extract(self, drawing_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if not self.runtime.florence2_model_path:
            return _unavailable(
                self.name,
                "No local Florence-2 model path supplied. Use --florence2-model-path.",
                output_path,
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            return _unavailable(self.name, f"Missing optional Florence-2 dependency: {exc}", output_path)

        model_path = str(self.runtime.florence2_model_path)
        device = self.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        with Image.open(drawing_path) as image:
            image = image.convert("RGB")
            prompt = GEMMA_EVIDENCE_USER_PROMPT
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        try:
            evidence = _normalize_evidence(extract_json_object(generated_text))
            success = True
        except Exception:
            evidence = {
                "physical_features": [],
                "dimensions": [],
                "gd_t": [],
                "annotation_regions": [],
                "title_block_or_sheet_regions": [],
                "reconstruction_hints": [generated_text],
                "uncertainties": ["Florence-2 output was not strict JSON."],
            }
            success = False
        result = {
            "backend": self.name,
            "success": success,
            "available": True,
            "model_path": model_path,
            "raw_content": generated_text,
            "evidence": evidence,
        }
        (output_path / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


class YoloDonutEvidenceExtractor:
    """Experimental YOLOv11-OBB detector plus optional Donut patch parser."""

    name = "yolo_donut"

    def __init__(self, runtime: ExtractorRuntime):
        self.runtime = runtime

    def extract(self, drawing_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        output_path = Path(output_dir)
        crops_dir = output_path / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        if not self.runtime.yolo_obb_model_path:
            return _unavailable(
                self.name,
                "No local YOLO OBB model path supplied. Use --yolo-obb-model-path.",
                output_path,
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            return _unavailable(self.name, f"Missing optional YOLO dependency: {exc}", output_path)

        model = YOLO(str(self.runtime.yolo_obb_model_path))
        detections: list[dict[str, Any]] = []
        with Image.open(drawing_path) as image:
            image = image.convert("RGB")
            results = model.predict(str(drawing_path), verbose=False)
            names = getattr(model, "names", {}) or {}
            for result_index, result in enumerate(results):
                detections.extend(_extract_yolo_detections(result, names, result_index))
            for index, detection in enumerate(detections):
                box = detection.get("xyxy")
                if box:
                    crop = image.crop(tuple(int(round(coord)) for coord in box))
                    crop_path = crops_dir / f"det_{index:03d}_{_safe_file_label(detection['label'])}.png"
                    crop.save(crop_path)
                    detection["crop_path"] = str(crop_path)

        evidence = _evidence_from_yolo_detections(detections)
        if self.runtime.donut_model_path:
            evidence["reconstruction_hints"].append(
                "Donut model path supplied; patch parsing is reserved for fine-tuned local model adapters."
            )
        result = {
            "backend": self.name,
            "success": True,
            "available": True,
            "model_path": str(self.runtime.yolo_obb_model_path),
            "donut_model_path": str(self.runtime.donut_model_path) if self.runtime.donut_model_path else None,
            "detections": detections,
            "evidence": evidence,
        }
        (output_path / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def _normalize_evidence(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "physical_features": _string_items(value.get("physical_features")),
        "view_layout": str(value.get("view_layout", "")),
        "dimensions": _string_items(value.get("dimensions")),
        "gd_t": _string_items(value.get("gd_t", value.get("gd&t"))),
        "annotation_regions": _string_items(value.get("annotation_regions")),
        "title_block_or_sheet_regions": _string_items(value.get("title_block_or_sheet_regions")),
        "reconstruction_hints": _string_items(value.get("reconstruction_hints")),
        "uncertainties": _string_items(value.get("uncertainties")),
    }


def _has_useful_evidence(evidence: dict[str, Any]) -> bool:
    return any(
        evidence.get(key)
        for key in (
            "physical_features",
            "dimensions",
            "gd_t",
            "annotation_regions",
            "title_block_or_sheet_regions",
            "reconstruction_hints",
        )
    )


def _merge_single_evidence(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key, value in fallback.items():
        if isinstance(value, list):
            merged[key] = list(dict.fromkeys([*_string_items(merged.get(key)), *value]))
        elif value and not merged.get(key):
            merged[key] = value
    return merged


def _fallback_evidence_from_text(
    content: str,
    reason: str,
    drawing_path: str | Path | None = None,
) -> dict[str, Any]:
    """Coerce free-form model output into the evidence schema."""
    lines = [
        re.sub(r"\s+", " ", line.strip(" -*\t"))
        for line in content.splitlines()
        if line.strip(" -*\t")
    ]
    compact_lines = [line for line in lines if 4 <= len(line) <= 220][:24]
    physical_keywords = re.compile(
        r"\b(hole|bore|slot|cutout|pocket|boss|shaft|flange|rib|chamfer|fillet|thread|"
        r"cylinder|rectangular|circle|profile|plate|block)\b",
        re.IGNORECASE,
    )
    dimension_keywords = re.compile(r"\b(diameter|radius|width|height|depth|thickness|mm|inch|in\.|±|\+/-|phi|ø)\b", re.IGNORECASE)
    gdt_keywords = re.compile(
        r"\b(datum|position|flatness|parallelism|perpendicularity|profile|runout|concentricity|gd&t|gdt)\b",
        re.IGNORECASE,
    )
    annotation_keywords = re.compile(r"\b(title|border|note|tolerance|revision|material|finish|sheet|scale)\b", re.IGNORECASE)

    image_hints = _fallback_image_hints(drawing_path) if drawing_path else {}
    return {
        "physical_features": [line for line in compact_lines if physical_keywords.search(line)][:12],
        "view_layout": _first_matching_line(
            compact_lines,
            re.compile(r"\b(front|right|top|side|view|orthographic)\b", re.IGNORECASE),
        )
        or str(image_hints.get("view_layout", "")),
        "dimensions": [line for line in compact_lines if dimension_keywords.search(line)][:12],
        "gd_t": [line for line in compact_lines if gdt_keywords.search(line)][:12],
        "annotation_regions": [
            *[line for line in compact_lines if annotation_keywords.search(line)][:12],
            *image_hints.get("annotation_regions", []),
        ],
        "title_block_or_sheet_regions": [
            line
            for line in compact_lines
            if re.search(r"\b(title|border|revision|sheet)\b", line, re.IGNORECASE)
        ][:8]
        + image_hints.get("title_block_or_sheet_regions", []),
        "reconstruction_hints": [
            "Gemma evidence was coerced from free-form output; use it as hints only.",
            "Model the main physical part and ignore title blocks, borders, notes, dimensions, and GD&T frames as geometry.",
            *image_hints.get("reconstruction_hints", []),
            *compact_lines[:6],
        ],
        "uncertainties": [f"Structured JSON parse fallback used: {reason}"],
    }


def _first_matching_line(lines: list[str], pattern: re.Pattern[str]) -> str:
    for line in lines:
        if pattern.search(line):
            return line
    return ""


def _fallback_image_hints(drawing_path: str | Path | None) -> dict[str, list[str] | str]:
    if not drawing_path:
        return {}
    try:
        with Image.open(drawing_path) as image:
            gray = ImageOps.grayscale(image)
            width, height = gray.size
            dark_bbox = gray.point(lambda value: 255 if value < 210 else 0).getbbox()
    except Exception:
        return {}
    aspect = width / max(height, 1)
    layout = "landscape engineering drawing" if aspect >= 1.1 else "portrait/square engineering drawing"
    hints = [
        f"source raster size {width}x{height} px",
        "Gemma returned no parseable details; inspect the image directly for holes, slots, bosses, and section views.",
    ]
    annotation_regions = [f"overall dark-line bounding region {dark_bbox}"] if dark_bbox else []
    sheet_regions = ["landscape sheet; title block often appears near lower-right border"] if aspect >= 1.1 else []
    return {
        "view_layout": layout,
        "annotation_regions": annotation_regions,
        "title_block_or_sheet_regions": sheet_regions,
        "reconstruction_hints": hints,
    }


def _string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalized = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(json.dumps(item, sort_keys=True))
        else:
            normalized.append(str(item))
    return [item for item in normalized if item.strip()]


def _sheet_region_hints(aspect: float, border_dark_ratio: float) -> list[str]:
    hints = []
    if aspect > 1.15:
        hints.append("landscape sheet; title block often appears near lower-right border")
    if border_dark_ratio > 0.25:
        hints.append("substantial border/title-block linework likely present")
    return hints


def _unavailable(backend: str, reason: str, output_path: Path) -> dict[str, Any]:
    result = {
        "backend": backend,
        "success": False,
        "available": False,
        "error": reason,
        "evidence": {
            "physical_features": [],
            "dimensions": [],
            "gd_t": [],
            "annotation_regions": [],
            "title_block_or_sheet_regions": [],
            "reconstruction_hints": [],
            "uncertainties": [reason],
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _extract_yolo_detections(result: Any, names: dict[int, str], result_index: int) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if getattr(result, "obb", None) is not None and result.obb is not None:
        xywhr = _tolist(getattr(result.obb, "xywhr", None))
        cls = _tolist(getattr(result.obb, "cls", None))
        conf = _tolist(getattr(result.obb, "conf", None))
        xyxy = _tolist(getattr(result.obb, "xyxy", None))
        for index, row in enumerate(xywhr):
            class_id = int(cls[index]) if index < len(cls) else -1
            detections.append(
                {
                    "result_index": result_index,
                    "index": index,
                    "label": str(names.get(class_id, class_id)),
                    "confidence": float(conf[index]) if index < len(conf) else None,
                    "xywhr": [float(value) for value in row],
                    "xyxy": [float(value) for value in xyxy[index]] if index < len(xyxy) else None,
                }
            )
        return detections

    if getattr(result, "boxes", None) is not None and result.boxes is not None:
        xyxy = _tolist(getattr(result.boxes, "xyxy", None))
        cls = _tolist(getattr(result.boxes, "cls", None))
        conf = _tolist(getattr(result.boxes, "conf", None))
        for index, row in enumerate(xyxy):
            class_id = int(cls[index]) if index < len(cls) else -1
            detections.append(
                {
                    "result_index": result_index,
                    "index": index,
                    "label": str(names.get(class_id, class_id)),
                    "confidence": float(conf[index]) if index < len(conf) else None,
                    "xyxy": [float(value) for value in row],
                }
            )
    return detections


def _evidence_from_yolo_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    for detection in detections:
        label = str(detection.get("label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1
    gd_t_labels = {"gd&t", "gdt", "feature_control_frame"}
    dimension_labels = {"measure", "measures", "dimension", "radii", "threads", "surface_roughness"}
    sheet_labels = {"title_block", "title blocks", "general_tolerances", "notes", "materials"}
    return {
        "physical_features": [],
        "dimensions": [
            f"{label}: {count}"
            for label, count in label_counts.items()
            if label.lower() in dimension_labels
        ],
        "gd_t": [
            f"{label}: {count}"
            for label, count in label_counts.items()
            if label.lower() in gd_t_labels
        ],
        "annotation_regions": [
            f"{label}: {count}"
            for label, count in sorted(label_counts.items())
        ],
        "title_block_or_sheet_regions": [
            f"{label}: {count}"
            for label, count in label_counts.items()
            if label.lower() in sheet_labels
        ],
        "reconstruction_hints": [
            "Use YOLO/Donut detections to mask annotation regions and preserve only physical drawing linework as CAD geometry."
        ],
        "uncertainties": [
            "YOLO/Donut extractor identifies annotation categories but does not directly reconstruct 3D physical features."
        ],
    }


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [value]
    return []


def _safe_file_label(value: Any) -> str:
    text = str(value)
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return cleaned[:48] or "unknown"
