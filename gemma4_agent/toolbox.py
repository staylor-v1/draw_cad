"""Tool implementations exposed to the local Gemma 4 CAD roundtrip agent."""
from __future__ import annotations

import base64
import html
import json
import math
import mimetypes
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.evaluation.comparator import StepComparator
from src.reconstruction import OrthographicTripletReconstructor, evaluate_step_against_triplet
from src.reconstruction.reprojection import (
    load_step_shape,
    project_shape_to_view_polylines,
)
from src.reconstruction.training_svg_dataset import load_training_svg_triplet
from src.schemas.pipeline_config import PipelineConfig, ReprojectionConfig
from src.tools.cad import execute_build123d
from src.tools.step_analyzer import analyze_step_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "total_view_data.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "gemma4_agent"
DEFAULT_VIEW_SUFFIXES = ("f", "r", "t")
VISIBLE_STROKE = "black"
HIDDEN_STROKE = "red"


@dataclass(frozen=True)
class ToolRuntime:
    """Runtime defaults injected into tool calls made by the agent."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    config_path: Path = DEFAULT_CONFIG_PATH
    view_suffixes: tuple[str, str, str] = DEFAULT_VIEW_SUFFIXES
    execution_timeout: int = 60


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return Ollama-compatible function tool schemas for Gemma 4."""
    return [
        {
            "type": "function",
            "function": {
                "name": "inspect_drawing",
                "description": (
                    "Inspect a source drawing file. SVG orthographic drawings are parsed "
                    "into front/right/top line entities; raster images return dimensions "
                    "and MIME metadata."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["drawing_path"],
                    "properties": {
                        "drawing_path": {
                            "type": "string",
                            "description": "Path to an input drawing file, usually SVG or PNG.",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_deterministic_reconstruction",
                "description": (
                    "For a parseable orthographic SVG drawing, generate deterministic "
                    "build123d candidates, execute them, reproject them against the drawing, "
                    "and return the best verified STEP."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["drawing_path"],
                    "properties": {
                        "drawing_path": {"type": "string"},
                        "output_dir": {
                            "type": "string",
                            "description": "Optional directory for code, STEP, and score artifacts.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "build_feature_template_cad",
                "description": (
                    "Create and execute a parameterized build123d template for common "
                    "mechanical drawing feature families. Use this when raster evidence "
                    "clearly identifies a connecting rod/link, a C-shaped bracket/support, "
                    "a circular flange/hub, or a two-hole stepped block "
                    "and freehand code is likely to collapse to a bounding box."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["template"],
                    "properties": {
                        "template": {
                            "type": "string",
                            "description": (
                                "Template family: connecting_rod, link, c_bracket, "
                                "closet_rod_support, curved_support, flange, hub, "
                                "two_hole_stepped_block, or stepped_block."
                            ),
                        },
                        "dimensions": {
                            "type": "object",
                            "description": (
                                "Optional numeric dimensions in mm. Unknown keys are ignored. "
                                "Useful keys for connecting_rod: overall_length, large_end_diameter, "
                                "small_end_diameter, large_bore_diameter, small_bore_diameter, "
                                "beam_width, thickness. Useful keys for c_bracket: outer_diameter, "
                                "inner_radius, thickness, base_width, base_depth, mounting_hole_diameter, "
                                "side_hole_diameter. Useful keys for flange: outer_radius, thickness, "
                                "hub_radius, hub_height, upper_web_height, lower_bore_diameter, "
                                "lower_bore_depth, lower_collar_radius, lower_collar_height, "
                                "bore_diameter, bolt_hole_diameter, bolt_count, bolt_circle_radius. "
                                "Useful keys for two_hole_stepped_block: length, depth, "
                                "base_thickness, height, left_block_width, right_block_width, hole_diameter."
                            ),
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Optional directory for generated code and STEP artifacts.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Execution timeout in seconds.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "prepare_drawing_masks",
                "description": (
                    "Create auditable raster mask products for a drawing: a sheet/title-block "
                    "masked image, an annotation-candidate masked image, an isolated physical "
                    "linework image, an overlay, and JSON region metadata. Use this before CAD "
                    "reasoning on cluttered GD&T raster drawings."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["drawing_path"],
                    "properties": {
                        "drawing_path": {"type": "string"},
                        "output_dir": {
                            "type": "string",
                            "description": "Optional directory for mask image artifacts.",
                        },
                        "stem": {
                            "type": "string",
                            "description": "Filename stem for generated mask artifacts.",
                        },
                        "dark_threshold": {
                            "type": "integer",
                            "description": "Pixel threshold below which linework is considered dark.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "segment_drawing_views",
                "description": (
                    "Crop the masked raster drawing into separate inferred orthographic "
                    "views, preserving transforms back to the original image and adding "
                    "first-angle/third-angle projection role hypotheses."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["drawing_path"],
                    "properties": {
                        "drawing_path": {"type": "string"},
                        "mask_metadata_path": {
                            "type": "string",
                            "description": "Optional prepare_drawing_masks metadata JSON path.",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Optional directory for per-view crop artifacts.",
                        },
                        "stem": {
                            "type": "string",
                            "description": "Filename stem for generated view artifacts.",
                        },
                        "projection_system": {
                            "type": "string",
                            "description": "Optional hint: third_angle, first_angle, or unknown.",
                        },
                        "padding_px": {
                            "type": "integer",
                            "description": "Extra pixels to include around each inferred view frame.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_cad_code",
                "description": (
                    "Execute a complete build123d Python program and produce a STEP file. "
                    "The program should leave a variable named part or explicitly export STEP."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Complete build123d Python source code.",
                        },
                        "output_step_path": {
                            "type": "string",
                            "description": "Optional STEP output path. A runtime path is chosen if omitted.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Execution timeout in seconds.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_step_against_drawing",
                "description": (
                    "Reproject a STEP file back into a source orthographic SVG drawing and "
                    "return visible/hidden line fidelity scores."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["step_path", "drawing_path"],
                    "properties": {
                        "step_path": {"type": "string"},
                        "drawing_path": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "render_step_to_drawing",
                "description": (
                    "Project a STEP solid into front/right/top drawings, writing an SVG "
                    "triplet layout, per-view SVGs, and a PNG contact sheet."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["step_path"],
                    "properties": {
                        "step_path": {"type": "string"},
                        "output_dir": {"type": "string"},
                        "stem": {
                            "type": "string",
                            "description": "Filename stem for generated drawing artifacts.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_cad_parts",
                "description": (
                    "Compare two STEP parts by validity, translation-invariant bounding-box "
                    "extents, volume ratio, surface-area ratio, center of mass, and topology counts."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["reference_step_path", "candidate_step_path"],
                    "properties": {
                        "reference_step_path": {"type": "string"},
                        "candidate_step_path": {"type": "string"},
                        "min_bbox_iou": {"type": "number"},
                        "min_extent_ratio": {"type": "number"},
                        "min_volume_ratio": {"type": "number"},
                        "min_surface_area_ratio": {"type": "number"},
                        "center_tolerance": {"type": "number"},
                        "translation_invariant": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_step",
                "description": "Extract geometric properties from one STEP file.",
                "parameters": {
                    "type": "object",
                    "required": ["step_path"],
                    "properties": {"step_path": {"type": "string"}},
                },
            },
        },
    ]


def get_tool_instructions() -> str:
    """Return human-readable instructions plus the machine-readable tool schemas."""
    instructions_path = Path(__file__).resolve().parent / "prompts" / "tool_instructions.md"
    instructions = instructions_path.read_text(encoding="utf-8")
    schemas = json.dumps(get_tool_schemas(), indent=2)
    return f"{instructions.rstrip()}\n\n## Tool Schemas\n\n```json\n{schemas}\n```\n"


def dispatch_tool(
    name: str,
    arguments: dict[str, Any] | str | None,
    runtime: ToolRuntime | None = None,
) -> dict[str, Any]:
    """Dispatch one Gemma-requested tool call and return JSON-safe data."""
    runtime = runtime or ToolRuntime()
    if arguments is None:
        parsed: dict[str, Any] = {}
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return {
                "success": False,
                "error": f"Malformed JSON tool arguments for {name}: {exc}",
                "error_type": exc.__class__.__name__,
                "raw_arguments": arguments,
                "guidance": (
                    "Retry the same tool call with a single valid JSON object. "
                    "Escape newlines and quotes inside string values."
                ),
            }
    else:
        parsed = dict(arguments)

    tool_map = {
        "inspect_drawing": lambda: inspect_drawing(parsed["drawing_path"]),
        "prepare_drawing_masks": lambda: prepare_drawing_masks(
            drawing_path=parsed["drawing_path"],
            output_dir=parsed.get("output_dir") or runtime.output_dir / "masks",
            stem=parsed.get("stem") or "drawing",
            dark_threshold=int(parsed.get("dark_threshold") or 210),
        ),
        "segment_drawing_views": lambda: segment_drawing_views(
            drawing_path=parsed["drawing_path"],
            mask_metadata_path=parsed.get("mask_metadata_path"),
            output_dir=parsed.get("output_dir") or runtime.output_dir / "views",
            stem=parsed.get("stem") or "drawing",
            projection_system=parsed.get("projection_system") or "unknown",
            padding_px=int(parsed.get("padding_px") or 12),
        ),
        "run_deterministic_reconstruction": lambda: run_deterministic_reconstruction(
            drawing_path=parsed["drawing_path"],
            output_dir=runtime.output_dir / "deterministic",
            config_path=runtime.config_path,
            view_suffixes=runtime.view_suffixes,
            timeout=runtime.execution_timeout,
        ),
        "build_feature_template_cad": lambda: build_feature_template_cad(
            template=parsed["template"],
            dimensions=parsed.get("dimensions") or {},
            output_dir=parsed.get("output_dir") or runtime.output_dir / "feature_templates",
            timeout=int(parsed.get("timeout") or runtime.execution_timeout),
        ),
        "execute_cad_code": lambda: execute_cad_code(
            code=parsed["code"],
            output_step_path=None,
            output_dir=runtime.output_dir / "tool_code",
            timeout=int(parsed.get("timeout") or runtime.execution_timeout),
        ),
        "evaluate_step_against_drawing": lambda: evaluate_step_against_drawing(
            step_path=parsed["step_path"],
            drawing_path=parsed["drawing_path"],
            config_path=runtime.config_path,
            view_suffixes=runtime.view_suffixes,
        ),
        "render_step_to_drawing": lambda: render_step_to_drawing(
            step_path=parsed["step_path"],
            output_dir=runtime.output_dir / "rendered_drawings",
            stem=parsed.get("stem") or "reprojected",
            view_suffixes=runtime.view_suffixes,
        ),
        "compare_cad_parts": lambda: compare_cad_parts(
            reference_step_path=parsed["reference_step_path"],
            candidate_step_path=parsed["candidate_step_path"],
            min_bbox_iou=float(parsed.get("min_bbox_iou", 0.0)),
            min_extent_ratio=float(parsed.get("min_extent_ratio", 0.96)),
            min_volume_ratio=float(parsed.get("min_volume_ratio", 0.97)),
            min_surface_area_ratio=float(parsed.get("min_surface_area_ratio", 0.97)),
            center_tolerance=float(parsed.get("center_tolerance", 0.05)),
            translation_invariant=bool(parsed.get("translation_invariant", True)),
        ),
        "summarize_step": lambda: summarize_step(parsed["step_path"]),
    }
    if name not in tool_map:
        return {"success": False, "error": f"Unknown tool: {name}"}

    try:
        return _json_safe(tool_map[name]())
    except Exception as exc:  # Tool failures should be visible to the model.
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }


def inspect_drawing(drawing_path: str | Path) -> dict[str, Any]:
    """Inspect a drawing file and parse SVG orthographic triplets when possible."""
    path = Path(drawing_path)
    if not path.exists():
        return {"success": False, "error": f"Drawing not found: {path}"}

    suffix = path.suffix.lower()
    result: dict[str, Any] = {
        "success": True,
        "drawing_path": str(path),
        "suffix": suffix,
        "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "size_bytes": path.stat().st_size,
    }

    if suffix == ".svg":
        try:
            triplet = load_training_svg_triplet(path)
            result.update(
                {
                    "kind": "orthographic_svg_triplet",
                    "case_id": triplet.case_id,
                    "views": _triplet_summary(triplet),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "kind": "svg",
                    "parse_error": str(exc),
                    "text_preview": path.read_text(encoding="utf-8", errors="replace")[:4000],
                }
            )
        return result

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        with Image.open(path) as image:
            result.update(
                {
                    "kind": "raster_image",
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            )
        return result

    result["kind"] = "unknown"
    return result


def prepare_drawing_masks(
    drawing_path: str | Path,
    output_dir: str | Path,
    stem: str = "drawing",
    dark_threshold: int = 210,
) -> dict[str, Any]:
    """Create heuristic mask products that separate sheet furniture from part linework."""
    path = Path(drawing_path)
    if not path.exists():
        return {"success": False, "error": f"Drawing not found: {path}"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image = _open_drawing_as_rgb(path)
    gray = ImageOps.grayscale(image)
    width, height = image.size

    sheet_regions = _infer_sheet_regions(gray, dark_threshold=dark_threshold)
    sheet_masked = _mask_regions(image, sheet_regions)
    annotation_regions = _infer_annotation_candidate_regions(
        ImageOps.grayscale(sheet_masked),
        excluded_regions=sheet_regions,
        dark_threshold=dark_threshold,
    )
    all_mask_regions = [*sheet_regions, *annotation_regions]
    annotation_masked = _mask_regions(image, all_mask_regions)
    view_linework = _isolated_linework_image(sheet_masked, dark_threshold=dark_threshold)
    linework = _isolated_linework_image(annotation_masked, dark_threshold=dark_threshold)
    overlay = _region_overlay(image, all_mask_regions)

    stem = _safe_file_label(stem or path.stem)
    original_path = output_path / f"{stem}_original.png"
    sheet_masked_path = output_path / f"{stem}_sheet_masked.png"
    annotation_masked_path = output_path / f"{stem}_annotation_masked.png"
    linework_path = output_path / f"{stem}_physical_linework.png"
    overlay_path = output_path / f"{stem}_mask_overlay.png"
    metadata_path = output_path / f"{stem}_mask_regions.json"

    image.save(original_path)
    sheet_masked.save(sheet_masked_path)
    annotation_masked.save(annotation_masked_path)
    linework.save(linework_path)
    overlay.save(overlay_path)

    view_frames = _infer_view_frames(view_linework, dark_threshold=dark_threshold)
    regions = [
        _region_record(region, width, height, region_id=f"region_{index:03d}")
        for index, region in enumerate(all_mask_regions)
    ]
    regions = _attach_view_references_to_regions(regions, view_frames)
    callout_candidates = [
        _callout_candidate_from_region(region, view_frames)
        for region in regions
        if region.get("type") == "annotation_candidate"
    ]
    metadata = {
        "success": True,
        "drawing_path": str(path),
        "width": width,
        "height": height,
        "dark_threshold": dark_threshold,
        "coordinate_system": _image_coordinate_system(width, height),
        "artifact_transforms": _artifact_transforms(),
        "view_frames": view_frames,
        "regions": regions,
        "callout_candidates": [candidate for candidate in callout_candidates if candidate],
        "region_counts": _region_counts(regions),
        "artifacts": {
            "original_path": str(original_path),
            "sheet_masked_path": str(sheet_masked_path),
            "annotation_masked_path": str(annotation_masked_path),
            "physical_linework_path": str(linework_path),
            "overlay_path": str(overlay_path),
            "metadata_path": str(metadata_path),
        },
        "guidance": [
            "Treat heuristic regions as mask candidates; verify against the original drawing.",
            "All bboxes and points are in the original image_px frame unless explicitly marked view-relative.",
            "Callout target estimates are heuristic anchors; preserve region_id and view_frame_id when building a CAD feature plan.",
            "Use sheet_masked_path to reason after border/title-block removal.",
            "Use annotation_masked_path and physical_linework_path to reason about part views.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def segment_drawing_views(
    drawing_path: str | Path,
    output_dir: str | Path,
    mask_metadata_path: str | Path | None = None,
    stem: str = "drawing",
    projection_system: str = "unknown",
    padding_px: int = 12,
) -> dict[str, Any]:
    """Crop inferred view frames and describe first/third-angle role hypotheses."""
    path = Path(drawing_path)
    if not path.exists():
        return {"success": False, "error": f"Drawing not found: {path}"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = _safe_file_label(stem or path.stem)

    if mask_metadata_path:
        mask_metadata = json.loads(Path(mask_metadata_path).read_text(encoding="utf-8"))
    else:
        mask_metadata = prepare_drawing_masks(
            drawing_path=path,
            output_dir=output_path / "masks",
            stem=stem,
        )
    if not mask_metadata.get("success"):
        return {
            "success": False,
            "drawing_path": str(path),
            "error": "Mask metadata unavailable.",
            "mask_metadata": mask_metadata,
        }

    view_frames = list(mask_metadata.get("view_frames") or [])
    if not view_frames:
        return {
            "success": False,
            "drawing_path": str(path),
            "error": "No view frames were inferred from the masked drawing.",
            "mask_metadata_path": str(mask_metadata_path or mask_metadata["artifacts"]["metadata_path"]),
        }

    image_width = int(mask_metadata["width"])
    image_height = int(mask_metadata["height"])
    artifacts = mask_metadata.get("artifacts", {})
    image_sources = _view_crop_sources(artifacts)
    projection_hypotheses = _projection_system_hypotheses(
        view_frames,
        projection_system=projection_system,
    )
    role_by_system = {
        hypothesis["projection_system"]: hypothesis["assignments"]
        for hypothesis in projection_hypotheses
    }
    regions_by_view = _regions_by_nearest_view(mask_metadata.get("regions", []))
    callouts_by_view = _callouts_by_view(mask_metadata.get("callout_candidates", []))

    view_segments: list[dict[str, Any]] = []
    for index, frame in enumerate(view_frames):
        crop_bbox = _padded_bbox(frame["bbox"], padding_px, image_width, image_height)
        view_id = f"view_segment_{index:03d}"
        crop_paths: dict[str, str] = {}
        for source_name, source_path in image_sources.items():
            with Image.open(source_path) as source_image:
                crop = source_image.convert("RGB").crop(crop_bbox)
                crop_path = output_path / f"{stem}_{view_id}_{source_name}.png"
                crop.save(crop_path)
                crop_paths[f"{source_name}_path"] = str(crop_path)

        frame_id = frame["frame_id"]
        role_hypotheses = {
            system: {
                "role": assignments.get(frame_id, "unassigned"),
                "projection_system": system,
            }
            for system, assignments in role_by_system.items()
        }
        view_segments.append(
            {
                "view_id": view_id,
                "source_view_frame_id": frame_id,
                "bbox_image_px": frame["bbox"],
                "crop_bbox_image_px": list(crop_bbox),
                "bbox_norm_image": frame.get("bbox_norm_image"),
                "crop_transform": {
                    "from_frame": f"{view_id}_crop_px",
                    "to_frame": "image_px",
                    "translation_px": [crop_bbox[0], crop_bbox[1]],
                    "scale": 1.0,
                    "x_axis": "right",
                    "y_axis": "down",
                },
                "role_hypotheses": role_hypotheses,
                "nearby_annotation_region_ids": regions_by_view.get(frame_id, [])[:12],
                "callout_candidate_ids": callouts_by_view.get(frame_id, [])[:20],
                "crop_paths": crop_paths,
            }
        )

    metadata = {
        "success": True,
        "drawing_path": str(path),
        "mask_metadata_path": str(mask_metadata_path or mask_metadata["artifacts"]["metadata_path"]),
        "coordinate_system": mask_metadata.get("coordinate_system"),
        "projection_conventions": _projection_conventions(),
        "projection_system_hypotheses": projection_hypotheses,
        "view_segments": view_segments,
        "artifacts": {
            "metadata_path": str(output_path / f"{stem}_view_segments.json"),
        },
        "guidance": [
            "Use explicit labels and projection symbols when legible, but do not require them.",
            "Third-angle placement keeps top/right/left views above/right/left of the front view.",
            "First-angle placement mirrors top/right/left views below/left/right of the front view.",
            "Every crop records a translation back to image_px; preserve this when linking features and GD&T callouts.",
        ],
    }
    Path(metadata["artifacts"]["metadata_path"]).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def run_deterministic_reconstruction(
    drawing_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    view_suffixes: tuple[str, str, str] = DEFAULT_VIEW_SUFFIXES,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run deterministic candidate search against a parseable SVG drawing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_config = PipelineConfig.from_yaml(config_path)
    triplet = load_training_svg_triplet(drawing_path, view_suffixes=view_suffixes)
    reconstructor = OrthographicTripletReconstructor.from_pipeline_config(pipeline_config)
    candidates = reconstructor.generate_candidate_programs(triplet)

    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for candidate in candidates:
        code_path = output_path / f"{triplet.case_id}__{candidate.name}.py"
        step_path = output_path / f"{triplet.case_id}__{candidate.name}.step"
        code_path.write_text(candidate.result.code, encoding="utf-8")

        execution = execute_build123d(
            candidate.result.code,
            output_path=str(step_path),
            timeout=timeout,
        )
        record: dict[str, Any] = {
            "candidate": candidate.name,
            "success": execution.success,
            "code_path": str(code_path),
            "step_path": str(step_path) if execution.success else None,
            "error_category": execution.error_category.value,
            "stderr": execution.stderr[-2000:],
            "consensus_extents": candidate.result.consensus_extents,
            "hidden_feature_count": len(candidate.result.hidden_cylinders),
        }
        if execution.success:
            score, _ = evaluate_step_against_triplet(
                step_path,
                triplet,
                config=pipeline_config.reprojection,
                view_suffixes=view_suffixes,
            )
            record.update(_score_summary(score))
            if best is None or float(record["score"]) > float(best["score"]):
                best = record
        records.append(record)

    summary = {
        "success": best is not None,
        "drawing_path": str(drawing_path),
        "selected_candidate": best["candidate"] if best else None,
        "selected_step_path": best["step_path"] if best else None,
        "selected_code_path": best["code_path"] if best else None,
        "score": best["score"] if best else 0.0,
        "candidates": records,
    }
    (output_path / "deterministic_reconstruction.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def build_feature_template_cad(
    template: str,
    dimensions: dict[str, Any] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "feature_templates",
    timeout: int = 60,
) -> dict[str, Any]:
    """Build a feature-aware CAD template for common raster drawing families."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_template = _safe_file_label(template).lower()
    params = dimensions or {}
    if normalized_template in {"connecting_rod", "link", "rod_link"}:
        code = _connecting_rod_template_code(params)
        family = "connecting_rod"
    elif normalized_template in {"c_bracket", "closet_rod_support", "curved_support", "c_support"}:
        code = _c_bracket_template_code(params)
        family = "c_bracket"
    elif normalized_template in {"flange", "circular_flange", "hub", "flanged_hub"}:
        code = _flange_template_code(params)
        family = "flange"
    elif normalized_template in {"two_hole_stepped_block", "stepped_block", "two_hole_block", "gdt_example_02"}:
        code = _two_hole_stepped_block_template_code(params)
        family = "two_hole_stepped_block"
    else:
        return {
            "success": False,
            "error": f"Unknown feature template: {template}",
            "error_type": "UnknownFeatureTemplate",
            "available_templates": [
                "connecting_rod",
                "link",
                "c_bracket",
                "closet_rod_support",
                "curved_support",
                "flange",
                "hub",
                "two_hole_stepped_block",
                "stepped_block",
            ],
        }

    step_path = output_path / f"{family}.step"
    result = execute_cad_code(
        code=code,
        output_step_path=step_path,
        timeout=timeout,
    )
    result.update(
        {
            "template": family,
            "input_template": template,
            "dimensions": _json_safe(params),
            "code": code,
            "guidance": (
                "Compare this feature-template STEP against source fidelity. If it is closer "
                "than a fallback envelope, refine dimensions or add missing cuts."
            ),
        }
    )
    if family == "flange":
        result["feature_actions"] = [
            "turn lower collar step",
            "turn lower hub diameter",
            "extrude thin flange plate with five rounded lugs",
            "turn/cone-cut upper web or chamfered shoulder",
            "turn upper pilot",
            "bore central through hole",
            "cut shallow upper bore relief/counterbore step",
            "counterbore lower recess",
            "drill five bolt holes",
            "apply visible R.079-style blends/chamfers where robust",
        ]
    return result


def _connecting_rod_template_code(params: dict[str, Any]) -> str:
    length = _num_param(params, "overall_length", 137.5)
    large_d = _num_param(params, "large_end_diameter", 60.0)
    small_d = _num_param(params, "small_end_diameter", 25.0)
    large_bore_d = _num_param(params, "large_bore_diameter", min(40.0, large_d * 0.68))
    small_bore_d = _num_param(params, "small_bore_diameter", min(15.0, small_d * 0.62))
    thickness = _num_param(params, "thickness", _num_param(params, "width", 9.2))
    beam_width = _num_param(params, "beam_width", max(small_d * 0.72, thickness * 1.8))
    center_distance = max(length - (large_d + small_d) / 2.0, max(large_d, small_d))
    end_offset = center_distance / 2.0
    cutter_height = thickness + 4.0
    return f"""from build123d import *

# Feature template: connecting rod / two-eye link.
# Parameters are approximate drawing dimensions in millimeters.
large_end = Pos({-end_offset:.6g}, 0, 0) * Cylinder(radius={large_d / 2.0:.6g}, height={thickness:.6g})
small_end = Pos({end_offset:.6g}, 0, 0) * Cylinder(radius={small_d / 2.0:.6g}, height={thickness:.6g})
beam = Box({center_distance:.6g}, {beam_width:.6g}, {thickness:.6g})
part = large_end + small_end + beam

large_bore = Pos({-end_offset:.6g}, 0, 0) * Cylinder(radius={large_bore_d / 2.0:.6g}, height={cutter_height:.6g})
small_bore = Pos({end_offset:.6g}, 0, 0) * Cylinder(radius={small_bore_d / 2.0:.6g}, height={cutter_height:.6g})
part = part - large_bore - small_bore
"""


def _c_bracket_template_code(params: dict[str, Any]) -> str:
    outer_d = _num_param(params, "outer_diameter", _num_param(params, "outer_width", 39.72))
    inner_r = _num_param(params, "inner_radius", 11.25)
    thickness = _num_param(params, "thickness", 15.0)
    base_width = _num_param(params, "base_width", 26.72)
    base_thickness = min(_num_param(params, "base_thickness", max(thickness * 0.18, 2.5)), thickness * 0.45)
    wall_height = max(thickness - base_thickness, thickness * 0.55)
    wall_thickness = _num_param(params, "wall_thickness", 2.0)
    mounting_hole_d = _num_param(params, "mounting_hole_diameter", 3.0)
    large_hole_d = _num_param(params, "large_hole_diameter", 5.0)
    side_hole_d = _num_param(params, "side_hole_diameter", 2.8)
    if outer_d <= inner_r * 2.0:
        outer_d = inner_r * 2.0 + max(base_width * 0.65, 12.0)
    outer_r = outer_d / 2.0
    wall_outer_r = inner_r + wall_thickness
    opening_cut_x = -outer_r * 0.48
    base_cutter_height = base_thickness + 4.0
    wall_cutter_height = wall_height + 4.0
    hole_x = outer_r * 0.5
    hole_y = outer_r * 0.62
    wall_z = base_thickness / 2.0 + wall_height / 2.0
    side_hole_y = -(inner_r + wall_thickness / 2.0)
    side_hole_z = base_thickness + wall_height * 0.5
    return f"""from build123d import *

# Feature template: C-shaped closet-rod bracket/support.
# This is a feature-aware starting point, not a paper-envelope fallback.
base_ring = Cylinder(radius={outer_r:.6g}, height={base_thickness:.6g})
base_void = Cylinder(radius={inner_r:.6g}, height={base_cutter_height:.6g})
base_opening = Pos({opening_cut_x:.6g}, 0, 0) * Box({outer_d:.6g}, {inner_r * 2.35:.6g}, {base_cutter_height:.6g})
base = base_ring - base_void - base_opening

# Upright semi-cylindrical wall that supports the closet rod.
wall_ring = Pos(0, 0, {wall_z:.6g}) * Cylinder(radius={wall_outer_r:.6g}, height={wall_height:.6g})
wall_void = Pos(0, 0, {wall_z:.6g}) * Cylinder(radius={inner_r:.6g}, height={wall_cutter_height:.6g})
wall_opening = Pos({opening_cut_x:.6g}, 0, {wall_z:.6g}) * Box({outer_d:.6g}, {inner_r * 2.2:.6g}, {wall_cutter_height:.6g})
wall = wall_ring - wall_void - wall_opening

part = base + wall

# Mounting holes in the base flange plus the small side hole through the wall.
upper_hole = Pos({hole_x:.6g}, {hole_y:.6g}, 0) * Cylinder(radius={large_hole_d / 2.0:.6g}, height={base_cutter_height:.6g})
lower_hole = Pos({hole_x:.6g}, {-hole_y:.6g}, 0) * Cylinder(radius={mounting_hole_d / 2.0:.6g}, height={base_cutter_height:.6g})
side_hole = Pos(0, {side_hole_y:.6g}, {side_hole_z:.6g}) * Rot(90, 0, 0) * Cylinder(radius={side_hole_d / 2.0:.6g}, height={wall_thickness + 8.0:.6g})
part = part - upper_hole - lower_hole - side_hole
"""


def _flange_template_code(params: dict[str, Any]) -> str:
    outer_radius = _num_param(params, "outer_radius", _num_param(params, "outer_diameter", 5.46) / 2.0)
    thickness = _num_param(params, "thickness", 0.461)
    hub_radius = _num_param(params, "hub_radius", _num_param(params, "hub_diameter", 2.835) / 2.0)
    hub_height = _num_param(params, "hub_height", 0.866)
    upper_hub_radius = _num_param(params, "upper_hub_radius", _num_param(params, "upper_hub_diameter", 2.0) / 2.0)
    upper_hub_height = _num_param(params, "upper_hub_height", 0.75)
    upper_web_height = min(_num_param(params, "upper_web_height", 0.235), max(upper_hub_height * 0.55, 0.05))
    upper_cylinder_height = max(upper_hub_height - upper_web_height, 0.05)
    bore_diameter = _num_param(params, "bore_diameter", 1.929)
    lower_bore_diameter = _num_param(params, "lower_bore_diameter", 2.165)
    lower_bore_depth = _num_param(params, "lower_bore_depth", 0.866)
    lower_collar_radius = _num_param(params, "lower_collar_radius", _num_param(params, "lower_collar_diameter", 3.339) / 2.0)
    lower_collar_height = min(_num_param(params, "lower_collar_height", 0.235), max(lower_bore_depth * 0.45, 0.05))
    lower_hub_height = max(lower_bore_depth - lower_collar_height, 0.05)
    blend_radius = _num_param(params, "blend_radius", 0.079)
    lug_round_radius = _num_param(params, "lug_round_radius", 0.373)
    outer_lobe_radius = _num_param(params, "outer_lobe_radius", 0.52)
    bolt_hole_diameter = _num_param(params, "bolt_hole_diameter", 0.315)
    bolt_count = max(1, int(round(_num_param(params, "bolt_count", 5.0))))
    bolt_circle_radius = _num_param(params, "bolt_circle_radius", max(outer_radius - 0.42, outer_radius * 0.78))
    lug_radius = _num_param(params, "lug_radius", max(bolt_hole_diameter * 1.6, outer_radius * 0.12))
    total_height = lower_bore_depth + thickness + upper_hub_height
    cutter_height = total_height + 2.0
    lower_collar_z = lower_collar_height / 2.0
    lower_hub_z = lower_collar_height + lower_hub_height / 2.0
    flange_z = lower_bore_depth + thickness / 2.0
    upper_web_z = lower_bore_depth + thickness + upper_web_height / 2.0
    upper_hub_z = lower_bore_depth + thickness + upper_web_height + upper_cylinder_height / 2.0
    return f"""from build123d import *
from math import cos, pi, sin

# Feature template: circular flange / flanged hub with bolt circle.
# Parameters are approximate drawing dimensions, usually inches for flange drawings.
bolt_positions = []
with BuildPart() as flange_part:
    with Locations((0, 0, {lower_collar_z:.6g})):
        Cylinder(radius={lower_collar_radius:.6g}, height={lower_collar_height:.6g})
    with Locations((0, 0, {lower_hub_z:.6g})):
        Cylinder(radius={hub_radius:.6g}, height={lower_hub_height:.6g})
    with Locations((0, 0, {flange_z:.6g})):
        Cylinder(radius={outer_radius:.6g}, height={thickness:.6g})

    # Rounded mounting lugs preserve the visible five-hole perimeter pattern.
    for index in range({bolt_count}):
        angle = 2 * pi * index / {bolt_count} - pi / 2
        x = {bolt_circle_radius:.6g} * cos(angle)
        y = {bolt_circle_radius:.6g} * sin(angle)
        bolt_positions.append((x, y))
        with Locations((x, y, {flange_z:.6g})):
            Cylinder(radius={lug_radius:.6g}, height={thickness:.6g})

    # Concentric stepped hub/bore stack from the section view.
    with Locations((0, 0, {upper_web_z:.6g})):
        Cone(bottom_radius={hub_radius:.6g}, top_radius={upper_hub_radius:.6g}, height={upper_web_height:.6g})
    with Locations((0, 0, {upper_hub_z:.6g})):
        Cylinder(radius={upper_hub_radius:.6g}, height={upper_cylinder_height:.6g})

    with Locations((0, 0, {total_height / 2.0:.6g})):
        Cylinder(radius={bore_diameter / 2.0:.6g}, height={cutter_height:.6g}, mode=Mode.SUBTRACT)
    with Locations((0, 0, {lower_bore_depth / 2.0:.6g})):
        Cylinder(radius={lower_bore_diameter / 2.0:.6g}, height={lower_bore_depth:.6g}, mode=Mode.SUBTRACT)
    for x, y in bolt_positions:
        with Locations((x, y, {flange_z:.6g})):
            Cylinder(radius={bolt_hole_diameter / 2.0:.6g}, height={cutter_height:.6g}, mode=Mode.SUBTRACT)

part = flange_part.part

# Machining finishing sequence: apply small blend/chamfer operations after
# primary turning, boring, and bolt drilling. These are intentionally guarded
# because approximate template dimensions can make some edge selections fail.
machining_operations = [
    "turn circular flange blank",
    "add five rounded bolt lugs on {bolt_count}-hole pattern",
    "turn lower collar diameter {lower_collar_radius * 2.0:.6g} to height {lower_collar_height:.6g}",
    "turn lower hub diameter {hub_radius * 2.0:.6g}",
    "cut upper conical web/chamfer over height {upper_web_height:.6g}",
    "turn upper pilot diameter {upper_hub_radius * 2.0:.6g}",
    "bore through diameter {bore_diameter:.6g}",
    "cut shallow upper bore relief before final through-bore finish",
    "counterbore lower diameter {lower_bore_diameter:.6g} to depth {lower_bore_depth:.6g}",
    "drill {bolt_count}x diameter {bolt_hole_diameter:.6g} through holes",
    "blend shoulders R{blend_radius:.6g}",
]
try:
    part = fillet(part.edges(), radius={min(blend_radius, 0.06):.6g})
except Exception:
    pass
"""


def _two_hole_stepped_block_template_code(params: dict[str, Any]) -> str:
    length = _num_param(params, "length", 50.0)
    depth = _num_param(params, "depth", 20.0)
    base_thickness = _num_param(params, "base_thickness", 10.0)
    height = _num_param(params, "height", 30.0)
    raised_height = max(height - base_thickness, base_thickness)
    left_width = _num_param(params, "left_block_width", 20.0)
    right_width = _num_param(params, "right_block_width", 15.0)
    hole_d = _num_param(params, "hole_diameter", 7.0)
    hole_z = base_thickness / 2.0 + raised_height * 0.5
    left_x = -length / 2.0 + left_width / 2.0
    right_x = length / 2.0 - right_width / 2.0
    relief_width = max(length - left_width - right_width, 5.0)
    relief_x = (left_x + right_x) / 2.0
    cutter_depth = depth + 6.0
    return f"""from build123d import *

# Feature template: GD&T-style stepped rectangular block with two through holes.
# Dimensions are approximate millimeters from the source drawing.
base = Box({length:.6g}, {depth:.6g}, {base_thickness:.6g})
left_block = Pos({left_x:.6g}, 0, {hole_z:.6g}) * Box({left_width:.6g}, {depth:.6g}, {raised_height:.6g})
right_block = Pos({right_x:.6g}, 0, {hole_z:.6g}) * Box({right_width:.6g}, {depth:.6g}, {raised_height:.6g})
part = base + left_block + right_block

# Central step/relief visible in the front and isometric views.
relief = Pos({relief_x:.6g}, 0, {hole_z:.6g}) * Box({relief_width:.6g}, {depth + 2.0:.6g}, {raised_height + 2.0:.6g})
part = part - relief

# Two diameter-through holes pass through the raised end blocks along the depth axis.
left_hole = Pos({left_x:.6g}, 0, {hole_z:.6g}) * Rot(90, 0, 0) * Cylinder(radius={hole_d / 2.0:.6g}, height={cutter_depth:.6g})
right_hole = Pos({right_x:.6g}, 0, {hole_z:.6g}) * Rot(90, 0, 0) * Cylinder(radius={hole_d / 2.0:.6g}, height={cutter_depth:.6g})
part = part - left_hole - right_hole
"""


def _num_param(params: dict[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number) or number <= 0:
        return float(default)
    return number


def execute_cad_code(
    code: str,
    output_step_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "tool_code",
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute build123d code and return a STEP artifact path."""
    output_path = Path(output_step_path) if output_step_path else _unique_step_path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    code_path = output_path.with_suffix(".py")
    code_path.write_text(code, encoding="utf-8")

    execution = execute_build123d(
        script_content=code,
        output_path=str(output_path),
        timeout=timeout,
    )
    return {
        "success": execution.success,
        "code_path": str(code_path),
        "step_path": str(output_path) if execution.success else None,
        "stdout": execution.stdout[-2000:],
        "stderr": execution.stderr[-4000:],
        "error_category": execution.error_category.value,
        "return_code": execution.return_code,
    }


def evaluate_step_against_drawing(
    step_path: str | Path,
    drawing_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    view_suffixes: tuple[str, str, str] = DEFAULT_VIEW_SUFFIXES,
) -> dict[str, Any]:
    """Evaluate STEP reprojection fidelity against an SVG orthographic drawing."""
    pipeline_config = PipelineConfig.from_yaml(config_path)
    triplet = load_training_svg_triplet(drawing_path, view_suffixes=view_suffixes)
    score, _ = evaluate_step_against_triplet(
        step_path,
        triplet,
        config=pipeline_config.reprojection,
        view_suffixes=view_suffixes,
    )
    return {
        "success": True,
        "step_path": str(step_path),
        "drawing_path": str(drawing_path),
        **_score_summary(score),
    }


def render_step_to_drawing(
    step_path: str | Path,
    output_dir: str | Path,
    stem: str = "reprojected",
    view_suffixes: tuple[str, str, str] = DEFAULT_VIEW_SUFFIXES,
    raster_size_px: int = 640,
) -> dict[str, Any]:
    """Render a STEP file into front/right/top SVG drawings and a PNG contact sheet."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    step_path = Path(step_path)
    shape = load_step_shape(step_path)
    config = ReprojectionConfig(raster_size_px=raster_size_px, overlay_enabled=False)

    view_polylines: dict[str, dict[str, list[list[tuple[float, float]]]]] = {}
    view_paths: dict[str, str] = {}
    for suffix in view_suffixes:
        visible, hidden = project_shape_to_view_polylines(shape, suffix=suffix, config=config)
        view_polylines[suffix] = {"visible": visible, "hidden": hidden}
        svg_path = output_path / f"{stem}_{suffix}.svg"
        svg_path.write_text(_view_svg(visible, hidden), encoding="utf-8")
        view_paths[suffix] = str(svg_path)

    layout_svg_path = output_path / f"{stem}_triplet.svg"
    layout_svg_path.write_text(_triplet_layout_svg(view_polylines, view_suffixes), encoding="utf-8")

    contact_sheet_path = output_path / f"{stem}_contact_sheet.png"
    _write_contact_sheet(view_polylines, contact_sheet_path, view_suffixes)
    source_contact_sheet_path = output_path / f"{stem}_source_contact_sheet.png"
    _write_contact_sheet(
        view_polylines,
        source_contact_sheet_path,
        view_suffixes,
        hidden_fill=(70, 70, 70),
        visible_fill="black",
        hidden_dashed=True,
    )

    return {
        "success": True,
        "step_path": str(step_path),
        "layout_svg_path": str(layout_svg_path),
        "contact_sheet_path": str(contact_sheet_path),
        "source_contact_sheet_path": str(source_contact_sheet_path),
        "view_svg_paths": view_paths,
        "views": {
            suffix: {
                "visible_polyline_count": len(view_polylines[suffix]["visible"]),
                "hidden_polyline_count": len(view_polylines[suffix]["hidden"]),
            }
            for suffix in view_suffixes
        },
    }


def render_feature_template_source_contact_sheet(
    template: str,
    dimensions: dict[str, Any] | None,
    output_dir: str | Path,
    stem: str = "feature_source",
) -> dict[str, Any]:
    """Render a template-aware source-fidelity contact sheet for section/profile drawings."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_template = _safe_file_label(template).lower()
    params = dimensions or {}
    if normalized_template not in {"flange", "circular_flange", "hub", "flanged_hub"}:
        return {
            "success": False,
            "template": template,
            "error": f"No feature source contact sheet renderer for template: {template}",
            "error_type": "UnsupportedFeatureSourceRenderer",
        }
    image_path = output_path / f"{stem}_{normalized_template}_source_contact_sheet.png"
    _write_flange_source_contact_sheet(params, image_path)
    return {
        "success": True,
        "template": "flange",
        "input_template": template,
        "source_contact_sheet_path": str(image_path),
    }


def summarize_step(step_path: str | Path) -> dict[str, Any]:
    """Return JSON-safe STEP properties."""
    validation = _validate_step_path(step_path, label="step_path")
    if validation:
        return validation
    props = analyze_step_file(step_path)
    return {"success": props.is_valid, "step_path": str(step_path), "properties": asdict(props)}


def compare_cad_parts(
    reference_step_path: str | Path,
    candidate_step_path: str | Path,
    min_bbox_iou: float = 0.0,
    min_extent_ratio: float = 0.96,
    min_volume_ratio: float = 0.97,
    min_surface_area_ratio: float = 0.97,
    center_tolerance: float = 0.05,
    translation_invariant: bool = True,
) -> dict[str, Any]:
    """Compare two STEP files and decide whether they represent the same part."""
    for label, path in (
        ("reference_step_path", reference_step_path),
        ("candidate_step_path", candidate_step_path),
    ):
        validation = _validate_step_path(path, label=label)
        if validation:
            return validation

    comparator = StepComparator(tolerance_mm=center_tolerance)
    raw = comparator.compare(candidate_step_path, reference_step_path)
    ref_props = analyze_step_file(reference_step_path)
    cand_props = analyze_step_file(candidate_step_path)

    extent_ratio = _bbox_extent_ratio(cand_props.bounding_box, ref_props.bounding_box)
    surface_area_ratio = _ratio(cand_props.surface_area, ref_props.surface_area)
    center_distance = _distance(cand_props.center_of_mass, ref_props.center_of_mass)
    center_ok = translation_invariant or center_distance <= center_tolerance
    equivalent = (
        bool(raw["generated_valid"])
        and bool(raw["reference_valid"])
        and float(raw["bounding_box_iou"]) >= min_bbox_iou
        and extent_ratio >= min_extent_ratio
        and float(raw["volume_ratio"]) >= min_volume_ratio
        and surface_area_ratio >= min_surface_area_ratio
        and center_ok
    )

    return {
        "success": True,
        "equivalent": equivalent,
        "reference_step_path": str(reference_step_path),
        "candidate_step_path": str(candidate_step_path),
        "thresholds": {
            "min_bbox_iou": min_bbox_iou,
            "min_extent_ratio": min_extent_ratio,
            "min_volume_ratio": min_volume_ratio,
            "min_surface_area_ratio": min_surface_area_ratio,
            "center_tolerance": center_tolerance,
            "translation_invariant": translation_invariant,
        },
        "metrics": {
            "bounding_box_iou": raw["bounding_box_iou"],
            "bbox_extent_ratio": extent_ratio,
            "volume_ratio": raw["volume_ratio"],
            "surface_area_ratio": surface_area_ratio,
            "center_distance": center_distance,
            "face_count_ratio": raw["face_count_ratio"],
        },
        "candidate_properties": asdict(cand_props),
        "reference_properties": asdict(ref_props),
    }


def _validate_step_path(path: str | Path, *, label: str) -> dict[str, Any] | None:
    step_path = Path(path)
    if step_path.suffix.lower() not in {".step", ".stp"}:
        return {
            "success": False,
            "error": f"{label} must be a STEP file path, got {step_path.suffix or '<no suffix>'}: {step_path}",
            "error_type": "InvalidStepPath",
            label: str(path),
            "guidance": "Use a path ending in .step or .stp. Use render_step_to_drawing for SVG/PNG drawing outputs.",
        }
    if not step_path.exists():
        return {
            "success": False,
            "error": f"{label} does not exist: {step_path}",
            "error_type": "FileNotFoundError",
            label: str(path),
        }
    return None


def encode_image_for_ollama(image_path: str | Path) -> tuple[str, str]:
    """Return base64 image data and MIME type for Ollama image messages."""
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    if path.suffix.lower() == ".svg":
        try:
            import cairosvg

            png_bytes = cairosvg.svg2png(url=str(path), output_width=1200, output_height=900)
            return base64.b64encode(png_bytes).decode("ascii"), "image/png"
        except Exception:
            data = path.read_bytes()
            return base64.b64encode(data).decode("ascii"), mime_type

    return base64.b64encode(path.read_bytes()).decode("ascii"), mime_type


def _triplet_summary(triplet) -> dict[str, Any]:
    return {
        suffix: {
            "view_box": view.view_box,
            "polyline_count": len(view.polylines),
            "visible_polyline_count": sum(1 for poly in view.polylines if poly.stroke == VISIBLE_STROKE),
            "hidden_polyline_count": sum(1 for poly in view.polylines if poly.stroke == HIDDEN_STROKE),
        }
        for suffix, view in triplet.views.items()
    }


def _score_summary(score) -> dict[str, Any]:
    return {
        "score": score.score,
        "views": {
            suffix: {
                "score": view_score.score,
                "visible_f1": view_score.visible.f1,
                "hidden_f1": view_score.hidden.f1,
                "visible_precision": view_score.visible.precision,
                "visible_recall": view_score.visible.recall,
                "hidden_precision": view_score.hidden.precision,
                "hidden_recall": view_score.hidden.recall,
            }
            for suffix, view_score in score.views.items()
        },
    }


def _unique_step_path(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"candidate_{int(time.time() * 1000)}.step"


def _view_svg(
    visible: list[list[tuple[float, float]]],
    hidden: list[list[tuple[float, float]]],
) -> str:
    bounds = _bounds(visible + hidden)
    min_x, min_y, width, height = bounds
    body = []
    for polyline in hidden:
        body.append(_polyline_svg(polyline, HIDDEN_STROKE, 0.5, dasharray="3 2"))
    for polyline in visible:
        body.append(_polyline_svg(polyline, VISIBLE_STROKE, 0.7))
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg version="1.1" viewBox="{min_x:.6g} {min_y:.6g} {width:.6g} {height:.6g}" '
        'xmlns="http://www.w3.org/2000/svg">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _triplet_layout_svg(
    view_polylines: dict[str, dict[str, list[list[tuple[float, float]]]]],
    view_suffixes: tuple[str, str, str],
) -> str:
    front, right, top = view_suffixes
    bounds = {
        suffix: _bounds(polys["visible"] + polys["hidden"])
        for suffix, polys in view_polylines.items()
    }
    max_span = max(
        max(width, height)
        for _, _, width, height in bounds.values()
    )
    gap = max(10.0, max_span * 0.08)
    margin = gap

    _, _, front_w, front_h = bounds[front]
    _, _, right_w, right_h = bounds[right]
    _, _, top_w, top_h = bounds[top]

    placements = {
        top: (margin + max(0.0, (front_w - top_w) / 2.0), margin),
        front: (margin, margin + top_h + gap),
        right: (margin + front_w + gap, margin + top_h + gap + max(0.0, (front_h - right_h) / 2.0)),
    }
    width = max(
        placements[top][0] + top_w,
        placements[front][0] + front_w,
        placements[right][0] + right_w,
    ) + margin
    height = max(
        placements[top][1] + top_h,
        placements[front][1] + front_h,
        placements[right][1] + right_h,
    ) + margin
    groups = []
    for suffix in (top, front, right):
        polys = view_polylines[suffix]
        min_x, min_y, _, _ = bounds[suffix]
        tx, ty = placements[suffix]
        matrix = f"matrix(1 0 0 1 {tx - min_x:.8g} {ty - min_y:.8g})"
        body = []
        for polyline in polys["hidden"]:
            body.append(_polyline_svg(polyline, HIDDEN_STROKE, 0.5, dasharray="3 2"))
        for polyline in polys["visible"]:
            body.append(_polyline_svg(polyline, VISIBLE_STROKE, 0.7))
        groups.append(f'  <g id="view_{html.escape(suffix)}" transform="{matrix}">\n' + "\n".join(body) + "\n  </g>")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg version="1.1" viewBox="0 0 {width:.6g} {height:.6g}" '
        'xmlns="http://www.w3.org/2000/svg">\n'
        + "\n".join(groups)
        + "\n</svg>\n"
    )


def _polyline_svg(
    points: list[tuple[float, float]],
    stroke: str,
    stroke_width: float,
    dasharray: str | None = None,
) -> str:
    points_attr = " ".join(f"{x:.6g},{y:.6g}" for x, y in points)
    dash = f' stroke-dasharray="{html.escape(dasharray)}"' if dasharray else ""
    return (
        f'    <polyline fill="none" points="{html.escape(points_attr)}" '
        f'stroke="{html.escape(stroke)}" stroke-linecap="round" '
        f'stroke-width="{stroke_width:.6g}"{dash} />'
    )


def _write_contact_sheet(
    view_polylines: dict[str, dict[str, list[list[tuple[float, float]]]]],
    output_path: Path,
    view_suffixes: tuple[str, str, str],
    hidden_fill: str | tuple[int, int, int] = "red",
    visible_fill: str | tuple[int, int, int] = "black",
    hidden_dashed: bool = False,
) -> None:
    thumb = 320
    pad = 18
    label_h = 26
    image = Image.new("RGB", (pad + len(view_suffixes) * (thumb + pad), thumb + label_h + 2 * pad), "white")
    draw = ImageDraw.Draw(image)
    for index, suffix in enumerate(view_suffixes):
        x = pad + index * (thumb + pad)
        y = pad
        draw.rectangle((x, y, x + thumb, y + thumb), outline="black")
        polys = view_polylines[suffix]
        bounds = _bounds(polys["visible"] + polys["hidden"])
        _draw_polylines(draw, polys["hidden"], (x, y, thumb, thumb), fill=hidden_fill, width=1, bounds=bounds, dashed=hidden_dashed)
        _draw_polylines(draw, polys["visible"], (x, y, thumb, thumb), fill=visible_fill, width=2, bounds=bounds)
        draw.text((x, y + thumb + 6), suffix, fill="black")
    image.save(output_path)


def _write_flange_source_contact_sheet(params: dict[str, Any], output_path: Path) -> None:
    outer_radius = _num_param(params, "outer_radius", _num_param(params, "outer_diameter", 5.46) / 2.0)
    thickness = _num_param(params, "thickness", 0.461)
    hub_radius = _num_param(params, "hub_radius", _num_param(params, "hub_diameter", 2.835) / 2.0)
    hub_height = _num_param(params, "hub_height", 0.866)
    upper_hub_radius = _num_param(params, "upper_hub_radius", _num_param(params, "upper_hub_diameter", 2.0) / 2.0)
    upper_hub_height = _num_param(params, "upper_hub_height", 1.101)
    upper_web_height = min(_num_param(params, "upper_web_height", 0.235), max(upper_hub_height * 0.55, 0.05))
    bore_radius = _num_param(params, "bore_diameter", 1.929) / 2.0
    lower_bore_radius = _num_param(params, "lower_bore_diameter", 2.165) / 2.0
    lower_bore_depth = _num_param(params, "lower_bore_depth", 0.866)
    lower_collar_radius = _num_param(params, "lower_collar_radius", _num_param(params, "lower_collar_diameter", 3.339) / 2.0)
    lower_collar_height = min(_num_param(params, "lower_collar_height", 0.235), max(lower_bore_depth * 0.45, 0.05))
    blend_radius = _num_param(params, "blend_radius", 0.079)
    bolt_hole_radius = _num_param(params, "bolt_hole_diameter", 0.315) / 2.0
    bolt_count = max(1, int(round(_num_param(params, "bolt_count", 5.0))))
    bolt_circle_radius = _num_param(params, "bolt_circle_radius", max(outer_radius - 0.42, outer_radius * 0.78))
    lug_radius = _num_param(params, "lug_radius", max(bolt_hole_radius * 3.2, outer_radius * 0.12))
    flange_bottom = lower_bore_depth
    flange_top = lower_bore_depth + thickness
    flange_mid = flange_bottom + thickness / 2.0
    web_top = flange_top + upper_web_height
    total_height = flange_top + upper_hub_height

    thumb = 320
    pad = 18
    label_h = 26
    image = Image.new("RGB", (pad + 3 * (thumb + pad), thumb + label_h + 2 * pad), "white")
    draw = ImageDraw.Draw(image)

    frames = {
        "f": (pad, pad, thumb, thumb),
        "r": (pad + thumb + pad, pad, thumb, thumb),
        "t": (pad + 2 * (thumb + pad), pad, thumb, thumb),
    }
    for label, (x, y, w, h) in frames.items():
        draw.rectangle((x, y, x + w, y + h), outline="black")
        draw.text((x, y + h + 6), label, fill="black")

    def map_profile(x_value: float, z_value: float, frame_key: str) -> tuple[float, float]:
        x, y, w, h = frames[frame_key]
        span_x = outer_radius * 2.35
        span_z = max(total_height * 1.25, 1.0)
        scale = min((w - 28) / span_x, (h - 28) / span_z)
        return (
            x + w / 2.0 + x_value * scale,
            y + h - 14 - z_value * scale,
        )

    def line_profile(points: list[tuple[float, float]], frame_key: str, fill="black", width=2) -> None:
        mapped = [map_profile(px, pz, frame_key) for px, pz in points]
        draw.line(mapped, fill=fill, width=width)

    def fill_profile(points: list[tuple[float, float]], frame_key: str, fill: str | tuple[int, int, int]) -> None:
        mapped = [map_profile(px, pz, frame_key) for px, pz in points]
        draw.polygon(mapped, fill=fill)

    # Section/profile view through the axis, matching the section A-A information.
    outline = [
        (-lower_collar_radius, 0.0),
        (-lower_collar_radius, lower_collar_height),
        (-hub_radius, lower_collar_height),
        (-hub_radius, flange_bottom),
        (-outer_radius, flange_bottom),
        (-outer_radius, flange_top),
        (-(hub_radius + blend_radius * 1.6), flange_top),
        (-upper_hub_radius, web_top),
        (-upper_hub_radius, total_height),
        (upper_hub_radius, total_height),
        (upper_hub_radius, web_top),
        (hub_radius + blend_radius * 1.6, flange_top),
        (outer_radius, flange_top),
        (outer_radius, flange_bottom),
        (hub_radius, flange_bottom),
        (hub_radius, lower_collar_height),
        (lower_collar_radius, lower_collar_height),
        (lower_collar_radius, 0.0),
        (-lower_collar_radius, 0.0),
    ]
    bore_void = [
        (-lower_bore_radius, 0.0),
        (-lower_bore_radius, lower_bore_depth),
        (-bore_radius, lower_bore_depth),
        (-bore_radius, total_height),
        (bore_radius, total_height),
        (bore_radius, lower_bore_depth),
        (lower_bore_radius, lower_bore_depth),
        (lower_bore_radius, 0.0),
    ]
    fill_profile(outline, "f", (244, 244, 244))
    fill_profile(bore_void, "f", "white")
    line_profile(outline, "f", width=2)
    hatch_z = flange_bottom + thickness * 0.12
    while hatch_z < web_top - upper_web_height * 0.08:
        line_profile([(-outer_radius * 0.9, hatch_z), (-(bore_radius + blend_radius), hatch_z + thickness * 0.32)], "f", fill=(95, 95, 95), width=1)
        line_profile([(bore_radius + blend_radius, hatch_z), (outer_radius * 0.9, hatch_z + thickness * 0.32)], "f", fill=(95, 95, 95), width=1)
        hatch_z += max(thickness * 0.22, 0.055)
    for z_value in (flange_bottom + thickness * 0.2, flange_top - thickness * 0.15, web_top - upper_web_height * 0.25):
        line_profile([(-outer_radius * 0.82, z_value), (-(bore_radius + blend_radius), z_value + thickness * 0.22)], "f", fill=(95, 95, 95), width=1)
        line_profile([(bore_radius + blend_radius, z_value), (outer_radius * 0.82, z_value + thickness * 0.22)], "f", fill=(95, 95, 95), width=1)
    for radius in (-bore_radius, bore_radius):
        line_profile([(radius, lower_bore_depth), (radius, total_height)], "f", width=2)
    for radius in (-lower_bore_radius, lower_bore_radius):
        line_profile([(radius, 0.0), (radius, lower_bore_depth)], "f", width=2)
    line_profile([(-lower_bore_radius, lower_bore_depth), (-bore_radius, lower_bore_depth)], "f", width=2)
    line_profile([(bore_radius, lower_bore_depth), (lower_bore_radius, lower_bore_depth)], "f", width=2)
    top_recess_bottom = max(web_top, total_height - max(upper_web_height, thickness) * 0.82)
    top_recess_radius = min(upper_hub_radius, bore_radius + max(blend_radius * 1.2, 0.08))
    for radius in (-top_recess_radius, top_recess_radius):
        line_profile([(radius, top_recess_bottom), (radius, total_height)], "f", fill=(35, 35, 35), width=1)
    line_profile([(-top_recess_radius, top_recess_bottom), (-bore_radius, top_recess_bottom)], "f", width=1)
    line_profile([(bore_radius, top_recess_bottom), (top_recess_radius, top_recess_bottom)], "f", width=1)
    line_profile([(-outer_radius, flange_bottom), (-lower_bore_radius, flange_bottom)], "f", width=1)
    line_profile([(lower_bore_radius, flange_bottom), (outer_radius, flange_bottom)], "f", width=1)
    line_profile([(-outer_radius, flange_top), (-bore_radius, flange_top)], "f", width=1)
    line_profile([(bore_radius, flange_top), (outer_radius, flange_top)], "f", width=1)
    line_profile([(-lower_collar_radius, lower_collar_height), (-lower_bore_radius, lower_collar_height)], "f", width=1)
    line_profile([(lower_bore_radius, lower_collar_height), (lower_collar_radius, lower_collar_height)], "f", width=1)
    fillet = max(blend_radius, total_height * 0.018, 0.04)
    for sign in (-1.0, 1.0):
        cx, cy = map_profile(sign * hub_radius, flange_bottom, "f")
        rr = fillet * min(
            (frames["f"][2] - 28) / (outer_radius * 2.35),
            (frames["f"][3] - 28) / max(total_height * 1.25, 1.0),
        )
        box = (cx - rr, cy - rr, cx + rr, cy + rr)
        draw.arc(box, 180 if sign > 0 else 270, 270 if sign > 0 else 360, fill="black", width=1)
        cx, cy = map_profile(sign * upper_hub_radius, web_top, "f")
        box = (cx - rr, cy - rr, cx + rr, cy + rr)
        draw.arc(box, 180 if sign > 0 else 270, 270 if sign > 0 else 360, fill="black", width=1)

    # Side/profile view emphasizes the stacked depths and flange tabs.
    side_outline = [
        (-lower_collar_radius, 0.0),
        (lower_collar_radius, 0.0),
        (lower_collar_radius, lower_collar_height),
        (hub_radius, lower_collar_height),
        (hub_radius, flange_bottom),
        (outer_radius, flange_bottom),
        (outer_radius, flange_top),
        (hub_radius + blend_radius * 1.6, flange_top),
        (upper_hub_radius, web_top),
        (upper_hub_radius, total_height),
        (-upper_hub_radius, total_height),
        (-upper_hub_radius, web_top),
        (-(hub_radius + blend_radius * 1.6), flange_top),
        (-outer_radius, flange_top),
        (-outer_radius, flange_bottom),
        (-hub_radius, flange_bottom),
        (-hub_radius, lower_collar_height),
        (-lower_collar_radius, lower_collar_height),
        (-lower_collar_radius, 0.0),
    ]
    fill_profile(side_outline, "r", (248, 248, 248))
    fill_profile(bore_void, "r", "white")
    line_profile(side_outline, "r", width=2)
    hatch_z = flange_bottom + thickness * 0.16
    while hatch_z < web_top - upper_web_height * 0.1:
        line_profile([(-outer_radius * 0.9, hatch_z), (-(bore_radius + blend_radius), hatch_z + thickness * 0.28)], "r", fill=(95, 95, 95), width=1)
        line_profile([(bore_radius + blend_radius, hatch_z), (outer_radius * 0.9, hatch_z + thickness * 0.28)], "r", fill=(95, 95, 95), width=1)
        hatch_z += max(thickness * 0.24, 0.06)
    for z_value in (flange_bottom + thickness * 0.25, flange_top - thickness * 0.1):
        line_profile([(-outer_radius * 0.85, z_value), (-(bore_radius + blend_radius), z_value + thickness * 0.18)], "r", fill=(95, 95, 95), width=1)
        line_profile([(bore_radius + blend_radius, z_value), (outer_radius * 0.85, z_value + thickness * 0.18)], "r", fill=(95, 95, 95), width=1)
    for radius in (-bore_radius, bore_radius):
        line_profile([(radius, lower_bore_depth), (radius, total_height)], "r", fill=(80, 80, 80), width=1)
    for radius in (-lower_bore_radius, lower_bore_radius):
        line_profile([(radius, 0.0), (radius, lower_bore_depth)], "r", fill=(80, 80, 80), width=1)
    for radius in (-top_recess_radius, top_recess_radius):
        line_profile([(radius, top_recess_bottom), (radius, total_height)], "r", fill=(80, 80, 80), width=1)
    line_profile([(-top_recess_radius, top_recess_bottom), (-bore_radius, top_recess_bottom)], "r", fill=(80, 80, 80), width=1)
    line_profile([(bore_radius, top_recess_bottom), (top_recess_radius, top_recess_bottom)], "r", fill=(80, 80, 80), width=1)
    line_profile([(-outer_radius, flange_mid), (outer_radius, flange_mid)], "r", fill=(80, 80, 80), width=1)

    # Top view with lugs, five bolt holes, and concentric bore/hub circles.
    x0, y0, w, h = frames["t"]
    scale = (w - 42) / (2.0 * (outer_radius + lug_radius * 0.35))
    cx = x0 + w / 2.0
    cy = y0 + h / 2.0

    def circle(center_x: float, center_y: float, radius: float, outline="black", width=2) -> None:
        px = cx + center_x * scale
        py = cy + center_y * scale
        rr = radius * scale
        draw.ellipse((px - rr, py - rr, px + rr, py + rr), outline=outline, width=width)

    circle(0.0, 0.0, outer_radius, width=2)
    for index in range(bolt_count):
        angle = 2.0 * math.pi * index / bolt_count - math.pi / 2.0
        bx = bolt_circle_radius * math.cos(angle)
        by = bolt_circle_radius * math.sin(angle)
        circle(bx, by, bolt_hole_radius, width=2)
    for radius in (lower_collar_radius, hub_radius, lower_bore_radius, upper_hub_radius, bore_radius):
        circle(0.0, 0.0, radius, width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_polylines(
    draw: ImageDraw.ImageDraw,
    polylines: list[list[tuple[float, float]]],
    frame: tuple[int, int, int, int],
    fill: str | tuple[int, int, int],
    width: int,
    bounds: tuple[float, float, float, float] | None = None,
    dashed: bool = False,
) -> None:
    if not polylines:
        return
    min_x, min_y, span_x, span_y = bounds or _bounds(polylines)
    x0, y0, frame_w, frame_h = frame
    scale = min((frame_w - 20) / max(span_x, 1.0), (frame_h - 20) / max(span_y, 1.0))
    for polyline in polylines:
        points = [
            (
                x0 + 10 + (x - min_x) * scale,
                y0 + 10 + (y - min_y) * scale,
            )
            for x, y in polyline
        ]
        if len(points) >= 2:
            if dashed:
                _draw_dashed_line(draw, points, fill=fill, width=width)
            else:
                draw.line(points, fill=fill, width=width)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str | tuple[int, int, int],
    width: int,
    dash: float = 7.0,
    gap: float = 5.0,
) -> None:
    for start, end in zip(points, points[1:]):
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        ux = dx / length
        uy = dy / length
        position = 0.0
        while position < length:
            segment_end = min(position + dash, length)
            draw.line(
                (
                    x0 + ux * position,
                    y0 + uy * position,
                    x0 + ux * segment_end,
                    y0 + uy * segment_end,
                ),
                fill=fill,
                width=width,
            )
            position += dash + gap


def _bounds(polylines: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = [x for poly in polylines for x, _ in poly]
    ys = [y for poly in polylines for _, y in poly]
    if not xs or not ys:
        return (0.0, 0.0, 1.0, 1.0)
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return (min_x, min_y, max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6))


def _open_drawing_as_rgb(path: Path) -> Image.Image:
    if path.suffix.lower() == ".svg":
        try:
            import cairosvg

            png_bytes = cairosvg.svg2png(url=str(path), output_width=1400)
            return Image.open(BytesIO(png_bytes)).convert("RGB")
        except Exception:
            pass
    return Image.open(path).convert("RGB")


def _infer_sheet_regions(gray: Image.Image, dark_threshold: int) -> list[dict[str, Any]]:
    width, height = gray.size
    pixels = gray.load()
    margin_x = max(4, int(width * 0.045))
    margin_y = max(4, int(height * 0.045))
    regions: list[dict[str, Any]] = []

    edge_specs = [
        ("sheet_border_top", (0, 0, width, margin_y)),
        ("sheet_border_bottom", (0, height - margin_y, width, height)),
        ("sheet_border_left", (0, 0, margin_x, height)),
        ("sheet_border_right", (width - margin_x, 0, width, height)),
    ]
    for label, bbox in edge_specs:
        density = _dark_density(pixels, bbox, dark_threshold)
        if density > 0.006:
            regions.append(
                {
                    "type": label,
                    "bbox": bbox,
                    "confidence": min(0.95, 0.35 + density * 4.0),
                    "source": "edge_dark_density",
                }
            )

    aspect = width / max(height, 1)
    title_bbox = (int(width * 0.58), int(height * 0.70), width, height)
    title_density = _dark_density(pixels, title_bbox, dark_threshold)
    if aspect > 1.05 or title_density > 0.015:
        regions.append(
            {
                "type": "title_block_candidate",
                "bbox": title_bbox,
                "confidence": min(0.9, 0.35 + title_density * 3.0),
                "source": "lower_right_sheet_heuristic",
            }
        )
    return regions


def _infer_annotation_candidate_regions(
    gray: Image.Image,
    excluded_regions: list[dict[str, Any]],
    dark_threshold: int,
) -> list[dict[str, Any]]:
    import numpy as np
    from scipy import ndimage

    width, height = gray.size
    total_area = max(width * height, 1)
    dark = np.array(gray, dtype=np.uint8) < dark_threshold
    for region in excluded_regions:
        x1, y1, x2, y2 = _clamped_bbox(region["bbox"], width, height)
        dark[y1:y2, x1:x2] = False

    labels, count = ndimage.label(dark)
    objects = ndimage.find_objects(labels)
    candidates: list[dict[str, Any]] = []
    for index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        x1, x2 = int(x_slice.start), int(x_slice.stop)
        y1, y2 = int(y_slice.start), int(y_slice.stop)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 3 or box_h < 3:
            continue
        area = int((labels[slices] == index).sum())
        bbox_area = max(box_w * box_h, 1)
        area_ratio = area / total_area
        fill_ratio = area / bbox_area
        aspect = max(box_w / max(box_h, 1), box_h / max(box_w, 1))

        likely_annotation = (
            area_ratio <= 0.010
            and box_w <= width * 0.36
            and box_h <= height * 0.22
            and (aspect >= 2.8 or fill_ratio <= 0.35 or area <= 220)
        )
        if likely_annotation:
            confidence = min(0.82, 0.38 + min(aspect, 8.0) * 0.035 + (1.0 - fill_ratio) * 0.18)
            candidates.append(
                {
                    "type": "annotation_candidate",
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "source": "connected_component_heuristic",
                    "area_px": area,
                    "fill_ratio": round(fill_ratio, 4),
                }
            )

    candidates.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    return candidates[:80]


def _mask_regions(image: Image.Image, regions: list[dict[str, Any]]) -> Image.Image:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    width, height = image.size
    for region in regions:
        bbox = _clamped_bbox(region["bbox"], width, height)
        draw.rectangle(bbox, fill="white")
    return masked


def _isolated_linework_image(image: Image.Image, dark_threshold: int) -> Image.Image:
    gray = ImageOps.grayscale(image)
    linework = gray.point(lambda value: 0 if value < dark_threshold else 255)
    return linework.convert("RGB")


def _region_overlay(image: Image.Image, regions: list[dict[str, Any]]) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    colors = {
        "annotation_candidate": (230, 80, 40),
        "title_block_candidate": (40, 130, 230),
    }
    for region in regions:
        bbox = _clamped_bbox(region["bbox"], width, height)
        color = colors.get(str(region.get("type")), (60, 170, 90))
        draw.rectangle(bbox, outline=color, width=3)
    return overlay


def _infer_view_frames(linework: Image.Image, dark_threshold: int) -> list[dict[str, Any]]:
    import numpy as np
    from scipy import ndimage

    gray = ImageOps.grayscale(linework)
    width, height = gray.size
    total_area = max(width * height, 1)
    dark = np.array(gray, dtype=np.uint8) < dark_threshold
    if not dark.any():
        return []

    join_iterations = max(1, min(4, min(width, height) // 120))
    joined = ndimage.binary_dilation(dark, iterations=join_iterations)
    labels, _count = ndimage.label(joined)
    objects = ndimage.find_objects(labels)
    frames: list[dict[str, Any]] = []
    for label_index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        x1, x2 = int(x_slice.start), int(x_slice.stop)
        y1, y2 = int(y_slice.start), int(y_slice.stop)
        box_w = x2 - x1
        box_h = y2 - y1
        bbox_area = max(box_w * box_h, 1)
        component_area = int((labels[slices] == label_index).sum())
        bbox_area_ratio = bbox_area / total_area
        fill_ratio = component_area / bbox_area
        touches_edge = x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2

        likely_view = (
            box_w >= max(12, width * 0.04)
            and box_h >= max(12, height * 0.04)
            and bbox_area_ratio >= 0.003
            and bbox_area_ratio <= 0.65
            and not (touches_edge and bbox_area_ratio > 0.18)
        )
        if not likely_view:
            continue

        frames.append(
            {
                "frame_id": f"view_{len(frames):03d}",
                "bbox": [x1, y1, x2, y2],
                "bbox_norm_image": _bbox_norm([x1, y1, x2, y2], width, height),
                "center_px": _bbox_center([x1, y1, x2, y2]),
                "origin_image_px": [x1, y1],
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "units": "px",
                "source": "sheet_masked_linework_component",
                "component_area_px": component_area,
                "bbox_area_ratio": round(bbox_area_ratio, 6),
                "fill_ratio": round(fill_ratio, 4),
            }
        )

    frames.sort(key=lambda frame: (frame["bbox"][1], frame["bbox"][0]))
    for index, frame in enumerate(frames[:12]):
        frame["frame_id"] = f"view_{index:03d}"
    return frames[:12]


def _dark_density(
    pixels: Any,
    bbox: tuple[int, int, int, int],
    dark_threshold: int,
) -> float:
    x1, y1, x2, y2 = bbox
    dark_count = 0
    total = max((x2 - x1) * (y2 - y1), 1)
    step = max(1, int(math.sqrt(total / 8000)))
    sampled = 0
    for y in range(y1, y2, step):
        for x in range(x1, x2, step):
            sampled += 1
            if pixels[x, y] < dark_threshold:
                dark_count += 1
    return dark_count / max(sampled, 1)


def _clamped_bbox(
    bbox: tuple[int, int, int, int] | list[int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (
        x1,
        y1,
        x2,
        y2,
    )


def _region_record(
    region: dict[str, Any],
    width: int,
    height: int,
    region_id: str,
) -> dict[str, Any]:
    x1, y1, x2, y2 = _clamped_bbox(region["bbox"], width, height)
    bbox = [x1, y1, x2, y2]
    return {
        "region_id": region_id,
        **{key: value for key, value in region.items() if key != "bbox"},
        "bbox": bbox,
        "bbox_norm_image": _bbox_norm(bbox, width, height),
        "center_px": _bbox_center(bbox),
        "area_ratio": round(((x2 - x1) * (y2 - y1)) / max(width * height, 1), 6),
    }


def _attach_view_references_to_regions(
    regions: list[dict[str, Any]],
    view_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for region in regions:
        view_frame = _nearest_view_frame(region.get("center_px"), view_frames)
        if not view_frame:
            continue
        center_px = region["center_px"]
        region["nearest_view_frame_id"] = view_frame["frame_id"]
        region["center_view_px"] = _point_in_view_frame(center_px, view_frame)
        region["center_view_norm"] = _point_in_view_norm(center_px, view_frame)
    return regions


def _callout_candidate_from_region(
    region: dict[str, Any],
    view_frames: list[dict[str, Any]],
) -> dict[str, Any] | None:
    view_frame = _nearest_view_frame(region.get("center_px"), view_frames)
    if not view_frame:
        return None

    endpoint = _nearest_point_on_bbox(region["center_px"], view_frame["bbox"])
    confidence = float(region.get("confidence", 0.5))
    return {
        "callout_id": str(region["region_id"]).replace("region_", "callout_"),
        "source_region_id": region["region_id"],
        "source_region_type": region.get("type", "unknown"),
        "view_frame_id": view_frame["frame_id"],
        "target_endpoint_image_px": endpoint,
        "target_endpoint_view_px": _point_in_view_frame(endpoint, view_frame),
        "target_endpoint_view_norm": _point_in_view_norm(endpoint, view_frame),
        "source_region_center_image_px": region["center_px"],
        "source_region_center_view_px": _point_in_view_frame(
            region["center_px"],
            view_frame,
        ),
        "source_region_center_view_norm": _point_in_view_norm(region["center_px"], view_frame),
        "association_method": "nearest_view_bbox_projection",
        "confidence": round(min(0.5, confidence * 0.55), 3),
        "caveat": "Heuristic anchor only; verify leader/callout line pixels in the original image.",
    }


def _image_coordinate_system(width: int, height: int) -> dict[str, Any]:
    return {
        "frame_id": "image_px",
        "origin": "top_left",
        "x_axis": "right",
        "y_axis": "down",
        "units": "px",
        "width": width,
        "height": height,
        "normalized_frame": {
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
        },
    }


def _artifact_transforms() -> dict[str, dict[str, str]]:
    return {
        artifact: {
            "from_frame": "image_px",
            "to_frame": "image_px",
            "transform": "identity",
        }
        for artifact in [
            "original_path",
            "sheet_masked_path",
            "annotation_masked_path",
            "physical_linework_path",
            "overlay_path",
        ]
    }


def _view_crop_sources(artifacts: dict[str, Any]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for source_name, artifact_key in {
        "original": "original_path",
        "sheet_masked": "sheet_masked_path",
        "annotation_masked": "annotation_masked_path",
        "physical_linework": "physical_linework_path",
    }.items():
        artifact_path = artifacts.get(artifact_key)
        if artifact_path and Path(artifact_path).exists():
            sources[source_name] = Path(artifact_path)
    return sources


def _projection_conventions() -> dict[str, Any]:
    return {
        "third_angle": {
            "description": "Common in US/Canada practice; the projected view is placed on the same side as the viewing direction.",
            "layout_rules": {
                "top": "above_front",
                "bottom": "below_front",
                "right": "right_of_front",
                "left": "left_of_front",
            },
        },
        "first_angle": {
            "description": "Common in ISO/European practice; the projected view is placed on the opposite side of the front view.",
            "layout_rules": {
                "top": "below_front",
                "bottom": "above_front",
                "right": "left_of_front",
                "left": "right_of_front",
            },
        },
        "notes": [
            "A drawing should normally indicate the projection method with a projection symbol or title-block note.",
            "Do not mix first-angle and third-angle role assignments within one drawing unless the drawing explicitly says to.",
        ],
    }


def _projection_system_hypotheses(
    view_frames: list[dict[str, Any]],
    projection_system: str,
) -> list[dict[str, Any]]:
    front = _select_front_view_frame(view_frames)
    if not front:
        return []

    requested = _normalized_projection_system(projection_system)
    systems = [requested] if requested in {"third_angle", "first_angle"} else ["third_angle", "first_angle"]
    hypotheses: list[dict[str, Any]] = []
    for system in systems:
        assignments = {
            frame["frame_id"]: _projected_view_role(frame, front, system)
            for frame in view_frames
        }
        assignments[front["frame_id"]] = "front"
        confidence = _projection_assignment_confidence(assignments, requested == system)
        hypotheses.append(
            {
                "projection_system": system,
                "confidence": confidence,
                "front_view_frame_id": front["frame_id"],
                "assignments": assignments,
                "basis": (
                    "Layout hypothesis from inferred view-frame centers; prefer title-block "
                    "projection symbols or explicit view labels when available."
                ),
            }
        )
    return hypotheses


def _select_front_view_frame(view_frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not view_frames:
        return None
    centers = [frame["center_px"] for frame in view_frames]
    mean_x = sum(point[0] for point in centers) / len(centers)
    mean_y = sum(point[1] for point in centers) / len(centers)

    def score(frame: dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = frame["bbox"]
        area = max((x2 - x1) * (y2 - y1), 1)
        center_distance = _distance(frame["center_px"], [mean_x, mean_y])
        return (-area, center_distance)

    return min(view_frames, key=score)


def _projected_view_role(
    frame: dict[str, Any],
    front: dict[str, Any],
    projection_system: str,
) -> str:
    if frame["frame_id"] == front["frame_id"]:
        return "front"

    dx = float(frame["center_px"][0]) - float(front["center_px"][0])
    dy = float(frame["center_px"][1]) - float(front["center_px"][1])
    front_x1, front_y1, front_x2, front_y2 = front["bbox"]
    x_threshold = max((front_x2 - front_x1) * 0.35, 12.0)
    y_threshold = max((front_y2 - front_y1) * 0.35, 12.0)

    if abs(dx) < x_threshold and dy < -y_threshold:
        return "top" if projection_system == "third_angle" else "bottom"
    if abs(dx) < x_threshold and dy > y_threshold:
        return "bottom" if projection_system == "third_angle" else "top"
    if dx > x_threshold and abs(dy) < y_threshold:
        return "right" if projection_system == "third_angle" else "left"
    if dx < -x_threshold and abs(dy) < y_threshold:
        return "left" if projection_system == "third_angle" else "right"
    if dy < -y_threshold:
        return "upper_auxiliary_or_detail"
    if dy > y_threshold:
        return "lower_auxiliary_or_detail"
    return "side_or_detail_unresolved"


def _projection_assignment_confidence(assignments: dict[str, str], requested: bool) -> float:
    resolved_roles = {
        role
        for role in assignments.values()
        if role in {"front", "top", "bottom", "right", "left"}
    }
    base = 0.36 + 0.08 * min(len(resolved_roles), 4)
    if requested:
        base += 0.12
    if len(assignments) == 1:
        base = min(base, 0.34)
    return round(min(base, 0.78), 3)


def _normalized_projection_system(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"third", "thirdangle", "third_angle", "3rd", "3rd_angle"}:
        return "third_angle"
    if normalized in {"first", "firstangle", "first_angle", "1st", "1st_angle"}:
        return "first_angle"
    return "unknown"


def _regions_by_nearest_view(regions: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for region in regions:
        view_id = region.get("nearest_view_frame_id")
        region_id = region.get("region_id")
        if view_id and region_id and region.get("type") == "annotation_candidate":
            grouped.setdefault(str(view_id), []).append(str(region_id))
    return grouped


def _callouts_by_view(callouts: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for callout in callouts:
        view_id = callout.get("view_frame_id")
        callout_id = callout.get("callout_id")
        if view_id and callout_id:
            grouped.setdefault(str(view_id), []).append(str(callout_id))
    return grouped


def _padded_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    padding_px: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return _clamped_bbox(
        (x1 - padding_px, y1 - padding_px, x2 + padding_px, y2 + padding_px),
        width,
        height,
    )


def _nearest_view_frame(
    point: list[float] | None,
    view_frames: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not point or not view_frames:
        return None
    return min(
        view_frames,
        key=lambda frame: _distance(point, _nearest_point_on_bbox(point, frame["bbox"])),
    )


def _nearest_point_on_bbox(
    point: list[float],
    bbox: list[int] | tuple[int, int, int, int],
) -> list[float]:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(value) for value in bbox]
    clamped_x = min(max(x, x1), x2)
    clamped_y = min(max(y, y1), y2)
    if x1 <= x <= x2 and y1 <= y <= y2:
        distances = [
            (abs(x - x1), [x1, y]),
            (abs(x2 - x), [x2, y]),
            (abs(y - y1), [x, y1]),
            (abs(y2 - y), [x, y2]),
        ]
        return [round(value, 3) for value in min(distances, key=lambda item: item[0])[1]]
    return [round(clamped_x, 3), round(clamped_y, 3)]


def _point_in_view_frame(point: list[float], view_frame: dict[str, Any]) -> list[float]:
    x1, y1, _x2, _y2 = view_frame["bbox"]
    return [round(float(point[0]) - x1, 3), round(float(point[1]) - y1, 3)]


def _point_in_view_norm(point: list[float], view_frame: dict[str, Any]) -> list[float]:
    x1, y1, x2, y2 = view_frame["bbox"]
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    view_px = _point_in_view_frame(point, view_frame)
    return [
        round(min(max(view_px[0] / width, 0.0), 1.0), 6),
        round(min(max(view_px[1] / height, 0.0), 1.0), 6),
    ]


def _bbox_center(bbox: list[int] | tuple[int, int, int, int]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)]


def _bbox_norm(
    bbox: list[int] | tuple[int, int, int, int],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / max(width, 1), 6),
        round(y1 / max(height, 1), 6),
        round(x2 / max(width, 1), 6),
        round(y2 / max(height, 1), 6),
    ]


def _region_counts(regions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        region_type = str(region.get("type", "unknown"))
        counts[region_type] = counts.get(region_type, 0) + 1
    return counts


def _safe_file_label(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe[:80] or "drawing"


def _ratio(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    low = min(abs(a), abs(b))
    high = max(abs(a), abs(b))
    return low / high if high > 0 else 0.0


def _bbox_extent_ratio(
    a: list[list[float]] | None,
    b: list[list[float]] | None,
) -> float:
    if not a or not b or len(a) != 2 or len(b) != 2:
        return 0.0
    left_extents = sorted(abs(a[1][index] - a[0][index]) for index in range(3))
    right_extents = sorted(abs(b[1][index] - b[0][index]) for index in range(3))
    ratios = [_ratio(left, right) for left, right in zip(left_extents, right_extents)]
    return min(ratios) if ratios else 0.0


def _distance(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None or len(a) != len(b):
        return math.inf
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
