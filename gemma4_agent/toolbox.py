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
        parsed = json.loads(arguments or "{}")
    else:
        parsed = dict(arguments)

    tool_map = {
        "inspect_drawing": lambda: inspect_drawing(parsed["drawing_path"]),
        "run_deterministic_reconstruction": lambda: run_deterministic_reconstruction(
            drawing_path=parsed["drawing_path"],
            output_dir=runtime.output_dir / "deterministic",
            config_path=runtime.config_path,
            view_suffixes=runtime.view_suffixes,
            timeout=runtime.execution_timeout,
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

    return {
        "success": True,
        "step_path": str(step_path),
        "layout_svg_path": str(layout_svg_path),
        "contact_sheet_path": str(contact_sheet_path),
        "view_svg_paths": view_paths,
        "views": {
            suffix: {
                "visible_polyline_count": len(view_polylines[suffix]["visible"]),
                "hidden_polyline_count": len(view_polylines[suffix]["hidden"]),
            }
            for suffix in view_suffixes
        },
    }


def summarize_step(step_path: str | Path) -> dict[str, Any]:
    """Return JSON-safe STEP properties."""
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
    for polyline in visible:
        body.append(_polyline_svg(polyline, VISIBLE_STROKE, 0.7))
    for polyline in hidden:
        body.append(_polyline_svg(polyline, HIDDEN_STROKE, 0.5, dasharray="3 2"))
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
        for polyline in polys["visible"]:
            body.append(_polyline_svg(polyline, VISIBLE_STROKE, 0.7))
        for polyline in polys["hidden"]:
            body.append(_polyline_svg(polyline, HIDDEN_STROKE, 0.5, dasharray="3 2"))
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
) -> None:
    thumb = 256
    pad = 18
    label_h = 26
    image = Image.new("RGB", (pad + len(view_suffixes) * (thumb + pad), thumb + label_h + 2 * pad), "white")
    draw = ImageDraw.Draw(image)
    for index, suffix in enumerate(view_suffixes):
        x = pad + index * (thumb + pad)
        y = pad
        draw.rectangle((x, y, x + thumb, y + thumb), outline="black")
        polys = view_polylines[suffix]
        _draw_polylines(draw, polys["visible"], (x, y, thumb, thumb), fill="black", width=2)
        _draw_polylines(draw, polys["hidden"], (x, y, thumb, thumb), fill="red", width=1)
        draw.text((x, y + thumb + 6), suffix, fill="black")
    image.save(output_path)


def _draw_polylines(
    draw: ImageDraw.ImageDraw,
    polylines: list[list[tuple[float, float]]],
    frame: tuple[int, int, int, int],
    fill: str,
    width: int,
) -> None:
    if not polylines:
        return
    min_x, min_y, span_x, span_y = _bounds(polylines)
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
            draw.line(points, fill=fill, width=width)


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
