"""Reproject generated STEP solids back to orthographic line drawings."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from src.reconstruction.total_view_dataset import OrthographicTriplet, SvgOrthographicView
from src.schemas.pipeline_config import ReprojectionConfig


VISIBLE_STROKES = frozenset({"black", "#000000"})
HIDDEN_STROKE = "red"


@dataclass(frozen=True)
class LineMatchMetrics:
    """Tolerance-aware line raster comparison metrics."""

    source_pixels: int
    predicted_pixels: int
    precision: float
    recall: float
    f1: float
    iou: float


@dataclass(frozen=True)
class ViewReprojectionScore:
    """Visible and hidden line scores for one orthographic view."""

    suffix: str
    visible: LineMatchMetrics
    hidden: LineMatchMetrics
    score: float


@dataclass(frozen=True)
class CaseReprojectionScore:
    """Aggregate reprojection score for one case."""

    case_id: str
    views: dict[str, ViewReprojectionScore]
    score: float


@dataclass(frozen=True)
class RenderedComparison:
    """Rasterized source and prediction masks for debugging."""

    source_visible: np.ndarray
    source_hidden: np.ndarray
    predicted_visible: np.ndarray
    predicted_hidden: np.ndarray
    overlay: Image.Image | None


def evaluate_step_against_triplet(
    step_path: str | Path,
    triplet: OrthographicTriplet,
    config: ReprojectionConfig | None = None,
    view_suffixes: tuple[str, str, str] = ("f", "r", "t"),
) -> tuple[CaseReprojectionScore, dict[str, RenderedComparison]]:
    """Compare a generated STEP file against the source line drawings."""
    reprojection_config = config or ReprojectionConfig()
    shape = load_step_shape(step_path)

    view_scores: dict[str, ViewReprojectionScore] = {}
    rendered: dict[str, RenderedComparison] = {}
    for suffix in view_suffixes:
        view = triplet.views[suffix]
        source_visible = [
            polyline.points
            for polyline in view.polylines
            if polyline.stroke in VISIBLE_STROKES
        ]
        source_hidden = [
            polyline.points
            for polyline in view.polylines
            if polyline.stroke == HIDDEN_STROKE
        ]

        projected_visible, projected_hidden = project_shape_to_view_polylines(
            shape,
            suffix=suffix,
            config=reprojection_config,
        )
        comparison = render_comparison(
            source_view=view,
            source_visible_polylines=source_visible,
            source_hidden_polylines=source_hidden,
            predicted_visible_polylines=projected_visible,
            predicted_hidden_polylines=projected_hidden,
            config=reprojection_config,
        )

        visible_metrics = compare_line_masks(
            comparison.source_visible,
            comparison.predicted_visible,
            tolerance_px=reprojection_config.match_tolerance_px,
        )
        hidden_metrics = compare_line_masks(
            comparison.source_hidden,
            comparison.predicted_hidden,
            tolerance_px=reprojection_config.match_tolerance_px,
        )
        score = (visible_metrics.f1 + hidden_metrics.f1) / 2.0
        view_scores[suffix] = ViewReprojectionScore(
            suffix=suffix,
            visible=visible_metrics,
            hidden=hidden_metrics,
            score=score,
        )
        rendered[suffix] = comparison

    overall = sum(view.score for view in view_scores.values()) / len(view_scores)
    return CaseReprojectionScore(case_id=triplet.case_id, views=view_scores, score=overall), rendered


def load_step_shape(step_path: str | Path):
    """Load a STEP B-Rep shape using OpenCASCADE."""
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise ValueError(f"Could not read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP file contains no shape: {step_path}")
    return shape


def project_shape_to_view_polylines(
    shape,
    suffix: str,
    config: ReprojectionConfig,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Project visible and hidden edges into 2D polylines."""
    from OCP.HLRAlgo import HLRAlgo_Projector
    from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(HLRAlgo_Projector(_projector_axis(suffix)))
    algo.Update()
    algo.Hide()

    projected = HLRBRep_HLRToShape(algo)
    visible = _shape_to_polylines(projected.VCompound(), config)
    hidden_raw = _shape_to_polylines(projected.HCompound(), config)
    return visible, hidden_raw


def render_comparison(
    source_view: SvgOrthographicView,
    source_visible_polylines: list[list[tuple[float, float]]],
    source_hidden_polylines: list[list[tuple[float, float]]],
    predicted_visible_polylines: list[list[tuple[float, float]]],
    predicted_hidden_polylines: list[list[tuple[float, float]]],
    config: ReprojectionConfig,
) -> RenderedComparison:
    """Render aligned raster masks for source and predicted line work."""
    source_bbox = _polyline_bounds(source_visible_polylines)
    predicted_bbox = _polyline_bounds(predicted_visible_polylines) or _polyline_bounds(
        predicted_hidden_polylines
    )
    if source_bbox is None:
        source_bbox = _view_box_to_bounds(source_view.view_box)
    if predicted_bbox is None:
        predicted_bbox = source_bbox

    normalized_source_visible = _translate_polylines(source_visible_polylines, source_bbox[:2])
    normalized_source_hidden = _translate_polylines(source_hidden_polylines, source_bbox[:2])
    normalized_predicted_visible = _translate_polylines(
        predicted_visible_polylines,
        predicted_bbox[:2],
    )
    normalized_predicted_hidden = _translate_polylines(
        predicted_hidden_polylines,
        predicted_bbox[:2],
    )

    canvas_width = max(source_bbox[2], predicted_bbox[2], 1.0)
    canvas_height = max(source_bbox[3], predicted_bbox[3], 1.0)
    image_size = _image_size(canvas_width, canvas_height, config)

    source_visible_mask = _draw_mask(
        normalized_source_visible,
        canvas_width,
        canvas_height,
        image_size=image_size,
        config=config,
    )
    source_hidden_mask = _draw_mask(
        normalized_source_hidden,
        canvas_width,
        canvas_height,
        image_size=image_size,
        config=config,
    )
    predicted_visible_mask = _draw_mask(
        normalized_predicted_visible,
        canvas_width,
        canvas_height,
        image_size=image_size,
        config=config,
    )
    predicted_hidden_raw = _draw_mask(
        normalized_predicted_hidden,
        canvas_width,
        canvas_height,
        image_size=image_size,
        config=config,
    )
    subtract = ndimage.binary_dilation(
        predicted_visible_mask,
        iterations=config.hidden_line_subtract_dilation_px,
    )
    predicted_hidden_mask = predicted_hidden_raw & ~subtract

    overlay = None
    if config.overlay_enabled:
        overlay = render_overlay(
            source_visible_mask,
            source_hidden_mask,
            predicted_visible_mask,
            predicted_hidden_mask,
        )

    return RenderedComparison(
        source_visible=source_visible_mask,
        source_hidden=source_hidden_mask,
        predicted_visible=predicted_visible_mask,
        predicted_hidden=predicted_hidden_mask,
        overlay=overlay,
    )


def compare_line_masks(
    source_mask: np.ndarray,
    predicted_mask: np.ndarray,
    tolerance_px: int,
) -> LineMatchMetrics:
    """Compute tolerance-aware line similarity metrics."""
    source = source_mask.astype(bool)
    predicted = predicted_mask.astype(bool)

    source_count = int(source.sum())
    predicted_count = int(predicted.sum())

    dilated_source = ndimage.binary_dilation(source, iterations=tolerance_px)
    dilated_predicted = ndimage.binary_dilation(predicted, iterations=tolerance_px)

    matched_predicted = int((predicted & dilated_source).sum())
    matched_source = int((source & dilated_predicted).sum())

    if predicted_count == 0:
        precision = 1.0 if source_count == 0 else 0.0
    else:
        precision = matched_predicted / predicted_count

    if source_count == 0:
        recall = 1.0 if predicted_count == 0 else 0.0
    else:
        recall = matched_source / source_count

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    union = int((source | predicted).sum())
    iou = (int((source & predicted).sum()) / union) if union else 1.0

    return LineMatchMetrics(
        source_pixels=source_count,
        predicted_pixels=predicted_count,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        iou=float(iou),
    )


def render_overlay(
    source_visible: np.ndarray,
    source_hidden: np.ndarray,
    predicted_visible: np.ndarray,
    predicted_hidden: np.ndarray,
) -> Image.Image:
    """Render a false-color overlay for debugging."""
    rgb = np.full((*source_visible.shape, 3), 255, dtype=np.uint8)

    rgb[source_visible] = np.array([45, 45, 45], dtype=np.uint8)
    rgb[source_hidden] = np.array([210, 50, 50], dtype=np.uint8)

    predicted_visible_only = predicted_visible & ~source_visible
    predicted_hidden_only = predicted_hidden & ~source_hidden
    rgb[predicted_visible_only] = np.array([40, 120, 220], dtype=np.uint8)
    rgb[predicted_hidden_only] = np.array([240, 165, 35], dtype=np.uint8)

    shared_visible = source_visible & predicted_visible
    shared_hidden = source_hidden & predicted_hidden
    rgb[shared_visible] = np.array([50, 160, 90], dtype=np.uint8)
    rgb[shared_hidden] = np.array([120, 90, 200], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _shape_to_polylines(shape, config: ReprojectionConfig) -> list[list[tuple[float, float]]]:
    """Sample every projected edge into a 2D polyline."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    if shape.IsNull():
        return []

    polylines: list[list[tuple[float, float]]] = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        adaptor = BRepAdaptor_Curve(edge)
        first = adaptor.FirstParameter()
        last = adaptor.LastParameter()
        if not math.isfinite(first) or not math.isfinite(last) or first == last:
            explorer.Next()
            continue

        sample_count = _sample_count(adaptor.GetType().name, config.curve_samples)
        points: list[tuple[float, float]] = []
        for index in range(sample_count):
            t = first + (last - first) * index / (sample_count - 1)
            point = adaptor.Value(t)
            points.append((float(point.X()), float(point.Y())))

        if len(points) >= 2:
            polyline = _dedupe_adjacent_points(points)
            polylines.append(_snap_nearly_axis_aligned(polyline))
        explorer.Next()

    return [polyline for polyline in polylines if len(polyline) >= 2]


def _sample_count(curve_type: str, curve_samples: int) -> int:
    if curve_type == "GeomAbs_Line":
        return 2
    if curve_type in {"GeomAbs_Circle", "GeomAbs_Ellipse"}:
        return max(curve_samples, 64)
    return max(curve_samples, 24)


def _projector_axis(suffix: str):
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    if suffix == "f":
        return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0), gp_Dir(1, 0, 0))
    if suffix == "t":
        return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    if suffix == "r":
        return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0))
    raise ValueError(f"Unsupported orthographic suffix: {suffix}")


def _polyline_bounds(polylines: list[list[tuple[float, float]]]) -> tuple[float, float, float, float] | None:
    if not polylines:
        return None
    xs = [x for polyline in polylines for x, _ in polyline]
    ys = [y for polyline in polylines for _, y in polyline]
    min_x = min(xs)
    min_y = min(ys)
    return min_x, min_y, max(xs) - min_x, max(ys) - min_y


def _view_box_to_bounds(view_box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    min_x, min_y, width, height = view_box
    return min_x, min_y, width, height


def _translate_polylines(
    polylines: list[list[tuple[float, float]]],
    origin: tuple[float, float],
) -> list[list[tuple[float, float]]]:
    origin_x, origin_y = origin
    return [
        [(x - origin_x, y - origin_y) for x, y in polyline]
        for polyline in polylines
    ]


def _image_size(
    canvas_width: float,
    canvas_height: float,
    config: ReprojectionConfig,
) -> tuple[int, int]:
    drawable = max(config.raster_size_px - 2 * config.raster_padding_px, 16)
    scale = drawable / max(canvas_width, canvas_height, 1.0)
    width = int(math.ceil(canvas_width * scale)) + 2 * config.raster_padding_px
    height = int(math.ceil(canvas_height * scale)) + 2 * config.raster_padding_px
    return max(width, 16), max(height, 16)


def _draw_mask(
    polylines: list[list[tuple[float, float]]],
    canvas_width: float,
    canvas_height: float,
    image_size: tuple[int, int],
    config: ReprojectionConfig,
) -> np.ndarray:
    image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(image)

    drawable = max(config.raster_size_px - 2 * config.raster_padding_px, 16)
    scale = drawable / max(canvas_width, canvas_height, 1.0)
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        transformed = [
            (
                config.raster_padding_px + x * scale,
                config.raster_padding_px + (canvas_height - y) * scale,
            )
            for x, y in polyline
        ]
        draw.line(
            transformed,
            fill=255,
            width=config.line_width_px,
            joint="curve",
        )
    return np.array(image, dtype=bool)


def _dedupe_adjacent_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped = [points[0]]
    for point in points[1:]:
        if math.dist(point, deduped[-1]) > 1e-6:
            deduped.append(point)
    return deduped


def _snap_nearly_axis_aligned(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove tiny projection shear from near-horizontal/vertical line segments."""
    if len(points) != 2:
        return points

    (x1, y1), (x2, y2) = points
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dominant = max(dx, dy)
    if dominant <= 0:
        return points

    if min(dx, dy) / dominant > 0.02:
        return points

    if dx >= dy:
        y = (y1 + y2) / 2.0
        return [(x1, y), (x2, y)]

    x = (x1 + x2) / 2.0
    return [(x, y1), (x, y2)]
