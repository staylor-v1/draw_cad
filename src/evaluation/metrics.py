"""Metric computation for evaluating generated CAD models."""
from __future__ import annotations

from typing import Optional

from src.schemas.evaluation_result import (
    BenchmarkCaseResult,
    DimensionCheck,
    EvaluationMetrics,
    FeatureCheck,
    ValidationResult,
)
from src.tools.mesh_validator import compute_bounding_box_iou
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_dimension_fidelity(checks: list[DimensionCheck]) -> float:
    """Fraction of expected dimensions within tolerance."""
    if not checks:
        return 0.0
    passed = sum(1 for c in checks if c.within_tolerance)
    return passed / len(checks)


def compute_feature_recall(checks: list[FeatureCheck]) -> float:
    """Fraction of expected features present."""
    if not checks:
        return 0.0
    found = sum(1 for c in checks if c.found)
    return found / len(checks)


def compute_feature_precision(checks: list[FeatureCheck]) -> float:
    """Fraction of generated features that are correct."""
    if not checks:
        return 0.0
    correct = sum(1 for c in checks if c.found and c.match_confidence > 0.5)
    return correct / len(checks) if checks else 0.0


def compute_bounding_box_iou_metric(
    generated_bbox: Optional[list[list[float]]],
    reference_bbox: Optional[list[list[float]]],
) -> float:
    """Compute 3D IoU between generated and reference bounding boxes."""
    if generated_bbox is None or reference_bbox is None:
        return 0.0
    return compute_bounding_box_iou(generated_bbox, reference_bbox)


def compute_volume_ratio(
    generated_volume: Optional[float],
    reference_volume: Optional[float],
) -> float:
    """Compute volume ratio: min(V_gen, V_ref) / max(V_gen, V_ref)."""
    if generated_volume is None or reference_volume is None:
        return 0.0
    if generated_volume <= 0 or reference_volume <= 0:
        return 0.0
    return min(generated_volume, reference_volume) / max(generated_volume, reference_volume)


def compute_geometry_valid(validation: Optional[ValidationResult]) -> float:
    """1.0 if geometry is valid (watertight and manifold), 0.0 otherwise."""
    if validation is None:
        return 0.0
    if validation.is_valid and validation.is_watertight:
        return 1.0
    if validation.is_valid:
        return 0.5
    return 0.0


def compute_retry_efficiency(retry_count: int, max_retries: int = 3) -> float:
    """1.0 - min(retry_count, max_retries) / max_retries."""
    return 1.0 - min(retry_count, max_retries) / max_retries


def compute_face_count_ratio(generated: int, reference: int) -> float:
    """Compute face-count ratio: min(gen, ref) / max(gen, ref)."""
    if generated <= 0 or reference <= 0:
        return 0.0
    return min(generated, reference) / max(generated, reference)


def compute_all_metrics(
    dim_checks: list[DimensionCheck],
    feat_checks: list[FeatureCheck],
    validation: Optional[ValidationResult],
    reference_bbox: Optional[list[list[float]]] = None,
    reference_volume: Optional[float] = None,
    reference_face_count: int = 0,
    retry_count: int = 0,
    execution_success: bool = False,
    weights: Optional[dict[str, float]] = None,
) -> EvaluationMetrics:
    """Compute all metrics and the composite score."""
    gen_bbox = validation.bounding_box if validation else None
    gen_volume = validation.volume if validation else None
    gen_face_count = validation.face_count if validation else 0

    metrics = EvaluationMetrics(
        dimension_fidelity=compute_dimension_fidelity(dim_checks),
        feature_recall=compute_feature_recall(feat_checks),
        feature_precision=compute_feature_precision(feat_checks),
        bounding_box_iou=compute_bounding_box_iou_metric(gen_bbox, reference_bbox),
        volume_ratio=compute_volume_ratio(gen_volume, reference_volume),
        face_count_ratio=compute_face_count_ratio(gen_face_count, reference_face_count),
        geometry_valid=compute_geometry_valid(validation),
        retry_efficiency=compute_retry_efficiency(retry_count),
        dimension_checks=dim_checks,
        feature_checks=feat_checks,
        validation=validation,
        execution_success=execution_success,
        retry_count=retry_count,
    )
    metrics.compute_composite(weights)
    return metrics
