"""Evaluation result schemas."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of validating a generated STEP file."""
    is_valid: bool = False
    is_watertight: bool = False
    is_manifold: bool = False
    volume: Optional[float] = None
    bounding_box: Optional[list[list[float]]] = None  # [[min_x,min_y,min_z],[max_x,max_y,max_z]]
    face_count: int = 0
    edge_count: int = 0
    vertex_count: int = 0
    errors: list[str] = Field(default_factory=list)


class DimensionCheck(BaseModel):
    """Result of checking a single dimension."""
    label: str
    expected: float
    actual: Optional[float] = None
    tolerance: float = 0.5
    within_tolerance: bool = False
    error_mm: Optional[float] = None


class FeatureCheck(BaseModel):
    """Result of checking a single feature."""
    feature_type: str
    expected_description: str
    found: bool = False
    match_confidence: float = 0.0


class EvaluationMetrics(BaseModel):
    """Full evaluation metrics for a single pipeline run."""
    dimension_fidelity: float = 0.0
    feature_recall: float = 0.0
    feature_precision: float = 0.0
    bounding_box_iou: float = 0.0
    volume_ratio: float = 0.0
    face_count_ratio: float = 0.0
    geometry_valid: float = 0.0
    retry_efficiency: float = 1.0
    composite_score: float = 0.0

    dimension_checks: list[DimensionCheck] = Field(default_factory=list)
    feature_checks: list[FeatureCheck] = Field(default_factory=list)
    validation: Optional[ValidationResult] = None

    execution_success: bool = False
    retry_count: int = 0
    error_message: str = ""

    def compute_composite(self, weights: Optional[dict[str, float]] = None) -> float:
        """Compute weighted composite score."""
        if weights is None:
            weights = {
                "dimension_fidelity": 0.30,
                "feature_recall": 0.25,
                "bounding_box_iou": 0.15,
                "volume_ratio": 0.10,
                "geometry_valid": 0.10,
                "feature_precision": 0.05,
                "retry_efficiency": 0.05,
            }
        score = 0.0
        for metric, weight in weights.items():
            score += getattr(self, metric, 0.0) * weight
        self.composite_score = score
        return score


class BenchmarkCaseResult(BaseModel):
    """Result of running the pipeline on a single benchmark case."""
    case_id: str
    drawing_path: str
    reference_step_path: Optional[str] = None
    generated_step_path: Optional[str] = None
    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    generated_code: str = ""
    error: str = ""
    retry_count: int = 0


class BenchmarkReport(BaseModel):
    """Aggregate report for a benchmark suite run."""
    suite_name: str = ""
    case_results: list[BenchmarkCaseResult] = Field(default_factory=list)
    aggregate_metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0

    def compute_aggregate(self, weights: Optional[dict[str, float]] = None) -> None:
        """Compute aggregate metrics across all cases."""
        if not self.case_results:
            return

        self.total_cases = len(self.case_results)
        self.successful_cases = sum(1 for c in self.case_results if c.metrics.execution_success)
        self.failed_cases = self.total_cases - self.successful_cases

        metric_fields = [
            "dimension_fidelity", "feature_recall", "feature_precision",
            "bounding_box_iou", "volume_ratio", "face_count_ratio",
            "geometry_valid", "retry_efficiency",
        ]
        for field_name in metric_fields:
            values = [getattr(c.metrics, field_name) for c in self.case_results]
            avg = sum(values) / len(values) if values else 0.0
            setattr(self.aggregate_metrics, field_name, avg)

        self.aggregate_metrics.compute_composite(weights)
