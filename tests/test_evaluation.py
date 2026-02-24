"""Tests for evaluation framework."""
import pytest

from src.schemas.evaluation_result import (
    BenchmarkCaseResult,
    BenchmarkReport,
    DimensionCheck,
    EvaluationMetrics,
    FeatureCheck,
    ValidationResult,
)
from src.evaluation.metrics import (
    compute_dimension_fidelity,
    compute_feature_recall,
    compute_feature_precision,
    compute_volume_ratio,
    compute_geometry_valid,
    compute_retry_efficiency,
    compute_all_metrics,
)
from src.tools.mesh_validator import compute_bounding_box_iou


class TestDimensionFidelity:
    def test_all_pass(self):
        checks = [
            DimensionCheck(label="L", expected=100, actual=100.1, tolerance=0.5, within_tolerance=True),
            DimensionCheck(label="W", expected=50, actual=49.9, tolerance=0.5, within_tolerance=True),
        ]
        assert compute_dimension_fidelity(checks) == 1.0

    def test_partial_pass(self):
        checks = [
            DimensionCheck(label="L", expected=100, actual=100.1, tolerance=0.5, within_tolerance=True),
            DimensionCheck(label="W", expected=50, actual=55.0, tolerance=0.5, within_tolerance=False),
        ]
        assert compute_dimension_fidelity(checks) == 0.5

    def test_empty(self):
        assert compute_dimension_fidelity([]) == 0.0


class TestFeatureRecall:
    def test_all_found(self):
        checks = [
            FeatureCheck(feature_type="Hole", expected_description="hole", found=True, match_confidence=0.9),
            FeatureCheck(feature_type="Plate", expected_description="plate", found=True, match_confidence=0.8),
        ]
        assert compute_feature_recall(checks) == 1.0

    def test_none_found(self):
        checks = [
            FeatureCheck(feature_type="Hole", expected_description="hole", found=False),
        ]
        assert compute_feature_recall(checks) == 0.0


class TestVolumeRatio:
    def test_equal_volumes(self):
        assert compute_volume_ratio(1000.0, 1000.0) == 1.0

    def test_half_volume(self):
        assert compute_volume_ratio(500.0, 1000.0) == 0.5

    def test_zero_volume(self):
        assert compute_volume_ratio(0.0, 1000.0) == 0.0

    def test_none_volume(self):
        assert compute_volume_ratio(None, 1000.0) == 0.0


class TestGeometryValid:
    def test_watertight(self):
        v = ValidationResult(is_valid=True, is_watertight=True)
        assert compute_geometry_valid(v) == 1.0

    def test_valid_not_watertight(self):
        v = ValidationResult(is_valid=True, is_watertight=False)
        assert compute_geometry_valid(v) == 0.5

    def test_invalid(self):
        v = ValidationResult(is_valid=False)
        assert compute_geometry_valid(v) == 0.0


class TestRetryEfficiency:
    def test_no_retries(self):
        assert compute_retry_efficiency(0) == 1.0

    def test_one_retry(self):
        assert abs(compute_retry_efficiency(1) - 0.6667) < 0.01

    def test_max_retries(self):
        assert compute_retry_efficiency(3) == 0.0

    def test_over_max(self):
        assert compute_retry_efficiency(5) == 0.0


class TestBoundingBoxIoU:
    def test_identical_boxes(self):
        box = [[0, 0, 0], [10, 10, 10]]
        assert compute_bounding_box_iou(box, box) == 1.0

    def test_no_overlap(self):
        box_a = [[0, 0, 0], [5, 5, 5]]
        box_b = [[10, 10, 10], [20, 20, 20]]
        assert compute_bounding_box_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        box_a = [[0, 0, 0], [10, 10, 10]]
        box_b = [[5, 5, 5], [15, 15, 15]]
        iou = compute_bounding_box_iou(box_a, box_b)
        assert 0.0 < iou < 1.0


class TestCompositeMetrics:
    def test_compute_all_metrics(self):
        dim_checks = [
            DimensionCheck(label="L", expected=100, actual=100, tolerance=0.5, within_tolerance=True),
        ]
        feat_checks = [
            FeatureCheck(feature_type="Plate", expected_description="plate", found=True, match_confidence=0.9),
        ]
        validation = ValidationResult(is_valid=True, is_watertight=True, volume=50000.0)

        metrics = compute_all_metrics(
            dim_checks=dim_checks,
            feat_checks=feat_checks,
            validation=validation,
            execution_success=True,
        )
        assert metrics.dimension_fidelity == 1.0
        assert metrics.feature_recall == 1.0
        assert metrics.geometry_valid == 1.0
        assert metrics.composite_score > 0.0


class TestBenchmarkReport:
    def test_aggregate_computation(self):
        report = BenchmarkReport(suite_name="test")
        case1 = BenchmarkCaseResult(
            case_id="001",
            drawing_path="drawing1.png",
            metrics=EvaluationMetrics(
                dimension_fidelity=1.0,
                feature_recall=0.8,
                geometry_valid=1.0,
                execution_success=True,
            ),
        )
        case2 = BenchmarkCaseResult(
            case_id="002",
            drawing_path="drawing2.png",
            metrics=EvaluationMetrics(
                dimension_fidelity=0.5,
                feature_recall=0.6,
                geometry_valid=0.5,
                execution_success=True,
            ),
        )
        report.case_results = [case1, case2]
        report.compute_aggregate()

        assert report.total_cases == 2
        assert report.successful_cases == 2
        assert report.aggregate_metrics.dimension_fidelity == 0.75
        assert report.aggregate_metrics.feature_recall == 0.7
