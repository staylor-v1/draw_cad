"""STEP file comparison for evaluation against reference models."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.schemas.evaluation_result import DimensionCheck, FeatureCheck, ValidationResult
from src.tools.mesh_validator import compute_bounding_box_iou, validate_step_file
from src.tools.step_analyzer import StepProperties, analyze_step_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class StepComparator:
    """Compares a generated STEP file against a reference STEP file."""

    def __init__(self, tolerance_mm: float = 0.5):
        self.tolerance = tolerance_mm

    def compare(
        self,
        generated_path: str | Path,
        reference_path: str | Path,
    ) -> dict:
        """Compare generated STEP file against reference.
        
        Returns:
            Dict with comparison results including metrics.
        """
        gen_props = analyze_step_file(generated_path)
        ref_props = analyze_step_file(reference_path)

        gen_validation = validate_step_file(generated_path)

        result = {
            "generated_valid": gen_props.is_valid,
            "reference_valid": ref_props.is_valid,
            "bounding_box_iou": 0.0,
            "volume_ratio": 0.0,
            "face_count_ratio": 0.0,
            "gen_props": gen_props,
            "ref_props": ref_props,
            "gen_validation": gen_validation,
        }

        if gen_props.bounding_box and ref_props.bounding_box:
            result["bounding_box_iou"] = compute_bounding_box_iou(
                gen_props.bounding_box, ref_props.bounding_box
            )

        if gen_props.volume and ref_props.volume:
            v_min = min(gen_props.volume, ref_props.volume)
            v_max = max(gen_props.volume, ref_props.volume)
            result["volume_ratio"] = v_min / v_max if v_max > 0 else 0.0

        if gen_props.face_count > 0 and ref_props.face_count > 0:
            f_min = min(gen_props.face_count, ref_props.face_count)
            f_max = max(gen_props.face_count, ref_props.face_count)
            result["face_count_ratio"] = f_min / f_max

        logger.info(
            "step_comparison_complete",
            bbox_iou=result["bounding_box_iou"],
            volume_ratio=result["volume_ratio"],
        )
        return result

    def check_dimensions_from_reference(
        self,
        gen_props: StepProperties,
        ref_props: StepProperties,
    ) -> list[DimensionCheck]:
        """Check dimensions by comparing bounding box extents."""
        checks = []
        if not gen_props.bounding_box or not ref_props.bounding_box:
            return checks

        gen_min, gen_max = gen_props.bounding_box
        ref_min, ref_max = ref_props.bounding_box

        labels = ["X extent", "Y extent", "Z extent"]
        for i, label in enumerate(labels):
            expected = abs(ref_max[i] - ref_min[i])
            actual = abs(gen_max[i] - gen_min[i])
            error = abs(actual - expected)
            checks.append(DimensionCheck(
                label=label,
                expected=expected,
                actual=actual,
                tolerance=self.tolerance,
                within_tolerance=error <= self.tolerance,
                error_mm=error,
            ))

        return checks
