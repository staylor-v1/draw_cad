"""Validation stage: check generated STEP file against expected geometry."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.schemas.evaluation_result import DimensionCheck, FeatureCheck, ValidationResult
from src.schemas.geometry import ReconciledGeometry
from src.schemas.pipeline_config import PipelineConfig
from src.tools.mesh_validator import validate_step_file
from src.tools.step_analyzer import StepProperties, analyze_step_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationStage:
    """Validates generated STEP file against expected geometry."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        step_path: str | Path,
        reconciled: ReconciledGeometry,
    ) -> tuple[ValidationResult, list[DimensionCheck], list[FeatureCheck]]:
        """Validate the generated STEP file.
        
        Returns:
            Tuple of (ValidationResult, dimension_checks, feature_checks).
        """
        logger.info("validation_stage_start", step_path=str(step_path))

        # Basic mesh validation
        validation = validate_step_file(step_path)

        # Analyze STEP properties
        props = analyze_step_file(step_path)

        # Check dimensions
        dim_checks = self._check_dimensions(props, reconciled)

        # Check features (basic)
        feat_checks = self._check_features(props, reconciled)

        logger.info(
            "validation_stage_complete",
            is_valid=validation.is_valid,
            dim_checks_passed=sum(1 for d in dim_checks if d.within_tolerance),
            feat_checks_found=sum(1 for f in feat_checks if f.found),
        )

        return validation, dim_checks, feat_checks

    def _check_dimensions(
        self,
        props: StepProperties,
        reconciled: ReconciledGeometry,
    ) -> list[DimensionCheck]:
        """Check extracted dimensions against expected values."""
        checks = []
        tolerance = self.config.evaluation.dimension_tolerance_mm

        if not props.bounding_box:
            # Can't check without bounding box
            for dim in reconciled.features:
                pass
            return checks

        bbox_min, bbox_max = props.bounding_box
        bbox_dims = {
            "x": abs(bbox_max[0] - bbox_min[0]),
            "y": abs(bbox_max[1] - bbox_min[1]),
            "z": abs(bbox_max[2] - bbox_min[2]),
        }
        # Sort bbox dimensions descending
        sorted_dims = sorted(bbox_dims.values(), reverse=True)

        # Map overall dimensions to sorted bbox dimensions
        overall = reconciled.overall_dimensions
        sorted_expected = sorted(overall.values(), reverse=True) if overall else []

        for i, (expected_val, actual_val) in enumerate(
            zip(sorted_expected, sorted_dims)
        ):
            label = list(overall.keys())[
                list(overall.values()).index(sorted(overall.values(), reverse=True)[i])
            ] if i < len(overall) else f"dim_{i}"
            error = abs(actual_val - expected_val)
            checks.append(DimensionCheck(
                label=label,
                expected=expected_val,
                actual=actual_val,
                tolerance=tolerance,
                within_tolerance=error <= tolerance,
                error_mm=error,
            ))

        return checks

    def _check_features(
        self,
        props: StepProperties,
        reconciled: ReconciledGeometry,
    ) -> list[FeatureCheck]:
        """Basic feature check based on topology counts."""
        checks = []

        for feat in reconciled.features:
            feat_type = feat.type.lower()

            if "hole" in feat_type:
                # A hole adds faces (cylindrical surface), so check face count > 6 (box has 6)
                found = props.face_count > 6
                checks.append(FeatureCheck(
                    feature_type=feat.type,
                    expected_description=feat.description,
                    found=found,
                    match_confidence=0.7 if found else 0.0,
                ))
            elif "fillet" in feat_type or "chamfer" in feat_type:
                found = props.face_count > 6
                checks.append(FeatureCheck(
                    feature_type=feat.type,
                    expected_description=feat.description,
                    found=found,
                    match_confidence=0.5 if found else 0.0,
                ))
            else:
                # Base feature - always "found" if geometry is valid
                checks.append(FeatureCheck(
                    feature_type=feat.type,
                    expected_description=feat.description,
                    found=props.is_valid,
                    match_confidence=0.8 if props.is_valid else 0.0,
                ))

        return checks
