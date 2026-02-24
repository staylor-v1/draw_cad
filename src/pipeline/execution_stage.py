"""Execution stage: run build123d code and validate output."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.schemas.evaluation_result import ValidationResult
from src.schemas.pipeline_config import PipelineConfig
from src.tools.cad import ExecutionResult, execute_build123d
from src.tools.mesh_validator import validate_step_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExecutionStage:
    """Executes generated build123d code and produces STEP output."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, code: str, output_path: str | None = None) -> ExecutionResult:
        """Execute build123d code.
        
        Args:
            code: Generated Python code using build123d.
            output_path: Path for the STEP file output.
        
        Returns:
            ExecutionResult with success status and details.
        """
        if output_path is None:
            output_path = self.config.output.default_step_path

        logger.info("execution_stage_start", output_path=output_path)

        result = execute_build123d(
            script_content=code,
            output_path=output_path,
            timeout=self.config.pipeline.execution_timeout,
        )

        logger.info(
            "execution_stage_complete",
            success=result.success,
            error_category=result.error_category.value,
        )
        return result
