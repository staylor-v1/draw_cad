"""Benchmark runner: runs the pipeline against a benchmark suite."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.evaluation.comparator import StepComparator
from src.evaluation.metrics import compute_all_metrics
from src.inference.base import BaseLLMClient, BaseVisionClient
from src.pipeline.runner import PipelineRunner
from src.schemas.evaluation_result import BenchmarkCaseResult, BenchmarkReport
from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_yaml
from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.training.ground_truth import StepGroundTruth

logger = get_logger(__name__)


class BenchmarkRunner:
    """Runs the pipeline against a benchmark suite and collects results."""

    def __init__(
        self,
        config: PipelineConfig,
        llm_client: BaseLLMClient | None = None,
        vision_client: BaseVisionClient | None = None,
        use_mock: bool = False,
    ):
        self.config = config
        self.llm_client = llm_client
        self.vision_client = vision_client
        self.use_mock = use_mock
        self.comparator = StepComparator(
            tolerance_mm=config.evaluation.dimension_tolerance_mm,
        )

    def run_suite(
        self,
        suite_path: str | Path,
        output_dir: str | Path = "experiments/benchmark_output",
    ) -> BenchmarkReport:
        """Run the pipeline against all cases in a benchmark suite.
        
        Args:
            suite_path: Path to the benchmark suite YAML file.
            output_dir: Directory for generated STEP files.
        
        Returns:
            BenchmarkReport with all results.
        """
        suite = load_yaml(suite_path)
        suite_name = suite.get("name", "unknown")
        cases = suite.get("cases", [])

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("benchmark_suite_start", suite=suite_name, cases=len(cases))

        report = BenchmarkReport(suite_name=suite_name)

        for case_def in cases:
            case_result = self._run_case(case_def, suite_path, output_dir)
            report.case_results.append(case_result)

        # Compute aggregate
        weights = self.config.evaluation.weights.model_dump()
        report.compute_aggregate(weights)

        logger.info(
            "benchmark_suite_complete",
            suite=suite_name,
            total=report.total_cases,
            successful=report.successful_cases,
            composite=report.aggregate_metrics.composite_score,
        )
        return report

    def run_cases(
        self,
        cases: list[dict],
        output_dir: str | Path = "experiments/benchmark_output",
    ) -> BenchmarkReport:
        """Run the pipeline against a programmatic list of cases.

        Each case dict should contain:
          - id: str
          - drawing: str | Path  (PNG or SVG path)
          - reference: str | Path | None  (STEP reference path)
          - ground_truth: StepGroundTruth | None  (precomputed)

        Returns:
            BenchmarkReport with all results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("benchmark_cases_start", cases=len(cases))
        report = BenchmarkReport(suite_name="training_data")

        for case_def in cases:
            case_result = self._run_case_programmatic(case_def, output_dir)
            report.case_results.append(case_result)

        weights = self.config.evaluation.weights.model_dump()
        report.compute_aggregate(weights)

        logger.info(
            "benchmark_cases_complete",
            total=report.total_cases,
            successful=report.successful_cases,
            composite=report.aggregate_metrics.composite_score,
        )
        return report

    def _run_case_programmatic(
        self,
        case_def: dict,
        output_dir: Path,
    ) -> BenchmarkCaseResult:
        """Run a single programmatic case (from training data)."""
        case_id = case_def.get("id", "unknown")
        drawing_path = Path(case_def["drawing"])
        reference_path = case_def.get("reference")
        ground_truth: StepGroundTruth | None = case_def.get("ground_truth")

        output_path = output_dir / f"{case_id}_output.step"

        logger.info("benchmark_case_start", case_id=case_id, drawing=str(drawing_path))

        case_result = BenchmarkCaseResult(
            case_id=case_id,
            drawing_path=str(drawing_path),
            reference_step_path=str(reference_path) if reference_path else None,
        )

        try:
            pipeline = PipelineRunner(
                config=self.config,
                llm_client=self.llm_client,
                vision_client=self.vision_client,
                use_mock=self.use_mock,
            )

            pipeline_result = pipeline.run(
                image_path=drawing_path,
                output_path=str(output_path),
            )

            case_result.generated_step_path = pipeline_result.step_file
            case_result.generated_code = pipeline_result.generated_code
            case_result.retry_count = pipeline_result.retry_count
            case_result.metrics = pipeline_result.metrics

            if ground_truth and pipeline_result.step_file:
                comparison = self.comparator.compare_with_ground_truth(
                    pipeline_result.step_file, ground_truth
                )
                case_result.metrics.bounding_box_iou = comparison["bounding_box_iou"]
                case_result.metrics.volume_ratio = comparison["volume_ratio"]
                case_result.metrics.face_count_ratio = comparison["face_count_ratio"]
                case_result.metrics.compute_composite(
                    self.config.evaluation.weights.model_dump()
                )
            elif reference_path and Path(reference_path).exists() and pipeline_result.step_file:
                comparison = self.comparator.compare(
                    pipeline_result.step_file, reference_path
                )
                case_result.metrics.bounding_box_iou = comparison["bounding_box_iou"]
                case_result.metrics.volume_ratio = comparison["volume_ratio"]
                case_result.metrics.face_count_ratio = comparison["face_count_ratio"]
                case_result.metrics.compute_composite(
                    self.config.evaluation.weights.model_dump()
                )

            if not pipeline_result.success:
                case_result.error = pipeline_result.error

        except Exception as e:
            logger.error("benchmark_case_error", case_id=case_id, error=str(e))
            case_result.error = str(e)

        logger.info(
            "benchmark_case_complete",
            case_id=case_id,
            success=case_result.metrics.execution_success,
            score=case_result.metrics.composite_score,
        )
        return case_result

    def _run_case(
        self,
        case_def: dict,
        suite_path: str | Path,
        output_dir: Path,
    ) -> BenchmarkCaseResult:
        """Run the pipeline on a single benchmark case."""
        case_id = case_def.get("id", "unknown")
        suite_dir = Path(suite_path).parent

        drawing_path = suite_dir / case_def.get("drawing", "")
        reference_path = case_def.get("reference")
        if reference_path:
            reference_path = suite_dir / reference_path

        output_path = output_dir / f"{case_id}_output.step"

        logger.info("benchmark_case_start", case_id=case_id, drawing=str(drawing_path))

        case_result = BenchmarkCaseResult(
            case_id=case_id,
            drawing_path=str(drawing_path),
            reference_step_path=str(reference_path) if reference_path else None,
        )

        try:
            pipeline = PipelineRunner(
                config=self.config,
                llm_client=self.llm_client,
                vision_client=self.vision_client,
                use_mock=self.use_mock,
            )

            pipeline_result = pipeline.run(
                image_path=drawing_path,
                output_path=str(output_path),
            )

            case_result.generated_step_path = pipeline_result.step_file
            case_result.generated_code = pipeline_result.generated_code
            case_result.retry_count = pipeline_result.retry_count
            case_result.metrics = pipeline_result.metrics

            # If we have a reference, do STEP comparison
            if reference_path and reference_path.exists() and pipeline_result.step_file:
                comparison = self.comparator.compare(
                    pipeline_result.step_file, reference_path
                )
                case_result.metrics.bounding_box_iou = comparison["bounding_box_iou"]
                case_result.metrics.volume_ratio = comparison["volume_ratio"]
                case_result.metrics.compute_composite(
                    self.config.evaluation.weights.model_dump()
                )

            if not pipeline_result.success:
                case_result.error = pipeline_result.error

        except Exception as e:
            logger.error("benchmark_case_error", case_id=case_id, error=str(e))
            case_result.error = str(e)

        logger.info(
            "benchmark_case_complete",
            case_id=case_id,
            success=case_result.metrics.execution_success,
            score=case_result.metrics.composite_score,
        )
        return case_result
