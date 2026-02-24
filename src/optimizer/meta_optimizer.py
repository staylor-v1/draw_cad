"""Meta-optimizer: Loop 2 controller that optimizes the Loop 3 pipeline."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from src.evaluation.benchmark_runner import BenchmarkRunner
from src.inference.base import BaseLLMClient, BaseVisionClient
from src.optimizer.convergence import ConvergenceDetector
from src.optimizer.experiment_tracker import ExperimentTracker
from src.optimizer.failure_analyzer import FailureAnalyzer
from src.optimizer.fewshot_selector import FewShotSelector
from src.optimizer.parameter_tuner import ParameterCandidate, ParameterTuner
from src.optimizer.prompt_optimizer import PromptOptimizer
from src.schemas.evaluation_result import BenchmarkReport
from src.schemas.experiment import ConfigDelta, ExperimentRecord
from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_prompt
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MetaOptimizer:
    """Loop 2: meta-optimizer that iteratively improves the Loop 3 pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        llm_client: BaseLLMClient | None = None,
        vision_client: BaseVisionClient | None = None,
        use_mock: bool = False,
        experiments_dir: str = "experiments",
        benchmark_suite: str = "benchmarks/suite.yaml",
    ):
        self.config = config
        self.llm_client = llm_client
        self.vision_client = vision_client
        self.use_mock = use_mock
        self.benchmark_suite = benchmark_suite

        # Sub-components
        self.tracker = ExperimentTracker(experiments_dir)
        self.failure_analyzer = FailureAnalyzer()
        self.prompt_optimizer = PromptOptimizer(config, llm_client)
        self.fewshot_selector = FewShotSelector(config)
        self.param_tuner = ParameterTuner()
        self.convergence = ConvergenceDetector(
            threshold=0.01,
            patience=3,
        )

    def run(self, max_iterations: int = 20) -> dict:
        """Run the meta-optimization loop.
        
        Args:
            max_iterations: Maximum number of optimization iterations.
        
        Returns:
            Dict with optimization results.
        """
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.tracker.start_run(run_id)
        self.convergence.reset()

        logger.info("meta_optimizer_start", run_id=run_id, max_iterations=max_iterations)

        best_config = self.config.deep_copy()
        best_score = 0.0

        # Baseline run
        logger.info("meta_optimizer_baseline")
        baseline_report = self._run_benchmark(self.config)
        baseline_score = baseline_report.aggregate_metrics.composite_score
        self.convergence.update(baseline_score)
        best_score = baseline_score

        self._record_experiment(
            iteration=0,
            config=self.config,
            report=baseline_report,
            delta=ConfigDelta(description="Baseline"),
        )

        for iteration in range(1, max_iterations + 1):
            logger.info("meta_optimizer_iteration", iteration=iteration)

            # Analyze failures from last report
            patterns = self.failure_analyzer.analyze(baseline_report)
            failure_summary = self.failure_analyzer.summarize(patterns)

            # Generate candidates
            candidates = self._generate_candidates(
                best_config, failure_summary, patterns, iteration
            )

            if not candidates:
                logger.info("no_candidates_generated")
                break

            # Evaluate each candidate
            candidate_results: list[tuple[PipelineConfig, BenchmarkReport, ConfigDelta]] = []
            for candidate_config, delta in candidates:
                report = self._run_benchmark(candidate_config)
                score = report.aggregate_metrics.composite_score
                candidate_results.append((candidate_config, report, delta))

                logger.info(
                    "candidate_evaluated",
                    description=delta.description,
                    score=score,
                    improvement=score - best_score,
                )

            # Select best candidate
            best_candidate = max(
                candidate_results,
                key=lambda x: x[1].aggregate_metrics.composite_score,
            )
            candidate_config, candidate_report, candidate_delta = best_candidate
            candidate_score = candidate_report.aggregate_metrics.composite_score

            # Accept if improvement
            if candidate_score > best_score:
                logger.info(
                    "candidate_accepted",
                    old_score=best_score,
                    new_score=candidate_score,
                )
                best_config = candidate_config
                best_score = candidate_score
                baseline_report = candidate_report
            else:
                logger.info("candidate_rejected", candidate_score=candidate_score, best_score=best_score)
                # Rollback any prompt patches
                candidate_delta = ConfigDelta(description=f"Rejected (score {candidate_score:.4f})")

            self.convergence.update(candidate_score)
            self._record_experiment(
                iteration=iteration,
                config=candidate_config,
                report=candidate_report,
                delta=candidate_delta,
            )

            # Check convergence
            converged, reason = self.convergence.has_converged()
            if converged:
                logger.info("meta_optimizer_converged", reason=reason)
                self.tracker.complete_run(converged=True, reason=reason)
                break
        else:
            self.tracker.complete_run(
                converged=False,
                reason=f"Max iterations ({max_iterations}) reached",
            )

        # Save best config
        best_config.to_yaml(f"experiments/best_config_{run_id}.yaml")

        result = {
            "run_id": run_id,
            "iterations": min(iteration, max_iterations) if 'iteration' in dir() else 0,
            "best_score": best_score,
            "baseline_score": baseline_score,
            "improvement": best_score - baseline_score,
            "converged": self.convergence.has_converged()[0],
        }
        logger.info("meta_optimizer_complete", **result)
        return result

    def _run_benchmark(self, config: PipelineConfig) -> BenchmarkReport:
        """Run the benchmark suite with the given config."""
        runner = BenchmarkRunner(
            config=config,
            llm_client=self.llm_client,
            vision_client=self.vision_client,
            use_mock=self.use_mock,
        )
        return runner.run_suite(self.benchmark_suite)

    def _generate_candidates(
        self,
        base_config: PipelineConfig,
        failure_summary: str,
        patterns: list,
        iteration: int,
    ) -> list[tuple[PipelineConfig, ConfigDelta]]:
        """Generate candidate configs for evaluation."""
        candidates = []

        # 1. Prompt patch candidate
        strategies = ["add_error_prevention", "add_constraint", "add_example_pattern", "clarify_ambiguity"]
        strategy = strategies[iteration % len(strategies)]
        patch = self.prompt_optimizer.generate_patch(failure_summary, strategy)
        if patch:
            patched_config = base_config.deep_copy()
            # Save the original prompt for potential rollback
            delta = ConfigDelta(
                prompt_patches=[patch],
                description=f"Prompt patch ({strategy})",
            )
            candidates.append((patched_config, delta))

        # 2. Few-shot variation candidate
        fewshot_strategies = ["fixed", "random", "coverage", "failure_targeted"]
        fs_strategy = fewshot_strategies[iteration % len(fewshot_strategies)]
        fewshot_config = base_config.deep_copy()
        delta = ConfigDelta(
            fewshot_changes=[f"strategy={fs_strategy}"],
            description=f"Few-shot strategy: {fs_strategy}",
        )
        candidates.append((fewshot_config, delta))

        # 3. Parameter tuning candidate
        pattern_cats = [p.category for p in patterns] if patterns else []
        param_candidates = self.param_tuner.generate_targeted_candidates(
            base_config, pattern_cats
        )
        for pc in param_candidates[:2]:  # Limit to 2 param candidates
            tuned_config = pc.apply_to_config(base_config)
            delta = ConfigDelta(
                parameter_changes=pc.params,
                description=f"Parameter: {pc.description}",
            )
            candidates.append((tuned_config, delta))

        return candidates

    def _record_experiment(
        self,
        iteration: int,
        config: PipelineConfig,
        report: BenchmarkReport,
        delta: ConfigDelta,
    ) -> None:
        """Record an experiment to the tracker."""
        record = ExperimentRecord(
            experiment_id=str(uuid.uuid4())[:8],
            iteration=iteration,
            config_snapshot=config.model_dump(),
            config_delta=delta,
            composite_score=report.aggregate_metrics.composite_score,
            metrics={
                "dimension_fidelity": report.aggregate_metrics.dimension_fidelity,
                "feature_recall": report.aggregate_metrics.feature_recall,
                "feature_precision": report.aggregate_metrics.feature_precision,
                "bounding_box_iou": report.aggregate_metrics.bounding_box_iou,
                "volume_ratio": report.aggregate_metrics.volume_ratio,
                "geometry_valid": report.aggregate_metrics.geometry_valid,
                "retry_efficiency": report.aggregate_metrics.retry_efficiency,
            },
            case_scores={
                c.case_id: c.metrics.composite_score
                for c in report.case_results
            },
            failure_categories={
                c.case_id: c.error[:100]
                for c in report.case_results
                if c.error
            },
            status="completed",
        )
        self.tracker.record_experiment(record)
