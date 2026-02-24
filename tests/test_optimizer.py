"""Tests for Loop 2 meta-optimizer components."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.schemas.pipeline_config import PipelineConfig
from src.schemas.evaluation_result import BenchmarkCaseResult, BenchmarkReport, EvaluationMetrics
from src.schemas.experiment import ExperimentRecord, OptimizationHistory, ConfigDelta
from src.optimizer.convergence import ConvergenceDetector
from src.optimizer.experiment_tracker import ExperimentTracker
from src.optimizer.failure_analyzer import FailureAnalyzer, FailurePattern
from src.optimizer.fewshot_selector import FewShotSelector, FewShotExample
from src.optimizer.parameter_tuner import ParameterCandidate, ParameterTuner
from src.optimizer.prompt_optimizer import PromptOptimizer


class TestConvergenceDetector:
    def test_not_enough_data(self):
        det = ConvergenceDetector(threshold=0.01, patience=3)
        det.update(0.5)
        converged, _ = det.has_converged()
        assert converged is False

    def test_convergence_detected(self):
        det = ConvergenceDetector(threshold=0.01, patience=3)
        for score in [0.50, 0.60, 0.65, 0.651, 0.652, 0.653]:
            det.update(score)
        converged, reason = det.has_converged()
        assert converged is True
        assert "Improvement" in reason or "improvement" in reason.lower()

    def test_no_convergence(self):
        det = ConvergenceDetector(threshold=0.01, patience=3)
        for score in [0.50, 0.55, 0.60, 0.65, 0.70]:
            det.update(score)
        converged, _ = det.has_converged()
        assert converged is False

    def test_best_score(self):
        det = ConvergenceDetector()
        det.update(0.5)
        det.update(0.7)
        det.update(0.6)
        assert det.best_score == 0.7

    def test_reset(self):
        det = ConvergenceDetector()
        det.update(0.5)
        det.reset()
        assert len(det.scores) == 0
        assert det.best_score == 0.0


class TestExperimentTracker:
    def test_start_and_record(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("test_run")
        
        record = ExperimentRecord(
            experiment_id="exp_001",
            iteration=1,
            composite_score=0.75,
            status="completed",
        )
        tracker.record_experiment(record)
        
        history = tracker.get_history()
        assert history is not None
        assert len(history.experiments) == 1
        assert history.best_composite_score == 0.75

    def test_complete_run(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("test_run")
        tracker.complete_run(converged=True, reason="threshold reached")
        
        history = tracker.get_history()
        assert history.converged is True

    def test_list_runs(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_a")
        tracker.complete_run()
        tracker.start_run("run_b")
        tracker.complete_run()
        
        runs = tracker.list_runs()
        assert len(runs) == 2

    def test_get_best_experiment(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("test")
        
        tracker.record_experiment(ExperimentRecord(
            experiment_id="a", iteration=1, composite_score=0.5, status="completed",
        ))
        tracker.record_experiment(ExperimentRecord(
            experiment_id="b", iteration=2, composite_score=0.8, status="completed",
        ))
        
        best = tracker.get_best_experiment()
        assert best.experiment_id == "b"


class TestFailureAnalyzer:
    def test_analyze_syntax_errors(self):
        report = BenchmarkReport(suite_name="test")
        report.case_results = [
            BenchmarkCaseResult(
                case_id="001",
                drawing_path="d.png",
                error="SyntaxError: invalid syntax on line 5",
                metrics=EvaluationMetrics(execution_success=False),
            ),
            BenchmarkCaseResult(
                case_id="002",
                drawing_path="d.png",
                error="SyntaxError: unexpected EOF",
                metrics=EvaluationMetrics(execution_success=False),
            ),
        ]
        
        analyzer = FailureAnalyzer()
        patterns = analyzer.analyze(report)
        assert len(patterns) > 0
        assert patterns[0].category == "syntax"
        assert patterns[0].count == 2

    def test_analyze_low_quality(self):
        report = BenchmarkReport(suite_name="test")
        report.case_results = [
            BenchmarkCaseResult(
                case_id="001",
                drawing_path="d.png",
                metrics=EvaluationMetrics(execution_success=True, composite_score=0.3),
            ),
        ]
        
        analyzer = FailureAnalyzer()
        patterns = analyzer.analyze(report)
        assert any(p.category == "low_quality" for p in patterns)

    def test_summarize(self):
        analyzer = FailureAnalyzer()
        patterns = [
            FailurePattern("syntax", "Syntax errors", count=5, severity=0.8),
        ]
        summary = analyzer.summarize(patterns)
        assert "syntax" in summary
        assert "5" in summary


class TestPromptOptimizer:
    def test_mock_patch_generation(self):
        config = PipelineConfig()
        optimizer = PromptOptimizer(config, llm_client=None)
        patch = optimizer.generate_patch("Test failures", strategy="add_error_prevention")
        assert len(patch) > 0
        assert "Pitfalls" in patch or "Avoid" in patch

    def test_different_strategies(self):
        config = PipelineConfig()
        optimizer = PromptOptimizer(config, llm_client=None)
        
        for strategy in ["add_constraint", "add_example_pattern", "clarify_ambiguity"]:
            patch = optimizer.generate_patch("Test", strategy=strategy)
            assert len(patch) > 0


class TestFewShotSelector:
    def test_empty_selector(self):
        config = PipelineConfig()
        config.prompts.fewshot_index = "/nonexistent/index.yaml"
        selector = FewShotSelector(config)
        assert selector.select(count=3) == []

    def test_add_and_select(self):
        config = PipelineConfig()
        config.prompts.fewshot_index = "/nonexistent/index.yaml"
        selector = FewShotSelector(config)
        
        selector.add_example("test1", "example code 1", tags=["basic"])
        selector.add_example("test2", "example code 2", tags=["advanced"])
        
        selected = selector.select(count=2, strategy="fixed")
        assert len(selected) == 2

    def test_remove_example(self):
        config = PipelineConfig()
        config.prompts.fewshot_index = "/nonexistent/index.yaml"
        selector = FewShotSelector(config)
        
        selector.add_example("test", "content")
        assert selector.remove_example("test") is True
        assert selector.remove_example("nonexistent") is False


class TestParameterTuner:
    def test_targeted_candidates_timeout(self):
        tuner = ParameterTuner(optimizer_config_path="/nonexistent.yaml")
        candidates = tuner.generate_targeted_candidates(
            PipelineConfig(), ["timeout errors"]
        )
        assert len(candidates) > 0
        assert any(c.params.get("execution_timeout") == 120 for c in candidates)

    def test_apply_to_config(self):
        candidate = ParameterCandidate(
            params={"temperature": 0.1, "max_retries": 5},
            description="test",
        )
        config = PipelineConfig()
        new_config = candidate.apply_to_config(config)
        assert new_config.models.reasoning.temperature == 0.1
        assert new_config.pipeline.max_retries == 5
        # Original should be unchanged
        assert config.models.reasoning.temperature == 0.2
