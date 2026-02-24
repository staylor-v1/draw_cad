"""Tests for the Loop 3 pipeline."""
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.schemas.pipeline_config import PipelineConfig
from src.schemas.geometry import Dimension, Feature, GeometryData, EnrichedGeometry, ReconciledGeometry, TextRegion
from src.pipeline.vision_stage import VisionStage
from src.pipeline.ocr_stage import OCRStage
from src.pipeline.reconciliation import ReconciliationStage
from src.pipeline.reasoning_stage import ReasoningStage, extract_code
from src.pipeline.execution_stage import ExecutionStage
from src.pipeline.validation_stage import ValidationStage
from src.pipeline.retry_controller import RetryController
from src.pipeline.runner import PipelineRunner
from src.tools.cad import ErrorCategory, ExecutionResult


@pytest.fixture
def config():
    return PipelineConfig()


@pytest.fixture
def sample_geometry():
    return GeometryData(
        views=["Top", "Front", "Right"],
        dimensions=[
            Dimension(label="Length", value=100.0, unit="mm"),
            Dimension(label="Width", value=50.0, unit="mm"),
            Dimension(label="Thickness", value=10.0, unit="mm"),
        ],
        features=[
            Feature(type="Base Plate", description="Rectangular plate 100x50x10mm"),
            Feature(type="Through Hole", description="5mm diameter centered hole"),
        ],
        notes="Standard tolerance +/- 0.1mm",
    )


@pytest.fixture
def sample_reconciled():
    return ReconciledGeometry(
        overall_dimensions={"length": 100.0, "width": 50.0, "thickness": 10.0},
        features=[
            Feature(type="Base Plate", description="Rectangular plate 100x50x10mm"),
            Feature(type="Through Hole", description="5mm diameter centered hole"),
        ],
        notes=["Standard tolerance +/- 0.1mm"],
    )


class TestExtractCode:
    def test_python_block(self):
        text = "Here is the code:\n```python\nfrom build123d import *\nBox(10,10,10)\n```"
        code = extract_code(text)
        assert "from build123d import *" in code
        assert "Box(10,10,10)" in code

    def test_no_code_block(self):
        text = "No code here"
        assert extract_code(text) == ""

    def test_generic_block_with_build123d(self):
        text = "```\nfrom build123d import *\nBox(5,5,5)\n```"
        code = extract_code(text)
        assert "from build123d import *" in code


class TestVisionStage:
    def test_mock_mode(self, config):
        stage = VisionStage(config, vision_client=None)
        result = stage.run("test_drawing.png")
        assert isinstance(result, GeometryData)
        assert len(result.views) > 0
        assert len(result.dimensions) > 0


class TestOCRStage:
    def test_disabled(self, config):
        config.pipeline.ocr_enabled = False
        stage = OCRStage(config)
        geometry = GeometryData(views=["Top"])
        result = stage.run("test.png", geometry)
        assert isinstance(result, EnrichedGeometry)
        assert len(result.text_regions) == 0

    def test_mock_mode(self, config):
        stage = OCRStage(config, use_mock=True)
        geometry = GeometryData(views=["Top"])
        result = stage.run("test.png", geometry)
        assert isinstance(result, EnrichedGeometry)
        assert len(result.text_regions) > 0


class TestReconciliationStage:
    def test_basic_reconciliation(self, config, sample_geometry):
        stage = ReconciliationStage(config)
        enriched = EnrichedGeometry(geometry=sample_geometry)
        result = stage.run(enriched)
        assert isinstance(result, ReconciledGeometry)
        assert len(result.overall_dimensions) > 0
        assert len(result.features) > 0


class TestReasoningStage:
    def test_mock_generate(self, config, sample_reconciled):
        stage = ReasoningStage(config, llm_client=None)
        code = stage.run(sample_reconciled)
        assert "from build123d import *" in code
        assert "Rectangle" in code or "Extrude" in code

    def test_mock_generate_with_error_context(self, config, sample_reconciled):
        stage = ReasoningStage(config, llm_client=None)
        code = stage.run(
            sample_reconciled,
            error_context="SyntaxError on line 5",
            previous_code="invalid code",
        )
        assert len(code) > 0


class TestRetryController:
    def test_initial_state(self, config):
        ctrl = RetryController(config)
        assert ctrl.attempt == 0
        assert ctrl.should_retry(ExecutionResult(success=False)) is True

    def test_success_no_retry(self, config):
        ctrl = RetryController(config)
        result = ExecutionResult(success=True)
        assert ctrl.should_retry(result) is False

    def test_max_retries(self, config):
        config.pipeline.max_retries = 2
        ctrl = RetryController(config)
        fail_result = ExecutionResult(success=False, error_category=ErrorCategory.RUNTIME_ERROR)
        ctrl.record_attempt("code1", fail_result)
        ctrl.record_attempt("code2", fail_result)
        assert ctrl.should_retry(fail_result) is False

    def test_import_error_no_retry(self, config):
        ctrl = RetryController(config)
        result = ExecutionResult(success=False, error_category=ErrorCategory.IMPORT_ERROR)
        assert ctrl.should_retry(result) is False

    def test_error_context_building(self, config):
        ctrl = RetryController(config)
        fail_result = ExecutionResult(
            success=False,
            stderr="TypeError: missing argument",
            error_category=ErrorCategory.RUNTIME_ERROR,
        )
        ctrl.record_attempt("bad code", fail_result)
        context = ctrl.build_error_context()
        assert "TypeError" in context
        assert "Attempt" in context

    def test_reset(self, config):
        ctrl = RetryController(config)
        ctrl.record_attempt("code", ExecutionResult(success=False))
        ctrl.reset()
        assert ctrl.attempt == 0
        assert len(ctrl.history) == 0


class TestPipelineRunner:
    def test_mock_pipeline(self, config, tmp_path):
        output = str(tmp_path / "test_output.step")
        runner = PipelineRunner(config=config, use_mock=True)
        result = runner.run(image_path="test_drawing.png", output_path=output)
        # In mock mode, execution depends on build123d being installed
        assert result.generated_code != ""
        assert result.reconciled_geometry is not None

    def test_pipeline_result_metrics(self, config):
        runner = PipelineRunner(config=config, use_mock=True)
        result = runner.run(image_path="test_drawing.png")
        assert result.metrics.retry_efficiency >= 0.0
        assert result.metrics.retry_efficiency <= 1.0
