"""Vision stage: extract geometry data from engineering drawing."""
from __future__ import annotations

from pathlib import Path

from src.inference.base import BaseVisionClient
from src.schemas.geometry import GeometryData
from src.schemas.pipeline_config import PipelineConfig
from src.tools.vision import analyze_drawing, analyze_drawing_mock
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class VisionStage:
    """Extracts geometry data from an engineering drawing image."""

    def __init__(self, config: PipelineConfig, vision_client: BaseVisionClient | None = None):
        self.config = config
        self.vision_client = vision_client

    def run(self, image_path: str | Path) -> GeometryData:
        """Run vision analysis on the drawing image."""
        logger.info("vision_stage_start", image_path=str(image_path))

        if self.vision_client is None:
            logger.info("vision_stage_using_mock")
            return analyze_drawing_mock(image_path)

        model_cfg = self.config.models.vision
        geometry = analyze_drawing(
            image_path=image_path,
            vision_client=self.vision_client,
            model=model_cfg.name,
            prompt_path=self.config.prompts.vision_prompt,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
        logger.info("vision_stage_complete", dimensions=len(geometry.dimensions), features=len(geometry.features))
        return geometry
