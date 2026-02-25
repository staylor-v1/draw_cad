"""Vision stage: extract geometry data from engineering drawing."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.inference.base import BaseVisionClient
from src.schemas.geometry import GeometryData
from src.schemas.pipeline_config import PipelineConfig
from src.tools.vision import analyze_drawing, analyze_drawing_mock
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default location for pre-rendered PNGs from training data
_DEFAULT_RENDERED_PNG_DIR = Path("training_data/rendered_png")


class VisionStage:
    """Extracts geometry data from an engineering drawing image."""

    def __init__(self, config: PipelineConfig, vision_client: BaseVisionClient | None = None):
        self.config = config
        self.vision_client = vision_client

    def run(self, image_path: str | Path) -> GeometryData:
        """Run vision analysis on the drawing image.

        If *image_path* is an SVG file the method first resolves it to a
        PNG (either from the pre-rendered cache or by rendering on-the-fly)
        before passing it to the vision model.
        """
        image_path = Path(image_path)
        logger.info("vision_stage_start", image_path=str(image_path))

        # Handle SVG inputs: convert to PNG for the vision model
        if image_path.suffix.lower() == ".svg":
            image_path = self._resolve_svg_to_png(image_path)

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

    @staticmethod
    def _resolve_svg_to_png(svg_path: Path) -> Path:
        """Find a pre-rendered PNG or render on-the-fly."""
        # Check pre-rendered directory
        png_candidate = _DEFAULT_RENDERED_PNG_DIR / f"{svg_path.stem}.png"
        if png_candidate.exists():
            logger.debug("svg_using_prerendered", png=str(png_candidate))
            return png_candidate

        # Render on-the-fly to a temp file
        from src.training.svg_renderer import render_svg_to_png

        tmp_dir = Path(tempfile.mkdtemp(prefix="svg_render_"))
        png_path = tmp_dir / f"{svg_path.stem}.png"
        render_svg_to_png(svg_path, png_path)
        logger.info("svg_rendered_on_the_fly", svg=str(svg_path), png=str(png_path))
        return png_path
