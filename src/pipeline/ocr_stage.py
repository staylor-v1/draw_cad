"""OCR stage: supplement vision output with PaddleOCR text extraction."""
from __future__ import annotations

from pathlib import Path

from src.schemas.geometry import Dimension, EnrichedGeometry, GeometryData, TextRegion
from src.schemas.pipeline_config import PipelineConfig
from src.tools.ocr import extract_dimensions_from_ocr, extract_text_regions, extract_text_regions_mock
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class OCRStage:
    """Supplements vision-extracted geometry with OCR-detected text."""

    def __init__(self, config: PipelineConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock

    def run(self, image_path: str | Path, geometry: GeometryData) -> EnrichedGeometry:
        """Run OCR on the drawing and merge with vision geometry."""
        if not self.config.pipeline.ocr_enabled:
            logger.info("ocr_stage_disabled")
            return EnrichedGeometry(geometry=geometry)

        logger.info("ocr_stage_start", image_path=str(image_path))

        if self.use_mock:
            text_regions = extract_text_regions_mock(image_path)
        else:
            text_regions = extract_text_regions(image_path)

        ocr_dimensions = extract_dimensions_from_ocr(text_regions)

        enriched = EnrichedGeometry(
            geometry=geometry,
            text_regions=text_regions,
            ocr_dimensions=ocr_dimensions,
        )

        logger.info(
            "ocr_stage_complete",
            text_regions=len(text_regions),
            ocr_dimensions=len(ocr_dimensions),
        )
        return enriched
