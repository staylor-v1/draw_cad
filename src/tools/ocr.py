"""PaddleOCR wrapper for dimension text extraction from engineering drawings."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.schemas.geometry import Dimension, TextRegion
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Lazy-load PaddleOCR to avoid import overhead when OCR is disabled
_ocr_engine = None


def _get_ocr_engine():
    """Get or create the PaddleOCR engine (lazy singleton)."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except ImportError:
            logger.warning("paddleocr_not_installed")
            return None
    return _ocr_engine


def extract_text_regions(image_path: str | Path) -> list[TextRegion]:
    """Extract text regions from an engineering drawing using PaddleOCR.
    
    Args:
        image_path: Path to the drawing image.
    
    Returns:
        List of TextRegion objects with detected text and bounding boxes.
    """
    engine = _get_ocr_engine()
    if engine is None:
        logger.info("ocr_skipped_no_engine")
        return []

    image_path = Path(image_path)
    if not image_path.exists():
        logger.warning("ocr_image_not_found", path=str(image_path))
        return []

    logger.info("ocr_processing", image_path=str(image_path))
    results = engine.ocr(str(image_path), cls=True)

    text_regions = []
    if results and results[0]:
        for line in results[0]:
            bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = line[1][0]
            confidence = float(line[1][1])

            # Convert 4-point bbox to [x_min, y_min, x_max, y_max]
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            category = classify_text_region(text)
            text_regions.append(TextRegion(
                text=text,
                bbox=bbox,
                confidence=confidence,
                category=category,
            ))

    logger.info("ocr_complete", regions_found=len(text_regions))
    return text_regions


def classify_text_region(text: str) -> str:
    """Classify a text region into categories."""
    import re
    text_stripped = text.strip()

    # Dimension pattern: numbers with optional units and tolerances
    if re.match(r'^[\d.,]+\s*(mm|cm|in|"|\'|°)?(\s*[±+\-][\d.,]+)?$', text_stripped):
        return "dimension"

    # Tolerance pattern
    if re.match(r'^[±+\-][\d.,]+', text_stripped):
        return "tolerance"

    # GD&T symbols or feature control frames
    if any(sym in text_stripped for sym in ["⌀", "∅", "Ø", "⊕", "⊖", "◎", "⊘"]):
        return "gdt"

    # Common drawing notes
    note_keywords = ["unless", "noted", "tolerance", "material", "finish", "scale", "all dimensions"]
    if any(kw in text_stripped.lower() for kw in note_keywords):
        return "note"

    # Label/callout
    if re.match(r'^[A-Z]{1,3}(-[A-Z]{1,3})?$', text_stripped):
        return "label"

    return "unknown"


def extract_dimensions_from_ocr(text_regions: list[TextRegion]) -> list[Dimension]:
    """Extract dimension values from OCR text regions.
    
    Args:
        text_regions: List of TextRegion objects.
    
    Returns:
        List of Dimension objects parsed from text.
    """
    import re
    dimensions = []

    for region in text_regions:
        if region.category != "dimension":
            continue

        # Parse numeric value
        match = re.match(r'([\d.,]+)\s*(mm|cm|in)?', region.text.strip())
        if match:
            value_str = match.group(1).replace(",", ".")
            try:
                value = float(value_str)
            except ValueError:
                continue

            unit = match.group(2) or "mm"
            dimensions.append(Dimension(
                label=f"OCR_{len(dimensions) + 1}",
                value=value,
                unit=unit,
                confidence=region.confidence,
            ))

    logger.info("ocr_dimensions_extracted", count=len(dimensions))
    return dimensions


def extract_text_regions_mock(image_path: str | Path) -> list[TextRegion]:
    """Mock OCR for testing without PaddleOCR installed."""
    logger.info("mock_ocr_analysis", image_path=str(image_path))
    return [
        TextRegion(text="100", bbox=[10, 10, 50, 30], confidence=0.95, category="dimension"),
        TextRegion(text="50", bbox=[60, 10, 90, 30], confidence=0.93, category="dimension"),
        TextRegion(text="10", bbox=[100, 50, 130, 70], confidence=0.91, category="dimension"),
        TextRegion(text="∅5", bbox=[50, 50, 75, 70], confidence=0.88, category="gdt"),
        TextRegion(text="UNLESS NOTED ALL DIMS IN MM", bbox=[10, 200, 300, 220], confidence=0.85, category="note"),
    ]
