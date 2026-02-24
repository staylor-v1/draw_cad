"""Vision analysis tool using the inference abstraction layer."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from src.inference.base import BaseVisionClient, LLMResponse
from src.schemas.geometry import Dimension, Feature, GeometryData
from src.utils.image_utils import encode_image_base64, get_image_mime_type
from src.utils.file_utils import load_prompt
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM response text, handling markdown code blocks."""
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("failed_to_extract_json", text_length=len(text))
    return {}


def analyze_drawing(
    image_path: str | Path,
    vision_client: BaseVisionClient,
    model: str,
    prompt_path: str = "prompts/vision_prompt.md",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> GeometryData:
    """Analyze an engineering drawing using the vision model."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = load_prompt(prompt_path)
    image_b64 = encode_image_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    logger.info("analyzing_drawing", image_path=str(image_path), model=model)

    response: LLMResponse = vision_client.analyze_image(
        image_base64=image_b64,
        prompt=prompt,
        model=model,
        mime_type=mime_type,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_json = extract_json_from_response(response.content)
    if not raw_json:
        logger.warning("vision_empty_json", raw_content=response.content[:500])
        return GeometryData(raw_response=response.content)

    geometry = GeometryData.from_vision_dict(raw_json)
    geometry.raw_response = response.content
    logger.info(
        "vision_analysis_complete",
        views=len(geometry.views),
        dimensions=len(geometry.dimensions),
        features=len(geometry.features),
    )
    return geometry


def analyze_drawing_mock(image_path: str | Path) -> GeometryData:
    """Mock vision analysis for testing without a live model backend."""
    logger.info("mock_vision_analysis", image_path=str(image_path))
    return GeometryData(
        views=["Top", "Front", "Right"],
        dimensions=[
            Dimension(label="Length", value=100.0, unit="mm"),
            Dimension(label="Width", value=50.0, unit="mm"),
            Dimension(label="Thickness", value=10.0, unit="mm"),
            Dimension(label="Hole Diameter", value=5.0, unit="mm", count=1),
        ],
        features=[
            Feature(type="Base Plate", description="Rectangular plate 100x50x10mm"),
            Feature(type="Through Hole", description="1x 5mm hole centered on the top face"),
        ],
        notes="Standard tolerance +/- 0.1mm",
    )
