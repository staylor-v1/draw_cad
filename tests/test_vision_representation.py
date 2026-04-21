"""Tests for drawing text representation extraction."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.schemas.geometry import TextRegion
from src.tools import vision


def test_extract_drawing_text_representation_masks_and_vectorizes(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "drawing.png"
    _create_test_drawing(image_path)

    mock_regions = [
        TextRegion(text="TITLE: BRACKET", bbox=[140, 145, 196, 156], confidence=0.99, category="note"),
        TextRegion(text="AUTHOR: JANE", bbox=[140, 157, 196, 167], confidence=0.99, category="note"),
        TextRegion(text="⌀0.2 A B", bbox=[30, 20, 70, 28], confidence=0.97, category="gdt"),
        TextRegion(text="50", bbox=[80, 80, 90, 90], confidence=0.95, category="dimension"),
    ]
    monkeypatch.setattr(vision, "extract_text_regions", lambda _: mock_regions)

    masked_path = tmp_path / "masked.png"
    representation = vision.extract_drawing_text_representation(image_path, masked_path)

    assert representation.metadata.title == "BRACKET"
    assert representation.metadata.author == "JANE"
    assert len(representation.gdt_markups) == 1
    assert "⌀0.2 A B" in representation.to_text()
    assert "<svg" in representation.svg
    assert masked_path.exists()


def _create_test_drawing(path: Path) -> None:
    img = Image.new("RGB", (200, 180), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 120, 120], outline="black", width=2)
    draw.rectangle([130, 140, 198, 178], outline="black", width=1)  # title block
    draw.text((136, 146), "TITLE: BRACKET", fill="black")
    img.save(path)
