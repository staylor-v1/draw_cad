from pathlib import Path

import pytest

from src.segmentation.title_block import analyze_drawing_structure
from src.segmentation.vision_callouts import (
    DEFAULT_GDT_VISION_MODEL,
    detect_vision_callouts,
    usable_vision_callouts_for_masks,
)


pytestmark = pytest.mark.skipif(
    not DEFAULT_GDT_VISION_MODEL.exists(),
    reason="fine-tuned GD&T vision model artifact is not present",
)


def test_fine_tuned_yolo_detects_simple1_callout_candidates():
    pytest.importorskip("ultralytics")

    detections = detect_vision_callouts(Path("training_data/gdt/simple1.webp"), conf=0.05)
    classes = {item["className"] for item in detections}

    assert len(detections) >= 13
    assert all(item["source"] == "fine_tuned_yolo" for item in detections)
    assert "fcf_position" in classes
    assert "fcf_perpendicularity" in classes
    assert "boxed_basic_dimension" in classes
    assert "diameter_dimension" in classes
    assert any(0.38 < item["crop"]["x"] < 0.42 and 0.33 < item["crop"]["y"] < 0.37 for item in detections)


def test_thread_texture_detection_is_not_promoted_to_mask_callout():
    pytest.importorskip("ultralytics")

    detections = detect_vision_callouts(Path("training_data/gdt/threaded_cap.jpg"), conf=0.05)
    maskable = usable_vision_callouts_for_masks(detections, [], min_confidence=0.05)

    assert any(item["className"] == "non_callout_thread_texture" for item in detections)
    assert all(item["className"] != "non_callout_thread_texture" for item in maskable)


def test_fine_tuned_yolo_is_available_in_analysis_toolset():
    pytest.importorskip("ultralytics")

    structure = analyze_drawing_structure(Path("training_data/gdt/threaded_cap.jpg"))

    assert len(structure["visionCallouts"]) >= 8
    assert any(item["source"] == "fine_tuned_yolo" for item in structure["visionCallouts"])
