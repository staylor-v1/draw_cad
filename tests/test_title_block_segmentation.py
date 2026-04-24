from pathlib import Path

from PIL import Image

from src.segmentation.title_block import analyze_drawing_structure, detect_title_block


def assert_box_close(actual, expected, tolerance=0.015):
    for key, value in expected.items():
        assert abs(actual[key] - value) <= tolerance, f"{key}: {actual[key]} != {value}"


def test_detects_full_flange_title_block_region():
    image = Image.open(Path("training_data/gdt/flange1.png"))
    candidate = detect_title_block(image)

    assert candidate.confidence >= 0.45
    assert candidate.crop.x < 0.5
    assert candidate.crop.y > 0.70
    assert candidate.crop.w > 0.45
    assert candidate.crop.h > 0.10
    assert candidate.present is True


def test_detects_iso_symbol_sheet_title_block_region():
    image = Image.open(Path("training_data/title_block_examples/iso_10628_symbols_sheet_title_block.png"))
    candidate = detect_title_block(image)

    assert candidate.confidence >= 0.45
    assert candidate.crop.x > 0.65
    assert candidate.crop.y > 0.82
    assert candidate.crop.w > 0.20
    assert candidate.crop.h > 0.06


def test_flange_projection_regions_are_nonblank():
    structure = analyze_drawing_structure(Path("training_data/gdt/flange1.png"))
    projections = structure["projections"]

    assert len(projections) >= 3
    assert all(projection["crop"]["w"] > 0.08 for projection in projections[:3])
    assert all(projection["crop"]["h"] > 0.05 for projection in projections[:3])


def test_flange1_regression_title_block_and_projection_crops():
    structure = analyze_drawing_structure(Path("training_data/gdt/flange1.png"))
    assert_box_close(
        structure["titleBlock"]["crop"],
        {"x": 0.409, "y": 0.770, "w": 0.557, "h": 0.197},
    )

    middle_center = next(
        item for item in structure["projections"] if item["label"] == "middle center projection candidate"
    )
    assert_box_close(
        middle_center["crop"],
        {"x": 0.472, "y": 0.419, "w": 0.263, "h": 0.347},
    )


def test_flange_middle_center_projection_keeps_lower_context():
    structure = analyze_drawing_structure(Path("training_data/gdt/flange1.png"))
    projection = next(item for item in structure["projections"] if item["label"] == "middle center projection candidate")
    crop = projection["crop"]

    assert crop["y"] <= 0.42
    assert crop["y"] + crop["h"] >= 0.765


def test_flange2_splits_overlapping_review_boxes_with_component_masks():
    structure = analyze_drawing_structure(Path("training_data/gdt/flange2.png"))
    projections = structure["projections"]

    assert structure["titleBlock"]["present"] is False
    assert structure["border"]["present"] is False
    assert len(projections) >= 2
    left = projections[0]["crop"]
    right = projections[1]["crop"]
    assert left["x"] < 0.05
    assert right["x"] > 0.50
    assert left["x"] + left["w"] > right["x"]
    assert projections[0]["segmentationMode"] == "component_mask"
    assert projections[1]["segmentationMode"] == "component_mask"


def test_flange2_detects_gdt_callouts_for_masking():
    structure = analyze_drawing_structure(Path("training_data/gdt/flange2.png"))
    callouts = structure["gdt"]

    assert len(callouts) >= 4
    assert any(item["kind"] == "feature_control_frame" for item in callouts)
    assert any(0.35 < item["crop"]["x"] < 0.56 and 0.12 < item["crop"]["y"] < 0.25 for item in callouts)
    assert all(item["crop"]["w"] > 0.03 and item["crop"]["h"] > 0.02 for item in callouts)


def test_simple1_teacher_fixture_identifies_all_visible_callouts():
    structure = analyze_drawing_structure(Path("training_data/gdt/simple1.webp"))
    callouts = structure["callouts"]
    front = [item for item in callouts if item["view"] == "front"]
    side = [item for item in callouts if item["view"] == "side"]

    assert structure["titleBlock"]["present"] is False
    assert structure["border"]["present"] is False
    assert len(structure["projections"]) == 2
    assert structure["projections"][0]["axis"] == "front"
    assert structure["projections"][1]["axis"] == "side"
    assert len(front) == 10
    assert len(side) == 3
    assert any(item["id"] == "front-09" and "not visible" in item["note"] for item in front)
