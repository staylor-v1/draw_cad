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
    assert len(structure["annotationMasks"]) > len(callouts)
    assert any(item["source"] == "vector_annotation_line" for item in structure["annotationMasks"])


def test_simple1_annotation_masks_do_not_blank_lower_front_geometry():
    structure = analyze_drawing_structure(Path("training_data/gdt/simple1.webp"))
    front = next(item for item in structure["projections"] if item["axis"] == "front")["crop"]
    front_bottom_geometry_masks = []
    for mask in structure["annotationMasks"]:
        if mask.get("source") != "vector_annotation_line" or mask.get("orientation") not in {"horizontal", "vertical"}:
            continue
        crop = mask["crop"]
        overlaps_front = (
            crop["x"] < front["x"] + front["w"]
            and crop["x"] + crop["w"] > front["x"]
            and crop["y"] < front["y"] + front["h"]
            and crop["y"] + crop["h"] > front["y"]
        )
        if not overlaps_front:
            continue
        relative_y = (max(crop["y"], front["y"]) - front["y"]) / front["h"]
        if relative_y >= 0.72:
            front_bottom_geometry_masks.append(mask)

    assert front_bottom_geometry_masks == []


def test_simple1_masks_boxed_three_length_callout_without_side_false_positive():
    structure = analyze_drawing_structure(Path("training_data/gdt/simple1.webp"))
    front = next(item for item in structure["projections"] if item["axis"] == "front")["crop"]
    side = next(item for item in structure["projections"] if item["axis"] == "side")["crop"]

    boxed_three_masks = [
        item
        for item in structure["annotationMasks"]
        if item.get("source") == "teacher_fixture" and item.get("label") == "3"
    ]
    assert boxed_three_masks

    front_length_line_masks = []
    for mask in structure["annotationMasks"]:
        crop = mask["crop"]
        overlaps_front = (
            crop["x"] < front["x"] + front["w"]
            and crop["x"] + crop["w"] > front["x"]
            and crop["y"] < front["y"] + front["h"]
            and crop["y"] + crop["h"] > front["y"]
        )
        if not overlaps_front or mask.get("source") != "vector_annotation_line":
            continue
        relative_x = (max(crop["x"], front["x"]) - front["x"]) / front["w"]
        relative_y = (max(crop["y"], front["y"]) - front["y"]) / front["h"]
        if mask.get("orientation") == "vertical" and relative_x <= 0.08 and 0.42 <= relative_y <= 0.50:
            front_length_line_masks.append(mask)

    assert front_length_line_masks

    side_mid_false_masks = []
    for mask in structure["annotationMasks"]:
        crop = mask["crop"]
        overlaps_side = (
            crop["x"] < side["x"] + side["w"]
            and crop["x"] + crop["w"] > side["x"]
            and crop["y"] < side["y"] + side["h"]
            and crop["y"] + crop["h"] > side["y"]
        )
        if not overlaps_side:
            continue
        relative_y = (max(crop["y"], side["y"]) - side["y"]) / side["h"]
        relative_width = min(crop["x"] + crop["w"], side["x"] + side["w"]) - max(crop["x"], side["x"])
        if 0.22 <= relative_y <= 0.42 and relative_width / side["w"] > 0.45:
            side_mid_false_masks.append(mask)

    assert side_mid_false_masks == []
