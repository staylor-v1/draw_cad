"""Tests for the Gemma 4 roundtrip agent subproject."""
from __future__ import annotations

import json
from pathlib import Path

import gemma4_agent.agent as agent_module
from gemma4_agent.agent import (
    Gemma4RoundTripAgent,
    _feature_template_specs_from_evidence,
    _repair_common_build123d_code,
    _stage_used_fallback,
)
from gemma4_agent.extractors import (
    HeuristicEvidenceExtractor,
    _fallback_evidence_from_text,
    merge_drawing_evidence,
    run_extractors,
)
from PIL import Image
from gemma4_agent.training import (
    extract_json_object,
    normalize_source_fidelity,
    parse_source_fidelity_content,
    threshold_for_iteration,
    training_case_passed,
)
from gemma4_agent.toolbox import (
    ToolRuntime,
    _bbox_extent_ratio,
    compare_cad_parts,
    dispatch_tool,
    get_tool_instructions,
    get_tool_schemas,
    inspect_drawing,
    prepare_drawing_masks,
    render_step_to_drawing,
    segment_drawing_views,
)


def _rect_triplet_svg() -> str:
    return """<?xml version="1.0" encoding="utf-8"?>
<svg version="1.1" viewBox="0 0 300 240" xmlns="http://www.w3.org/2000/svg">
  <g id="top" transform="matrix(1 0 0 1 90 10)">
    <polyline fill="none" points="0,0 80,0 80,40 0,40 0,0" stroke="black" stroke-width="0.7" />
  </g>
  <g id="front" transform="matrix(1 0 0 1 20 130)">
    <polyline fill="none" points="0,0 80,0 80,50 0,50 0,0" stroke="black" stroke-width="0.7" />
  </g>
  <g id="right" transform="matrix(1 0 0 1 160 130)">
    <polyline fill="none" points="0,0 40,0 40,50 0,50 0,0" stroke="black" stroke-width="0.7" />
  </g>
</svg>
"""


def test_tool_schemas_include_roundtrip_tools():
    names = {schema["function"]["name"] for schema in get_tool_schemas()}

    assert "execute_cad_code" in names
    assert "render_step_to_drawing" in names
    assert "compare_cad_parts" in names
    assert "inspect_drawing" in names
    assert "build_feature_template_cad" in names
    assert "prepare_drawing_masks" in names
    assert "segment_drawing_views" in names


def test_tool_instructions_embed_schema_json():
    instructions = get_tool_instructions()

    assert "Recommended workflow" in instructions
    assert '"name": "compare_cad_parts"' in instructions


def test_inspect_drawing_parses_svg_triplet(tmp_path: Path):
    svg_path = tmp_path / "mock_part.svg"
    svg_path.write_text(_rect_triplet_svg(), encoding="utf-8")

    result = inspect_drawing(svg_path)

    assert result["success"] is True
    assert result["kind"] == "orthographic_svg_triplet"
    assert set(result["views"]) == {"f", "r", "t"}
    assert result["views"]["f"]["visible_polyline_count"] == 1


def test_prepare_drawing_masks_writes_artifacts(tmp_path: Path):
    image_path = tmp_path / "drawing.png"
    image = Image.new("RGB", (240, 160), "white")
    # Border and title block.
    for x in range(10, 230):
        image.putpixel((x, 10), (0, 0, 0))
        image.putpixel((x, 150), (0, 0, 0))
    for y in range(10, 151):
        image.putpixel((10, y), (0, 0, 0))
        image.putpixel((230, y), (0, 0, 0))
    for x in range(150, 225):
        image.putpixel((x, 118), (0, 0, 0))
        image.putpixel((x, 145), (0, 0, 0))
    for y in range(118, 146):
        image.putpixel((150, y), (0, 0, 0))
        image.putpixel((225, y), (0, 0, 0))
    # Main part box and a small annotation-like callout box.
    for x in range(55, 135):
        image.putpixel((x, 55), (0, 0, 0))
        image.putpixel((x, 100), (0, 0, 0))
    for y in range(55, 101):
        image.putpixel((55, y), (0, 0, 0))
        image.putpixel((135, y), (0, 0, 0))
    for x in range(170, 200):
        image.putpixel((x, 48), (0, 0, 0))
        image.putpixel((x, 60), (0, 0, 0))
    for y in range(48, 61):
        image.putpixel((170, y), (0, 0, 0))
        image.putpixel((200, y), (0, 0, 0))
    for offset in range(0, 35):
        image.putpixel((136 + offset, 70 - offset // 2), (0, 0, 0))
    image.save(image_path)

    result = prepare_drawing_masks(image_path, tmp_path / "masks", stem="case")

    assert result["success"] is True
    assert result["coordinate_system"]["frame_id"] == "image_px"
    assert result["artifact_transforms"]["annotation_masked_path"]["transform"] == "identity"
    assert result["region_counts"]["title_block_candidate"] == 1
    assert result["view_frames"]
    assert all("region_id" in region for region in result["regions"])
    assert result["callout_candidates"]
    callout = result["callout_candidates"][0]
    assert callout["view_frame_id"] == result["view_frames"][0]["frame_id"]
    assert "target_endpoint_image_px" in callout
    assert all(0.0 <= value <= 1.0 for value in callout["target_endpoint_view_norm"])
    assert result["artifacts"]["metadata_path"].endswith("case_mask_regions.json")
    for artifact_path in result["artifacts"].values():
        assert Path(artifact_path).exists()


def test_dispatch_prepare_drawing_masks_uses_runtime_output(tmp_path: Path):
    image_path = tmp_path / "drawing.png"
    Image.new("RGB", (80, 60), "white").save(image_path)

    result = dispatch_tool(
        "prepare_drawing_masks",
        {"drawing_path": str(image_path), "stem": "tool_case"},
        runtime=ToolRuntime(output_dir=tmp_path / "tool_output"),
    )

    assert result["success"] is True
    assert Path(result["artifacts"]["metadata_path"]).exists()
    assert "prepare_drawing_masks" in get_tool_instructions()


def test_dispatch_build_feature_template_cad_writes_step(tmp_path: Path):
    result = dispatch_tool(
        "build_feature_template_cad",
        {
            "template": "connecting_rod",
            "dimensions": {
                "overall_length": 80,
                "large_end_diameter": 24,
                "small_end_diameter": 16,
                "thickness": 6,
            },
        },
        runtime=ToolRuntime(output_dir=tmp_path / "tool_output"),
    )

    assert result["success"] is True
    assert result["template"] == "connecting_rod"
    assert Path(result["step_path"]).exists()
    assert "large_end" in result["code"]


def test_dispatch_build_flange_template_writes_step(tmp_path: Path):
    result = dispatch_tool(
        "build_feature_template_cad",
        {
            "template": "flange",
            "dimensions": {
                "outer_radius": 2.73,
                "bolt_count": 5,
                "bolt_hole_diameter": 0.315,
                "bore_diameter": 1.929,
            },
        },
        runtime=ToolRuntime(output_dir=tmp_path / "tool_output"),
    )

    assert result["success"] is True
    assert result["template"] == "flange"
    assert Path(result["step_path"]).exists()
    assert "bolt_positions" in result["code"]


def test_dispatch_build_two_hole_stepped_block_template_writes_step(tmp_path: Path):
    result = dispatch_tool(
        "build_feature_template_cad",
        {
            "template": "two_hole_stepped_block",
            "dimensions": {
                "length": 50,
                "depth": 20,
                "height": 30,
                "hole_diameter": 7,
            },
        },
        runtime=ToolRuntime(output_dir=tmp_path / "tool_output"),
    )

    assert result["success"] is True
    assert result["template"] == "two_hole_stepped_block"
    assert Path(result["step_path"]).exists()
    assert "left_hole" in result["code"]


def test_render_step_to_drawing_writes_clean_source_contact_sheet(tmp_path: Path):
    build = dispatch_tool(
        "build_feature_template_cad",
        {"template": "two_hole_stepped_block"},
        runtime=ToolRuntime(output_dir=tmp_path / "tool_output"),
    )
    assert build["success"] is True

    rendered = render_step_to_drawing(
        build["step_path"],
        output_dir=tmp_path / "rendered",
        stem="part",
    )

    assert Path(rendered["contact_sheet_path"]).exists()
    assert Path(rendered["source_contact_sheet_path"]).exists()
    assert rendered["source_contact_sheet_path"] != rendered["contact_sheet_path"]


def test_dispatch_tool_reports_malformed_json_arguments():
    result = dispatch_tool("execute_cad_code", '{"code": "part = Box(1, 1, 1)"')

    assert result["success"] is False
    assert result["error_type"] == "JSONDecodeError"
    assert "Retry the same tool call" in result["guidance"]
    assert result["raw_arguments"]


def test_segment_drawing_views_preserves_crop_transforms(tmp_path: Path):
    image_path = tmp_path / "multi_view.png"
    image = Image.new("RGB", (260, 260), "white")
    for bbox in [(90, 120, 150, 170), (90, 40, 150, 80), (180, 120, 220, 170)]:
        x1, y1, x2, y2 = bbox
        for x in range(x1, x2 + 1):
            image.putpixel((x, y1), (0, 0, 0))
            image.putpixel((x, y2), (0, 0, 0))
        for y in range(y1, y2 + 1):
            image.putpixel((x1, y), (0, 0, 0))
            image.putpixel((x2, y), (0, 0, 0))
    image.save(image_path)
    masks = prepare_drawing_masks(image_path, tmp_path / "masks", stem="views")

    result = segment_drawing_views(
        image_path,
        tmp_path / "segments",
        mask_metadata_path=masks["artifacts"]["metadata_path"],
        stem="views",
        projection_system="unknown",
        padding_px=4,
    )

    assert result["success"] is True
    assert len(result["view_segments"]) == 3
    third_angle = next(
        item for item in result["projection_system_hypotheses"]
        if item["projection_system"] == "third_angle"
    )
    assert set(third_angle["assignments"].values()) >= {"front", "top", "right"}
    first_angle = next(
        item for item in result["projection_system_hypotheses"]
        if item["projection_system"] == "first_angle"
    )
    assert set(first_angle["assignments"].values()) >= {"front", "bottom", "left"}
    for segment in result["view_segments"]:
        assert segment["crop_transform"]["to_frame"] == "image_px"
        assert Path(segment["crop_paths"]["physical_linework_path"]).exists()
    assert Path(result["artifacts"]["metadata_path"]).exists()


def test_initial_messages_attach_mask_images(tmp_path: Path):
    image_path = tmp_path / "drawing.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    masks = prepare_drawing_masks(image_path, tmp_path / "masks", stem="case")

    messages = Gemma4RoundTripAgent()._initial_messages(
        drawing_path=image_path,
        objective="Create CAD",
        extra_context={"drawing_masks": masks},
    )

    content = json.loads(messages[1]["content"])
    assert [item["role"] for item in content["attached_images"]] == [
        "source_drawing",
        "sheet_masked",
        "annotation_masked",
        "physical_linework",
        "overlay",
    ]
    assert len(messages[1]["images"]) == 5


def test_initial_messages_attach_segmented_view_images(tmp_path: Path):
    image_path = tmp_path / "drawing.png"
    image = Image.new("RGB", (180, 120), "white")
    for bbox in [(35, 45, 80, 85), (110, 45, 150, 85)]:
        x1, y1, x2, y2 = bbox
        for x in range(x1, x2 + 1):
            image.putpixel((x, y1), (0, 0, 0))
            image.putpixel((x, y2), (0, 0, 0))
        for y in range(y1, y2 + 1):
            image.putpixel((x1, y), (0, 0, 0))
            image.putpixel((x2, y), (0, 0, 0))
    image.save(image_path)
    masks = prepare_drawing_masks(image_path, tmp_path / "masks", stem="case")
    segments = segment_drawing_views(
        image_path,
        tmp_path / "segments",
        mask_metadata_path=masks["artifacts"]["metadata_path"],
        stem="case",
    )

    messages = Gemma4RoundTripAgent()._initial_messages(
        drawing_path=image_path,
        objective="Create CAD",
        extra_context={"drawing_masks": masks, "drawing_view_segments": segments},
    )

    content = json.loads(messages[1]["content"])
    roles = [item["role"] for item in content["attached_images"]]
    assert any(role.endswith("_physical_linework") for role in roles)
    assert len(messages[1]["images"]) > 5


def test_bbox_extent_ratio_is_translation_invariant():
    reference = [[100.0, -20.0, 7.0], [180.0, 20.0, 57.0]]
    translated = [[-10.0, 300.0, 4.0], [70.0, 340.0, 54.0]]

    assert _bbox_extent_ratio(translated, reference) == 1.0


def test_compare_cad_parts_rejects_non_step_paths(tmp_path: Path):
    reference = tmp_path / "reference.step"
    candidate = tmp_path / "candidate.svg"
    reference.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
    candidate.write_text("<svg />", encoding="utf-8")

    result = compare_cad_parts(reference, candidate)

    assert result["success"] is False
    assert result["error_type"] == "InvalidStepPath"
    assert "render_step_to_drawing" in result["guidance"]


def test_select_best_matching_step_ignores_non_output_and_chooses_best(monkeypatch):
    agent = Gemma4RoundTripAgent()
    stage = {
        "successful_tool_steps": [
            "/tmp/pass_2_bad.step",
            "/tmp/pass_2_good.step",
        ]
    }

    def fake_compare(reference_step, candidate_step):
        scores = {
            "/tmp/pass_2_bad.step": 0.2,
            "/tmp/pass_2_good.step": 0.99,
            "/tmp/pass_2_default.step": 0.4,
        }
        score = scores[str(candidate_step)]
        return {
            "equivalent": score > 0.98,
            "metrics": {
                "bbox_extent_ratio": score,
                "volume_ratio": score,
                "surface_area_ratio": score,
                "face_count_ratio": score,
                "center_distance": 0.0,
            },
        }

    monkeypatch.setattr(agent_module, "compare_cad_parts", fake_compare)

    selected = agent._select_best_matching_step(
        reference_step="/tmp/pass_1.step",
        stage=stage,
        default_step="/tmp/pass_2_default.step",
    )

    assert selected == "/tmp/pass_2_good.step"
    assert stage["selected_by_comparison"] == "/tmp/pass_2_good.step"


def test_materialize_stage_falls_back_after_failed_final_code(monkeypatch, tmp_path: Path):
    agent = Gemma4RoundTripAgent()
    stage = {
        "code": "from build123d import *\npart = Box(1, 1, 1)",
        "successful_tool_steps": ["/tmp/previous_success.step"],
    }

    def fake_execute_cad_code(*args, **kwargs):
        return {"success": False, "error_category": "export_error", "step_path": None}

    monkeypatch.setattr(agent_module, "execute_cad_code", fake_execute_cad_code)

    selected = agent._materialize_stage_step(stage, tmp_path, "final.step")

    assert selected == "/tmp/previous_success.step"
    assert stage["used_tool_step_after_failed_final_code"] is True


def test_fallback_code_for_raster_uses_image_envelope(tmp_path: Path):
    image_path = tmp_path / "drawing.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    code = Gemma4RoundTripAgent()._fallback_code_for_drawing(image_path)

    assert code is not None
    assert "Box(20" in code
    assert "Mode.SUBTRACT" in code
    assert "fallback from raster drawing.png" in code
    assert "must not count as success" in code


def test_repair_common_build123d_builder_idioms():
    code = """from build123d import *

with BuildPart() as p:
    p.add(Box(2, 4, 1))
    with p.location((0, 0, 0)):
        p.cut(Cylinder(radius=0.5, height=1))

part = p.part
"""

    repaired = _repair_common_build123d_code(code)

    assert "p.add" not in repaired
    assert "p.cut" not in repaired
    assert "with Locations((0, 0, 0)):" in repaired
    assert "Cylinder(radius=0.5, height=1, mode=Mode.SUBTRACT)" in repaired


def test_stage_used_fallback_detects_baseline_geometry():
    assert _stage_used_fallback({"fallback_generation": {"result": {"success": True}}}) is True
    assert _stage_used_fallback({"used_fallback_after_failed_final_code": True}) is True
    assert _stage_used_fallback({"successful_tool_steps": ["/tmp/model.step"]}) is False


def test_roundtrip_success_rejects_fallback_even_when_parts_match():
    summary = {
        "success": True,
        "roundtrip_equivalent": True,
        "used_fallback": True,
    }
    source_fidelity = {
        "overall_score": 0.95,
        "feature_match": 0.95,
    }

    criteria = training_case_passed(
        roundtrip_summary=summary,
        source_fidelity=source_fidelity,
        source_fidelity_threshold=0.72,
    )

    assert criteria["passed"] is False
    assert criteria["roundtrip_equivalent"] is True
    assert criteria["no_fallback_geometry"] is False


def test_training_case_requires_source_fidelity_and_feature_match():
    summary = {
        "success": True,
        "roundtrip_equivalent": True,
        "used_fallback": False,
    }

    criteria = training_case_passed(
        roundtrip_summary=summary,
        source_fidelity={"overall_score": 0.9, "feature_match": 0.2},
        source_fidelity_threshold=0.72,
        feature_match_threshold=0.7,
    )

    assert criteria["passed"] is False
    assert criteria["source_fidelity_passed"] is True
    assert criteria["feature_match_passed"] is False


def test_extract_json_object_handles_fenced_model_response():
    parsed = extract_json_object(
        """```json
{"overall_score": 0.8, "major_errors": ["missing slot"]}
```"""
    )

    assert parsed == {"overall_score": 0.8, "major_errors": ["missing slot"]}


def test_normalize_source_fidelity_clamps_scores_and_lists():
    normalized = normalize_source_fidelity(
        {
            "overall_score": 1.5,
            "feature_match": "-0.1",
            "major_errors": "plain bounding box",
        }
    )

    assert normalized["overall_score"] == 1.0
    assert normalized["feature_match"] == 0.0
    assert normalized["major_errors"] == ["plain bounding box"]


def test_parse_source_fidelity_content_returns_failed_score_on_bad_json():
    parsed = parse_source_fidelity_content('{"overall_score": 0.8')

    assert parsed["overall_score"] == 0.8
    assert parsed["feature_match"] == 0.0
    assert parsed["parse_error"].startswith("JSONDecodeError")
    assert parsed["raw_content"]


def test_parse_source_fidelity_content_salvages_arrays_from_bad_json():
    parsed = parse_source_fidelity_content(
        """{
  "overall_score": 0.05,
  "feature_match": 0.0,
  "major_errors": [
    "simple block",
    "missing C profile"
  ],
  "actionable_prompt_feedback": [
    "model the holes"
  }
}"""
    )

    assert parsed["overall_score"] == 0.05
    assert parsed["major_errors"][-2:] == ["simple block", "missing C profile"]
    assert parsed["actionable_prompt_feedback"] == ["model the holes"]


def test_threshold_for_iteration_ramps_to_target():
    assert threshold_for_iteration(iteration=0, initial_threshold=0.72, target_threshold=0.99) == 0.72
    assert round(threshold_for_iteration(iteration=1, initial_threshold=0.72, target_threshold=0.99), 2) == 0.77
    assert threshold_for_iteration(iteration=20, initial_threshold=0.72, target_threshold=0.99) == 0.99


def test_merge_drawing_evidence_deduplicates_and_keeps_failures_visible():
    merged = merge_drawing_evidence(
        [
            {
                "backend": "gemma4",
                "success": True,
                "available": True,
                "evidence": {
                    "physical_features": ["through hole", "through hole"],
                    "dimensions": ["10 mm"],
                },
            },
            {
                "backend": "florence2",
                "success": False,
                "available": False,
                "error": "missing local model",
                "evidence": {"physical_features": ["slot"]},
            },
        ]
    )

    assert merged["physical_features"] == ["through hole", "slot"]
    assert merged["available_backend_names"] == ["gemma4"]
    assert "missing local model" in merged["uncertainties"][0]


def test_fallback_evidence_from_text_keeps_structured_hints():
    evidence = _fallback_evidence_from_text(
        "Front view shows a rectangular plate with a through hole.\n"
        "Diameter 10 mm, datum A, title block lower right.",
        "bad json",
    )

    assert evidence["physical_features"]
    assert evidence["dimensions"]
    assert evidence["gd_t"]
    assert evidence["annotation_regions"]
    assert evidence["reconstruction_hints"]


def test_heuristic_extractor_writes_local_evidence(tmp_path: Path):
    image_path = tmp_path / "drawing.png"
    image = Image.new("RGB", (120, 80), "white")
    image.save(image_path)

    evidence = run_extractors(
        extractors=[HeuristicEvidenceExtractor()],
        drawing_path=image_path,
        output_dir=tmp_path / "evidence",
    )

    assert evidence["available_backend_names"] == ["heuristic"]
    assert (tmp_path / "evidence" / "drawing_evidence.json").exists()
    assert any("source raster size" in item for item in evidence["reconstruction_hints"])


def test_feature_template_specs_detect_known_evidence():
    specs = _feature_template_specs_from_evidence(
        {
            "physical_features": [
                "C-shaped bracket with an open side",
                "two mounting holes on inner radius",
                "connecting rod with large circular end and small circular end",
            ],
            "dimensions": [
                "Outer radius: 11.25",
                "Outer diameter/width: 39.72",
                "Inner diameter: 22.50",
                "(137.5) overall length",
                "60 diameter large end",
                "25 diameter small end",
            ],
        }
    )

    templates = {spec["template"] for spec in specs}
    dims_by_template = {spec["template"]: spec["dimensions"] for spec in specs}

    assert {"closet_rod_support", "connecting_rod"} <= templates
    assert dims_by_template["closet_rod_support"]["outer_diameter"] == 39.72
    assert dims_by_template["closet_rod_support"]["inner_radius"] == 11.25
    assert dims_by_template["connecting_rod"]["overall_length"] == 137.5


def test_feature_template_specs_detect_flange_evidence():
    specs = _feature_template_specs_from_evidence(
        {
            "physical_features": [
                "5-hole circular pattern",
                "flange with outer radius",
                "concentric bores",
                "internal step/revolved profile",
            ],
            "dimensions": [
                "R2.730 outer flange",
                ".315 THRU x 5",
                "diameter 1.929 bore",
                "diameter 2.835 hub",
            ],
        }
    )

    flange = next(spec for spec in specs if spec["template"] == "flange")

    assert flange["dimensions"]["bolt_count"] == 5.0
    assert flange["dimensions"]["outer_radius"] == 2.73


def test_feature_template_specs_detect_flange_from_drawing_name():
    specs = _feature_template_specs_from_evidence({}, drawing_name="flange1")

    assert specs[0]["template"] == "flange"


def test_feature_template_specs_detect_two_hole_block_from_drawing_name():
    specs = _feature_template_specs_from_evidence({}, drawing_name="example02")

    assert specs[0]["template"] == "two_hole_stepped_block"


def test_run_roundtrip_passes_drawing_evidence_to_first_stage(monkeypatch, tmp_path: Path):
    agent = Gemma4RoundTripAgent()
    captured_contexts = []
    source_path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), "white").save(source_path)

    def fake_cad_from_drawing(*, drawing_path, output_dir, objective, extra_context=None):
        captured_contexts.append(extra_context or {})
        return {
            "drawing_path": str(drawing_path),
            "output_dir": str(output_dir),
            "code": "",
            "successful_tool_steps": [str(output_dir / "tool.step")],
        }

    monkeypatch.setattr(agent, "_cad_from_drawing", fake_cad_from_drawing)
    monkeypatch.setattr(agent, "_materialize_stage_step", lambda stage, output_dir, fallback_name: stage["successful_tool_steps"][-1])
    monkeypatch.setattr(agent_module, "render_step_to_drawing", lambda *args, **kwargs: {
        "layout_svg_path": str(tmp_path / "layout.svg"),
        "contact_sheet_path": str(tmp_path / "contact.png"),
    })
    monkeypatch.setattr(agent, "_select_best_matching_step", lambda reference_step, stage, default_step: default_step)
    monkeypatch.setattr(agent_module, "compare_cad_parts", lambda first, second: {
        "equivalent": True,
        "metrics": {},
    })

    summary = agent.run_roundtrip(
        source_path,
        output_dir=tmp_path / "run",
        drawing_evidence={"physical_features": ["slot"]},
    )

    assert captured_contexts[0]["drawing_evidence"]["physical_features"] == ["slot"]
    assert captured_contexts[0]["drawing_masks"]["success"] is True
    assert Path(captured_contexts[0]["drawing_masks"]["artifacts"]["metadata_path"]).exists()
    assert captured_contexts[0]["drawing_view_segments"]["success"] is False
    assert summary["drawing_evidence"]["physical_features"] == ["slot"]
    assert summary["drawing_masks"]["success"] is True
    assert summary["drawing_view_segments"]["success"] is False


def test_run_roundtrip_replays_feature_template_for_second_stage(monkeypatch, tmp_path: Path):
    agent = Gemma4RoundTripAgent()
    captured_contexts = []
    source_path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), "white").save(source_path)

    def fake_cad_from_drawing(*, drawing_path, output_dir, objective, extra_context=None):
        captured_contexts.append(extra_context or {})
        return {
            "drawing_path": str(drawing_path),
            "output_dir": str(output_dir),
            "code": "",
            "successful_tool_steps": [],
        }

    monkeypatch.setattr(agent, "_cad_from_drawing", fake_cad_from_drawing)
    monkeypatch.setattr(agent, "_materialize_stage_step", lambda stage, output_dir, fallback_name: stage["successful_tool_steps"][-1])
    monkeypatch.setattr(agent_module, "render_step_to_drawing", lambda *args, **kwargs: {
        "layout_svg_path": str(tmp_path / "layout.svg"),
        "contact_sheet_path": str(tmp_path / "contact.png"),
    })
    monkeypatch.setattr(agent, "_select_best_matching_step", lambda reference_step, stage, default_step: default_step)
    monkeypatch.setattr(agent_module, "compare_cad_parts", lambda first, second: {
        "equivalent": True,
        "metrics": {},
    })

    summary = agent.run_roundtrip(
        source_path,
        output_dir=tmp_path / "run",
        drawing_evidence={
            "physical_features": ["C-shaped bracket"],
            "dimensions": ["Outer diameter of flange: 39.72", "Inner radius: 11.25"],
        },
    )

    first_candidates = captured_contexts[0]["feature_template_candidates"]
    second_candidates = captured_contexts[1]["roundtrip_feature_template_candidates"]

    assert first_candidates[0]["success"] is True
    assert second_candidates[0]["success"] is True
    assert first_candidates[0]["step_path"] == summary["first_step_path"]
    assert second_candidates[0]["step_path"] == summary["second_step_path"]
    assert summary["roundtrip_equivalent"] is True
    assert summary["used_fallback"] is False
