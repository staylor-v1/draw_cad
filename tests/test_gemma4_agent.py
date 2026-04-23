"""Tests for the Gemma 4 roundtrip agent subproject."""
from __future__ import annotations

from pathlib import Path

import gemma4_agent.agent as agent_module
from gemma4_agent.agent import Gemma4RoundTripAgent
from PIL import Image
from gemma4_agent.toolbox import (
    _bbox_extent_ratio,
    get_tool_instructions,
    get_tool_schemas,
    inspect_drawing,
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


def test_bbox_extent_ratio_is_translation_invariant():
    reference = [[100.0, -20.0, 7.0], [180.0, 20.0, 57.0]]
    translated = [[-10.0, 300.0, 4.0], [70.0, 340.0, 54.0]]

    assert _bbox_extent_ratio(translated, reference) == 1.0


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
    assert "part = Box(20" in code
    assert "fallback from raster drawing.png" in code
