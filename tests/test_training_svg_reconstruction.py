"""Tests for training-data SVG orthographic reconstruction helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.comparator import StepComparator
from src.reconstruction import (
    OrthographicTripletReconstructor,
    load_training_svg_triplet,
    parse_training_svg_triplet,
)
from src.schemas.pipeline_config import PipelineConfig
from src.tools.cad import execute_build123d


def _mock_training_svg() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
  <g id="DrawingContent" stroke="#000000" fill="none">
    <g transform="matrix(1,0,0,1,80,200)">
      <path d="M-20,-20 L20,-20 L20,20 L-20,20 Z" stroke-width="2.5" />
      <path d="M-10,-10 L10,-10 L10,10 L-10,10 Z" stroke-width="2.5" />
    </g>
    <g transform="matrix(1,0,0,1,80,80)">
      <path d="M-20,-40 L20,-40 L20,40 L-20,40 Z" stroke-width="2.5" />
      <path d="M-10,-40 L-10,40" stroke-width="1.3" stroke-dasharray="5.2,2.6" />
      <path d="M10,-40 L10,40" stroke-width="1.3" stroke-dasharray="5.2,2.6" />
    </g>
    <g transform="matrix(1,0,0,1,200,200)">
      <path d="M-40,-20 L40,-20 L40,20 L-40,20 Z" stroke-width="2.5" />
      <path d="M-40,-10 L40,-10" stroke-width="1.3" stroke-dasharray="5.2,2.6" />
      <path d="M-40,10 L40,10" stroke-width="1.3" stroke-dasharray="5.2,2.6" />
    </g>
  </g>
</svg>
"""


def _candidate_reprojection_scores(case_id: str) -> dict[str, float]:
    svg_path = Path("training_data/drawings_svg") / f"{case_id}.svg"
    if not svg_path.exists():
        pytest.skip(f"{svg_path} not available")

    triplet = load_training_svg_triplet(svg_path)
    reconstructor = OrthographicTripletReconstructor.from_pipeline_config(
        PipelineConfig.from_yaml("config/total_view_data.yaml")
    )
    comparator = StepComparator()

    scores: dict[str, float] = {}
    for candidate in reconstructor.generate_candidate_programs(triplet):
        output_path = Path("/tmp") / f"{case_id}__{candidate.name}.step"
        execution = execute_build123d(
            candidate.result.code,
            output_path=str(output_path),
            timeout=60,
        )
        assert execution.success, execution.stderr
        reprojection = comparator.compare_with_orthographic_triplet(output_path, triplet)
        scores[candidate.name] = reprojection["reprojection_score"]
    return scores


def test_parse_training_svg_triplet_maps_standard_layout():
    triplet = parse_training_svg_triplet(_mock_training_svg(), case_id="mock_case")

    assert set(triplet.views) == {"f", "r", "t"}
    assert triplet.views["f"].view_box[0] < triplet.views["r"].view_box[0]
    assert triplet.views["t"].view_box[1] < triplet.views["f"].view_box[1]
    assert any(polyline.stroke == "red" for polyline in triplet.views["t"].polylines)

    reconstructor = OrthographicTripletReconstructor()
    candidate_names = [candidate.name for candidate in reconstructor.generate_candidate_programs(triplet)]
    assert any(name.startswith("profile_extrude_f") for name in candidate_names)


def test_profile_candidate_improves_selected_hollow_training_case():
    scores = _candidate_reprojection_scores("119452_114cbeb4_0000")

    profile_scores = {
        name: score for name, score in scores.items() if name.startswith("profile_extrude_f")
    }
    assert profile_scores
    assert "visual_hull_hidden" in scores
    assert max(profile_scores.values()) > scores["visual_hull_hidden"]


def test_candidate_search_does_not_degrade_triangle_plate_case():
    scores = _candidate_reprojection_scores("26764_ab35159a_0002")

    assert "visual_hull_hidden" in scores
    assert max(scores.values()) >= scores["visual_hull_hidden"]


def test_hybrid_profile_source_beats_raster_on_triangle_plate_case():
    scores = _candidate_reprojection_scores("26764_ab35159a_0002")

    assert scores["profile_extrude_t_hybrid"] > scores["profile_extrude_t_raster"]
