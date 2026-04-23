"""Tests for deterministic Total_view_data reconstruction helpers."""
from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from src.reconstruction import (
    OrthographicTripletReconstructor,
    TotalViewPngArchive,
    TotalViewArchive,
    evaluate_step_against_triplet,
)
from src.reconstruction.reprojection import compare_line_masks
from src.schemas.pipeline_config import ReprojectionConfig
from src.tools.cad import execute_build123d


def _write_svg_zip(zip_path: Path) -> None:
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("mock/", "")
        zf.writestr("mock/part001_f.svg", _rect_svg(10, 6))
        zf.writestr("mock/part001_r.svg", _rect_svg(4, 6))
        zf.writestr("mock/part001_t.svg", _rect_svg(10, 4))
        zf.writestr("mock/part002_f.svg", _rect_svg(8, 5))


def _rect_svg(width: int, height: int) -> str:
    return f"""<?xml version="1.0" encoding="utf-8" ?>
<svg version="1.1" viewBox="0,0,{width},{height}" xmlns="http://www.w3.org/2000/svg">
  <polyline fill="none" points="0,0 {width},0 {width},{height} 0,{height} 0,0" stroke="black" stroke-linecap="round" stroke-width="0.35" />
</svg>
"""


def _write_png_zip(zip_path: Path) -> None:
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("mock/", "")
        zf.writestr("mock/part001_f.png", _solid_png_bytes((255, 255, 255)))
        zf.writestr("mock/part001_r.png", _solid_png_bytes((200, 200, 200)))
        zf.writestr("mock/part001_t.png", _solid_png_bytes((150, 150, 150)))
        zf.writestr("mock/part002_f.png", _solid_png_bytes((100, 100, 100)))


def _solid_png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_hidden_feature_zip(zip_path: Path) -> None:
    circle_points = " ".join(
        f"{5 + math.cos(theta):.4f},{2 + math.sin(theta):.4f}"
        for theta in [index * 2 * math.pi / 36 for index in range(37)]
    )
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("mock/", "")
        zf.writestr(
            "mock/holecase_f.svg",
            """<?xml version="1.0" encoding="utf-8" ?>
<svg version="1.1" viewBox="0,0,10,6" xmlns="http://www.w3.org/2000/svg">
  <polyline fill="none" points="0,0 10,0 10,6 0,6 0,0" stroke="black" stroke-linecap="round" stroke-width="0.35" />
  <polyline fill="none" points="4,5 6,5" stroke="red" stroke-linecap="round" stroke-width="0.35" />
</svg>
""",
        )
        zf.writestr(
            "mock/holecase_r.svg",
            """<?xml version="1.0" encoding="utf-8" ?>
<svg version="1.1" viewBox="0,0,4,6" xmlns="http://www.w3.org/2000/svg">
  <polyline fill="none" points="0,0 4,0 4,6 0,6 0,0" stroke="black" stroke-linecap="round" stroke-width="0.35" />
  <polyline fill="none" points="1,5 3,5" stroke="red" stroke-linecap="round" stroke-width="0.35" />
</svg>
""",
        )
        zf.writestr(
            "mock/holecase_t.svg",
            f"""<?xml version="1.0" encoding="utf-8" ?>
<svg version="1.1" viewBox="0,0,10,4" xmlns="http://www.w3.org/2000/svg">
  <polyline fill="none" points="0,0 10,0 10,4 0,4 0,0" stroke="black" stroke-linecap="round" stroke-width="0.35" />
  <polyline fill="none" points="{circle_points}" stroke="red" stroke-linecap="round" stroke-width="0.35" />
</svg>
""",
        )


def test_total_view_archive_groups_complete_triplets(tmp_path: Path):
    zip_path = tmp_path / "mock_total_view.zip"
    _write_svg_zip(zip_path)

    archive = TotalViewArchive(zip_path)

    assert archive.case_ids() == ["part001"]
    assert archive.available_views("part001") == ["f", "r", "t"]
    assert set(archive.case_ids(require_complete=False)) == {"part001", "part002"}


def test_total_view_png_archive_groups_complete_triplets(tmp_path: Path):
    zip_path = tmp_path / "mock_total_view_png.zip"
    _write_png_zip(zip_path)

    archive = TotalViewPngArchive(zip_path)

    assert archive.case_ids() == ["part001"]
    assert archive.available_views("part001") == ["f", "r", "t"]
    triplet = archive.load_triplet("part001")
    assert set(triplet.views) == {"f", "r", "t"}
    assert triplet.views["f"].image_bytes.startswith(b"\x89PNG")


def test_reconstructor_generates_intersection_program(tmp_path: Path):
    zip_path = tmp_path / "mock_total_view.zip"
    _write_svg_zip(zip_path)

    archive = TotalViewArchive(zip_path)
    triplet = archive.load_triplet("part001")
    result = OrthographicTripletReconstructor().generate_program(triplet)

    assert result.consensus_extents["x"] == pytest.approx(10.0, rel=0.05)
    assert result.consensus_extents["y"] == pytest.approx(4.0, rel=0.05)
    assert result.consensus_extents["z"] == pytest.approx(6.0, rel=0.05)
    assert "Plane.XZ" in result.code
    assert "Plane.XY" in result.code
    assert "Plane.YZ" in result.code
    assert "part = (front & top) & right" in result.code


def test_reconstructor_infers_hidden_cylinder_cut(tmp_path: Path):
    zip_path = tmp_path / "mock_hidden_feature.zip"
    _write_hidden_feature_zip(zip_path)

    archive = TotalViewArchive(zip_path)
    triplet = archive.load_triplet("holecase")
    result = OrthographicTripletReconstructor().generate_program(triplet)

    assert len(result.hidden_cylinders) == 1
    cut = result.hidden_cylinders[0]
    assert cut.center_x == pytest.approx(5.0, rel=0.05)
    assert cut.center_y == pytest.approx(2.0, rel=0.05)
    assert cut.radius == pytest.approx(1.0, rel=0.1)
    assert cut.z_min == pytest.approx(5.0, rel=0.05)
    assert cut.z_max == pytest.approx(6.0, rel=0.05)
    assert "Cylinder(" in result.code
    assert "hidden_cylinders" in result.code


def test_compare_line_masks_empty_is_perfect():
    metrics = compare_line_masks(
        source_mask=np.zeros((2, 2), dtype=bool),
        predicted_mask=np.zeros((2, 2), dtype=bool),
        tolerance_px=1,
    )

    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)


def test_reprojection_scores_simple_box_step(tmp_path: Path):
    zip_path = tmp_path / "mock_total_view.zip"
    _write_svg_zip(zip_path)

    archive = TotalViewArchive(zip_path)
    triplet = archive.load_triplet("part001")
    program = OrthographicTripletReconstructor().generate_program(triplet)

    step_path = tmp_path / "part001.step"
    execution = execute_build123d(program.code, output_path=str(step_path), timeout=60)
    assert execution.success, execution.stderr

    score, _ = evaluate_step_against_triplet(
        step_path,
        triplet,
        config=ReprojectionConfig(overlay_enabled=False),
    )

    assert score.score >= 0.92
    assert score.views["f"].visible.f1 >= 0.95
    assert score.views["r"].visible.f1 >= 0.84
    assert score.views["t"].visible.f1 >= 0.95
