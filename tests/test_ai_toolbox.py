"""Tests for the AI toolbox manifest and wrapper interface."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import yaml

from src.ai_toolbox import (
    get_toolbox_manifest,
    invoke_tool,
    list_tool_names,
    write_toolbox_manifest,
)


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


def test_toolbox_manifest_includes_representative_symbols():
    manifest = get_toolbox_manifest()
    inventory = {entry["qualname"] for entry in manifest["inventory"]}
    stable_tools = {entry["registry_name"] for entry in manifest["stable_tools"]}

    assert "src.reconstruction.reprojection.evaluate_step_against_triplet" in inventory
    assert "src.reconstruction.orthographic_solver.fit_circle" in inventory
    assert "scripts.benchmark_total_view_data.main" in inventory
    assert "reconstruct_case_with_candidate_search" in stable_tools
    assert "write_toolbox_manifest" in stable_tools


def test_invoke_tool_lists_total_view_cases(tmp_path: Path):
    zip_path = tmp_path / "mock_total_view.zip"
    _write_svg_zip(zip_path)

    result = invoke_tool(
        "list_total_view_cases",
        svg_zip_path=zip_path,
        required_views=("f", "r", "t"),
        require_complete=True,
    )

    assert result == ["part001"]


def test_write_toolbox_manifest_writes_yaml(tmp_path: Path):
    output_path = tmp_path / "ai_toolbox_manifest.yaml"

    written_path = write_toolbox_manifest(output_path)
    manifest = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert written_path == output_path
    assert written_path.exists()
    assert "stable_tools" in manifest
    assert "inventory" in manifest
    assert "list_tool_names" in list_tool_names()
