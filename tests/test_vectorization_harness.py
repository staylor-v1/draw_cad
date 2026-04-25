from pathlib import Path

from src.segmentation.gdt import detect_gdt_callouts
from src.vectorization.raster_to_dxf import raster_to_vector, write_dxf


def test_simple1_vectorization_emits_line_evidence_and_dxf(tmp_path):
    result = raster_to_vector(Path("training_data/gdt/simple1.webp"))
    counts = result.to_dict()["counts"]

    assert counts["segments"] >= 40
    assert counts["horizontal"] >= 15
    assert counts["vertical"] >= 10
    assert counts["rectangles"] >= 3

    dxf_path = tmp_path / "simple1.dxf"
    write_dxf(result, dxf_path)
    assert dxf_path.exists()
    assert dxf_path.stat().st_size > 1000


def test_flange2_gdt_detector_includes_vector_frame_evidence():
    callouts = detect_gdt_callouts(Path("training_data/gdt/flange2.png"))

    assert len(callouts) >= 4
    assert any("vectorized DXF line evidence" in " ".join(item["notes"]) for item in callouts)
    assert any(item["kind"] == "feature_control_frame" for item in callouts)
