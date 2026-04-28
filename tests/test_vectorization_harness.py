from pathlib import Path

from src.segmentation.gdt import detect_gdt_callouts
from src.segmentation.masks import _largest_angle_cluster
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


def test_public_thread_reference_images_emit_repeated_thread_line_evidence():
    examples = [
        Path("training_data/thread_examples/mcgill_conventional_thread_representation.jpg"),
        Path("training_data/thread_examples/mcgill_sectional_external_internal_threads.jpg"),
        Path("training_data/thread_examples/mcgill_rolled_thread_representation.jpg"),
    ]

    for image_path in examples:
        result = raster_to_vector(image_path)
        diagonal_segments = [segment for segment in result.segments if segment.orientation == "diagonal"]
        horizontal_segments = [segment for segment in result.segments if segment.orientation == "horizontal"]

        assert result.to_dict()["counts"]["segments"] >= 20
        assert _largest_angle_cluster(diagonal_segments) >= 3 or len(horizontal_segments) >= 10
