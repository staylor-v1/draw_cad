"""Integration-like round-trip tests for drawing2text/text2drawing loop."""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.tools.drawing_roundtrip import draw_cad, drawing2text


LOCAL_SVG_CASES = [
    Path("training_data/drawings_svg/132685_6ff62835_0000.svg"),
    Path("training_data/drawings_svg/41032_ed481084_0008.svg"),
    Path("training_data/drawings_svg/22756_fc3fdda5_0027.svg"),
    Path("training_data/drawings_svg/77666_ba753255_0001.svg"),
    Path("training_data/drawings_svg/137141_748596d0_0008.svg"),
]

ONLINE_DRAWING_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/1/10/Technical_drawing_hole_center.svg",
    "https://upload.wikimedia.org/wikipedia/commons/9/99/ISO_128_example.svg",
]


@pytest.mark.parametrize("svg_path", LOCAL_SVG_CASES)
def test_roundtrip_preserves_text_record_on_local_drawings(tmp_path: Path, svg_path: Path):
    assert svg_path.exists(), f"missing test drawing: {svg_path}"
    from src.training.svg_renderer import render_svg_to_png

    source_png = tmp_path / f"{svg_path.stem}.png"
    render_svg_to_png(svg_path, source_png, dpi=170)
    _run_roundtrip_assertions(source_png, tmp_path / f"{svg_path.stem}_roundtrip.png")


def test_roundtrip_preserves_text_record_on_online_drawings(tmp_path: Path):
    downloaded: list[Path] = []
    for idx, url in enumerate(ONLINE_DRAWING_URLS, start=1):
        output = tmp_path / f"online_{idx}.svg"
        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                output.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError):
            continue
        downloaded.append(output)

    if not downloaded:
        pytest.skip("unable to download online engineering drawings in current environment")

    # Convert SVG sources to PNG via the same rendering path used by training data.
    from src.training.svg_renderer import render_svg_to_png

    for svg_path in downloaded:
        png_path = tmp_path / f"{svg_path.stem}.png"
        render_svg_to_png(svg_path, png_path, dpi=180)
        _run_roundtrip_assertions(png_path, tmp_path / f"{svg_path.stem}_roundtrip.png")


def _run_roundtrip_assertions(source_image: Path, reconstructed_path: Path) -> None:
    text_first = drawing2text(source_image)
    draw_cad(text_first, reconstructed_path)
    text_second = drawing2text(reconstructed_path)

    assert text_first["curve_signature"] == text_second["curve_signature"]
    assert text_first["dimensions"] == text_second["dimensions"]
    assert text_first["metadata"] == text_second["metadata"]
