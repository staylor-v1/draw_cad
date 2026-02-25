"""SVG-to-PNG rasterisation using cairosvg."""
from __future__ import annotations

import concurrent.futures
from pathlib import Path

from src.training.data_loader import TrainingDataIndex
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def render_svg_to_png(
    svg_path: str | Path,
    output_path: str | Path,
    dpi: int = 150,
    background_color: str = "white",
) -> Path:
    """Rasterise a single SVG file to PNG.

    FreeCAD TechDraw SVGs typically use ``viewBox="0 0 5000 3000"`` with
    ``width=297mm``; at 150 DPI this produces ~1754x1054 px images which
    fit within llama-3.2-vision's 2048 px limit.

    Returns:
        Path to the written PNG file.
    """
    import cairosvg  # lazy import so the module can be imported without the dep

    svg_path = Path(svg_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(output_path),
        dpi=dpi,
        background_color=background_color,
    )
    logger.debug("svg_rendered", svg=str(svg_path), png=str(output_path))
    return output_path


def _render_one(args: tuple[Path, Path, int, str]) -> Path | None:
    """Worker function for parallel rendering."""
    svg_path, output_path, dpi, bg = args
    try:
        return render_svg_to_png(svg_path, output_path, dpi=dpi, background_color=bg)
    except Exception as e:
        logger.error("svg_render_error", svg=str(svg_path), error=str(e))
        return None


def render_all_svgs(
    index: TrainingDataIndex,
    output_dir: str | Path,
    dpi: int = 150,
    background_color: str = "white",
    workers: int = 4,
) -> int:
    """Batch-render all SVGs in the index to PNG.

    Returns:
        Number of successfully rendered PNGs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, Path, int, str]] = []
    for pair in index.pairs:
        png_path = output_dir / f"{pair.pair_id}.png"
        if png_path.exists():
            pair.png_path = png_path
            continue
        tasks.append((pair.svg_path, png_path, dpi, background_color))

    already_done = index.size - len(tasks)
    rendered = 0

    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_render_one, tasks)
            for result in results:
                if result is not None:
                    rendered += 1

    # Update png_path on all pairs
    for pair in index.pairs:
        png_path = output_dir / f"{pair.pair_id}.png"
        if png_path.exists():
            pair.png_path = png_path

    total = already_done + rendered
    logger.info(
        "batch_render_complete",
        total=total,
        newly_rendered=rendered,
        skipped=already_done,
        failed=len(tasks) - rendered,
    )
    return total
