"""Run raster-to-DXF vectorization across drawing images."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import UnidentifiedImageError

from src.segmentation.title_block import iter_default_title_block_images
from src.vectorization.raster_to_dxf import draw_vector_overlay, raster_to_vector, write_dxf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vector/DXF review artifacts for drawing segmentation.")
    parser.add_argument("--output-dir", default="experiments/vector_harness")
    parser.add_argument("--image", action="append", help="Specific image path. May be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    dxf_dir = output_dir / "dxf"
    json_dir = output_dir / "json"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dxf_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    images = [Path(path) for path in args.image] if args.image else list(iter_default_title_block_images("."))
    if args.limit:
        images = images[: args.limit]

    records = []
    for image_path in images:
        try:
            records.append(process_image(image_path, output_dir, overlay_dir, dxf_dir, json_dir))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            records.append({"filename": image_path.name, "path": str(image_path), "skipped": True, "reason": str(exc)})

    summary = {
        "total": len(records),
        "skipped": sum(1 for record in records if record.get("skipped")),
        "segments": sum(record.get("counts", {}).get("segments", 0) for record in records),
        "rectangles": sum(record.get("counts", {}).get("rectangles", 0) for record in records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_index(output_dir, records)
    print(f"Wrote {len(records)} vector review records to {output_dir}")


def process_image(image_path: Path, output_dir: Path, overlay_dir: Path, dxf_dir: Path, json_dir: Path) -> dict:
    result = raster_to_vector(image_path)
    key = _safe_key(image_path)
    overlay_path = overlay_dir / f"{key}.png"
    dxf_path = dxf_dir / f"{key}.dxf"
    json_path = json_dir / f"{key}.json"

    write_dxf(result, dxf_path)
    draw_vector_overlay(image_path, result).save(overlay_path)
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    counts = result.to_dict()["counts"]
    return {
        "filename": image_path.name,
        "path": str(image_path),
        "size": {"width": result.width, "height": result.height},
        "overlay": str(overlay_path.relative_to(output_dir)),
        "dxf": str(dxf_path.relative_to(output_dir)),
        "json": str(json_path.relative_to(output_dir)),
        "counts": counts,
        "topRectangles": [rectangle.to_dict() for rectangle in sorted(result.rectangles, key=lambda item: item.confidence, reverse=True)[:12]],
        "feedbackTemplate": (
            f"filename={image_path.name} | area=vector-harness | proposed=DXF line extraction | "
            f"issue=<issue> | segments={counts['segments']} rectangles={counts['rectangles']}"
        ),
    }


def write_index(output_dir: Path, records: list[dict]) -> None:
    cards = []
    for record in records:
        if record.get("skipped"):
            cards.append(
                f"<article class='card'><h2>{html.escape(record['filename'])}</h2><p>skipped: {html.escape(record['reason'])}</p></article>"
            )
            continue
        rectangle_rows = "".join(
            f"<li>{html.escape(item['kind'])} conf={item['confidence']} rows={item['line_rows']} cols={item['line_cols']} crop={html.escape(str(item['crop']))}</li>"
            for item in record["topRectangles"]
        )
        counts = record["counts"]
        cards.append(
            f"""
            <article class="card">
              <h2>{html.escape(record["filename"])}</h2>
              <p>{counts["segments"]} segments: {counts["horizontal"]} horizontal, {counts["vertical"]} vertical, {counts["diagonal"]} diagonal. {counts["rectangles"]} rectangles.</p>
              <p><a href="{html.escape(record["dxf"])}">DXF</a> · <a href="{html.escape(record["json"])}">JSON</a></p>
              <img src="{html.escape(record["overlay"])}" alt="vector overlay for {html.escape(record["filename"])}">
              <ul>{rectangle_rows}</ul>
              <textarea>{html.escape(record["feedbackTemplate"])}</textarea>
            </article>
            """
        )
    (output_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Vector Harness Review</title>
  <style>
    body {{ margin: 0; padding: 20px; font-family: Segoe UI, sans-serif; background: #f4f1ea; color: #1c1f23; }}
    h1 {{ margin-top: 0; font-family: Georgia, serif; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(390px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #2e343b; background: #fffdf8; padding: 12px; display: grid; gap: 8px; }}
    img {{ width: 100%; border: 1px solid #c7c0b3; background: white; }}
    textarea {{ min-height: 64px; border: 1px solid #2e343b; padding: 8px; }}
    li {{ margin-bottom: 5px; }}
  </style>
</head>
<body>
  <h1>Vector Harness Review</h1>
  <p>Green lines are horizontal DXF segments, blue lines are vertical, red lines are diagonals, and amber boxes are rectangular frame candidates.</p>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>
""",
        encoding="utf-8",
    )


def _safe_key(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.with_suffix("").as_posix()).strip("_")


if __name__ == "__main__":
    main()
