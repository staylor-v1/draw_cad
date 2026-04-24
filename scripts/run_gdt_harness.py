"""Run GD&T/callout detection across drawing images and write review artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError

from src.segmentation.gdt import detect_gdt_callouts
from src.segmentation.title_block import NormalizedBox, iter_default_title_block_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GD&T callout review artifacts.")
    parser.add_argument("--output-dir", default="experiments/gdt_harness")
    parser.add_argument("--image", action="append", help="Specific image path. May be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    crop_dir = output_dir / "crops"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    images = [Path(path) for path in args.image] if args.image else list(iter_default_title_block_images("."))
    if args.limit:
        images = images[: args.limit]

    records = []
    for image_path in images:
        try:
            records.append(process_image(image_path, overlay_dir, crop_dir))
        except (UnidentifiedImageError, OSError) as exc:
            records.append({"filename": image_path.name, "path": str(image_path), "skipped": True, "reason": str(exc)})

    summary = {
        "total": len(records),
        "skipped": sum(1 for record in records if record.get("skipped")),
        "callouts": sum(len(record.get("callouts", [])) for record in records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_index(output_dir, records)
    print(f"Wrote {len(records)} GD&T review records to {output_dir}")


def process_image(image_path: Path, overlay_dir: Path, crop_dir: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    callouts = detect_gdt_callouts(image_path)
    key = _safe_key(image_path)
    overlay_path = overlay_dir / f"{key}.png"
    draw_overlay(image, callouts).save(overlay_path)

    crop_paths = []
    for callout in callouts:
        left, top, right, bottom = NormalizedBox(**callout["crop"]).to_pixels(*image.size)
        crop_path = crop_dir / f"{key}_{callout['id']}.png"
        image.crop((left, top, right, bottom)).save(crop_path)
        crop_paths.append(str(crop_path))

    return {
        "filename": image_path.name,
        "path": str(image_path),
        "overlay": str(overlay_path),
        "cropImages": crop_paths,
        "callouts": callouts,
    }


def draw_overlay(image: Image.Image, callouts: list[dict]) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview, "RGBA")
    for callout in callouts:
        left, top, right, bottom = NormalizedBox(**callout["crop"]).to_pixels(*image.size)
        color = (199, 70, 45, 230) if callout["kind"] == "feature_control_frame" else (24, 77, 115, 210)
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        draw.text((left + 3, max(0, top - 14)), callout["id"], fill=color)
    preview.thumbnail((1100, 780))
    return preview


def write_index(output_dir: Path, records: list[dict]) -> None:
    cards = []
    for record in records:
        if record.get("skipped"):
            cards.append(f"<article class='card'><h2>{html.escape(record['filename'])}</h2><p>skipped</p></article>")
            continue
        callout_rows = "".join(
            f"<li>{html.escape(item['id'])}: {html.escape(item['kind'])} conf={item['confidence']} crop={html.escape(str(item['crop']))}</li>"
            for item in record["callouts"]
        )
        overlay = Path(record["overlay"]).relative_to(output_dir).as_posix()
        cards.append(
            f"""
            <article class="card">
              <h2>{html.escape(record["filename"])}</h2>
              <p>{len(record["callouts"])} callouts</p>
              <img src="{html.escape(overlay)}" alt="GD&T overlay for {html.escape(record["filename"])}">
              <ul>{callout_rows}</ul>
            </article>
            """
        )
    (output_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GD&T Harness Review</title>
  <style>
    body {{ margin: 0; padding: 20px; font-family: Segoe UI, sans-serif; background: #f4f1ea; color: #1c1f23; }}
    h1 {{ margin-top: 0; font-family: Georgia, serif; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #2e343b; background: #fffdf8; padding: 12px; }}
    img {{ width: 100%; border: 1px solid #c7c0b3; background: white; }}
    li {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>GD&T Harness Review</h1>
  <p>Red boxes are feature-control-frame candidates. Blue boxes are boxed annotation candidates.</p>
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
