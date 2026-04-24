"""Run the title-block segmentation harness over the drawing test corpus."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.segmentation.title_block import (
    NormalizedBox,
    crop_image,
    detect_title_block,
    estimate_border,
    iter_default_title_block_images,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate title-block segmentation review artifacts.")
    parser.add_argument("--output-dir", default="experiments/titleblock_harness")
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
            record = process_image(image_path, overlay_dir, crop_dir)
        except (UnidentifiedImageError, OSError) as exc:
            record = {
                "filename": image_path.name,
                "path": str(image_path),
                "skipped": True,
                "reason": f"could not open image: {exc}",
                "feedbackTemplate": f"filename={image_path.name} | area=title-block-harness | issue=<issue>",
            }
        records.append(record)

    summary = {
        "records": records,
        "total": len(records),
        "skipped": sum(1 for record in records if record.get("skipped")),
        "low_confidence": sum(
            1 for record in records if not record.get("skipped") and record["titleBlock"]["confidence"] < 0.45
        ),
        "outputDir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "teacher_review.jsonl").write_text(
        "\n".join(json.dumps(_teacher_item(record)) for record in records) + "\n",
        encoding="utf-8",
    )
    write_index(output_dir, records)
    print(f"Wrote {len(records)} title-block review records to {output_dir}")


def process_image(image_path: Path, overlay_dir: Path, crop_dir: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    candidate = detect_title_block(image)
    border = estimate_border(image)
    key = _safe_key(image_path)
    overlay_path = overlay_dir / f"{key}.png"
    crop_path = crop_dir / f"{key}.png"

    draw_overlay(image, candidate.crop, [mask for mask in border.masks]).save(overlay_path)
    crop_image(image_path, candidate.crop).save(crop_path)

    return {
        "filename": image_path.name,
        "path": str(image_path),
        "size": {"width": image.width, "height": image.height},
        "overlay": str(overlay_path),
        "cropImage": str(crop_path),
        "titleBlock": candidate.to_dict(),
        "border": border.to_dict(),
        "feedbackTemplate": (
            f"filename={image_path.name} | area=title-block-mask | "
            f"proposed=title block isolated | issue=<issue> | "
            f"crop={_crop_text(candidate.crop)}"
        ),
    }


def draw_overlay(image: Image.Image, title_box: NormalizedBox, border_boxes: list[NormalizedBox]) -> Image.Image:
    preview = ImageOps.contain(image, (1100, 780)).convert("RGB")
    scale_x = preview.width / image.width
    scale_y = preview.height / image.height
    draw = ImageDraw.Draw(preview, "RGBA")

    for box in border_boxes:
        left, top, right, bottom = box.to_pixels(image.width, image.height)
        rect = [left * scale_x, top * scale_y, right * scale_x, bottom * scale_y]
        draw.rectangle(rect, fill=(24, 77, 115, 42), outline=(24, 77, 115, 210), width=2)

    left, top, right, bottom = title_box.to_pixels(image.width, image.height)
    rect = [left * scale_x, top * scale_y, right * scale_x, bottom * scale_y]
    draw.rectangle(rect, fill=(199, 70, 45, 36), outline=(199, 70, 45, 245), width=4)
    draw.text((rect[0] + 6, max(2, rect[1] - 18)), "title block candidate", fill=(199, 70, 45))
    return preview


def write_index(output_dir: Path, records: list[dict]) -> None:
    cards = []
    for record in records:
        if record.get("skipped"):
            cards.append(
                f"""
                <article class="card skipped">
                  <h2>{html.escape(record["filename"])}</h2>
                  <p><strong>skipped</strong> {html.escape(record["reason"])}</p>
                  <textarea>{html.escape(record["feedbackTemplate"])}</textarea>
                </article>
                """
            )
            continue
        candidate = record["titleBlock"]
        status = "needs review" if candidate["confidence"] < 0.45 else "candidate"
        cards.append(
            f"""
            <article class="card">
              <h2>{html.escape(record["filename"])}</h2>
              <p><strong>{status}</strong> confidence={candidate["confidence"]} rows={candidate["line_rows"]} cols={candidate["line_cols"]}</p>
              <img src="{html.escape(Path(record["overlay"]).relative_to(output_dir).as_posix())}" alt="overlay for {html.escape(record["filename"])}">
              <img src="{html.escape(Path(record["cropImage"]).relative_to(output_dir).as_posix())}" alt="title block crop for {html.escape(record["filename"])}">
              <textarea>{html.escape(record["feedbackTemplate"])}</textarea>
            </article>
            """
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Title Block Harness Review</title>
  <style>
    body {{ margin: 0; padding: 20px; font-family: Segoe UI, sans-serif; background: #f4f1ea; color: #1c1f23; }}
    h1 {{ margin-top: 0; font-family: Georgia, serif; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #2e343b; background: #fffdf8; padding: 12px; display: grid; gap: 10px; }}
    .card img {{ width: 100%; border: 1px solid #c7c0b3; background: white; }}
    textarea {{ min-height: 70px; border: 1px solid #2e343b; padding: 8px; }}
  </style>
</head>
<body>
  <h1>Title Block Harness Review</h1>
  <p>Red is the title-block candidate. Blue is the border mask.</p>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")


def _teacher_item(record: dict) -> dict:
    if record.get("skipped"):
        return {
            "filename": record["filename"],
            "image": record["path"],
            "skipped": True,
            "reason": record["reason"],
            "teacherInstruction": "Confirm whether this file should be removed from the title-block harness corpus.",
            "feedbackTemplate": record["feedbackTemplate"],
        }
    return {
        "filename": record["filename"],
        "image": record["path"],
        "candidateCrop": record["titleBlock"]["crop"],
        "confidence": record["titleBlock"]["confidence"],
        "teacherInstruction": (
            "Review whether the red rectangle isolates the complete title block. "
            "If incorrect, provide a corrected crop and a short issue description."
        ),
        "feedbackTemplate": record["feedbackTemplate"],
    }


def _safe_key(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.with_suffix("").as_posix()).strip("_")


def _crop_text(box: NormalizedBox) -> str:
    clipped = box.clipped()
    return ",".join(str(round(value * 100)) for value in (clipped.x, clipped.y, clipped.w, clipped.h))


if __name__ == "__main__":
    main()
