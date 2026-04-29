"""Split external GD&T assets by view type.

The external corpus intentionally contains two different kinds of useful data:

* 2D orthographic/detail drawings that are candidates for drawing-to-CAD work.
* 3D/model-based PMI views that are useful for GD&T vocabulary and callout
  grounding, but should not be mixed into 2D orthographic reconstruction sets.

This script writes explicit split lists and contact sheets under
`training_data/gdt_external/splits`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "training_data/gdt_external"
SPLIT_DIR = DATASET_DIR / "splits"

NIST_2D_ORTHOGRAPHIC = {
    "training_data/gdt_external/rendered/nist_pmi/nist_pmi_ftc_06-3.png",
    "training_data/gdt_external/rendered/nist_pmi/nist_pmi_ftc_10-4.png",
    "training_data/gdt_external/rendered/nist_pmi/nist_pmi_ftc_10-5.png",
    "training_data/gdt_external/rendered/nist_pmi/nist_pmi_ftc_11-2.png",
    "training_data/gdt_external/rendered/nist_am/nist_am_test_artifact_engineering_drawing-1.png",
    "training_data/gdt_external/rendered/nist_am/nist_am_test_artifact_engineering_drawing-2.png",
}


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for asset in manifest["assets"]:
        for rendered_path in asset.get("rendered_paths", []):
            if not Path(ROOT / rendered_path).exists():
                continue
            split, usefulness, reason = classify(asset, rendered_path)
            records.append(
                {
                    "path": rendered_path,
                    "assetId": asset["id"],
                    "title": asset["title"],
                    "group": asset["group"],
                    "sourceRole": asset["role"],
                    "split": split,
                    "usefulness": usefulness,
                    "reason": reason,
                    "sourceUrl": asset["source_url"],
                    "license": asset["license"],
                    "licenseUrl": asset["license_url"],
                }
            )

    records = sorted(records, key=lambda item: (item["split"], item["path"]))
    by_split: dict[str, list[dict]] = {}
    for record in records:
        by_split.setdefault(record["split"], []).append(record)

    for split, items in by_split.items():
        (SPLIT_DIR / f"{split}.txt").write_text(
            "\n".join(item["path"] for item in items) + "\n",
            encoding="utf-8",
        )
        write_contact_sheet(split, items)

    split_manifest = {
        "sourceManifest": "training_data/gdt_external/manifest.json",
        "policy": {
            "2d_orthographic_target": "Primary candidates for drawing-to-CAD training and evaluation.",
            "3d_pmi_reference": "Model-based/isometric PMI views; useful for GD&T/callout recognition but excluded from 2D reconstruction training.",
            "2d_reference": "2D diagrams, symbol sheets, and small drawing references; useful for symbol/callout augmentation.",
        },
        "counts": dict(Counter(item["split"] for item in records)),
        "records": records,
    }
    (DATASET_DIR / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    (SPLIT_DIR / "README.md").write_text(render_readme(split_manifest), encoding="utf-8")
    print(json.dumps(split_manifest["counts"], indent=2))


def classify(asset: dict, rendered_path: str) -> tuple[str, str, str]:
    group = asset["group"]
    role = asset["role"]
    if group.startswith("nist"):
        if rendered_path in NIST_2D_ORTHOGRAPHIC:
            return (
                "2d_orthographic_target",
                "primary",
                "Reviewed as a 2D orthographic/detail/section drawing suitable for drawing-to-CAD work.",
            )
        return (
            "3d_pmi_reference",
            "exclude_from_2d_reconstruction_training",
            "Reviewed as a 3D/model-based PMI or isometric view; keep for GD&T recognition only.",
        )
    if role == "drawing_reference":
        return (
            "2d_reference",
            "secondary",
            "Wikimedia 2D drawing/reference diagram; useful for callout vocabulary but not a full target sheet.",
        )
    return (
        "2d_reference",
        "symbol_or_callout_reference",
        "Wikimedia symbol/callout/reference image; use for augmentation and symbol vocabulary.",
    )


def write_contact_sheet(split: str, records: list[dict]) -> None:
    if not records:
        return
    thumb_w, thumb_h = 300, 230
    label_h = 42
    cols = 4 if len(records) > 4 else max(1, len(records))
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = None
    for index, record in enumerate(records):
        path = ROOT / record["path"]
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 10, thumb_h - 10))
        cell_x = (index % cols) * thumb_w
        cell_y = (index // cols) * (thumb_h + label_h)
        x = cell_x + (thumb_w - image.width) // 2
        y = cell_y + 4
        sheet.paste(image, (x, y))
        label = Path(record["path"]).stem[:38]
        draw.text((cell_x + 6, cell_y + thumb_h + 4), label, fill="black", font=font)
        draw.text((cell_x + 6, cell_y + thumb_h + 22), record["usefulness"][:38], fill=(80, 80, 80), font=font)
    sheet.save(SPLIT_DIR / f"{split}_contact_sheet.jpg", quality=90)


def render_readme(split_manifest: dict) -> str:
    counts = split_manifest["counts"]
    return f"""# External GD&T View-Type Splits

Generated by `scripts/split_external_gdt_view_types.py`.

## Counts

- 2D orthographic target: {counts.get('2d_orthographic_target', 0)}
- 3D PMI reference: {counts.get('3d_pmi_reference', 0)}
- 2D reference/symbol material: {counts.get('2d_reference', 0)}

## Use

Use `2d_orthographic_target.txt` for drawing-to-CAD reconstruction experiments.
Use `3d_pmi_reference.txt` only for GD&T mark/callout recognition and grounding.
Use `2d_reference.txt` for symbol vocabulary and augmentation, not as full
orthographic drawing reconstruction targets.

The NIST PMI corpus is mostly 3D/model-based PMI. Only a small number of NIST
pages are labelled as 2D orthographic/detail targets after visual review.
"""


if __name__ == "__main__":
    main()
