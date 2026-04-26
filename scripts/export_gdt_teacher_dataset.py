"""Export GD&T/callout candidate labels for teacher review or small-model tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.segmentation.callouts import load_callout_fixture, load_non_callout_fixture
from src.segmentation.gdt import detect_gdt_callouts
from src.segmentation.title_block import NormalizedBox, iter_default_title_block_images


POSITIVE_KINDS = {
    "feature_control_frame",
    "stacked_gdt_and_dimension",
    "datum_feature",
    "section_marker",
    "linear_dimension",
    "diameter_dimension",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled GD&T/callout candidate crops.")
    parser.add_argument("--output-dir", default="experiments/gdt_teacher_dataset")
    parser.add_argument("--image", action="append", help="Specific image path. May be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    images = [Path(path) for path in args.image] if args.image else list(iter_default_title_block_images("."))
    if args.limit:
        images = images[: args.limit]

    records = []
    for image_path in images:
        try:
            records.extend(process_image(image_path, crop_dir))
        except (OSError, UnidentifiedImageError):
            continue

    jsonl_path = output_dir / "candidates.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    summary = {
        "records": len(records),
        "labels": _counts(record["label"] for record in records),
        "output": str(jsonl_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def process_image(image_path: Path, crop_dir: Path) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    candidates = detect_gdt_callouts(image_path)
    positives = load_callout_fixture(image_path)
    negatives = load_non_callout_fixture(image_path)

    records = []
    key = _safe_key(image_path)
    for positive in positives:
        label = label_for_teacher_positive(positive)
        records.append(
            write_record(
                image=image,
                image_path=image_path,
                crop_dir=crop_dir,
                key=key,
                record_id=f"teacher-{positive['id']}",
                crop=positive["crop"],
                label=label,
                source="teacher_fixture",
                detector_kind="teacher_fixture",
                teacher=positive,
            )
        )
    for negative in negatives:
        records.append(
            write_record(
                image=image,
                image_path=image_path,
                crop_dir=crop_dir,
                key=key,
                record_id=f"negative-{negative['id']}",
                crop=negative["crop"],
                label="part_geometry_thread_texture"
                if negative.get("kind") == "thread_texture"
                else "part_geometry_other",
                source="teacher_negative_fixture",
                detector_kind=negative.get("kind", "teacher_negative"),
                teacher=negative,
            )
        )
    for candidate in candidates:
        label, teacher = classify_candidate(candidate, positives, negatives)
        records.append(
            write_record(
                image=image,
                image_path=image_path,
                crop_dir=crop_dir,
                key=key,
                record_id=candidate["id"],
                crop=candidate["crop"],
                label=label,
                source="detector_candidate",
                detector_kind=candidate["kind"],
                teacher=teacher,
            )
        )
    return records


def write_record(
    *,
    image: Image.Image,
    image_path: Path,
    crop_dir: Path,
    key: str,
    record_id: str,
    crop: dict,
    label: str,
    source: str,
    detector_kind: str,
    teacher: dict | None,
) -> dict:
    safe_id = _safe_key(Path(record_id))
    crop_path = crop_dir / f"{key}_{safe_id}_{label}.png"
    left, top, right, bottom = NormalizedBox(**crop).to_pixels(*image.size)
    image.crop((left, top, right, bottom)).save(crop_path)
    return {
        "image": str(image_path),
        "candidate_id": record_id,
        "source": source,
        "crop": crop,
        "crop_image": str(crop_path),
        "detector_kind": detector_kind,
        "label": label,
        "teacher_id": teacher.get("id") if teacher else None,
        "teacher_kind": teacher.get("kind") if teacher else None,
        "teacher_text": teacher.get("text") or teacher.get("value") if teacher else "",
        "prompt_label_space": [
            "real_gdt_frame",
            "real_dimension_or_datum",
            "part_geometry_thread_texture",
            "part_geometry_other",
            "uncertain",
        ],
    }


def label_for_teacher_positive(positive: dict) -> str:
    if positive["kind"] in {"feature_control_frame", "stacked_gdt_and_dimension"}:
        return "real_gdt_frame"
    return "real_dimension_or_datum"


def classify_candidate(candidate: dict, positives: list[dict], negatives: list[dict]) -> tuple[str, dict | None]:
    candidate_box = NormalizedBox(**candidate["crop"])
    for negative in negatives:
        if _overlap_fraction(candidate_box, NormalizedBox(**negative["crop"])) >= 0.25:
            return "part_geometry_thread_texture", negative
    best_positive = None
    best_overlap = 0.0
    for positive in positives:
        overlap = _overlap_fraction(candidate_box, NormalizedBox(**positive["crop"]))
        if overlap > best_overlap:
            best_positive = positive
            best_overlap = overlap
    if best_positive and best_overlap >= 0.18:
        if best_positive["kind"] == "feature_control_frame" or candidate["kind"] == "feature_control_frame":
            return "real_gdt_frame", best_positive
        if best_positive["kind"] in POSITIVE_KINDS:
            return "real_dimension_or_datum", best_positive
    return "uncertain", None


def _overlap_fraction(a: NormalizedBox, b: NormalizedBox) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return (ix * iy) / max(min(a.w * a.h, b.w * b.h), 1e-6)


def _counts(labels) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def _safe_key(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.with_suffix("").as_posix()).strip("_")


if __name__ == "__main__":
    main()
