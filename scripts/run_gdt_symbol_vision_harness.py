"""Benchmark Florence-2 and YOLO variants for GD&T symbol/callout detection."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.callouts import load_callout_fixture, load_non_callout_fixture


SYMBOL_CLASSES = [
    "datum_feature",
    "boxed_basic_dimension",
    "linear_dimension",
    "diameter_dimension",
    "fcf_perpendicularity",
    "fcf_position",
    "fcf_flatness",
    "fcf_profile",
    "section_marker",
    "stacked_callout",
    "non_callout_thread_texture",
]
CLASS_TO_ID = {name: index for index, name in enumerate(SYMBOL_CLASSES)}
DEFAULT_REAL_IMAGES = [
    Path("training_data/gdt/simple1.webp"),
    Path("training_data/gdt/threaded_cap.jpg"),
]
FONT_PATH = Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")
FCF_SYMBOLS = {
    "fcf_perpendicularity": "⊥",
    "fcf_position": "⌖",
    "fcf_flatness": "▱",
    "fcf_profile": "⌒",
}


@dataclass(frozen=True)
class Label:
    class_name: str
    crop: dict[str, float]
    text: str = ""
    source: str = "synthetic_reference"


def main() -> None:
    parser = argparse.ArgumentParser(description="GD&T symbol/callout vision benchmark and fine-tune harness.")
    parser.add_argument("--output-dir", default="experiments/gdt_symbol_vision")
    parser.add_argument("--real-image", action="append", help="Teacher-labelled real drawing image to include.")
    parser.add_argument("--synthetic-count", type=int, default=120)
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--time-hours", type=float, default=0.0)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--florence", action="store_true")
    parser.add_argument("--seed", type=int, default=20260426)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_images = [Path(path) for path in args.real_image] if args.real_image else DEFAULT_REAL_IMAGES
    dataset_dir = output_dir / "dataset"
    manifest = export_symbol_dataset(
        real_images=real_images,
        dataset_dir=dataset_dir,
        synthetic_count=args.synthetic_count,
        seed=args.seed,
    )

    records = []
    if args.florence:
        records.append(run_florence_probe(manifest, output_dir))

    if args.train:
        for model_name in args.models:
            records.append(
                train_and_eval_yolo(
                    model_name=model_name,
                    dataset_yaml=dataset_dir / "dataset.yaml",
                    output_dir=output_dir,
                    manifest=manifest,
                    epochs=args.epochs,
                    time_hours=args.time_hours,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                )
            )
    else:
        for model_name in args.models:
            records.append(eval_pretrained_yolo(model_name, manifest, args.imgsz))

    summary = {
        "classes": SYMBOL_CLASSES,
        "realImages": [str(path) for path in real_images],
        "syntheticCount": args.synthetic_count,
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


def export_symbol_dataset(
    *,
    real_images: list[Path],
    dataset_dir: Path,
    synthetic_count: int,
    seed: int,
) -> dict:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    manifest = {"train": [], "val": [], "real": [], "synthetic": []}

    synthetic_dir = dataset_dir / "synthetic_reference"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    for index in range(synthetic_count):
        split = "val" if index % 5 == 0 else "train"
        image, labels = render_reference_sheet(rng, index)
        image_path = synthetic_dir / f"gdt_reference_{index:04d}.jpg"
        image.save(image_path, quality=94)
        record = write_dataset_item(image_path, labels, dataset_dir, split, f"reference_{index:04d}")
        manifest[split].append(record)
        manifest["synthetic"].append(record)

    for image_path in real_images:
        labels = real_labels_for_image(image_path)
        if not labels:
            continue
        for split in ("train", "val"):
            record = write_dataset_item(image_path, labels, dataset_dir, split, f"real_{image_path.stem}")
            manifest[split].append(record)
            if split == "val":
                manifest["real"].append(record)

    (dataset_dir / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(SYMBOL_CLASSES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_dataset_item(image_path: Path, labels: list[Label], dataset_dir: Path, split: str, stem: str) -> dict:
    target_image = dataset_dir / "images" / split / f"{stem}.jpg"
    Image.open(image_path).convert("RGB").save(target_image, quality=95)
    target_label = dataset_dir / "labels" / split / f"{stem}.txt"
    target_label.write_text("\n".join(yolo_label(label) for label in labels) + "\n", encoding="utf-8")
    return {
        "image": str(target_image),
        "labels": [
            {"className": label.class_name, "crop": label.crop, "text": label.text, "source": label.source}
            for label in labels
        ],
    }


def real_labels_for_image(image_path: Path) -> list[Label]:
    labels = []
    for item in load_callout_fixture(image_path):
        labels.append(
            Label(
                class_name=class_for_real_callout(item),
                crop=item["crop"],
                text=item.get("value") or item.get("label", ""),
                source="real_teacher_fixture",
            )
        )
    for item in load_non_callout_fixture(image_path):
        labels.append(
            Label(
                class_name="non_callout_thread_texture",
                crop=item["crop"],
                text=item.get("kind", ""),
                source="real_teacher_fixture",
            )
        )
    return labels


def class_for_real_callout(item: dict) -> str:
    kind = item.get("kind", "")
    text = (item.get("value") or item.get("label") or "").lower()
    if kind == "feature_control_frame":
        if "perpendicular" in text:
            return "fcf_perpendicularity"
        if "position" in text:
            return "fcf_position"
        if "flatness" in text:
            return "fcf_flatness"
        if "profile" in text:
            return "fcf_profile"
    if kind == "stacked_gdt_and_dimension":
        return "stacked_callout"
    if kind in CLASS_TO_ID:
        return kind
    return "linear_dimension"


def render_reference_sheet(rng: random.Random, index: int) -> tuple[Image.Image, list[Label]]:
    width, height = 1180, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    small = load_font(20)
    labels: list[Label] = []

    draw_part_geometry(draw, rng, width, height)
    grid = [(60 + col * 265, 70 + row * 180) for row in range(4) for col in range(4)]
    rng.shuffle(grid)
    classes = [
        "datum_feature",
        "boxed_basic_dimension",
        "linear_dimension",
        "diameter_dimension",
        "fcf_perpendicularity",
        "fcf_position",
        "fcf_flatness",
        "fcf_profile",
        "section_marker",
        "stacked_callout",
    ]
    for class_name, (x, y) in zip(classes, grid):
        jitter_x = rng.randint(-22, 22)
        jitter_y = rng.randint(-18, 18)
        box, text = draw_symbol_callout(draw, class_name, x + jitter_x, y + jitter_y, font, small, rng)
        labels.append(Label(class_name=class_name, crop=pixel_box_to_norm(box, width, height), text=text))

    thread_box = (rng.randint(660, 780), rng.randint(410, 520), rng.randint(945, 1035), rng.randint(590, 690))
    draw_thread_texture(draw, thread_box)
    labels.append(Label("non_callout_thread_texture", pixel_box_to_norm(thread_box, width, height), "thread texture"))

    if index % 3 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
    return image, labels


def draw_part_geometry(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    for _ in range(8):
        x = rng.randint(180, width - 260)
        y = rng.randint(170, height - 230)
        w = rng.randint(90, 230)
        h = rng.randint(70, 180)
        draw.rectangle((x, y, x + w, y + h), outline=(25, 25, 25), width=2)
        if rng.random() < 0.75:
            cx = x + w // 2
            cy = y + h // 2
            r = min(w, h) // rng.randint(4, 6)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(25, 25, 25), width=2)
        if rng.random() < 0.45:
            draw.line((x - 30, y + h + 35, x + w + 30, y + h + 35), fill=(25, 25, 25), width=1)


def draw_symbol_callout(
    draw: ImageDraw.ImageDraw,
    class_name: str,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], str]:
    if class_name == "datum_feature":
        text = rng.choice(["A", "B", "C"])
        box = (x, y, x + 48, y + 58)
        draw.rectangle(box, outline="black", width=2)
        draw.text((x + 15, y + 12), text, fill="black", font=font)
        draw.line((x + 24, y + 58, x + 24, y + 92), fill="black", width=2)
        return expand(box, 6), text
    if class_name == "boxed_basic_dimension":
        text = rng.choice(["1", "2", "3", "12.5"])
        box = (x, y, x + 74, y + 48)
        draw.rectangle(box, outline="black", width=2)
        draw.text((x + 18, y + 8), text, fill="black", font=font)
        return expand(box, 6), text
    if class_name in {"linear_dimension", "diameter_dimension"}:
        prefix = "⌀ " if class_name == "diameter_dimension" else ""
        text = prefix + rng.choice(["7.71 ±0.15", "40.54 ±0.25", "2.0 ±0.1", "13.45 ±0.10"])
        text_box = draw.textbbox((x, y), text, font=small)
        box = (x - 4, y - 4, text_box[2] + 6, text_box[3] + 6)
        draw.text((x, y), text, fill="black", font=small)
        draw.line((x - 30, y + 12, x - 8, y + 12), fill="black", width=1)
        draw.line((box[2] + 8, y + 12, box[2] + 50, y + 12), fill="black", width=1)
        return expand(box, 10), text
    if class_name.startswith("fcf_"):
        symbol = FCF_SYMBOLS[class_name]
        tolerance = rng.choice(["0.005", "0.10", "0.25", "⌀0.003 M"])
        datum = rng.choice(["A", "A|B", "A|B|C", ""])
        text = f"{symbol}|{tolerance}" + (f"|{datum}" if datum else "")
        cells = [symbol, tolerance, *([datum] if datum else [])]
        cell_widths = [44] + [max(70, len(cell) * 14) for cell in cells[1:]]
        h = 42
        total_w = sum(cell_widths)
        cursor = x
        for cell, cell_w in zip(cells, cell_widths):
            draw.rectangle((cursor, y, cursor + cell_w, y + h), outline="black", width=2)
            draw.text((cursor + 10, y + 7), cell, fill="black", font=small)
            cursor += cell_w
        return expand((x, y, x + total_w, y + h), 7), text
    if class_name == "section_marker":
        text = "A-A"
        draw.line((x + 45, y, x + 45, y + 130), fill="black", width=2)
        draw.polygon([(x + 45, y), (x + 34, y + 24), (x + 56, y + 24)], outline="black", fill=None)
        draw.polygon([(x + 45, y + 130), (x + 34, y + 106), (x + 56, y + 106)], outline="black", fill=None)
        draw.text((x, y + 48), text, fill="black", font=font)
        return expand((x, y, x + 92, y + 130), 8), text
    text = "⌀3.85±0.1\n⌖|0.50|A|B\n◎|0.50"
    box = (x, y, x + 218, y + 126)
    draw.rectangle(box, outline="black", width=2)
    yy = y + 7
    for line in text.splitlines():
        draw.text((x + 10, yy), line, fill="black", font=small)
        yy += 36
    return expand(box, 7), text


def draw_thread_texture(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="black", width=2)
    for offset in range(-80, x1 - x0 + 80, 16):
        draw.line((x0 + offset, y1, x0 + offset + 90, y0), fill="black", width=1)
    for yy in range(y0 + 12, y1, 18):
        draw.line((x0, yy, x1, yy), fill="black", width=1)


def load_font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def expand(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)


def pixel_box_to_norm(box: tuple[int, int, int, int], width: int, height: int) -> dict[str, float]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return {"x": x0 / width, "y": y0 / height, "w": (x1 - x0) / width, "h": (y1 - y0) / height}


def yolo_label(label: Label) -> str:
    crop = label.crop
    cx = crop["x"] + crop["w"] / 2
    cy = crop["y"] + crop["h"] / 2
    return f"{CLASS_TO_ID[label.class_name]} {cx:.6f} {cy:.6f} {crop['w']:.6f} {crop['h']:.6f}"


def train_and_eval_yolo(
    *,
    model_name: str,
    dataset_yaml: Path,
    output_dir: Path,
    manifest: dict,
    epochs: int,
    time_hours: float,
    imgsz: int,
    batch: int,
    device: str | int,
) -> dict:
    from ultralytics import YOLO

    run_name = f"{Path(model_name).stem}_symbol"
    start = time.monotonic()
    model = YOLO(model_name)
    kwargs = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(output_dir / "runs"),
        "name": run_name,
        "exist_ok": True,
        "patience": max(epochs, 100),
        "plots": False,
        "verbose": False,
        "workers": 0,
    }
    if time_hours > 0:
        kwargs["time"] = time_hours
    results = model.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best))
    metrics = trained.val(data=str(dataset_yaml), imgsz=imgsz, batch=batch, device=device, verbose=False, plots=False)
    coverage = score_model_against_manifest(trained, manifest, imgsz=imgsz, device=device)
    return {
        "kind": "fine_tuned_yolo_symbol",
        "model": model_name,
        "best_model": str(best),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "epochs": epochs,
        "imgsz": imgsz,
        "metrics": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision_mean": float(metrics.box.mp),
            "recall_mean": float(metrics.box.mr),
        },
        "coverage": coverage,
    }


def eval_pretrained_yolo(model_name: str, manifest: dict, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_name)
    start = time.monotonic()
    coverage = score_model_against_manifest(model, manifest, imgsz=imgsz, device=0, conf=0.01)
    return {
        "kind": "pretrained_probe",
        "model": model_name,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "coverage": coverage,
        "note": "COCO-pretrained detector has no GD&T classes; this is a negative-control probe.",
    }


def score_model_against_manifest(model, manifest: dict, *, imgsz: int, device: str | int, conf: float = 0.05) -> dict:
    scored = {}
    for group in ("real", "synthetic"):
        records = manifest.get(group, [])
        total = 0
        matched = 0
        correct_class = 0
        for record in records:
            result = model.predict(record["image"], imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            predictions = [
                {
                    "className": class_name_for_prediction(int(box.cls[0])),
                    "crop": xyxy_to_norm(box.xyxy[0].tolist(), result.orig_shape),
                }
                for box in result.boxes or []
            ]
            for label in record["labels"]:
                if label["className"] == "non_callout_thread_texture" and group == "real":
                    continue
                total += 1
                best = best_prediction_match(label["crop"], predictions)
                if best:
                    matched += 1
                    if best["className"] == label["className"]:
                        correct_class += 1
        scored[group] = {
            "labels": total,
            "matched": matched,
            "classCorrect": correct_class,
            "boxRecall": round(matched / total, 3) if total else 0.0,
            "classRecall": round(correct_class / total, 3) if total else 0.0,
        }
    return scored


def class_name_for_prediction(class_id: int) -> str:
    if 0 <= class_id < len(SYMBOL_CLASSES):
        return SYMBOL_CLASSES[class_id]
    return "external_pretrained_class"


def best_prediction_match(label_crop: dict, predictions: list[dict]) -> dict | None:
    best = None
    best_score = 0.0
    for prediction in predictions:
        pred_crop = prediction["crop"]
        score = max(box_iou(label_crop, pred_crop), center_inside_score(label_crop, pred_crop))
        if score > best_score:
            best_score = score
            best = prediction
    return best if best_score >= 0.30 else None


def xyxy_to_norm(xyxy: list[float], orig_shape: tuple[int, int]) -> dict[str, float]:
    height, width = orig_shape
    x0, y0, x1, y1 = xyxy
    return {"x": x0 / width, "y": y0 / height, "w": (x1 - x0) / width, "h": (y1 - y0) / height}


def run_florence_probe(manifest: dict, output_dir: Path) -> dict:
    start = time.monotonic()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        return {"kind": "florence_probe", "ok": False, "error": f"import failed: {exc}"}
    try:
        model_id = "microsoft/Florence-2-base-ft"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(device)
        records = manifest.get("real", []) + manifest.get("synthetic", [])[:8]
        coverage = {"labels": 0, "ocrRegionMatched": 0}
        outputs = []
        for record in records:
            image = Image.open(record["image"]).convert("RGB")
            parsed = florence_task(model, processor, image, "<OCR_WITH_REGION>", device, dtype)
            boxes = [
                quad_to_norm(quad, image.size)
                for quad in parsed.get("<OCR_WITH_REGION>", {}).get("quad_boxes", [])
            ]
            matched = 0
            for label in record["labels"]:
                if label["className"] == "non_callout_thread_texture":
                    continue
                coverage["labels"] += 1
                if any(max(box_iou(label["crop"], box), center_inside_score(label["crop"], box)) >= 0.25 for box in boxes):
                    matched += 1
                    coverage["ocrRegionMatched"] += 1
            outputs.append({"image": record["image"], "labels": len(record["labels"]), "ocrRegions": len(boxes), "matched": matched})
        coverage["ocrRegionRecall"] = round(coverage["ocrRegionMatched"] / coverage["labels"], 3) if coverage["labels"] else 0.0
        (output_dir / "florence_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        return {
            "kind": "florence_ocr_probe",
            "ok": True,
            "model": model_id,
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "coverage": coverage,
            "note": "Florence-2 OCR_WITH_REGION is scored as text-region coverage, not callout-box segmentation.",
        }
    except Exception as exc:
        return {
            "kind": "florence_ocr_probe",
            "ok": False,
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def florence_task(model, processor, image: Image.Image, prompt: str, device: str, dtype):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=768,
        num_beams=3,
        do_sample=False,
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(text, task=prompt, image_size=image.size)


def quad_to_norm(quad: list[float], image_size: tuple[int, int]) -> dict[str, float]:
    width, height = image_size
    xs = quad[0::2]
    ys = quad[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {"x": x0 / width, "y": y0 / height, "w": (x1 - x0) / width, "h": (y1 - y0) / height}


def box_iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0.0, min(ax1, bx1) - max(a["x"], b["x"]))
    iy = max(0.0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / max(union, 1e-6)


def center_inside_score(label: dict, prediction: dict) -> float:
    cx = prediction["x"] + prediction["w"] / 2
    cy = prediction["y"] + prediction["h"] / 2
    if label["x"] <= cx <= label["x"] + label["w"] and label["y"] <= cy <= label["y"] + label["h"]:
        return min(1.0, (prediction["w"] * prediction["h"]) / max(label["w"] * label["h"], 1e-6))
    return 0.0


def render_report(summary: dict) -> str:
    lines = [
        "# GD&T Symbol Vision Harness",
        "",
        f"Classes: {', '.join(summary['classes'])}",
        f"Real images: {', '.join(Path(path).name for path in summary['realImages'])}",
        f"Synthetic reference sheets: {summary['syntheticCount']}",
        "",
        "| Kind | Model | mAP50 | mAP50-95 | Precision | Recall | Real box/class | Reference box/class | Time | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in summary["records"]:
        metrics = record.get("metrics", {})
        coverage = record.get("coverage", {})
        real = coverage.get("real", coverage)
        synthetic = coverage.get("synthetic", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    record.get("kind", ""),
                    record.get("model", ""),
                    format_metric(metrics.get("map50")),
                    format_metric(metrics.get("map50_95")),
                    format_metric(metrics.get("precision_mean")),
                    format_metric(metrics.get("recall_mean")),
                    format_pair(real.get("boxRecall"), real.get("classRecall")),
                    format_pair(synthetic.get("boxRecall"), synthetic.get("classRecall")),
                    f"{record.get('elapsed_seconds', 0)}s",
                    record.get("note") or record.get("error") or record.get("best_model", ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_metric(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_pair(box_recall, class_recall) -> str:
    if box_recall is None:
        return "n/a"
    if class_recall is None:
        return f"{box_recall:.3f}/n/a"
    return f"{box_recall:.3f}/{class_recall:.3f}"


if __name__ == "__main__":
    main()
