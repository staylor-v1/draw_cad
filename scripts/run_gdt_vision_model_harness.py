"""Train and evaluate vision detectors for GD&T/callout crops."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.callouts import load_callout_fixture, load_non_callout_fixture


CLASS_NAMES = ["callout", "non_callout_thread_texture"]
DEFAULT_IMAGES = [
    Path("training_data/gdt/simple1.webp"),
    Path("training_data/gdt/threaded_cap.jpg"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="GD&T vision model benchmark/fine-tune harness.")
    parser.add_argument("--output-dir", default="experiments/gdt_vision_models")
    parser.add_argument("--image", action="append", help="Image to include. Defaults to scored GD&T fixtures.")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--florence", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images = [Path(path) for path in args.image] if args.image else DEFAULT_IMAGES
    dataset_dir = output_dir / "dataset"
    export_yolo_dataset(images, dataset_dir)
    records = []

    if args.florence:
        records.append(run_florence_probe(images, output_dir))

    if args.train:
        for model_name in args.models:
            records.append(
                train_and_eval_yolo(
                    model_name=model_name,
                    dataset_yaml=dataset_dir / "dataset.yaml",
                    output_dir=output_dir,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                )
            )
    else:
        for model_name in args.models:
            records.append(eval_pretrained_yolo(model_name, images, output_dir))

    summary = {"images": [str(path) for path in images], "records": records}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


def export_yolo_dataset(images: list[Path], dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_path in images:
        for split in ("train", "val"):
            target_image = dataset_dir / "images" / split / f"{image_path.stem}.jpg"
            Image.open(image_path).convert("RGB").save(target_image, quality=95)
            labels = yolo_labels_for_image(image_path)
            (dataset_dir / "labels" / split / f"{image_path.stem}.txt").write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8",
            )

    (dataset_dir / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def yolo_labels_for_image(image_path: Path) -> list[str]:
    labels = []
    for item in load_callout_fixture(image_path):
        labels.append(yolo_label(0, item["crop"]))
    for item in load_non_callout_fixture(image_path):
        class_id = 1 if item.get("kind") == "thread_texture" else 0
        labels.append(yolo_label(class_id, item["crop"]))
    return labels


def yolo_label(class_id: int, crop: dict) -> str:
    cx = crop["x"] + crop["w"] / 2
    cy = crop["y"] + crop["h"] / 2
    return f"{class_id} {cx:.6f} {cy:.6f} {crop['w']:.6f} {crop['h']:.6f}"


def eval_pretrained_yolo(model_name: str, images: list[Path], output_dir: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_name)
    start = time.monotonic()
    detections = 0
    for image_path in images:
        result = model.predict(str(image_path), imgsz=1024, conf=0.01, verbose=False)[0]
        detections += len(result.boxes or [])
    return {
        "kind": "pretrained_probe",
        "model": model_name,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "detections": detections,
        "note": "COCO-pretrained detector has no GD&T/callout class, so detections are not expected to be useful.",
    }


def train_and_eval_yolo(
    *,
    model_name: str,
    dataset_yaml: Path,
    output_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | int,
) -> dict:
    from ultralytics import YOLO

    run_name = Path(model_name).stem.replace(".", "_")
    start = time.monotonic()
    model = YOLO(model_name)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir / "runs"),
        name=run_name,
        exist_ok=True,
        patience=epochs,
        plots=False,
        verbose=False,
        workers=0,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best))
    metrics = trained.val(data=str(dataset_yaml), imgsz=imgsz, batch=batch, device=device, verbose=False, plots=False)
    return {
        "kind": "fine_tuned_yolo",
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
    }


def run_florence_probe(images: list[Path], output_dir: Path) -> dict:
    start = time.monotonic()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        return {"kind": "florence_probe", "ok": False, "error": f"import failed: {exc}"}
    try:
        model_id = "microsoft/Florence-2-base-ft"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        outputs = []
        for image_path in images:
            image = Image.open(image_path).convert("RGB")
            prompt = "<OD>"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(text, task=prompt, image_size=image.size)
            outputs.append({"image": str(image_path), "parsed": parsed})
        return {
            "kind": "florence_probe",
            "ok": True,
            "model": model_id,
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "outputs": outputs,
        }
    except Exception as exc:
        return {
            "kind": "florence_probe",
            "ok": False,
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def render_report(summary: dict) -> str:
    lines = [
        "# GD&T Vision Model Harness",
        "",
        f"Images: {', '.join(Path(path).name for path in summary['images'])}",
        "",
        "| Kind | Model | mAP50 | mAP50-95 | Precision | Recall | Time | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in summary["records"]:
        metrics = record.get("metrics", {})
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
                    f"{record.get('elapsed_seconds', 0)}s",
                    record.get("note") or record.get("error") or record.get("best_model", ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_metric(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
