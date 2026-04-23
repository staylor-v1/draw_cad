"""Run comparable local extractor configurations for GD&T prompt training."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_SCRIPT = REPO_ROOT / "scripts" / "tune_gemma4_gdt_roundtrip.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local Gemma/Florence/YOLO-Donut evidence configurations."
    )
    parser.add_argument("--input-dir", default="training_data/gdt")
    parser.add_argument("--output-dir", default="experiments/gemma4_agent/gdt_extractor_experiments")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--initial-source-fidelity-threshold", type=float, default=0.72)
    parser.add_argument("--target-source-fidelity-threshold", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--florence2-model-path")
    parser.add_argument("--yolo-obb-model-path")
    parser.add_argument("--donut-model-path")
    parser.add_argument("--extractor-device")
    parser.add_argument(
        "--variant",
        action="append",
        help=(
            "Variant in name=extractor,extractor form. Defaults to gemma4, "
            "heuristic+gemma4, florence2+gemma4 when a Florence path exists, "
            "and yolo_donut+gemma4 when a YOLO path exists."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = _variants(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "output_dir": str(output_dir),
        "variants": variants,
        "target_source_fidelity_threshold": args.target_source_fidelity_threshold,
        "started_at_unix": time.time(),
    }
    (output_dir / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    seconds_per_variant = max(args.timeout_hours * 3600.0 / max(len(variants), 1), 60.0)
    for variant in variants:
        command = _command(args, variant, output_dir, seconds_per_variant / 3600.0)
        record: dict[str, Any] = {
            "name": variant["name"],
            "extractors": variant["extractors"],
            "command": command,
        }
        if args.dry_run:
            record["return_code"] = None
        else:
            started = time.monotonic()
            completed = subprocess.run(command, cwd=REPO_ROOT, text=True)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["return_code"] = completed.returncode
            summary_path = output_dir / variant["name"] / "training_loop_summary.json"
            if summary_path.exists():
                record["summary_path"] = str(summary_path)
                record["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append(record)
        (output_dir / "experiment_results.json").write_text(
            json.dumps({"results": results}, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"results": results}, indent=2))


def _variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.variant:
        variants = []
        for item in args.variant:
            name, _, extractors = item.partition("=")
            if not name or not extractors:
                raise ValueError(f"Invalid variant: {item}")
            variants.append(
                {
                    "name": _safe_name(name),
                    "extractors": [part.strip() for part in extractors.split(",") if part.strip()],
                }
            )
        return variants

    variants = [
        {"name": "gemma4", "extractors": ["gemma4"]},
        {"name": "heuristic_gemma4", "extractors": ["heuristic", "gemma4"]},
    ]
    if args.florence2_model_path:
        variants.append({"name": "florence2_gemma4", "extractors": ["florence2", "gemma4"]})
    if args.yolo_obb_model_path:
        variants.append({"name": "yolo_donut_gemma4", "extractors": ["yolo_donut", "gemma4"]})
    return variants


def _command(
    args: argparse.Namespace,
    variant: dict[str, Any],
    output_dir: Path,
    timeout_hours: float,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--input-dir",
        args.input_dir,
        "--output-dir",
        str(output_dir / variant["name"]),
        "--timeout-hours",
        f"{timeout_hours:.6f}",
        "--max-iterations",
        str(args.max_iterations),
        "--initial-source-fidelity-threshold",
        str(args.initial_source_fidelity_threshold),
        "--target-source-fidelity-threshold",
        str(args.target_source_fidelity_threshold),
        "--threshold-step",
        str(args.threshold_step),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    for pattern in args.include:
        command.extend(["--include", pattern])
    if args.model:
        command.extend(["--model", args.model])
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.florence2_model_path:
        command.extend(["--florence2-model-path", args.florence2_model_path])
    if args.yolo_obb_model_path:
        command.extend(["--yolo-obb-model-path", args.yolo_obb_model_path])
    if args.donut_model_path:
        command.extend(["--donut-model-path", args.donut_model_path])
    if args.extractor_device:
        command.extend(["--extractor-device", args.extractor_device])
    for extractor in variant["extractors"]:
        command.extend(["--extractor", extractor])
    return command


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)[:80]


if __name__ == "__main__":
    main()
