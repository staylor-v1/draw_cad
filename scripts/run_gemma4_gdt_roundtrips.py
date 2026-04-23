"""Run Gemma 4 roundtrip tuning cases from training_data/gdt."""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemma4_agent.agent import Gemma4RoundTripAgent, Gemma4RoundTripConfig


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 4 roundtrips for GD&T drawing images")
    parser.add_argument("--config", default="gemma4_agent/config.yaml")
    parser.add_argument("--input-dir", default="training_data/gdt")
    parser.add_argument("--output-dir", default="experiments/gemma4_agent/gdt_roundtrips")
    parser.add_argument("--model", help="Ollama model override")
    parser.add_argument("--base-url", help="Ollama base URL override")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--limit", type=int, help="Maximum number of images to run")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only run image names matching this shell-style pattern. May be repeated.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip cases with existing summaries")
    args = parser.parse_args()

    config = Gemma4RoundTripConfig.from_yaml(args.config)
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    if overrides:
        config = Gemma4RoundTripConfig(**{**config.__dict__, **overrides})

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    agent = Gemma4RoundTripAgent(config)

    image_paths = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if args.include:
        image_paths = [
            path for path in image_paths
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in args.include)
        ]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        case_dir = output_dir / _case_name(image_path)
        summary_path = case_dir / "roundtrip_summary.json"
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            records.append(_record_from_summary(image_path, summary, resumed=True))
            continue

        if time.monotonic() >= deadline:
            records.append(
                {
                    "case": image_path.name,
                    "success": False,
                    "skipped": True,
                    "reason": "timeout_hours reached before case start",
                }
            )
            break

        started = time.monotonic()
        try:
            summary = agent.run_roundtrip(image_path, output_dir=case_dir)
            record = _record_from_summary(image_path, summary)
        except Exception as exc:
            record = {
                "case": image_path.name,
                "success": False,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "output_dir": str(case_dir),
            }
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        records.append(record)

        aggregate = _aggregate(records, output_dir)
        print(json.dumps(aggregate, indent=2), flush=True)

    aggregate = _aggregate(records, output_dir)
    (output_dir / "gdt_roundtrip_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


def _case_name(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return stem[:96]


def _record_from_summary(
    image_path: Path,
    summary: dict[str, Any],
    resumed: bool = False,
) -> dict[str, Any]:
    comparison = summary.get("comparison", {})
    metrics = comparison.get("metrics", {})
    return {
        "case": image_path.name,
        "success": bool(summary.get("success")),
        "resumed": resumed,
        "output_dir": summary.get("output_dir"),
        "first_step_path": summary.get("first_step_path"),
        "second_step_path": summary.get("second_step_path"),
        "metrics": metrics,
    }


def _aggregate(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    completed = [record for record in records if not record.get("skipped")]
    passed = [record for record in completed if record.get("success")]
    return {
        "output_dir": str(output_dir),
        "total": len(records),
        "completed": len(completed),
        "passed": len(passed),
        "failed": len(completed) - len(passed),
        "records": records,
    }


if __name__ == "__main__":
    main()
