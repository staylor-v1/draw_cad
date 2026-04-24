"""Iteratively tune Gemma 4 prompts against GD&T drawing reconstruction cases."""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemma4_agent.agent import Gemma4RoundTripAgent, Gemma4RoundTripConfig
from gemma4_agent.extractors import ExtractorRuntime, build_extractors, run_extractors
from gemma4_agent.training import (
    DEFAULT_INITIAL_SOURCE_FIDELITY_THRESHOLD,
    DEFAULT_TARGET_SOURCE_FIDELITY_THRESHOLD,
    judge_source_fidelity,
    propose_profile_revision,
    training_case_passed,
    threshold_for_iteration,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an iterative prompt-training loop for raster GD&T drawings. A case passes "
            "only when it roundtrips, uses no fallback geometry, and passes source-drawing "
            "fidelity judged from the original drawing versus the generated contact sheet."
        )
    )
    parser.add_argument("--config", default="gemma4_agent/config.yaml")
    parser.add_argument("--input-dir", default="training_data/gdt")
    parser.add_argument("--output-dir", default="experiments/gemma4_agent/gdt_training_loop")
    parser.add_argument("--profile-path", default="gemma4_agent/prompts/agent.md")
    parser.add_argument("--model", help="Ollama model override")
    parser.add_argument("--base-url", help="Ollama base URL override")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Maximum number of images per iteration")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only run image names matching this shell-style pattern. May be repeated.",
    )
    parser.add_argument(
        "--source-fidelity-threshold",
        type=float,
        help="Use one fixed source-fidelity threshold. Overrides curriculum settings.",
    )
    parser.add_argument(
        "--initial-source-fidelity-threshold",
        type=float,
        default=DEFAULT_INITIAL_SOURCE_FIDELITY_THRESHOLD,
        help="Curriculum starting threshold used before ratcheting toward the target.",
    )
    parser.add_argument(
        "--target-source-fidelity-threshold",
        type=float,
        default=DEFAULT_TARGET_SOURCE_FIDELITY_THRESHOLD,
        help="Final source-fidelity goal. Defaults to 0.99.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Per-iteration source-fidelity threshold increase until the target is reached.",
    )
    parser.add_argument("--feature-match-threshold", type=float)
    parser.add_argument(
        "--extractor",
        action="append",
        default=[],
        help=(
            "Drawing evidence extractor to run before CAD reconstruction. May be repeated. "
            "Use none, heuristic, gemma4, florence2, or yolo_donut."
        ),
    )
    parser.add_argument("--florence2-model-path", help="Local fine-tuned Florence-2 model directory")
    parser.add_argument("--yolo-obb-model-path", help="Local YOLOv11-OBB model file")
    parser.add_argument("--donut-model-path", help="Local fine-tuned Donut model directory")
    parser.add_argument("--extractor-device", help="Optional torch device for local VLM extractors")
    parser.add_argument(
        "--skip-profile-revision",
        action="store_true",
        help="Run evaluation without asking Gemma to write a new profile.",
    )
    parser.add_argument(
        "--promote-best-profile",
        action="store_true",
        help="Overwrite --profile-path with the best profile produced by this run.",
    )
    args = parser.parse_args()
    if not args.extractor:
        args.extractor = ["gemma4"]

    config = Gemma4RoundTripConfig.from_yaml(args.config)
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    if overrides:
        config = Gemma4RoundTripConfig(**{**config.__dict__, **overrides})
    if args.source_fidelity_threshold is not None:
        args.initial_source_fidelity_threshold = args.source_fidelity_threshold
        args.target_source_fidelity_threshold = args.source_fidelity_threshold

    output_dir = Path(args.output_dir)
    profiles_dir = output_dir / "profiles"
    iterations_dir = output_dir / "iterations"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir.mkdir(parents=True, exist_ok=True)

    profile_path = Path(args.profile_path)
    seed_profile = profile_path.read_text(encoding="utf-8")
    current_profile_path = profiles_dir / "iteration_000.md"
    current_profile_path.write_text(seed_profile, encoding="utf-8")

    image_paths = _selected_images(Path(args.input_dir), args.include, args.limit)
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    history: list[dict[str, Any]] = []
    best_profile_path = current_profile_path
    best_passed = -1
    extractor_runtime = ExtractorRuntime(
        model=config.model,
        base_url=config.base_url,
        florence2_model_path=Path(args.florence2_model_path) if args.florence2_model_path else None,
        yolo_obb_model_path=Path(args.yolo_obb_model_path) if args.yolo_obb_model_path else None,
        donut_model_path=Path(args.donut_model_path) if args.donut_model_path else None,
        device=args.extractor_device,
    )
    extractors = build_extractors(args.extractor, extractor_runtime)

    for iteration in range(args.max_iterations):
        if time.monotonic() >= deadline:
            history.append(
                {
                    "iteration": iteration,
                    "stopped": True,
                    "reason": "timeout_hours reached before iteration start",
                }
            )
            break

        iteration_dir = iterations_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        active_profile_path = iteration_dir / "agent_profile.md"
        shutil.copyfile(current_profile_path, active_profile_path)
        active_config = Gemma4RoundTripConfig(
            **{
                **asdict(config),
                "agent_profile_path": active_profile_path,
                "output_dir": iteration_dir,
            }
        )
        agent = Gemma4RoundTripAgent(active_config)
        active_source_threshold = threshold_for_iteration(
            iteration=iteration,
            initial_threshold=args.initial_source_fidelity_threshold,
            target_threshold=args.target_source_fidelity_threshold,
            threshold_step=args.threshold_step,
        )
        records = _run_iteration(
            agent=agent,
            config=active_config,
            extractors=extractors,
            image_paths=image_paths,
            iteration_dir=iteration_dir,
            deadline=deadline,
            source_fidelity_threshold=active_source_threshold,
            feature_match_threshold=args.feature_match_threshold,
        )
        aggregate = _aggregate(
            iteration,
            records,
            iteration_dir,
            active_profile_path,
            active_source_threshold,
            args.target_source_fidelity_threshold,
        )
        history.append(aggregate)
        (iteration_dir / "iteration_summary.json").write_text(
            json.dumps(aggregate, indent=2),
            encoding="utf-8",
        )
        _write_run_summary(output_dir, history, best_profile_path)
        print(json.dumps(aggregate, indent=2), flush=True)

        passed = int(aggregate["passed"])
        if passed > best_passed:
            best_passed = passed
            best_profile_path = active_profile_path
        if (
            aggregate["failed"] == 0
            and aggregate["completed"] == len(image_paths)
            and active_source_threshold >= args.target_source_fidelity_threshold
        ):
            break
        if args.skip_profile_revision and active_source_threshold >= args.target_source_fidelity_threshold:
            break
        if time.monotonic() >= deadline:
            break

        failed_records = [record for record in records if not record.get("success")]
        passed_records = [record for record in records if record.get("success")]
        current_profile_path = profiles_dir / f"iteration_{iteration + 1:03d}.md"
        if failed_records and not args.skip_profile_revision:
            revised_profile = propose_profile_revision(
                model=active_config.model,
                base_url=active_config.base_url,
                current_profile=active_profile_path.read_text(encoding="utf-8"),
                failed_records=failed_records,
                passed_records=passed_records,
                temperature=0.2,
            )
            current_profile_path.write_text(revised_profile + "\n", encoding="utf-8")
        else:
            shutil.copyfile(active_profile_path, current_profile_path)

    _write_run_summary(output_dir, history, best_profile_path)
    if args.promote_best_profile and best_profile_path.exists():
        shutil.copyfile(best_profile_path, profile_path)


def _run_iteration(
    *,
    agent: Gemma4RoundTripAgent,
    config: Gemma4RoundTripConfig,
    extractors: list[Any],
    image_paths: list[Path],
    iteration_dir: Path,
    deadline: float,
    source_fidelity_threshold: float,
    feature_match_threshold: float | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        case_dir = iteration_dir / _case_name(image_path)
        if time.monotonic() >= deadline:
            records.append(
                {
                    "case": image_path.name,
                    "success": False,
                    "skipped": True,
                    "reason": "timeout_hours reached before case start",
                    "output_dir": str(case_dir),
                }
            )
            break

        started = time.monotonic()
        try:
            drawing_evidence = run_extractors(
                extractors=extractors,
                drawing_path=image_path,
                output_dir=case_dir / "evidence",
            )
            summary = agent.run_roundtrip(
                image_path,
                output_dir=case_dir,
                drawing_evidence=drawing_evidence,
            )
            contact_sheet = summary.get("rendered_drawing", {}).get("contact_sheet_path")
            if not contact_sheet:
                raise RuntimeError("Roundtrip did not produce a generated contact sheet")
            source_fidelity = judge_source_fidelity(
                model=config.model,
                base_url=config.base_url,
                original_drawing_path=image_path,
                generated_contact_sheet_path=contact_sheet,
                temperature=0.0,
            )
            criteria = training_case_passed(
                roundtrip_summary=summary,
                source_fidelity=source_fidelity,
                source_fidelity_threshold=source_fidelity_threshold,
                feature_match_threshold=feature_match_threshold,
            )
            summary["source_fidelity"] = source_fidelity
            summary["drawing_evidence"] = drawing_evidence
            summary["success_criteria"] = criteria
            summary["success"] = bool(criteria["passed"])
            summary_path = case_dir / "roundtrip_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    return records


def _selected_images(input_dir: Path, include: list[str], limit: int | None) -> list[Path]:
    image_paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if include:
        image_paths = [
            path
            for path in image_paths
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in include)
        ]
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def _case_name(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return stem[:96]


def _record_from_summary(image_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    comparison = summary.get("comparison", {})
    fidelity = summary.get("source_fidelity", {})
    return {
        "case": image_path.name,
        "success": bool(summary.get("success")),
        "output_dir": summary.get("output_dir"),
        "first_step_path": summary.get("first_step_path"),
        "second_step_path": summary.get("second_step_path"),
        "roundtrip_equivalent": bool(summary.get("roundtrip_equivalent")),
        "used_fallback": bool(summary.get("used_fallback")),
        "drawing_evidence": summary.get("drawing_evidence", {}),
        "success_criteria": summary.get("success_criteria", {}),
        "source_fidelity": fidelity,
        "comparison": comparison,
    }


def _aggregate(
    iteration: int,
    records: list[dict[str, Any]],
    iteration_dir: Path,
    profile_path: Path,
    source_fidelity_threshold: float,
    target_source_fidelity_threshold: float,
) -> dict[str, Any]:
    completed = [record for record in records if not record.get("skipped")]
    passed = [record for record in completed if record.get("success")]
    return {
        "iteration": iteration,
        "output_dir": str(iteration_dir),
        "profile_path": str(profile_path),
        "source_fidelity_threshold": source_fidelity_threshold,
        "target_source_fidelity_threshold": target_source_fidelity_threshold,
        "total": len(records),
        "completed": len(completed),
        "passed": len(passed),
        "failed": len(completed) - len(passed),
        "records": records,
    }


def _write_run_summary(
    output_dir: Path,
    history: list[dict[str, Any]],
    best_profile_path: Path,
) -> None:
    summary = {
        "output_dir": str(output_dir),
        "best_profile_path": str(best_profile_path),
        "iterations": history,
    }
    (output_dir / "training_loop_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
