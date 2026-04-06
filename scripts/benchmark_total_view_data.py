#!/usr/bin/env python3
"""Benchmark deterministic Total_view_data reconstruction by reprojection."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _maybe_reexec_into_project_venv() -> None:
    """Use the repo-local virtualenv when it exists."""
    repo_root = Path(__file__).resolve().parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


_maybe_reexec_into_project_venv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reconstruction import (
    OrthographicTripletReconstructor,
    TotalViewArchive,
    evaluate_step_against_triplet,
)
from src.schemas.pipeline_config import PipelineConfig
from src.tools.cad import execute_build123d
from src.tools.step_analyzer import analyze_step_file
from src.utils.file_utils import save_json
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Total_view_data reconstruction by reprojection"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/total_view_data.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help="Dataset archive to use (for example ABC or CSG)",
    )
    parser.add_argument("--case-id", help="Specific case id to benchmark")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of cases to benchmark when --case-id is not provided",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset into the sorted case list when sampling cases",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where generated code, STEP files, and overlays are written",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON path for the benchmark summary",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    config = PipelineConfig.from_yaml(args.config)

    dataset_name = args.dataset or config.total_view_data.default_dataset
    archive = TotalViewArchive(config.total_view_data.get_svg_archive(dataset_name))
    reconstructor = OrthographicTripletReconstructor.from_pipeline_config(config)
    preferred_views = tuple(config.total_view_data.preferred_views)

    if args.case_id:
        case_ids = [args.case_id]
    else:
        case_ids = archive.case_ids(
            required_views=preferred_views,
            require_complete=config.total_view_data.require_complete_triplet,
        )
        case_ids = case_ids[args.offset :]
        if args.limit > 0:
            case_ids = case_ids[: args.limit]

    if not case_ids:
        print("No matching cases found.")
        sys.exit(1)

    benchmark_dir = Path(args.output_dir or config.output.total_view_data_benchmark_dir)
    steps_dir = benchmark_dir / "steps"
    code_dir = benchmark_dir / "code"
    overlays_dir = benchmark_dir / "overlays"
    steps_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(
        args.summary_path or config.output.total_view_data_benchmark_summary_path
    )

    records: list[dict[str, object]] = []
    total_score = 0.0
    total_visible_f1 = 0.0
    total_hidden_f1 = 0.0
    successful = 0

    for case_id in case_ids:
        triplet = archive.load_triplet(case_id, preferred_views)
        candidate_records: list[dict[str, object]] = []
        best_record: dict[str, object] | None = None
        best_rendered = None
        for candidate in reconstructor.generate_candidate_programs(triplet):
            program = candidate.result
            code_path = code_dir / f"{case_id}__{candidate.name}.py"
            code_path.write_text(program.code, encoding="utf-8")

            step_path = steps_dir / f"{case_id}__{candidate.name}.step"
            execution = execute_build123d(
                script_content=program.code,
                output_path=str(step_path),
                timeout=config.pipeline.execution_timeout,
            )

            if execution.success:
                step_props = analyze_step_file(step_path)
                score, rendered = evaluate_step_against_triplet(
                    step_path,
                    triplet,
                    config=config.reprojection,
                    view_suffixes=preferred_views,  # type: ignore[arg-type]
                )
                mean_visible_f1 = sum(
                    view.visible.f1 for view in score.views.values()
                ) / len(score.views)
                mean_hidden_f1 = sum(
                    view.hidden.f1 for view in score.views.values()
                ) / len(score.views)
                candidate_record = {
                    "name": candidate.name,
                    "success": True,
                    "score": score.score,
                    "step_path": str(step_path),
                    "code_path": str(code_path),
                    "consensus_extents": program.consensus_extents,
                    "hidden_feature_count": len(program.hidden_cylinders),
                    "step_properties": step_props.__dict__,
                    "views": {
                        suffix: {
                            "score": view_score.score,
                            "visible": view_score.visible.__dict__,
                            "hidden": view_score.hidden.__dict__,
                        }
                        for suffix, view_score in score.views.items()
                    },
                }
                if best_record is None or score.score > float(best_record["score"]):
                    best_record = {
                        **candidate_record,
                        "mean_visible_f1": mean_visible_f1,
                        "mean_hidden_f1": mean_hidden_f1,
                    }
                    best_rendered = rendered
            else:
                candidate_record = {
                    "name": candidate.name,
                    "success": False,
                    "score": 0.0,
                    "step_path": None,
                    "code_path": str(code_path),
                    "consensus_extents": program.consensus_extents,
                    "hidden_feature_count": len(program.hidden_cylinders),
                    "error_category": execution.error_category.value,
                    "stderr": execution.stderr,
                }

            candidate_records.append(candidate_record)

        if best_record is not None:
            total_score += float(best_record["score"])
            total_visible_f1 += float(best_record["mean_visible_f1"])
            total_hidden_f1 += float(best_record["mean_hidden_f1"])
            successful += 1

            for suffix, comparison in best_rendered.items():
                if comparison.overlay is None:
                    continue
                comparison.overlay.save(overlays_dir / f"{case_id}_{suffix}.png")

            record = {
                "case_id": case_id,
                "dataset": dataset_name,
                "success": True,
                "score": best_record["score"],
                "selected_candidate": best_record["name"],
                "step_path": best_record["step_path"],
                "code_path": best_record["code_path"],
                "consensus_extents": best_record["consensus_extents"],
                "hidden_feature_count": best_record["hidden_feature_count"],
                "step_properties": best_record["step_properties"],
                "views": best_record["views"],
                "candidates": candidate_records,
            }
            print(
                f"{case_id} [{dataset_name}] score={float(best_record['score']):.3f} "
                f"visible={float(best_record['mean_visible_f1']):.3f} "
                f"hidden={float(best_record['mean_hidden_f1']):.3f} "
                f"candidate={best_record['name']} "
                f"cuts={int(best_record['hidden_feature_count'])}"
            )
        else:
            record = {
                "case_id": case_id,
                "dataset": dataset_name,
                "success": False,
                "score": 0.0,
                "selected_candidate": None,
                "candidates": candidate_records,
            }
            print(f"{case_id} [{dataset_name}] FAIL all_candidates")

        records.append(record)

    aggregate = {
        "dataset": dataset_name,
        "requested_cases": len(case_ids),
        "successful_cases": successful,
        "mean_score": total_score / successful if successful else 0.0,
        "mean_visible_f1": total_visible_f1 / successful if successful else 0.0,
        "mean_hidden_f1": total_hidden_f1 / successful if successful else 0.0,
        "worst_cases": [
            {
                "case_id": record["case_id"],
                "score": record["score"],
            }
            for record in sorted(records, key=lambda item: float(item["score"]))[:10]
        ],
    }

    save_json(
        {
            "aggregate": aggregate,
            "cases": records,
        },
        summary_path,
    )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
