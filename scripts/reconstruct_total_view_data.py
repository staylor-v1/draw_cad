#!/usr/bin/env python3
"""Deterministically reconstruct STEP files from Total_view_data triplets."""
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
from src.utils.file_utils import save_json
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct STEP files from Total_view_data f/r/t SVG triplets"
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
    parser.add_argument("--case-id", help="Specific case id to reconstruct")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of cases to reconstruct when --case-id is not provided",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List matching case ids without reconstructing them",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where generated STEP files and source code are written",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON path for the reconstruction summary",
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
        if args.limit > 0:
            case_ids = case_ids[: args.limit]

    if args.list_only:
        for case_id in case_ids:
            print(case_id)
        return

    if not case_ids:
        print("No matching cases found.")
        sys.exit(1)

    output_dir = Path(args.output_dir or config.output.total_view_data_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path or config.output.total_view_data_summary_path)

    summary_records: list[dict[str, object]] = []
    failed = False

    for case_id in case_ids:
        triplet = archive.load_triplet(case_id, preferred_views)
        candidates = (
            reconstructor.generate_candidate_programs(triplet)
            if config.orthographic_reconstruction.candidate_search_enabled
            else []
        )
        if not candidates:
            base_program = reconstructor.generate_program(triplet)
            from src.reconstruction.orthographic_solver import ReconstructionCandidate

            candidates = [ReconstructionCandidate(name="default", result=base_program)]

        best_record: dict[str, object] | None = None
        for candidate in candidates:
            candidate_code_path = output_dir / f"{case_id}__{candidate.name}.py"
            candidate_step_path = output_dir / f"{case_id}__{candidate.name}.step"
            candidate_code_path.write_text(candidate.result.code, encoding="utf-8")

            execution = execute_build123d(
                script_content=candidate.result.code,
                output_path=str(candidate_step_path),
                timeout=config.pipeline.execution_timeout,
            )
            if not execution.success:
                summary_records.append(
                    {
                        "case_id": case_id,
                        "dataset": dataset_name,
                        "candidate": candidate.name,
                        "success": False,
                        "step_path": None,
                        "code_path": str(candidate_code_path),
                        "error_category": execution.error_category.value,
                        "stderr": execution.stderr,
                        "consensus_extents": candidate.result.consensus_extents,
                        "hidden_feature_count": len(candidate.result.hidden_cylinders),
                    }
                )
                continue

            reprojection_score, _ = evaluate_step_against_triplet(
                candidate_step_path,
                triplet,
                config=config.reprojection,
                view_suffixes=preferred_views,  # type: ignore[arg-type]
            )
            candidate_record = {
                "case_id": case_id,
                "dataset": dataset_name,
                "candidate": candidate.name,
                "success": True,
                "score": reprojection_score.score,
                "step_path": str(candidate_step_path),
                "code_path": str(candidate_code_path),
                "consensus_extents": candidate.result.consensus_extents,
                "hidden_feature_count": len(candidate.result.hidden_cylinders),
                "point_counts": {
                    suffix: len(contour.points)
                    for suffix, contour in candidate.result.contours.items()
                },
                "views": {
                    suffix: {
                        "score": view_score.score,
                        "visible_f1": view_score.visible.f1,
                        "hidden_f1": view_score.hidden.f1,
                    }
                    for suffix, view_score in reprojection_score.views.items()
                },
            }
            summary_records.append(candidate_record)
            if best_record is None or reprojection_score.score > float(best_record["score"]):
                best_record = candidate_record

        if best_record is None:
            failed = True
            print(f"{case_id} [{dataset_name}] FAIL all_candidates")
            continue

        selected_code_path = output_dir / f"{case_id}.py"
        selected_step_path = output_dir / f"{case_id}.step"
        Path(best_record["code_path"]).replace(selected_code_path)
        Path(best_record["step_path"]).replace(selected_step_path)
        best_record["selected"] = True
        best_record["step_path"] = str(selected_step_path)
        best_record["code_path"] = str(selected_code_path)

        print(
            f"{case_id} [{dataset_name}] OK "
            f"candidate={best_record['candidate']} "
            f"score={float(best_record['score']):.3f} "
            f"x={best_record['consensus_extents']['x']:.3f} "
            f"y={best_record['consensus_extents']['y']:.3f} "
            f"z={best_record['consensus_extents']['z']:.3f} "
            f"cuts={int(best_record['hidden_feature_count'])}"
        )

    save_json({"cases": summary_records}, summary_path)
    print(f"Summary written to {summary_path}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
