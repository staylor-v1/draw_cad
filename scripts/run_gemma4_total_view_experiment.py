#!/usr/bin/env python3
"""Run the Gemma 4 Total_view_data experiment suite."""
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

from src.experiments.gemma4_total_view import run_experiment
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Gemma 4 multimodal/tool Total_view_data experiment"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/gemma4_total_view.yaml",
        help="Experiment config path",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    result = run_experiment(args.config)

    print("Gemma 4 experiment complete")
    print(f"Report: {result['report_path']}")
    for mode_name, mode_result in result["evaluation"].items():
        aggregate = mode_result["aggregate"]
        print(
            f"{mode_name}: score={aggregate['mean_score']:.3f} "
            f"visible={aggregate['mean_visible_f1']:.3f} "
            f"hidden={aggregate['mean_hidden_f1']:.3f} "
            f"success={aggregate['success_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
