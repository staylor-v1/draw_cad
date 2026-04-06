#!/usr/bin/env python3
"""Export the AI toolbox manifest for this repo."""
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

from src.ai_toolbox import DEFAULT_TOOLBOX_MANIFEST_PATH, write_toolbox_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the AI toolbox manifest")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TOOLBOX_MANIFEST_PATH),
        help="Path where the YAML manifest should be written",
    )
    args = parser.parse_args()

    output_path = write_toolbox_manifest(args.output)
    print(f"AI toolbox manifest written to {output_path}")


if __name__ == "__main__":
    main()
