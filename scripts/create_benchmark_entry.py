#!/usr/bin/env python3
"""Helper script to create a new benchmark entry."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_utils import save_yaml


def main():
    parser = argparse.ArgumentParser(description="Create a new benchmark entry")
    parser.add_argument("id", help="Case ID (e.g., 006_bracket)")
    parser.add_argument("--drawing", required=True, help="Path to drawing image")
    parser.add_argument("--reference", help="Path to reference STEP file")
    parser.add_argument("--description", default="", help="Case description")
    parser.add_argument("--tier", type=int, default=1, help="Difficulty tier (1-3)")
    args = parser.parse_args()

    case_dir = Path("benchmarks/drawings") / args.id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Copy drawing
    drawing_dest = case_dir / "drawing.png"
    shutil.copy2(args.drawing, drawing_dest)

    # Copy reference if provided
    if args.reference:
        ref_dest = case_dir / "reference.step"
        shutil.copy2(args.reference, ref_dest)

    # Create metadata
    metadata = {
        "id": args.id,
        "tier": args.tier,
        "description": args.description,
    }
    save_yaml(metadata, case_dir / "metadata.yaml")

    print(f"Created benchmark entry: {case_dir}")
    print(f"Remember to add this case to benchmarks/suite.yaml")


if __name__ == "__main__":
    main()
