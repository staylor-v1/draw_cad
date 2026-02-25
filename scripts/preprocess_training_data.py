#!/usr/bin/env python3
"""One-time preprocessing of training data.

Steps:
  1. Scan training_data/ to build a TrainingDataIndex
  2. Analyse each STEP file → StepGroundTruth
  3. Classify difficulty tiers
  4. Render all SVGs to PNG (parallel)
  5. Save manifest.json

Usage:
    python scripts/preprocess_training_data.py [--root training_data] [--workers 4] [--dpi 150]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.step_analyzer import analyze_step_file
from src.training.data_loader import TrainingDataIndex
from src.training.ground_truth import step_properties_to_ground_truth
from src.training.manifest import save_manifest
from src.training.svg_renderer import render_all_svgs
from src.training.tiering import assign_tiers, compute_tier_distribution
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--root", default="training_data", help="Training data root directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel render workers")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI")
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = root / "manifest.json"
    png_dir = root / "rendered_png"

    # 1. Build index from directory scan
    print(f"Scanning {root} for SVG/STEP pairs...")
    t0 = time.time()
    index = TrainingDataIndex.from_directory(root)
    print(f"  Found {index.size} matched pairs ({time.time() - t0:.1f}s)")

    if index.size == 0:
        print("ERROR: No matched pairs found. Check that drawings_svg/ and shapes_step/ exist.")
        sys.exit(1)

    # 2. Analyse STEP files and compute ground truth
    print("Analysing STEP files (this may take a while)...")
    t0 = time.time()
    analysed = 0
    failed = 0
    for i, pair in enumerate(index.pairs, 1):
        if i % 100 == 0 or i == index.size:
            print(f"  Progress: {i}/{index.size}", end="\r")
        try:
            props = analyze_step_file(pair.step_path)
            pair.ground_truth = step_properties_to_ground_truth(props)
            analysed += 1
        except Exception as e:
            logger.error("step_analysis_failed", pair_id=pair.pair_id, error=str(e))
            failed += 1

    print(f"\n  Analysed: {analysed}, Failed: {failed} ({time.time() - t0:.1f}s)")

    # 3. Classify difficulty tiers
    print("Classifying difficulty tiers...")
    assign_tiers(index)
    dist = compute_tier_distribution(index)
    print(f"  Tier distribution: {dict(dist)}")

    # 4. Render SVGs to PNG
    print(f"Rendering SVGs to PNG ({args.workers} workers, {args.dpi} DPI)...")
    t0 = time.time()
    rendered_count = render_all_svgs(
        index, output_dir=png_dir, dpi=args.dpi, workers=args.workers
    )
    print(f"  Rendered: {rendered_count} PNGs ({time.time() - t0:.1f}s)")

    # 5. Save manifest
    print(f"Saving manifest to {manifest_path}...")
    save_manifest(index, manifest_path)

    # Summary
    print("\n=== Preprocessing Complete ===")
    print(f"  Total pairs:   {index.size}")
    print(f"  STEP analysed: {analysed}")
    print(f"  PNGs rendered: {rendered_count}")
    print(f"  Tier 1 (Simple):  {dist.get(1, 0)}")
    print(f"  Tier 2 (Medium):  {dist.get(2, 0)}")
    print(f"  Tier 3 (Complex): {dist.get(3, 0)}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
