#!/usr/bin/env python3
"""Validate that all dependencies for the drawing-to-CAD system are installed."""
import sys


def main() -> int:
    errors = []

    # Core dependencies
    try:
        import pydantic
        print(f"  pydantic {pydantic.__version__}")
    except ImportError as e:
        errors.append(f"pydantic: {e}")

    try:
        import yaml
        print("  pyyaml OK")
    except ImportError as e:
        errors.append(f"pyyaml: {e}")

    try:
        import structlog
        print(f"  structlog {structlog.__version__}")
    except ImportError as e:
        errors.append(f"structlog: {e}")

    try:
        import httpx
        print(f"  httpx {httpx.__version__}")
    except ImportError as e:
        errors.append(f"httpx: {e}")

    try:
        import numpy
        print(f"  numpy {numpy.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")

    try:
        from PIL import Image
        print("  Pillow OK")
    except ImportError as e:
        errors.append(f"Pillow: {e}")

    try:
        import cairosvg
        print(f"  cairosvg {cairosvg.__version__}")
    except ImportError as e:
        errors.append(f"cairosvg: {e}")
    except OSError as e:
        errors.append(f"cairosvg: libcairo system library missing ({e})")

    # CAD dependencies
    try:
        import build123d
        print("  build123d OK")
    except ImportError as e:
        errors.append(f"build123d: {e}")

    try:
        import OCP
        print("  OCP (OpenCASCADE bindings) OK")
    except ImportError as e:
        errors.append(f"OCP: {e} (will fall back to trimesh)")

    try:
        import trimesh
        print(f"  trimesh {trimesh.__version__}")
    except ImportError as e:
        errors.append(f"trimesh: {e}")

    # Optional
    try:
        import optuna
        print(f"  optuna {optuna.__version__}")
    except ImportError:
        print("  optuna: not installed (optional)")

    try:
        import pytest
        print(f"  pytest {pytest.__version__}")
    except ImportError:
        print("  pytest: not installed (optional)")

    # Project module imports
    try:
        from src.training.ground_truth import StepGroundTruth
        from src.training.data_loader import TrainingDataIndex
        from src.training.sampler import BenchmarkSampler
        from src.training.curriculum import CurriculumScheduler
        from src.training.fewshot_miner import FewShotMiner
        from src.training.svg_renderer import render_svg_to_png
        from src.training.tiering import classify_tier
        from src.training.manifest import save_manifest, load_manifest
        from src.evaluation.comparator import StepComparator
        from src.evaluation.benchmark_runner import BenchmarkRunner
        from src.optimizer.meta_optimizer import MetaOptimizer
        from src.schemas.evaluation_result import EvaluationMetrics
        print("  All project modules: OK")
    except ImportError as e:
        errors.append(f"project modules: {e}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        return 1

    print("\n  All validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
