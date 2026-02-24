#!/usr/bin/env python3
"""Run the Loop 2 meta-optimizer."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.pipeline_config import PipelineConfig
from src.optimizer.meta_optimizer import MetaOptimizer
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run Loop 2 meta-optimizer")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max optimization iterations")
    parser.add_argument("-c", "--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("-s", "--suite", default="benchmarks/suite.yaml", help="Benchmark suite")
    parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM clients")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    config = PipelineConfig.from_yaml(args.config)

    llm_client = None
    vision_client = None
    if not args.mock:
        from src.inference.factory import create_llm_client, create_vision_client
        try:
            llm_client = create_llm_client(config)
            vision_client = create_vision_client(config)
        except Exception:
            print("Warning: Could not create inference clients. Using mock mode.")
            args.mock = True

    optimizer = MetaOptimizer(
        config=config,
        llm_client=llm_client,
        vision_client=vision_client,
        use_mock=args.mock,
        experiments_dir=args.experiments_dir,
        benchmark_suite=args.suite,
    )

    result = optimizer.run(max_iterations=args.max_iterations)

    print(f"\nOptimization complete:")
    print(f"  Run ID: {result['run_id']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Baseline score: {result['baseline_score']:.4f}")
    print(f"  Best score: {result['best_score']:.4f}")
    print(f"  Improvement: {result['improvement']:.4f}")
    print(f"  Converged: {result['converged']}")


if __name__ == "__main__":
    main()
