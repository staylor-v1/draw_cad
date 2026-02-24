"""CLI entry point for the drawing-to-CAD conversion pipeline."""
import argparse
import sys
import os

from agent_loop import run_agent_loop


def main():
    parser = argparse.ArgumentParser(
        description="Convert engineering drawings to 3D CAD models (STEP files)"
    )
    parser.add_argument("image", nargs="?", help="Path to engineering drawing image")
    parser.add_argument("output", nargs="?", default="output.step", help="Output STEP file path")
    parser.add_argument("-c", "--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument("--mock", action="store_true", help="Force mock inference mode")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark suite instead of single image"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run meta-optimizer (Loop 2)"
    )
    parser.add_argument("--max-iterations", type=int, default=20, help="Max optimizer iterations")
    args = parser.parse_args()

    if args.benchmark:
        _run_benchmark(args)
    elif args.optimize:
        _run_optimizer(args)
    elif args.image:
        _run_single(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_single(args):
    """Run pipeline on a single drawing."""
    if not os.path.exists(args.image):
        print(f"Error: File '{args.image}' not found.")
        sys.exit(1)

    result = run_agent_loop(args.image, args.output)
    if result and not result.success:
        sys.exit(1)


def _run_benchmark(args):
    """Run benchmark suite."""
    from src.schemas.pipeline_config import PipelineConfig
    from src.evaluation.benchmark_runner import BenchmarkRunner
    from src.evaluation.report import generate_report
    from src.utils.logging_config import setup_logging

    setup_logging()
    config = PipelineConfig.from_yaml(args.config)

    runner = BenchmarkRunner(config=config, use_mock=args.mock)
    report = runner.run_suite("benchmarks/suite.yaml")

    os.makedirs("experiments/benchmark_output", exist_ok=True)
    summary = generate_report(report, "experiments/benchmark_output/report.json")
    print("\n" + summary)


def _run_optimizer(args):
    """Run meta-optimizer."""
    from src.schemas.pipeline_config import PipelineConfig
    from src.optimizer.meta_optimizer import MetaOptimizer
    from src.utils.logging_config import setup_logging

    setup_logging()
    config = PipelineConfig.from_yaml(args.config)

    optimizer = MetaOptimizer(config=config, use_mock=args.mock)
    result = optimizer.run(max_iterations=args.max_iterations)

    print(f"\nOptimization complete:")
    print(f"  Best score: {result['best_score']:.4f}")
    print(f"  Improvement: {result['improvement']:.4f}")


if __name__ == "__main__":
    main()
