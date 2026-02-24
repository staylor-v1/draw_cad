#!/usr/bin/env python3
"""Run the Loop 3 pipeline against the benchmark suite."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.pipeline_config import PipelineConfig
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.report import generate_report
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument("-s", "--suite", default="benchmarks/suite.yaml", help="Benchmark suite YAML")
    parser.add_argument("-o", "--output-dir", default="experiments/benchmark_output", help="Output directory")
    parser.add_argument("-c", "--config", default="config/default.yaml", help="Config file path")
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

    runner = BenchmarkRunner(
        config=config,
        llm_client=llm_client,
        vision_client=vision_client,
        use_mock=args.mock,
    )

    report = runner.run_suite(args.suite, args.output_dir)

    # Generate report
    report_path = Path(args.output_dir) / "report.json"
    summary = generate_report(report, report_path)
    print("\n" + summary)


if __name__ == "__main__":
    main()
