#!/usr/bin/env python3
"""Run the Loop 3 pipeline on a single drawing."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.pipeline_config import PipelineConfig
from src.pipeline.runner import PipelineRunner
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run drawing-to-CAD pipeline")
    parser.add_argument("image", help="Path to engineering drawing image")
    parser.add_argument("-o", "--output", default="output.step", help="Output STEP file path")
    parser.add_argument("-c", "--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM clients")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    config = PipelineConfig.from_yaml(args.config)

    # Create clients
    llm_client = None
    vision_client = None
    if not args.mock:
        from src.inference.factory import create_llm_client, create_vision_client
        try:
            llm_client = create_llm_client(config)
            vision_client = create_vision_client(config)
        except Exception as e:
            print(f"Warning: Could not create inference clients: {e}")
            print("Falling back to mock mode.")
            args.mock = True

    runner = PipelineRunner(
        config=config,
        llm_client=llm_client,
        vision_client=vision_client,
        use_mock=args.mock,
    )

    result = runner.run(image_path=args.image, output_path=args.output)

    if result.success:
        print(f"\nSuccess! STEP file: {result.step_file}")
        print(f"Composite score: {result.metrics.composite_score:.4f}")
    else:
        print(f"\nFailed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
