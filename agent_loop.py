"""Backward-compatible agent loop wrapper around the new PipelineRunner.

This module preserves the original run_agent_loop() API for backward
compatibility while delegating to the new three-layer pipeline system.
"""
import json
import os

from src.schemas.pipeline_config import PipelineConfig
from src.pipeline.runner import PipelineRunner


def run_agent_loop(image_path: str, output_step: str = "output.step"):
    """Run the drawing-to-CAD pipeline on a single image.

    This is a thin wrapper around PipelineRunner for backward compatibility
    with the original CLI interface.

    Args:
        image_path: Path to the engineering drawing image.
        output_step: Path for the output STEP file.
    """
    print(f"Starting CAD Agent Loop for {image_path}...")

    # Load config (use defaults if config file doesn't exist)
    config_path = "config/default.yaml"
    if os.path.exists(config_path):
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    config.output.default_step_path = output_step

    # Try to create real inference clients, fall back to mock
    llm_client = None
    vision_client = None
    use_mock = True

    try:
        from src.inference.factory import create_llm_client, create_vision_client
        llm_client = create_llm_client(config)
        vision_client = create_vision_client(config)
        # Test connectivity
        if llm_client.health_check() and vision_client.health_check():
            use_mock = False
        else:
            print("Warning: Model backends not reachable. Using mock mode.")
            llm_client = None
            vision_client = None
    except Exception:
        print("Using mock inference mode.")

    runner = PipelineRunner(
        config=config,
        llm_client=llm_client,
        vision_client=vision_client,
        use_mock=use_mock,
    )

    result = runner.run(image_path=image_path, output_path=output_step)

    if result.success:
        print(f"Success! STEP file generated at {result.step_file}")
        print(f"Composite score: {result.metrics.composite_score:.4f}")
    else:
        print(f"Execution Failed: {result.error}")

    return result


if __name__ == "__main__":
    if not os.path.exists("test_drawing.png"):
        with open("test_drawing.png", "w") as f:
            f.write("dummy image content")

    run_agent_loop("test_drawing.png")
