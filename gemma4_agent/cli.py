"""Command line entrypoints for the Gemma 4 CAD roundtrip agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_agent.agent import Gemma4RoundTripAgent, Gemma4RoundTripConfig
from gemma4_agent.pi_loop import PiLoopConfig, run_pi_loop
from gemma4_agent.toolbox import (
    compare_cad_parts,
    get_tool_instructions,
    get_tool_schemas,
    render_step_to_drawing,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 drawing/CAD/drawing roundtrip agent")
    parser.add_argument("--config", default="gemma4_agent/config.yaml", help="Agent YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    roundtrip = subparsers.add_parser("roundtrip", help="Run drawing -> STEP -> drawing -> STEP")
    roundtrip.add_argument("drawing", help="Input drawing path")
    roundtrip.add_argument("--output-dir", help="Output directory")
    roundtrip.add_argument("--model", help="Ollama model override")
    roundtrip.add_argument("--base-url", help="Ollama base URL override")

    subparsers.add_parser("tools", help="Print Gemma tool instructions and schemas")
    subparsers.add_parser("schemas", help="Print raw Ollama tool schemas")

    render = subparsers.add_parser("render-step", help="Render a STEP file to orthographic drawings")
    render.add_argument("step", help="STEP file path")
    render.add_argument("--output-dir", default="experiments/gemma4_agent/rendered")
    render.add_argument("--stem", default="rendered")

    compare = subparsers.add_parser("compare", help="Compare two STEP parts")
    compare.add_argument("reference_step")
    compare.add_argument("candidate_step")
    improve = subparsers.add_parser("improve", help="Run PI-style iterative improvement loop")
    improve.add_argument("drawings", nargs="+", help="Input drawing paths")
    improve.add_argument("--output-dir", default="experiments/gemma4_agent/pi_loop")
    improve.add_argument("--max-iterations", type=int, default=3)
    improve.add_argument("--target-success-rate", type=float, default=0.8)

    args = parser.parse_args()

    if args.command == "tools":
        print(get_tool_instructions())
        return
    if args.command == "schemas":
        print(json.dumps(get_tool_schemas(), indent=2))
        return
    if args.command == "render-step":
        result = render_step_to_drawing(args.step, args.output_dir, stem=args.stem)
        print(json.dumps(result, indent=2))
        return
    if args.command == "compare":
        result = compare_cad_parts(args.reference_step, args.candidate_step)
        print(json.dumps(result, indent=2))
        return

    config = Gemma4RoundTripConfig.from_yaml(args.config)
    if args.model:
        config = Gemma4RoundTripConfig(**{**config.__dict__, "model": args.model})
    if args.base_url:
        config = Gemma4RoundTripConfig(**{**config.__dict__, "base_url": args.base_url})

    agent = Gemma4RoundTripAgent(config)
    if args.command == "improve":
        loop_result = run_pi_loop(
            drawing_paths=args.drawings,
            output_dir=args.output_dir,
            agent_config=config,
            loop_config=PiLoopConfig(
                max_iterations=args.max_iterations,
                min_success_rate=args.target_success_rate,
            ),
        )
        print(json.dumps(loop_result, indent=2))
        return

    result = agent.run_roundtrip(args.drawing, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
