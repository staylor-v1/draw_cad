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
    improve.add_argument("--source-fidelity-threshold", type=float)
    improve.add_argument("--feature-match-threshold", type=float)
    improve.add_argument("--summary-only", action="store_true", help="Print compact per-case PI loop results")

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
    if getattr(args, "model", None):
        config = Gemma4RoundTripConfig(**{**config.__dict__, "model": args.model})
    if getattr(args, "base_url", None):
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
                source_fidelity_threshold=args.source_fidelity_threshold,
                feature_match_threshold=args.feature_match_threshold,
            ),
        )
        if args.summary_only:
            loop_result = _compact_loop_result(loop_result)
        print(json.dumps(loop_result, indent=2))
        return

    result = agent.run_roundtrip(args.drawing, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))

def _compact_loop_result(loop_result: dict) -> dict:
    history = []
    for iteration in loop_result.get("history", []):
        cases = []
        for case in iteration.get("cases", []):
            source = case.get("source_fidelity") or {}
            criteria = case.get("success_criteria") or {}
            cases.append(
                {
                    "source_drawing": case.get("source_drawing"),
                    "success": case.get("success"),
                    "roundtrip_equivalent": case.get("roundtrip_equivalent"),
                    "used_fallback": case.get("used_fallback"),
                    "source_fidelity": {
                        "overall_score": source.get("overall_score"),
                        "feature_match": source.get("feature_match"),
                        "major_errors": source.get("major_errors"),
                        "missing_features": source.get("missing_features"),
                        "original_evaluation_path": source.get("original_evaluation_path"),
                    },
                    "success_criteria": {
                        "passed": criteria.get("passed"),
                        "source_fidelity_passed": criteria.get("source_fidelity_passed"),
                        "feature_match_passed": criteria.get("feature_match_passed"),
                    },
                    "first_step_path": case.get("first_step_path"),
                    "second_step_path": case.get("second_step_path"),
                    "source_contact_sheet_path": (case.get("rendered_drawing") or {}).get("source_contact_sheet_path"),
                }
            )
        history.append(
            {
                "iteration": iteration.get("iteration"),
                "success_rate": iteration.get("success_rate"),
                "successes": iteration.get("successes"),
                "total": iteration.get("total"),
                "failure_patterns": iteration.get("failure_patterns"),
                "cases": cases,
            }
        )
    return {
        "loop_config": loop_result.get("loop_config"),
        "profile_notes": loop_result.get("profile_notes"),
        "history": history,
    }


if __name__ == "__main__":
    main()
