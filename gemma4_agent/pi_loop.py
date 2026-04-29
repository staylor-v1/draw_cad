"""PI-style iterative loop for drawing-to-CAD improvement."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemma4_agent.agent import Gemma4RoundTripAgent, Gemma4RoundTripConfig


@dataclass(frozen=True)
class PiLoopConfig:
    max_iterations: int = 3
    min_success_rate: float = 0.8


def run_pi_loop(
    drawing_paths: list[str | Path],
    output_dir: str | Path,
    agent_config: Gemma4RoundTripConfig,
    loop_config: PiLoopConfig | None = None,
) -> dict[str, Any]:
    loop = loop_config or PiLoopConfig()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    profile_notes: list[str] = []
    history: list[dict[str, Any]] = []
    agent = Gemma4RoundTripAgent(agent_config)

    for iteration in range(1, loop.max_iterations + 1):
        iter_dir = root / f"iteration_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        case_results: list[dict[str, Any]] = []
        for drawing in drawing_paths:
            drawing_path = Path(drawing)
            result = agent.run_roundtrip(drawing_path, output_dir=iter_dir / drawing_path.stem)
            case_results.append(result)

        successes = sum(1 for r in case_results if r.get("success"))
        success_rate = successes / max(len(case_results), 1)
        failure_patterns = _summarize_failure_patterns(case_results)
        notes = (
            f"Iteration {iteration}: success_rate={success_rate:.3f}. "
            f"Top failure patterns: {failure_patterns or 'none'}"
        )
        profile_notes.append(notes)
        snapshot = {
            "iteration": iteration,
            "success_rate": success_rate,
            "successes": successes,
            "total": len(case_results),
            "failure_patterns": failure_patterns,
            "notes": notes,
            "cases": case_results,
        }
        history.append(snapshot)
        (iter_dir / "summary.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        if success_rate >= loop.min_success_rate:
            break

    final = {
        "loop_config": {"max_iterations": loop.max_iterations, "min_success_rate": loop.min_success_rate},
        "agent_config": agent_config.__dict__,
        "profile_notes": profile_notes,
        "history": history,
    }
    (root / "pi_loop_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def _summarize_failure_patterns(results: list[dict[str, Any]]) -> list[str]:
    patterns: dict[str, int] = {}
    for result in results:
        if result.get("success"):
            continue
        if result.get("used_fallback"):
            patterns["fallback_geometry"] = patterns.get("fallback_geometry", 0) + 1
        if not result.get("roundtrip_equivalent"):
            patterns["roundtrip_not_equivalent"] = patterns.get("roundtrip_not_equivalent", 0) + 1
        source = ((result.get("first_stage") or {}).get("code_source")) or "no_final_code"
        patterns[f"code_source:{source}"] = patterns.get(f"code_source:{source}", 0) + 1
    return [f"{name}={count}" for name, count in sorted(patterns.items(), key=lambda kv: kv[1], reverse=True)]
