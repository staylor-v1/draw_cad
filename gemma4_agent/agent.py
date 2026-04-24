"""Local Ollama Gemma 4 agent for drawing/CAD/drawing roundtrips."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from PIL import Image

from gemma4_agent.toolbox import (
    ToolRuntime,
    compare_cad_parts,
    dispatch_tool,
    encode_image_for_ollama,
    execute_cad_code,
    get_tool_schemas,
    render_step_to_drawing,
)
from src.pipeline.reasoning_stage import extract_code
from src.reconstruction.training_svg_dataset import load_training_svg_triplet


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM_PROMPT = Path(__file__).resolve().parent / "prompts" / "system.md"
DEFAULT_AGENT_PROFILE = Path(__file__).resolve().parent / "prompts" / "agent.md"


@dataclass(frozen=True)
class Gemma4RoundTripConfig:
    """Configuration for the local Ollama roundtrip agent."""

    model: str = "gemma4:26b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_tool_rounds: int = 6
    execution_timeout: int = 60
    config_path: Path = REPO_ROOT / "config" / "total_view_data.yaml"
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT
    agent_profile_path: Path = DEFAULT_AGENT_PROFILE
    output_dir: Path = REPO_ROOT / "experiments" / "gemma4_agent"
    view_suffixes: tuple[str, str, str] = ("f", "r", "t")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Gemma4RoundTripConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        agent = data.get("agent", data)
        return cls(
            model=agent.get("model", cls.model),
            base_url=agent.get("base_url", cls.base_url),
            temperature=float(agent.get("temperature", cls.temperature)),
            max_tokens=int(agent.get("max_tokens", cls.max_tokens)),
            max_tool_rounds=int(agent.get("max_tool_rounds", cls.max_tool_rounds)),
            execution_timeout=int(agent.get("execution_timeout", cls.execution_timeout)),
            config_path=Path(agent.get("config_path", cls.config_path)),
            system_prompt_path=Path(agent.get("system_prompt_path", cls.system_prompt_path)),
            agent_profile_path=Path(agent.get("agent_profile_path", cls.agent_profile_path)),
            output_dir=Path(agent.get("output_dir", cls.output_dir)),
            view_suffixes=tuple(agent.get("view_suffixes", cls.view_suffixes)),  # type: ignore[arg-type]
        )


class Gemma4RoundTripAgent:
    """Gemma 4 tool-calling agent that tests drawing-to-CAD roundtrip stability."""

    def __init__(self, config: Gemma4RoundTripConfig | None = None):
        self.config = config or Gemma4RoundTripConfig()
        self.api_url = f"{self.config.base_url.rstrip('/')}/api/chat"

    def run_roundtrip(
        self,
        drawing_path: str | Path,
        output_dir: str | Path | None = None,
        drawing_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run drawing -> STEP -> drawing -> STEP and compare the two parts."""
        root = Path(output_dir) if output_dir else self.config.output_dir / _run_id()
        root.mkdir(parents=True, exist_ok=True)
        source_drawing = Path(drawing_path)

        first_stage = self._cad_from_drawing(
            drawing_path=source_drawing,
            output_dir=root / "pass_1",
            objective=(
                "Create the first CAD part from the source drawing. Use tools to inspect, "
                "execute, and verify before returning final build123d code."
            ),
            extra_context={"drawing_evidence": drawing_evidence or {}},
        )
        first_step = self._materialize_stage_step(
            stage=first_stage,
            output_dir=root / "pass_1",
            fallback_name="pass_1_final.step",
        )

        rendered = render_step_to_drawing(
            first_step,
            output_dir=root / "rendered_from_pass_1",
            stem="pass_1_reprojected",
            view_suffixes=self.config.view_suffixes,
        )

        second_stage = self._cad_from_drawing(
            drawing_path=Path(rendered["layout_svg_path"]),
            output_dir=root / "pass_2",
            objective=(
                "Create the second CAD part from this drawing generated from the first STEP. "
                "The drawing can look different from the original, but the resulting part "
                "must be geometrically equivalent to the first STEP."
            ),
            extra_context={
                "first_step_path": first_step,
                "rendered_layout_svg_path": rendered["layout_svg_path"],
                "rendered_contact_sheet_path": rendered["contact_sheet_path"],
            },
        )
        second_step = self._materialize_stage_step(
            stage=second_stage,
            output_dir=root / "pass_2",
            fallback_name="pass_2_final.step",
        )
        second_step = self._select_best_matching_step(
            reference_step=first_step,
            stage=second_stage,
            default_step=second_step,
        )

        comparison = compare_cad_parts(first_step, second_step)
        roundtrip_equivalent = bool(comparison["equivalent"])
        used_fallback = _stage_used_fallback(first_stage) or _stage_used_fallback(second_stage)
        summary = {
            "success": roundtrip_equivalent and not used_fallback,
            "roundtrip_equivalent": roundtrip_equivalent,
            "used_fallback": used_fallback,
            "success_criteria": {
                "roundtrip_equivalent": roundtrip_equivalent,
                "no_fallback_geometry": not used_fallback,
                "source_fidelity_checked": False,
                "source_fidelity_passed": None,
            },
            "output_dir": str(root),
            "config": _json_safe(asdict(self.config)),
            "source_drawing": str(source_drawing),
            "drawing_evidence": drawing_evidence or {},
            "first_stage": first_stage,
            "first_step_path": first_step,
            "rendered_drawing": rendered,
            "second_stage": second_stage,
            "second_step_path": second_step,
            "comparison": comparison,
        }
        (root / "roundtrip_summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2),
            encoding="utf-8",
        )
        return _json_safe(summary)

    def _cad_from_drawing(
        self,
        drawing_path: Path,
        output_dir: Path,
        objective: str,
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime = ToolRuntime(
            output_dir=output_dir / "tools",
            config_path=self.config.config_path,
            view_suffixes=self.config.view_suffixes,
            execution_timeout=self.config.execution_timeout,
        )
        messages = self._initial_messages(drawing_path, objective, extra_context)
        transcript: list[dict[str, Any]] = []
        successful_steps: list[str] = []
        final_text = ""
        final_thinking = ""
        blank_response_repairs = 0

        for _ in range(self.config.max_tool_rounds):
            response = self._ollama_chat(messages=messages, tools=get_tool_schemas())
            assistant_message = response.get("message", {})
            messages.append(assistant_message)
            transcript.append({"assistant": assistant_message, "raw": response})

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                final_text = assistant_message.get("content", "")
                final_thinking = assistant_message.get("thinking", "")
                if (
                    not final_text.strip()
                    and assistant_message.get("thinking")
                    and not extract_code(final_thinking)
                    and blank_response_repairs < 1
                ):
                    blank_response_repairs += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response contained internal analysis but no visible "
                                "tool call or final code. Continue now with an actual tool call. "
                                "For a new drawing, call inspect_drawing first. If inspection is "
                                "already complete, call execute_cad_code with a simple valid "
                                "build123d model that defines part."
                            ),
                        }
                    )
                    continue
                break

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                tool_result = dispatch_tool(tool_name, function.get("arguments"), runtime=runtime)
                if tool_name in {"execute_cad_code", "run_deterministic_reconstruction"}:
                    step_path = tool_result.get("step_path") or tool_result.get("selected_step_path")
                else:
                    step_path = None
                if tool_result.get("success") and step_path:
                    successful_steps.append(str(step_path))
                transcript.append({"tool": tool_name, "result": tool_result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=True),
                    }
                )

        code = extract_code(final_text)
        code_source = "assistant_content" if code else None
        if not code and final_thinking:
            code = extract_code(final_thinking)
            if code:
                code_source = "assistant_thinking"
        result = {
            "drawing_path": str(drawing_path),
            "output_dir": str(output_dir),
            "final_text": final_text,
            "code": code,
            "code_source": code_source,
            "successful_tool_steps": successful_steps,
            "transcript_path": str(output_dir / "transcript.json"),
        }
        (output_dir / "transcript.json").write_text(
            json.dumps(_json_safe(transcript), indent=2),
            encoding="utf-8",
        )
        return result

    def _materialize_stage_step(
        self,
        stage: dict[str, Any],
        output_dir: Path,
        fallback_name: str,
    ) -> str:
        code = stage.get("code") or ""
        if code:
            result = execute_cad_code(
                code,
                output_step_path=output_dir / fallback_name,
                timeout=self.config.execution_timeout,
            )
            stage["final_execution"] = result
            if result["success"]:
                return str(result["step_path"])
            repaired_code = _repair_common_build123d_code(code)
            if repaired_code != code:
                repaired_result = execute_cad_code(
                    repaired_code,
                    output_step_path=output_dir / fallback_name,
                    timeout=self.config.execution_timeout,
                )
                stage["final_execution_repair"] = {
                    "repair": "common_build123d_builder_idioms",
                    "code": repaired_code,
                    "result": repaired_result,
                }
                if repaired_result["success"]:
                    stage["code"] = repaired_code
                    stage["used_repaired_final_code"] = True
                    return str(repaired_result["step_path"])
            successful = stage.get("successful_tool_steps") or []
            if successful:
                stage["used_tool_step_after_failed_final_code"] = True
                return str(successful[-1])
            fallback = self._fallback_step_from_drawing(stage, output_dir / fallback_name)
            if fallback:
                stage["used_fallback_after_failed_final_code"] = True
                return fallback
            raise RuntimeError(f"Final CAD code did not produce STEP: {result}")

        successful = stage.get("successful_tool_steps") or []
        if successful:
            stage["used_tool_step_without_final_code"] = True
            return str(successful[-1])
        fallback = self._fallback_step_from_drawing(stage, output_dir / fallback_name)
        if fallback:
            return fallback
        raise RuntimeError("Gemma did not return CAD code and no successful tool STEP is available.")

    def _fallback_step_from_drawing(
        self,
        stage: dict[str, Any],
        output_step_path: Path,
    ) -> str | None:
        drawing_path = Path(stage.get("drawing_path") or "")
        code = self._fallback_code_for_drawing(drawing_path)
        if code is None:
            return None
        result = execute_cad_code(
            code,
            output_step_path=output_step_path,
            timeout=self.config.execution_timeout,
        )
        stage["fallback_generation"] = {
            "reason": "no successful Gemma/tool STEP was available",
            "result": result,
        }
        if result["success"]:
            return str(result["step_path"])
        return None

    def _fallback_code_for_drawing(self, drawing_path: Path) -> str | None:
        suffix = drawing_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            with Image.open(drawing_path) as image:
                width = max(float(image.width) / 10.0, 10.0)
                height = max(float(image.height) / 10.0, 10.0)
            thickness = max(min(width, height) * 0.12, 3.0)
            return _diagnostic_envelope_code(
                width,
                height,
                thickness,
                f"fallback from raster {drawing_path.name}",
            )

        if suffix == ".svg":
            try:
                triplet = load_training_svg_triplet(drawing_path, view_suffixes=self.config.view_suffixes)
            except Exception:
                return None
            front, right, top = self.config.view_suffixes
            front_box = triplet.views[front].view_box
            right_box = triplet.views[right].view_box
            top_box = triplet.views[top].view_box
            width = max(float(front_box[2]), float(top_box[2]), 1.0)
            depth = max(float(right_box[2]), float(top_box[3]), 1.0)
            height = max(float(front_box[3]), float(right_box[3]), 1.0)
            return _diagnostic_envelope_code(
                width,
                depth,
                height,
                f"fallback from SVG triplet {drawing_path.name}",
            )

        return None

    def _select_best_matching_step(
        self,
        reference_step: str,
        stage: dict[str, Any],
        default_step: str,
    ) -> str:
        candidates = list(dict.fromkeys([*stage.get("successful_tool_steps", []), default_step]))
        comparisons: list[dict[str, Any]] = []
        best_step = default_step
        best_score: tuple[float, float, float, float] | None = None
        for candidate in candidates:
            try:
                comparison = compare_cad_parts(reference_step, candidate)
            except Exception as exc:
                comparisons.append(
                    {
                        "candidate_step_path": str(candidate),
                        "success": False,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    }
                )
                continue

            metrics = comparison["metrics"]
            score = (
                float(metrics["bbox_extent_ratio"]),
                float(metrics["volume_ratio"]),
                float(metrics["surface_area_ratio"]),
                float(metrics["face_count_ratio"]),
                -float(metrics["center_distance"]),
            )
            comparisons.append(
                {
                    "candidate_step_path": str(candidate),
                    "equivalent": comparison["equivalent"],
                    "metrics": metrics,
                }
            )
            if best_score is None or score > best_score:
                best_score = score
                best_step = str(candidate)
            if comparison["equivalent"]:
                best_step = str(candidate)
                break

        stage["candidate_comparisons"] = comparisons
        if best_step != default_step:
            stage["selected_by_comparison"] = best_step
        return best_step

    def _initial_messages(
        self,
        drawing_path: Path,
        objective: str,
        extra_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        system_prompt = self.config.system_prompt_path.read_text(encoding="utf-8")
        if self.config.agent_profile_path.exists():
            system_prompt = (
                f"{system_prompt.rstrip()}\n\n"
                f"{self.config.agent_profile_path.read_text(encoding='utf-8').strip()}"
            )
        user_text = {
            "objective": objective,
            "drawing_path": str(drawing_path),
            "requirements": [
                "Return final build123d Python code in a fenced python block.",
                "Use the tools to execute and inspect the CAD before finalizing.",
                "Name the final build123d object part or explicitly export a STEP file.",
                "For raster engineering drawings, reconstruct the main physical part only.",
                "Ignore title blocks, borders, notes, GD&T feature-control frames, and tolerance text as geometry.",
                "If drawing_evidence is provided in extra_context, use it as structured hints but verify against the image.",
            ],
            "extra_context": _json_safe(extra_context or {}),
        }
        message: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(user_text, indent=2),
        }
        if drawing_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"}:
            image_b64, _ = encode_image_for_ollama(drawing_path)
            message["images"] = [image_b64]
        return [
            {"role": "system", "content": system_prompt},
            message,
        ]

    def _ollama_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        with httpx.Client(timeout=600) as client:
            response = client.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _diagnostic_envelope_code(width: float, depth: float, height: float, reason: str) -> str:
    hole_radius = max(min(width, depth) * 0.16, 0.5)
    slot_width = max(width * 0.48, 1.0)
    slot_depth = max(depth * 0.12, 0.5)
    cutter_height = max(height * 1.4, height + 1.0)
    return f'''from build123d import *

# Non-passing baseline generated by the roundtrip harness: {reason}.
# Gemma did not produce an executable model, so this envelope is recorded only
# to keep downstream artifacts inspectable. It must not count as success. The
# diagnostic cutouts keep rendered contact sheets visibly non-empty.
with BuildPart() as baseline:
    Box({width:.6g}, {depth:.6g}, {height:.6g})
    Cylinder({hole_radius:.6g}, {cutter_height:.6g}, mode=Mode.SUBTRACT)
    Box({slot_width:.6g}, {slot_depth:.6g}, {cutter_height:.6g}, mode=Mode.SUBTRACT)

part = baseline.part
'''


def _repair_common_build123d_code(code: str) -> str:
    """Repair frequent BuildPart method hallucinations while keeping code recognizable."""
    repaired = code
    repaired = re.sub(
        r"^(\s*)with\s+([A-Za-z_]\w*)\.location\((.*?)\):\s*$",
        r"\1with Locations(\3):",
        repaired,
        flags=re.MULTILINE,
    )
    repaired = re.sub(
        r"^(\s*)[A-Za-z_]\w*\.add\((Box\(.*\))\)\s*$",
        r"\1\2",
        repaired,
        flags=re.MULTILINE,
    )

    def subtract_repl(match: re.Match[str]) -> str:
        indent = match.group(1)
        primitive = match.group(2)
        args = match.group(3)
        if "mode=" not in args:
            args = f"{args}, mode=Mode.SUBTRACT"
        return f"{indent}{primitive}({args})"

    return re.sub(
        r"^(\s*)[A-Za-z_]\w*\.cut\((Cylinder|Box|Sphere|Cone|Torus)\((.*)\)\)\s*$",
        subtract_repl,
        repaired,
        flags=re.MULTILINE,
    )


def _stage_used_fallback(stage: dict[str, Any]) -> bool:
    return bool(
        stage.get("fallback_generation")
        or stage.get("used_fallback_after_failed_final_code")
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
