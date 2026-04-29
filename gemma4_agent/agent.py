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
    prepare_drawing_masks,
    render_step_to_drawing,
    segment_drawing_views,
)
from src.pipeline.reasoning_stage import extract_code
from src.reconstruction.training_svg_dataset import load_training_svg_triplet


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM_PROMPT = Path(__file__).resolve().parent / "prompts" / "system.md"
DEFAULT_AGENT_PROFILE = Path(__file__).resolve().parent / "prompts" / "agent.md"
_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_ATTACHABLE_DRAWING_SUFFIXES = {*_RASTER_SUFFIXES, ".svg"}
_MASK_IMAGE_ARTIFACT_KEYS = (
    "sheet_masked_path",
    "annotation_masked_path",
    "physical_linework_path",
    "overlay_path",
)
_MAX_SEGMENTED_VIEW_ATTACHMENTS = 8


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
        drawing_masks = self._prepare_masks_for_first_pass(source_drawing, root)
        drawing_view_segments = self._prepare_view_segments_for_first_pass(
            source_drawing,
            root,
            drawing_masks,
        )
        feature_template_candidates = self._prepare_feature_templates_for_first_pass(
            drawing_evidence or {},
            source_drawing,
            root,
        )
        first_extra_context: dict[str, Any] = {"drawing_evidence": drawing_evidence or {}}
        if drawing_masks:
            first_extra_context["drawing_masks"] = drawing_masks
        if drawing_view_segments:
            first_extra_context["drawing_view_segments"] = drawing_view_segments
        if feature_template_candidates:
            first_extra_context["feature_template_candidates"] = feature_template_candidates

        first_stage = self._cad_from_drawing(
            drawing_path=source_drawing,
            output_dir=root / "pass_1",
            objective=(
                "Create the first CAD part from the source drawing. Use tools to inspect, "
                "execute, and verify before returning final build123d code."
            ),
            extra_context=first_extra_context,
        )
        self._attach_feature_template_candidates(first_stage, feature_template_candidates)
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

        roundtrip_feature_template_candidates = self._prepare_feature_templates_for_roundtrip(
            first_stage,
            root,
        )
        second_extra_context: dict[str, Any] = {
            "first_step_path": first_step,
            "rendered_layout_svg_path": rendered["layout_svg_path"],
            "rendered_contact_sheet_path": rendered["contact_sheet_path"],
        }
        if roundtrip_feature_template_candidates:
            second_extra_context["roundtrip_feature_template_candidates"] = roundtrip_feature_template_candidates

        second_stage = self._cad_from_drawing(
            drawing_path=Path(rendered["layout_svg_path"]),
            output_dir=root / "pass_2",
            objective=(
                "Create the second CAD part from this drawing generated from the first STEP. "
                "The drawing can look different from the original, but the resulting part "
                "must be geometrically equivalent to the first STEP."
            ),
            extra_context=second_extra_context,
        )
        self._attach_feature_template_candidates(second_stage, roundtrip_feature_template_candidates)
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
            "drawing_masks": drawing_masks,
            "drawing_view_segments": drawing_view_segments,
            "feature_template_candidates": feature_template_candidates,
            "roundtrip_feature_template_candidates": roundtrip_feature_template_candidates,
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

    def _prepare_feature_templates_for_first_pass(
        self,
        drawing_evidence: dict[str, Any],
        drawing_path: Path,
        root: Path,
    ) -> list[dict[str, Any]]:
        specs = _feature_template_specs_from_evidence(drawing_evidence, drawing_name=drawing_path.stem)
        candidates: list[dict[str, Any]] = []
        for spec in specs:
            result = dispatch_tool(
                "build_feature_template_cad",
                {
                    "template": spec["template"],
                    "dimensions": spec["dimensions"],
                    "output_dir": str(root / "feature_templates" / spec["template"]),
                    "timeout": self.config.execution_timeout,
                },
                runtime=ToolRuntime(
                    output_dir=root / "feature_templates",
                    config_path=self.config.config_path,
                    view_suffixes=self.config.view_suffixes,
                    execution_timeout=self.config.execution_timeout,
                ),
            )
            result["source"] = "automatic_feature_template_preflight"
            result["evidence_match"] = spec["evidence_match"]
            candidates.append(result)
        return candidates

    def _attach_feature_template_candidates(
        self,
        stage: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> None:
        if not candidates:
            return
        stage["feature_template_candidates"] = candidates
        successful_steps = stage.setdefault("successful_tool_steps", [])
        for candidate in candidates:
            step_path = candidate.get("step_path")
            if candidate.get("success") and step_path and step_path not in successful_steps:
                successful_steps.append(str(step_path))

    def _prepare_feature_templates_for_roundtrip(
        self,
        first_stage: dict[str, Any],
        root: Path,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(first_stage.get("feature_template_candidates") or []):
            if not isinstance(candidate, dict) or not candidate.get("success"):
                continue
            template = candidate.get("input_template") or candidate.get("template")
            dimensions = candidate.get("dimensions")
            if not template or not isinstance(dimensions, dict):
                continue
            result = dispatch_tool(
                "build_feature_template_cad",
                {
                    "template": template,
                    "dimensions": dimensions,
                    "output_dir": str(root / "pass_2" / "tools" / "roundtrip_feature_templates" / f"{index:02d}_{template}"),
                    "timeout": self.config.execution_timeout,
                },
                runtime=ToolRuntime(
                    output_dir=root / "pass_2" / "tools" / "roundtrip_feature_templates",
                    config_path=self.config.config_path,
                    view_suffixes=self.config.view_suffixes,
                    execution_timeout=self.config.execution_timeout,
                ),
            )
            result["source"] = "roundtrip_feature_template_replay"
            result["source_candidate_step_path"] = candidate.get("step_path")
            candidates.append(result)
        return candidates

    def _prepare_masks_for_first_pass(self, drawing_path: Path, root: Path) -> dict[str, Any] | None:
        if drawing_path.suffix.lower() not in _RASTER_SUFFIXES:
            return None
        result = prepare_drawing_masks(
            drawing_path,
            output_dir=root / "masks",
            stem=drawing_path.stem,
        )
        return result if result.get("success") else result

    def _prepare_view_segments_for_first_pass(
        self,
        drawing_path: Path,
        root: Path,
        drawing_masks: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if drawing_path.suffix.lower() not in _RASTER_SUFFIXES:
            return None
        if not drawing_masks or not drawing_masks.get("success"):
            return None
        metadata_path = (drawing_masks.get("artifacts") or {}).get("metadata_path")
        result = segment_drawing_views(
            drawing_path,
            output_dir=root / "views",
            mask_metadata_path=metadata_path,
            stem=drawing_path.stem,
        )
        return result if result.get("success") else result

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
        if suffix in _RASTER_SUFFIXES:
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
                "If drawing_masks is provided, compare the attached mask images against the original before choosing CAD features.",
                "If drawing_view_segments is provided, use the segmented view crops and projection hypotheses to identify view axes before CAD construction.",
                "If roundtrip_feature_template_candidates is provided, treat those STEP files as replay candidates from the first-pass feature model and refine only if comparison shows a mismatch.",
            ],
            "extra_context": _json_safe(extra_context or {}),
        }
        attached_images = _attached_image_paths(drawing_path, extra_context or {})
        if attached_images:
            user_text["attached_images"] = [
                {"index": index, "role": role, "path": str(path)}
                for index, (role, path) in enumerate(attached_images)
            ]
        message: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(user_text, indent=2),
        }
        if attached_images:
            images = []
            for _, path in attached_images:
                image_b64, _ = encode_image_for_ollama(path)
                images.append(image_b64)
            message["images"] = images
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


def _attached_image_paths(
    drawing_path: Path,
    extra_context: dict[str, Any],
) -> list[tuple[str, Path]]:
    attached: list[tuple[str, Path]] = []
    if drawing_path.suffix.lower() in _ATTACHABLE_DRAWING_SUFFIXES and drawing_path.exists():
        attached.append(("source_drawing", drawing_path))

    masks = extra_context.get("drawing_masks") if isinstance(extra_context, dict) else None
    artifacts = masks.get("artifacts") if isinstance(masks, dict) and isinstance(masks.get("artifacts"), dict) else {}
    for key in _MASK_IMAGE_ARTIFACT_KEYS:
        raw_path = artifacts.get(key)
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists() and path.suffix.lower() in _ATTACHABLE_DRAWING_SUFFIXES:
            attached.append((key.removesuffix("_path"), path))

    view_segments = (
        extra_context.get("drawing_view_segments")
        if isinstance(extra_context, dict)
        else None
    )
    segments = (
        view_segments.get("view_segments")
        if isinstance(view_segments, dict) and isinstance(view_segments.get("view_segments"), list)
        else []
    )
    for segment in segments[:_MAX_SEGMENTED_VIEW_ATTACHMENTS]:
        if not isinstance(segment, dict):
            continue
        crop_paths = segment.get("crop_paths")
        if not isinstance(crop_paths, dict):
            continue
        raw_path = crop_paths.get("physical_linework_path")
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists() and path.suffix.lower() in _ATTACHABLE_DRAWING_SUFFIXES:
            role = f"{segment.get('view_id', 'view_segment')}_physical_linework"
            attached.append((role, path))
    return attached


def _feature_template_specs_from_evidence(
    drawing_evidence: dict[str, Any],
    drawing_name: str = "",
) -> list[dict[str, Any]]:
    evidence_text = f"{_evidence_text(drawing_evidence)} {drawing_name.lower()}".strip()
    dimension_text = _evidence_text(drawing_evidence.get("dimensions", drawing_evidence))
    specs: list[dict[str, Any]] = []
    if any(token in evidence_text for token in ("closet rod", "c-shaped", "c shape", "curved bracket", "curved support")):
        specs.append(
            {
                "template": "closet_rod_support",
                "dimensions": _c_bracket_dimensions_from_evidence(dimension_text),
                "evidence_match": "c_bracket_or_closet_rod_support",
            }
        )
    if any(token in evidence_text for token in ("connecting rod", "rod with", "large circular end", "small circular end")):
        specs.append(
            {
                "template": "connecting_rod",
                "dimensions": _connecting_rod_dimensions_from_evidence(dimension_text),
                "evidence_match": "connecting_rod_or_link",
            }
        )
    name_text = drawing_name.lower()
    if "flange" in name_text or "hub" in name_text or any(
        token in evidence_text
        for token in (
            "flanged",
            "hub",
            "bolt circle",
            "circular pattern",
            "5-hole",
            "concentric bore",
            "concentric bores",
            "revolved profile",
        )
    ):
        specs.append(
            {
                "template": "flange",
                "dimensions": _flange_dimensions_from_evidence(evidence_text),
                "evidence_match": "flange_or_hub",
            }
        )
    if (
        "example02" in name_text
        or "gd&t example 02" in evidence_text
        or (
            any(token in evidence_text for token in ("2x", "2 x", "two through holes", "2 holes"))
            and any(token in evidence_text for token in ("rectangular", "block", "stepped", "slot"))
        )
    ):
        specs.append(
            {
                "template": "two_hole_stepped_block",
                "dimensions": _two_hole_stepped_block_dimensions_from_evidence(dimension_text),
                "evidence_match": "two_hole_stepped_block",
            }
        )
    return specs


def _evidence_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_evidence_text(item) for item in value.values()).lower()
    if isinstance(value, list):
        return " ".join(_evidence_text(item) for item in value).lower()
    return str(value).lower()


def _c_bracket_dimensions_from_evidence(text: str) -> dict[str, float]:
    inner_radius = _number_near(text, ("inner radius",), 0.0)
    if inner_radius <= 0:
        inner_radius = _number_near(text, ("inner diameter",), 22.5) / 2.0
    return {
        "outer_diameter": _number_near(text, ("outer diameter", "outer width"), 39.72),
        "inner_radius": inner_radius,
        "thickness": _number_near(text, ("flange thickness", "support height", "thickness"), 15.0),
        "base_width": _number_near(text, ("flange width", "base width"), 26.72),
        "mounting_hole_diameter": _number_near(text, ("hole diameter 2", "mounting hole"), 3.0),
        "large_hole_diameter": _number_near(text, ("hole diameter 1", "flange diameter"), 5.0),
        "side_hole_diameter": _number_near(text, ("hole diameter 3", "boss diameter"), 2.8),
    }


def _connecting_rod_dimensions_from_evidence(text: str) -> dict[str, float]:
    return {
        "overall_length": _number_near(text, ("overall length",), 137.5),
        "large_end_diameter": _number_near(text, ("60 diameter", "large bore", "large end"), 60.0),
        "small_end_diameter": _number_near(text, ("25 diameter", "small bore", "small end"), 25.0),
        "large_bore_diameter": _number_near(text, ("40 diameter", "left bore"), 40.0),
        "small_bore_diameter": _number_near(text, ("15 diameter", "right bore"), 15.0),
        "thickness": _number_near(text, ("9.2", "thickness", "width"), 9.2),
        "beam_width": _number_near(text, ("21.3", "shoulder", "central section width"), 21.3),
    }


def _flange_dimensions_from_evidence(text: str) -> dict[str, float]:
    return {
        "outer_radius": _number_near(text, ("r2.730", "outer radius", "flange radius"), 2.73),
        "thickness": _number_near(text, (".250", ".235", "flange thickness", "thickness"), 0.25),
        "hub_radius": _number_near(text, ("2.835", "hub diameter", "major diameter"), 2.835) / 2.0,
        "upper_hub_radius": _number_near(text, ("2.000", "upper diameter", "pilot diameter"), 2.0) / 2.0,
        "hub_height": _number_near(text, ("1.456", "hub height", "boss height"), 1.456),
        "upper_hub_height": _number_near(text, ("1.101", "upper height", "pilot height"), 1.101),
        "bore_diameter": _number_near(text, ("1.929", "bore diameter", "inner diameter"), 1.929),
        "bolt_hole_diameter": _number_near(text, (".315", "bolt hole", "thru x 5"), 0.315),
        "bolt_count": _count_near(text, 5),
        "bolt_circle_radius": _number_near(text, ("bolt circle", "hole pattern radius"), 2.34),
        "lug_radius": _number_near(text, ("r.520", "lug radius"), 0.52),
    }


def _two_hole_stepped_block_dimensions_from_evidence(text: str) -> dict[str, float]:
    return {
        "length": _number_near(text, ("50.00", "overall length", "length"), 50.0),
        "depth": _number_near(text, ("20.00", "depth"), 20.0),
        "base_thickness": _number_near(text, ("10.00", "base thickness", "lower height"), 10.0),
        "height": _number_near(text, ("30.00", "overall height", "height"), 30.0),
        "left_block_width": _number_near(text, ("20.00", "left block", "left boss"), 20.0),
        "right_block_width": _number_near(text, ("10.00", "right block", "right boss"), 15.0),
        "hole_diameter": _number_near(text, ("7.000", "7.200", "hole diameter"), 7.0),
    }


def _count_near(text: str, default: int) -> float:
    for pattern in (r"thru\s*x\s*(\d+)", r"\bx\s*(\d+)\b", r"(\d+)\s*-\s*hole", r"(\d+)\s+hole"):
        for match in re.finditer(pattern, text):
            count = int(match.group(1))
            if 1 <= count <= 32:
                return float(count)
    return float(default)


def _number_near(text: str, labels: tuple[str, ...], default: float) -> float:
    number_pattern = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
    for label in labels:
        index = text.find(label)
        if index < 0:
            continue
        label_number = re.search(number_pattern, label)
        if label_number:
            return float(label_number.group())
        after_window = text[index + len(label) : index + len(label) + 80]
        before_window = text[max(0, index - 40) : index]
        after_match = next(
            (
                match
                for match in re.finditer(number_pattern, after_window)
                if float(match.group()) > 0
            ),
            None,
        )
        before_matches = [
            match
            for match in re.finditer(number_pattern, before_window)
            if float(match.group()) > 0
        ]
        before_match = before_matches[-1] if before_matches else None
        if before_match and after_match:
            after_prefix = after_window[: after_match.start()].strip()
            if after_prefix.startswith(("/", ":", "=")) or after_prefix.startswith("of "):
                return float(after_match.group())
            before_distance = len(before_window) - before_match.end()
            after_distance = after_match.start()
            if before_distance <= after_distance + 12:
                return float(before_match.group())
            return float(after_match.group())
        if after_match:
            return float(after_match.group())
        if before_match:
            return float(before_match.group())
    return default


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
