"""Training-loop helpers for Gemma 4 mechanical drawing reconstruction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from gemma4_agent.toolbox import encode_image_for_ollama


SOURCE_FIDELITY_KEYS = (
    "overall_score",
    "main_part_match",
    "view_consistency",
    "feature_match",
    "dimension_plausibility",
    "annotation_filtering",
)

DEFAULT_INITIAL_SOURCE_FIDELITY_THRESHOLD = 0.72
DEFAULT_TARGET_SOURCE_FIDELITY_THRESHOLD = 0.99


SOURCE_FIDELITY_SYSTEM_PROMPT = """You are an independent mechanical drawing fidelity judge.

Compare the original engineering drawing with the generated orthographic contact sheet.
Judge whether the generated part represents the physical manufactured part in the source
drawing. Ignore title blocks, borders, notes, dimension text, GD&T frames, paper styling,
and scan artifacts as geometry. Penalize missing or invented physical features such as
holes, bosses, slots, cutouts, ribs, shafts, flanges, chamfers, fillets, steps, and
revolved profiles. A plain envelope box or slab that only matches the sheet or bounding
box must receive an overall_score below 0.35 even if it roundtrips perfectly.

Return strict JSON only. Do not include markdown."""


SOURCE_FIDELITY_USER_PROMPT = """The first image is the original source drawing.
The second image is the generated front/right/top contact sheet rendered from the STEP.

Return this JSON shape:
{
  "overall_score": 0.0,
  "main_part_match": 0.0,
  "view_consistency": 0.0,
  "feature_match": 0.0,
  "dimension_plausibility": 0.0,
  "annotation_filtering": 0.0,
  "major_errors": [],
  "missing_features": [],
  "spurious_geometry": [],
  "actionable_prompt_feedback": []
}

Use scores from 0.0 to 1.0. Be strict about source drawing resemblance."""


PROFILE_REVISION_SYSTEM_PROMPT = """You improve a CAD reconstruction agent prompt.

Given the current agent profile and failed training records, produce a complete revised
agent profile in Markdown. Keep the profile concise and operational. Add rules that
would have prevented the observed failures. Do not mention benchmark scores or the
training harness in the profile."""


def judge_source_fidelity(
    *,
    model: str,
    base_url: str,
    original_drawing_path: str | Path,
    generated_contact_sheet_path: str | Path,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Ask local Gemma to judge source drawing fidelity from two images."""
    original_b64, original_mime = encode_image_for_ollama(original_drawing_path)
    generated_b64, generated_mime = encode_image_for_ollama(generated_contact_sheet_path)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SOURCE_FIDELITY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SOURCE_FIDELITY_USER_PROMPT,
                "images": [original_b64, generated_b64],
            },
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    api_url = f"{base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=600) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data.get("message", {}).get("content", "")
    parsed = normalize_source_fidelity(extract_json_object(content))
    parsed["raw_content"] = content
    parsed["image_mime_types"] = {
        "original": original_mime,
        "generated": generated_mime,
    }
    return parsed


def normalize_source_fidelity(value: dict[str, Any]) -> dict[str, Any]:
    """Clamp and complete a source-fidelity judge response."""
    normalized: dict[str, Any] = {}
    for key in SOURCE_FIDELITY_KEYS:
        normalized[key] = _clamp_score(value.get(key, 0.0))
    for key in ("major_errors", "missing_features", "spurious_geometry", "actionable_prompt_feedback"):
        items = value.get(key, [])
        if isinstance(items, list):
            normalized[key] = [str(item) for item in items]
        elif items:
            normalized[key] = [str(items)]
        else:
            normalized[key] = []
    return normalized


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        loaded = json.loads(cleaned[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("Expected a JSON object")
    return loaded


def training_case_passed(
    *,
    roundtrip_summary: dict[str, Any],
    source_fidelity: dict[str, Any],
    source_fidelity_threshold: float,
    feature_match_threshold: float | None = None,
) -> dict[str, Any]:
    """Return pass/fail criteria for one drawing training case."""
    feature_threshold = (
        source_fidelity_threshold if feature_match_threshold is None else feature_match_threshold
    )
    roundtrip_equivalent = bool(
        roundtrip_summary.get("roundtrip_equivalent", roundtrip_summary.get("success"))
    )
    no_fallback = not bool(roundtrip_summary.get("used_fallback"))
    source_score = float(source_fidelity.get("overall_score", 0.0))
    feature_score = float(source_fidelity.get("feature_match", 0.0))
    passed = (
        roundtrip_equivalent
        and no_fallback
        and source_score >= source_fidelity_threshold
        and feature_score >= feature_threshold
    )
    return {
        "passed": passed,
        "roundtrip_equivalent": roundtrip_equivalent,
        "no_fallback_geometry": no_fallback,
        "source_fidelity_checked": True,
        "source_fidelity_passed": source_score >= source_fidelity_threshold,
        "feature_match_passed": feature_score >= feature_threshold,
        "source_fidelity_threshold": source_fidelity_threshold,
        "feature_match_threshold": feature_threshold,
    }


def threshold_for_iteration(
    *,
    iteration: int,
    initial_threshold: float = DEFAULT_INITIAL_SOURCE_FIDELITY_THRESHOLD,
    target_threshold: float = DEFAULT_TARGET_SOURCE_FIDELITY_THRESHOLD,
    threshold_step: float = 0.05,
) -> float:
    """Return the curriculum threshold for an iteration, capped at the target."""
    initial = _clamp_score(initial_threshold)
    target = _clamp_score(target_threshold)
    if target < initial:
        return target
    step = max(0.0, float(threshold_step))
    return min(target, initial + max(0, iteration) * step)


def build_profile_revision_prompt(
    *,
    current_profile: str,
    failed_records: list[dict[str, Any]],
    passed_records: list[dict[str, Any]] | None = None,
) -> str:
    """Build a text prompt asking Gemma to revise the agent profile."""
    compact_failures = [_compact_record(record) for record in failed_records]
    compact_passes = [_compact_record(record) for record in (passed_records or [])]
    return json.dumps(
        {
            "task": "Revise the agent profile to improve drawing-to-CAD reconstruction.",
            "current_profile": current_profile,
            "failed_records": compact_failures,
            "passed_records": compact_passes,
            "requirements": [
                "Preserve rules that are still correct.",
                "Add concrete reconstruction tactics for the observed drawing failures.",
                "Reject envelope-only or title-block-derived geometry.",
                "Use drawing_evidence fields when present, but avoid copying annotation text into geometry.",
                "Keep the profile directly usable as gemma4_agent/prompts/agent.md.",
            ],
        },
        indent=2,
    )


def propose_profile_revision(
    *,
    model: str,
    base_url: str,
    current_profile: str,
    failed_records: list[dict[str, Any]],
    passed_records: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """Ask local Gemma to produce a revised agent profile."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROFILE_REVISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_profile_revision_prompt(
                    current_profile=current_profile,
                    failed_records=failed_records,
                    passed_records=passed_records,
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    api_url = f"{base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=600) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
    return _strip_markdown_fence(data.get("message", {}).get("content", "")).strip()


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source_fidelity") or {}
    criteria = record.get("success_criteria") or {}
    return {
        "case": record.get("case"),
        "passed": record.get("success"),
        "criteria": criteria,
        "source_fidelity": {
            key: source.get(key)
            for key in (
                "overall_score",
                "main_part_match",
                "view_consistency",
                "feature_match",
                "dimension_plausibility",
                "annotation_filtering",
            )
        },
        "major_errors": source.get("major_errors", []),
        "missing_features": source.get("missing_features", []),
        "spurious_geometry": source.get("spurious_geometry", []),
        "actionable_prompt_feedback": source.get("actionable_prompt_feedback", []),
        "drawing_evidence": {
            key: (record.get("drawing_evidence") or {}).get(key, [])
            for key in (
                "physical_features",
                "dimensions",
                "gd_t",
                "annotation_regions",
                "title_block_or_sheet_regions",
                "reconstruction_hints",
                "uncertainties",
            )
        },
        "roundtrip_metrics": (record.get("comparison") or {}).get("metrics", {}),
    }


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped
