"""Use Gemma 4 as a supervised callout-identification agent."""

from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.request
from pathlib import Path

from src.segmentation.callouts import load_callout_fixture, split_callouts_by_view
from src.segmentation.title_block import analyze_drawing_structure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 4 on a drawing callout-identification task.")
    parser.add_argument("image")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--output-dir", default="experiments/gemma4_callout_agent")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tool_context = analyze_drawing_structure(image_path)
    teacher = load_callout_fixture(image_path)
    response = call_gemma(args.model, image_path, tool_context)
    parsed = parse_json_object(response)
    score = score_response(parsed, teacher)

    record = {
        "image": str(image_path),
        "model": args.model,
        "toolContext": {
            "projectionCount": len(tool_context["projections"]),
            "gdtCount": len(tool_context["gdt"]),
            "teacherCalloutCount": len(teacher),
            "teacherCountsByView": {key: len(value) for key, value in split_callouts_by_view(teacher).items()},
        },
        "rawResponse": response,
        "parsedResponse": parsed,
        "score": score,
    }
    out = output_dir / f"{image_path.stem}_callout_agent.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record["score"], indent=2))
    print(f"Wrote {out}")


def call_gemma(model: str, image_path: Path, tool_context: dict) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    prompt = f"""You are a tool-using engineering drawing callout agent.

The deterministic tools found:
- titleBlock.present={tool_context['titleBlock']['present']}
- border.present={tool_context['border']['present']}
- projection_count={len(tool_context['projections'])}
- gdt_candidate_count={len(tool_context['gdt'])}

Task:
Identify every visible callout attached to the two visible projections.
The left projection is front_view. The right projection is side_view.
The image is cropped from a larger drawing; do not invent the missing C projection.

Return strict JSON only:
{{
  "front_view_callouts": [{{"id": "...", "kind": "...", "text": "...", "reason": "..."}}],
  "side_view_callouts": [{{"id": "...", "kind": "...", "text": "...", "reason": "..."}}],
  "missing_or_cropped_context": ["..."]
}}

Expected visible count from the teacher tool: 10 front callouts and 3 side callouts.
"""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read()).get("response", "")


def parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"parse_error": "no JSON object found"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "raw": text}


def score_response(parsed: dict, teacher: list[dict]) -> dict:
    expected = split_callouts_by_view(teacher)
    front = parsed.get("front_view_callouts", []) if isinstance(parsed, dict) else []
    side = parsed.get("side_view_callouts", []) if isinstance(parsed, dict) else []
    return {
        "front_expected": len(expected.get("front", [])),
        "front_found": len(front),
        "side_expected": len(expected.get("side", [])),
        "side_found": len(side),
        "count_match": len(front) == len(expected.get("front", [])) and len(side) == len(expected.get("side", [])),
        "parse_ok": "parse_error" not in parsed,
    }


if __name__ == "__main__":
    main()
