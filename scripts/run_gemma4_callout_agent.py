"""Use Gemma 4 as a supervised callout-identification agent."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

from src.segmentation.callouts import load_callout_fixture, split_callouts_by_view
from src.segmentation.title_block import analyze_drawing_structure
from src.vectorization.raster_to_dxf import raster_to_vector


TEACHER_PROMPT_PATH = Path("prompts/gemma4_gdt_callout_teacher.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 4 on a drawing callout-identification task.")
    parser.add_argument("image")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--output-dir", default="experiments/gemma4_callout_agent")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for the local Gemma response.")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable Ollama streaming and wait for one complete response body.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Optional Ollama num_predict limit. Omit for the model default.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tool_context = analyze_drawing_structure(image_path)
    teacher = load_callout_fixture(image_path)
    vector_context = raster_to_vector(image_path)
    response = call_gemma(
        args.model,
        image_path,
        tool_context,
        vector_context.to_dict(),
        teacher,
        args.timeout,
        not args.no_stream,
        args.num_predict,
    )
    parsed = parse_json_object(response)
    score = score_response(parsed, teacher)

    record = {
        "image": str(image_path),
        "model": args.model,
        "toolContext": {
            "projectionCount": len(tool_context["projections"]),
            "gdtCount": len(tool_context["gdt"]),
            "vectorCounts": vector_context.to_dict()["counts"],
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


def call_gemma(
    model: str,
    image_path: Path,
    tool_context: dict,
    vector_context: dict,
    teacher: list[dict],
    timeout: int,
    stream: bool,
    num_predict: int | None,
) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    teacher_rubric = TEACHER_PROMPT_PATH.read_text(encoding="utf-8") if TEACHER_PROMPT_PATH.exists() else ""
    gdt_candidates = [
        {
            "id": item["id"],
            "kind": item["kind"],
            "confidence": item["confidence"],
            "crop": item["crop"],
        }
        for item in tool_context["gdt"]
    ]
    vector_rectangles = [
        {
            "kind": item["kind"],
            "confidence": item["confidence"],
            "crop": item["crop"],
            "source": item["source"],
        }
        for item in sorted(vector_context["rectangles"], key=lambda entry: entry["confidence"], reverse=True)[:12]
    ]
    expected_by_view = {key: len(value) for key, value in split_callouts_by_view(teacher).items()}
    view_names = [projection.get("axis") or projection.get("id") for projection in tool_context["projections"]]
    non_callout_regions = tool_context.get("nonCalloutRegions", [])
    prompt = f"""You are a tool-using engineering drawing callout agent.

The deterministic tools found:
- titleBlock.present={tool_context['titleBlock']['present']}
- border.present={tool_context['border']['present']}
- projection_count={len(tool_context['projections'])}
- gdt_candidate_count={len(tool_context['gdt'])}
- vector_segment_count={vector_context['counts']['segments']}
- vector_rectangle_count={vector_context['counts']['rectangles']}
- segmented_gdt_candidates={json.dumps(gdt_candidates)}
- vector_rectangles_top12={json.dumps(vector_rectangles)}
- expected_callout_counts_by_view={json.dumps(expected_by_view)}
- projection_view_names={json.dumps(view_names)}
- negative_non_callout_regions={json.dumps(non_callout_regions)}

Teacher rubric:
{teacher_rubric}

Task:
Identify every visible callout attached to the visible projections.
Use the vector rectangles as teacher-tool evidence for feature-control frames
and boxed/basic dimensions, but still verify against the image before assigning
text.
For each segmented_gdt_candidate, decide whether it is a real callout or a
false candidate. Reject large boxes around repeated thread lines or hatching.
Do not classify part geometry as a callout. In particular, repeated parallel
thread lines, hatching, and part edge texture inside a projection are negative
examples unless there is nearby readable text or a formal boxed frame.

Return strict JSON only:
{{
  "views": {{
    "view_name": [{{"id": "...", "kind": "...", "text": "...", "reason": "..."}}]
  }},
  "candidate_labels": [
    {{"id": "gdt-1", "label": "real_gdt_frame | real_dimension_or_datum | part_geometry_thread_texture | part_geometry_other | uncertain", "reason": "..."}}
  ],
  "rejected_candidates": [{{"id": "...", "reason": "..."}}],
  "non_callout_regions": [{{"id": "...", "kind": "...", "reason": "..."}}],
  "missing_or_cropped_context": ["..."]
}}

Expected visible counts from the teacher tool are given above. Match those counts
when a teacher fixture exists.
"""
    options = {"temperature": 0}
    if num_predict is not None:
        options["num_predict"] = num_predict
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": stream,
            "options": options,
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if not stream:
            return json.loads(response.read()).get("response", "")
        chunks = []
        for line in response:
            if not line.strip():
                continue
            payload = json.loads(line)
            chunk = payload.get("response", "")
            if chunk:
                chunks.append(chunk)
                if len(chunks) % 25 == 0:
                    print(f"received {len(chunks)} response chunks...", file=sys.stderr, flush=True)
            if payload.get("done"):
                break
        return "".join(chunks)


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
    found_by_view = _parsed_views(parsed)
    count_by_view = {key: len(value) for key, value in found_by_view.items()}
    expected_counts = {key: len(value) for key, value in expected.items()}
    return {
        "expected_by_view": expected_counts,
        "found_by_view": count_by_view,
        "total_expected": sum(expected_counts.values()),
        "total_found": sum(count_by_view.values()),
        "count_match": count_by_view == expected_counts,
        "parse_ok": "parse_error" not in parsed,
    }


def _parsed_views(parsed: dict) -> dict[str, list[dict]]:
    if not isinstance(parsed, dict):
        return {}
    if isinstance(parsed.get("views"), dict):
        return {
            str(view): items if isinstance(items, list) else []
            for view, items in parsed["views"].items()
        }
    legacy = {}
    if isinstance(parsed.get("front_view_callouts"), list):
        legacy["front"] = parsed["front_view_callouts"]
    if isinstance(parsed.get("side_view_callouts"), list):
        legacy["side"] = parsed["side_view_callouts"]
    return legacy


if __name__ == "__main__":
    main()
