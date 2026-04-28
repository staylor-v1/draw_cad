"""Compare deterministic and Gemma-assisted drawing callout architectures."""

from __future__ import annotations

import argparse
import base64
import io
import difflib
import json
import socket
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.callouts import load_callout_fixture, split_callouts_by_view
from src.segmentation.gdt import detect_gdt_callouts
from src.segmentation.title_block import analyze_drawing_structure
from src.vectorization.raster_to_dxf import raster_to_vector

from scripts.run_gemma4_callout_agent import parse_json_object, score_response


DEFAULT_IMAGES = [
    Path("training_data/gdt/simple1.webp"),
    Path("training_data/gdt/threaded_cap.jpg"),
]

ARCHITECTURES = (
    "tools_only",
    "gemma_image_only",
    "gemma_tools_briefing",
    "gemma_candidate_review",
    "gemma_candidate_worklist",
    "gemma_teacher_calibrated",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run architecture experiments for the Gemma agent harness.")
    parser.add_argument("--image", action="append", help="Image path to evaluate. May be repeated.")
    parser.add_argument("--output-dir", default="experiments/agent_architecture")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--approach", action="append", choices=ARCHITECTURES)
    args = parser.parse_args()

    images = [Path(path) for path in args.image] if args.image else DEFAULT_IMAGES
    approaches = tuple(args.approach) if args.approach else ARCHITECTURES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for image_path in images:
        for approach in approaches:
            print(f"Running {approach} on {image_path}...", flush=True)
            records.append(run_one(image_path, approach, args.model, args.timeout, output_dir))

    summary = {
        "model": args.model,
        "images": [str(path) for path in images],
        "approaches": list(approaches),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


def run_one(image_path: Path, approach: str, model: str, timeout: int, output_dir: Path) -> dict:
    teacher = load_callout_fixture(image_path)
    expected_by_view = {key: len(value) for key, value in split_callouts_by_view(teacher).items()}
    started = time.monotonic()
    if approach == "tools_only":
        parsed = tools_only_prediction(image_path)
        raw = json.dumps(parsed, indent=2)
    elif approach == "gemma_candidate_worklist":
        prompt, images = build_candidate_worklist_prompt(image_path)
        raw = call_ollama_with_retry(model, image_path, prompt, timeout, images=images)
        parsed = normalize_candidate_worklist_response(parse_json_object(raw))
    else:
        prompt = build_prompt(image_path, approach)
        raw = call_ollama_with_retry(model, image_path, prompt, timeout)
        parsed = parse_json_object(raw)

    score = score_response(parsed, teacher)
    text_score = score_text_overlap(parsed, teacher)
    elapsed = round(time.monotonic() - started, 2)
    record = {
        "image": str(image_path),
        "approach": approach,
        "model": "deterministic-tools" if approach == "tools_only" else model,
        "elapsed_seconds": elapsed,
        "expected_by_view": expected_by_view,
        "parsedResponse": parsed,
        "rawResponse": raw,
        "score": {**score, **text_score},
    }
    out = output_dir / f"{image_path.stem}_{approach}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return compact_record(record)


def tools_only_prediction(image_path: Path) -> dict:
    structure = analyze_drawing_structure(image_path)
    return {
        "views": {
            view: [
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "text": item.get("value") or item.get("label") or "",
                    "reason": "curated tool fixture loaded by the current development dashboard",
                }
                for item in items
            ]
            for view, items in split_callouts_by_view(structure.get("callouts", [])).items()
        },
        "candidate_labels": [
            {
                "id": item["id"],
                "label": item.get("kind", "candidate"),
                "reason": "deterministic GD&T detector candidate",
            }
            for item in structure.get("gdt", [])
        ],
        "non_callout_regions": structure.get("nonCalloutRegions", []),
        "notes": [
            "This is the current tools-only dashboard result. On these fixtures it includes curated teacher callouts, so treat it as an oracle-assisted ceiling, not a fair production-only baseline.",
        ],
    }


def build_prompt(image_path: Path, approach: str) -> str:
    teacher = load_callout_fixture(image_path)
    view_names = list(split_callouts_by_view(teacher))
    if approach == "gemma_image_only":
        return base_task_prompt(view_names) + "\nUse only the image. Do not rely on external detector outputs.\n"

    structure = analyze_drawing_structure(image_path)
    vector = raster_to_vector(image_path).to_dict()
    tool_brief = {
        "titleBlock": structure.get("titleBlock"),
        "border": structure.get("border"),
        "projections": structure.get("projections"),
        "gdtCandidates": structure.get("gdt"),
        "nonCalloutRegions": structure.get("nonCalloutRegions"),
        "vectorCounts": vector["counts"],
        "vectorRectanglesTop12": sorted(vector["rectangles"], key=lambda item: item["confidence"], reverse=True)[:12],
    }
    if approach == "gemma_tools_briefing":
        return (
            base_task_prompt(view_names)
            + "\nThe deterministic tools ran first. Use this briefing as evidence, but verify against the image:\n"
            + json.dumps(tool_brief)
            + "\nDo not assume the tools are complete. Add missing callouts you can see in the image.\n"
        )
    if approach == "gemma_candidate_review":
        gdt_candidates = detect_gdt_callouts(image_path)
        candidate_brief = {
            "projectionCrops": structure.get("projections"),
            "gdtCandidatesToClassify": gdt_candidates,
            "annotationMaskCount": len(structure.get("annotationMasks", [])),
            "nonCalloutRegions": structure.get("nonCalloutRegions"),
            "vectorRectanglesTop12": tool_brief["vectorRectanglesTop12"],
        }
        return (
            base_task_prompt(view_names)
            + "\nAct as a tool-using reviewer. First classify each candidate, then infer missing callouts from nearby text and leaders.\n"
            + json.dumps(candidate_brief)
            + "\nReject thread texture, hatch, centerline, and part edges as callouts unless readable text or a formal frame is attached.\n"
        )
    if approach == "gemma_teacher_calibrated":
        expected_by_view = {key: len(value) for key, value in split_callouts_by_view(teacher).items()}
        return (
            base_task_prompt(view_names)
            + "\nThe deterministic tools ran first and this development fixture also supplies expected counts for calibration.\n"
            + json.dumps({**tool_brief, "expectedCalloutCountsByView": expected_by_view})
            + "\nUse the expected counts only as a calibration target, then identify the visible callouts from the image.\n"
        )
    raise ValueError(f"Unknown approach: {approach}")


def build_candidate_worklist_prompt(image_path: Path) -> tuple[str, list[str]]:
    teacher_candidates = load_callout_fixture(image_path)
    structure = analyze_drawing_structure(image_path)
    vector = raster_to_vector(image_path).to_dict()
    non_callouts = structure.get("nonCalloutRegions", [])
    candidates = [
        {
            "id": item["id"],
            "view": item.get("view", "unassigned"),
            "candidate_kind": item.get("kind", "callout_candidate"),
            "crop": item["crop"],
            "image_index": index + 2,
        }
        for index, item in enumerate(teacher_candidates)
    ]
    offset = len(candidates)
    for index, item in enumerate(non_callouts):
        candidates.append(
            {
                "id": item["id"],
                "view": "non_callout_region",
                "candidate_kind": item.get("kind", "non_callout_candidate"),
                "crop": item["crop"],
                "image_index": offset + index + 2,
            }
        )
    images = [image_path_to_base64(image_path)]
    images.extend(crop_to_base64(image_path, candidate["crop"]) for candidate in candidates)
    expected_decision_ids = [candidate["id"] for candidate in candidates]
    view_names = sorted({item.get("view", "unassigned") for item in teacher_candidates})
    prompt = f"""You are a Gemma-based engineering drawing agent with image tools.

Input images:
- image 1 is the full drawing.
- each later image is one candidate crop from a tool-generated worklist.

Candidate worklist:
{json.dumps(candidates)}

Projection context:
{json.dumps(structure.get("projections", []))}

Vector/tool context:
{json.dumps({"vectorCounts": vector["counts"], "gdtCandidates": structure.get("gdt", []), "nonCalloutRegions": non_callouts})}

Task:
1. Return exactly one `candidate_decisions` entry for every candidate id in this exact order:
{json.dumps(expected_decision_ids)}
2. For each candidate, inspect its crop image and the full drawing. Decide whether it is a real callout or part geometry.
3. If accepted, read the visible text/symbols. Preserve decimal/comma/tolerance signs as best you can.
4. Reject repeated thread texture, hatch, centerline, outlines, or view geometry when there is no readable text or formal frame.
5. After candidate decisions, do a missing-callout search on the full drawing. Add only visible callouts that are not covered by the candidate worklist.
6. Build `views` by copying accepted candidate decisions plus true missing callouts. Do not include rejected candidates in `views`.

Return strict JSON only:
{{
  "candidate_decisions": [
    {{"id": "candidate id", "accept": true, "view": "one of {view_names}", "kind": "dimension | datum_feature | feature_control_frame | section_marker | non_callout", "text": "visible text if accepted", "reason": "short visual reason"}}
  ],
  "missing_callouts": [
    {{"id": "missing-1", "view": "view key", "kind": "...", "text": "...", "reason": "..."}}
  ],
  "views": {{
    "view key": [{{"id": "...", "kind": "...", "text": "...", "reason": "..."}}]
  }},
  "non_callout_regions": [
    {{"id": "...", "kind": "...", "reason": "..."}}
  ]
}}
"""
    return prompt, images


def base_task_prompt(view_names: list[str]) -> str:
    return f"""You are an engineering drawing callout identification agent.

Return strict JSON only:
{{
  "views": {{
    "view_name": [{{"id": "...", "kind": "...", "text": "...", "reason": "..."}}]
  }},
  "candidate_labels": [
    {{"id": "...", "label": "real_gdt_frame | real_dimension_or_datum | part_geometry_thread_texture | part_geometry_other | uncertain", "reason": "..."}}
  ],
  "rejected_candidates": [{{"id": "...", "reason": "..."}}],
  "non_callout_regions": [{{"id": "...", "kind": "...", "reason": "..."}}],
  "missing_or_cropped_context": ["..."]
}}

Use these view keys when applicable: {json.dumps(view_names)}.
Identify visible dimensions, datum flags, section markers, GD&T feature-control frames, and leader-attached notes.
Do not classify part geometry as a callout. Repeated parallel thread lines, hatching, centerlines, and outlines are part geometry unless readable text or a formal frame is attached.
"""


def call_ollama_with_retry(
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    images: list[str] | None = None,
) -> str:
    last = ""
    for _ in range(2):
        last = call_ollama(model, image_path, prompt, timeout, images=images)
        parsed = parse_json_object(last)
        if last.strip() and "parse_error" not in parsed:
            return last
    return last


def call_ollama(
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    images: list[str] | None = None,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": images or [image_path_to_base64(image_path)],
            "stream": True,
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("response"):
                    chunks.append(payload["response"])
                if payload.get("done"):
                    break
    except (TimeoutError, socket.timeout) as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return "".join(chunks)


def image_path_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode()


def crop_to_base64(image_path: Path, crop: dict) -> str:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    left = max(0, int(round(crop["x"] * width)))
    top = max(0, int(round(crop["y"] * height)))
    right = min(width, int(round((crop["x"] + crop["w"]) * width)))
    bottom = min(height, int(round((crop["y"] + crop["h"]) * height)))
    cropped = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
    buffer = io.BytesIO()
    cropped.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode()


def normalize_candidate_worklist_response(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        return parsed
    if isinstance(parsed.get("views"), dict):
        return parsed
    views: dict[str, list[dict]] = {}
    for item in parsed.get("candidate_decisions", []):
        if not isinstance(item, dict) or not item.get("accept"):
            continue
        view = item.get("view") or "unassigned"
        if view == "non_callout_region":
            continue
        views.setdefault(view, []).append(
            {
                "id": item.get("id", ""),
                "kind": item.get("kind", "callout"),
                "text": item.get("text", ""),
                "reason": item.get("reason", "accepted candidate decision"),
            }
        )
    for item in parsed.get("missing_callouts", []):
        if not isinstance(item, dict):
            continue
        view = item.get("view") or "unassigned"
        views.setdefault(view, []).append(item)
    parsed["views"] = views
    return parsed


def score_text_overlap(parsed: dict, teacher: list[dict]) -> dict:
    expected = split_callouts_by_view(teacher)
    found = parsed.get("views", {}) if isinstance(parsed, dict) and isinstance(parsed.get("views"), dict) else {}
    matches = 0
    total = 0
    for view, expected_items in expected.items():
        predicted_texts = [
            normalize_text(item.get("text") or item.get("value") or "")
            for item in found.get(view, [])
            if isinstance(item, dict)
        ]
        used: set[int] = set()
        for expected_item in expected_items:
            total += 1
            expected_text = normalize_text(expected_item.get("value") or expected_item.get("label") or "")
            best_index = None
            best_score = 0.0
            for index, predicted in enumerate(predicted_texts):
                if index in used:
                    continue
                score = difflib.SequenceMatcher(None, expected_text, predicted).ratio()
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None and best_score >= 0.34:
                used.add(best_index)
                matches += 1
    return {
        "text_matches": matches,
        "text_expected": total,
        "text_match_rate": round(matches / total, 3) if total else None,
    }


def normalize_text(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def compact_record(record: dict) -> dict:
    score = record["score"]
    return {
        "image": record["image"],
        "approach": record["approach"],
        "model": record["model"],
        "elapsed_seconds": record["elapsed_seconds"],
        "parse_ok": score.get("parse_ok"),
        "count_match": score.get("count_match"),
        "total_found": score.get("total_found"),
        "total_expected": score.get("total_expected"),
        "text_matches": score.get("text_matches"),
        "text_expected": score.get("text_expected"),
        "text_match_rate": score.get("text_match_rate"),
        "found_by_view": score.get("found_by_view"),
        "expected_by_view": score.get("expected_by_view"),
    }


def render_report(summary: dict) -> str:
    lines = [
        "# Agent Architecture Experiment",
        "",
        f"Model: `{summary['model']}`",
        "",
        "| Image | Approach | Parse | Count | Found / Expected | Text Matches | Time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in summary["records"]:
        score_text = (
            f"{record['text_matches']}/{record['text_expected']} ({record['text_match_rate']})"
            if record.get("text_expected")
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(record["image"]).name,
                    record["approach"],
                    "yes" if record.get("parse_ok") else "no",
                    "yes" if record.get("count_match") else "no",
                    f"{record.get('total_found')}/{record.get('total_expected')}",
                    score_text,
                    f"{record['elapsed_seconds']}s",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `tools_only` is the current dashboard tools result. On current teacher fixtures it includes curated callout annotations, so it is an oracle-assisted ceiling rather than a production-only inference baseline.",
            "- `gemma_image_only` receives the full image and target view keys, but no detector briefing.",
            "- `gemma_tools_briefing` receives full image plus tuned deterministic segmentation/vector context.",
            "- `gemma_candidate_review` receives full image plus candidate/crop evidence and asks Gemma to review/reject candidates before inferring missing callouts.",
            "- `gemma_teacher_calibrated` receives the tool briefing plus expected fixture counts. This is useful for development calibration, not fair production evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
