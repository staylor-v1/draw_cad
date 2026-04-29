"""Development web dashboard for drawing-to-CAD inspection workflows."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import tempfile
import uuid
import socket
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from scripts.run_agent_architecture_experiments import build_prompt, call_ollama_with_retry
from scripts.run_gemma4_callout_agent import parse_json_object, score_response
from src.segmentation.callouts import load_callout_fixture, split_callouts_by_view
from src.segmentation.title_block import analyze_drawing_structure
from src.vectorization.raster_to_dxf import raster_to_vector


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web_dashboard" / "static"
RUN_DIR = ROOT / "experiments" / "web_dashboard"
UPLOAD_DIR = RUN_DIR / "uploads"
GRABCAD_REVIEW_DIR = ROOT / "training_data" / "gdt_grabcad"
GRABCAD_CANDIDATE_DIR = GRABCAD_REVIEW_DIR / "orthographic_2d_candidates"
GRABCAD_EXTRACTION_MANIFEST = GRABCAD_REVIEW_DIR / "extraction_manifest.json"
GRABCAD_SELECTION_JSON = GRABCAD_REVIEW_DIR / "selected_training_candidates.json"
GRABCAD_SELECTION_TXT = GRABCAD_REVIEW_DIR / "selected_training_candidates.txt"
GEMMA_MODELS = {"gemma4:26b", "gemma4:e4b"}
ANALYSIS_STRATEGIES = {
    "tools_only",
    "gemma_image_only",
    "gemma_tools_briefing",
    "gemma_candidate_review",
    "gemma_teacher_calibrated",
}


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard and its small JSON API using only stdlib pieces."""

    server_version = "DrawCADDashboard/0.1"

    def translate_path(self, path: str) -> str:
        route = unquote(urlparse(path).path)
        if route == "/":
            return str(STATIC_DIR / "index.html")
        if route == "/training-review":
            return str(STATIC_DIR / "training-review.html")
        if route.startswith("/uploads/"):
            relative = route.removeprefix("/uploads/")
            return str(_contained_path(UPLOAD_DIR, relative))
        if route.startswith("/training-candidates/"):
            relative = route.removeprefix("/training-candidates/")
            return str(_contained_path(GRABCAD_CANDIDATE_DIR, relative))
        return str(_contained_path(STATIC_DIR, route))

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/upload":
            self._handle_upload()
            return
        if route == "/api/analyze":
            self._handle_analyze()
            return
        if route == "/api/training-selection":
            self._handle_training_selection_save()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/training-candidates":
            self._handle_training_candidates()
            return
        if route == "/api/training-selection":
            self._handle_training_selection_get()
            return
        super().do_GET()

    def _handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        if not content_type.startswith("image/") or length <= 0:
            self._send_json({"error": "Expected a non-empty image upload."}, HTTPStatus.BAD_REQUEST)
            return

        original_name = self.headers.get("X-Filename", "drawing")
        filename = _safe_filename(original_name)
        upload_id = uuid.uuid4().hex[:12]
        target_dir = UPLOAD_DIR / upload_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename

        with tempfile.NamedTemporaryFile(delete=False, dir=target_dir) as tmp:
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                tmp.write(chunk)
                remaining -= len(chunk)
            temp_name = tmp.name
        os.replace(temp_name, target)

        self._send_json(
            {
                "id": upload_id,
                "filename": filename,
                "contentType": content_type,
                "size": target.stat().st_size,
                "url": f"/uploads/{upload_id}/{filename}",
            }
        )

    def _handle_analyze(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Expected a JSON analysis request."}, HTTPStatus.BAD_REQUEST)
            return
        filename = payload.get("filename") or "drawing"
        image_url = payload.get("imageUrl") or ""
        mode = payload.get("mode") or "tools"
        gemma_model = _safe_gemma_model(payload.get("gemmaModel"))
        strategy = _safe_analysis_strategy(payload.get("analysisStrategy"), mode)
        mode = "tools" if strategy == "tools_only" else "gemma"
        include_3d = _probably_has_3d_view(filename)
        local_path = _local_upload_path(image_url)
        structure = _analysis_structure_for_path(local_path) if local_path else {}
        analysis = _mock_analysis(
            filename,
            image_url,
            include_3d,
            structure,
            mode=mode,
            gemma_model=gemma_model,
            strategy=strategy,
        )
        if mode == "gemma" and local_path:
            analysis["gemma"] = _run_gemma_runtime(local_path, structure, gemma_model, strategy)
            if analysis["gemma"]["ok"]:
                analysis["status"] = "gemma-analysis"
                analysis["model"] = analysis["gemma"]["model"]
            else:
                analysis["status"] = "gemma-error"
        self._send_json(analysis)

    def _handle_training_candidates(self) -> None:
        self._send_json(
            {
                "root": str(GRABCAD_REVIEW_DIR.relative_to(ROOT)),
                "candidateDir": str(GRABCAD_CANDIDATE_DIR.relative_to(ROOT)),
                "selectionPath": str(GRABCAD_SELECTION_JSON.relative_to(ROOT)),
                "candidates": _load_training_candidates(),
                "selection": _load_training_selection(),
            }
        )

    def _handle_training_selection_get(self) -> None:
        self._send_json(_load_training_selection())

    def _handle_training_selection_save(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Expected a JSON selection payload."}, HTTPStatus.BAD_REQUEST)
            return
        selected = payload.get("selected")
        if not isinstance(selected, list):
            self._send_json({"error": "Expected selected to be a list of candidate paths."}, HTTPStatus.BAD_REQUEST)
            return
        known = {candidate["path"] for candidate in _load_training_candidates()}
        cleaned = sorted({path for path in selected if isinstance(path, str) and path in known})
        GRABCAD_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "selected": cleaned,
            "count": len(cleaned),
            "updatedAt": _utc_timestamp(),
        }
        GRABCAD_SELECTION_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        GRABCAD_SELECTION_TXT.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")
        self._send_json(payload)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str) -> str:
        if path.endswith(".js"):
            return "text/javascript"
        if path.endswith(".css"):
            return "text/css"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _safe_filename(filename: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in filename)
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned or "drawing"


def _contained_path(base: Path, route: str) -> Path:
    safe = posixpath.normpath(route).lstrip("/")
    candidate = (base / safe).resolve()
    root = base.resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    return root / "__not_found__"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_training_candidates() -> list[dict]:
    manifest = {}
    if GRABCAD_EXTRACTION_MANIFEST.exists():
        try:
            manifest = json.loads(GRABCAD_EXTRACTION_MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    manifest_candidates = manifest.get("candidates") or []
    if manifest_candidates:
        candidates = []
        for item in manifest_candidates:
            relative_path = item.get("extracted_path") or ""
            local_path = ROOT / relative_path
            if not local_path.exists() or GRABCAD_CANDIDATE_DIR.resolve() not in local_path.resolve().parents:
                continue
            extension = local_path.suffix.lower().lstrip(".")
            candidates.append(
                {
                    "path": relative_path,
                    "name": local_path.name,
                    "url": f"/training-candidates/{local_path.name}",
                    "sourceArchive": item.get("source_archive"),
                    "modelSlug": item.get("model_slug"),
                    "originalMember": item.get("original_member"),
                    "extension": extension,
                    "sizeBytes": item.get("size_bytes", local_path.stat().st_size),
                    "confidence": item.get("confidence", "unrated"),
                    "notes": item.get("notes", []),
                    "previewKind": _candidate_preview_kind(extension),
                }
            )
        return sorted(candidates, key=lambda item: (item["modelSlug"] or "", item["name"]))

    candidates = []
    for local_path in sorted(GRABCAD_CANDIDATE_DIR.glob("*")):
        if not local_path.is_file():
            continue
        extension = local_path.suffix.lower().lstrip(".")
        candidates.append(
            {
                "path": str(local_path.relative_to(ROOT)),
                "name": local_path.name,
                "url": f"/training-candidates/{local_path.name}",
                "sourceArchive": None,
                "modelSlug": local_path.name.split("__", 1)[0],
                "originalMember": local_path.name,
                "extension": extension,
                "sizeBytes": local_path.stat().st_size,
                "confidence": "unrated",
                "notes": [],
                "previewKind": _candidate_preview_kind(extension),
            }
        )
    return candidates


def _candidate_preview_kind(extension: str) -> str:
    if extension in {"jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"}:
        return "image"
    if extension == "pdf":
        return "pdf"
    return "file"


def _load_training_selection() -> dict:
    if not GRABCAD_SELECTION_JSON.exists():
        return {"selected": [], "count": 0, "updatedAt": None}
    try:
        selection = json.loads(GRABCAD_SELECTION_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"selected": [], "count": 0, "updatedAt": None}
    selected = [path for path in selection.get("selected", []) if isinstance(path, str)]
    return {
        "selected": selected,
        "count": len(selected),
        "updatedAt": selection.get("updatedAt"),
    }


def _probably_has_3d_view(filename: str) -> bool:
    lower = filename.lower()
    return any(token in lower for token in ("iso", "3d", "perspective", "assembly", "rod", "bracket"))


def _safe_gemma_model(model: str | None) -> str:
    if model in GEMMA_MODELS:
        return model
    return os.environ.get("DRAW_CAD_GEMMA_MODEL", "gemma4:26b")


def _safe_analysis_strategy(strategy: str | None, mode: str) -> str:
    if strategy in ANALYSIS_STRATEGIES:
        return strategy
    return "tools_only" if mode == "tools" else "gemma_teacher_calibrated"


def _local_upload_path(image_url: str) -> Path | None:
    route = unquote(urlparse(image_url).path)
    if not route.startswith("/uploads/"):
        return None
    local_path = _contained_path(UPLOAD_DIR, route.removeprefix("/uploads/"))
    if not local_path.exists():
        return None
    return local_path


def _analysis_structure_for_path(local_path: Path | None) -> dict:
    if not local_path:
        return {}
    try:
        return analyze_drawing_structure(local_path)
    except Exception as exc:
        return {
            "titleBlock": {
                "crop": {"x": 0.52, "y": 0.72, "w": 0.43, "h": 0.22},
                "confidence": 0.05,
                "candidate": {"notes": [f"title-block detector failed: {exc}"]},
            }
        }


def _run_gemma_runtime(local_path: Path, structure: dict, model: str, strategy: str) -> dict:
    timeout = int(os.environ.get("DRAW_CAD_GEMMA_TIMEOUT", "300"))
    try:
        teacher = load_callout_fixture(local_path)
        vector = raster_to_vector(local_path)
        prompt = build_prompt(local_path, strategy)
        raw = ""
        parsed: dict = {"parse_error": "Gemma was not called"}
        attempts = 0
        for attempt in range(2):
            attempts = attempt + 1
            raw = call_ollama_with_retry(model, local_path, prompt, timeout)
            parsed = parse_json_object(raw)
            if raw.strip() and "parse_error" not in parsed:
                break
        score = score_response(parsed, teacher)
        return {
            "ok": bool(raw.strip()) and "parse_error" not in parsed,
            "model": model,
            "strategy": strategy,
            "attempts": attempts,
            "error": "empty Gemma response" if not raw.strip() else parsed.get("parse_error"),
            "rawResponse": raw,
            "parsedResponse": parsed,
            "score": score,
            "teacherCalloutCount": len(teacher),
            "teacherCountsByView": {key: len(value) for key, value in split_callouts_by_view(teacher).items()},
            "vectorCounts": vector.to_dict()["counts"],
        }
    except (OSError, ValueError, TimeoutError, socket.timeout, ConnectionError) as exc:
        return {
            "ok": False,
            "model": model,
            "strategy": strategy,
            "error": f"{type(exc).__name__}: {exc}",
            "score": {"parse_ok": False, "count_match": False},
        }


def _mock_analysis(
    filename: str,
    image_url: str,
    include_3d: bool,
    structure: dict | None = None,
    *,
    mode: str = "tools",
    gemma_model: str = "gemma4:26b",
    strategy: str = "tools_only",
) -> dict:
    """Return a stable fixture shaped like the future Gemma 4 analysis output."""

    structure = structure or {}
    title_block = structure.get(
        "titleBlock",
        {
            "present": False,
            "crop": {"x": 0.52, "y": 0.72, "w": 0.43, "h": 0.22},
            "confidence": 0.12,
            "candidate": {"notes": ["lower-right prior fallback"]},
        },
    )
    border = structure.get(
        "border",
        {
            "present": False,
            "confidence": 0.35,
            "masks": [],
            "notes": [
                "no sheet border detected",
                "do not mask border/title-block regions unless another detector finds them",
            ],
        },
    )

    projections = [
        {
            "id": "front",
            "label": "Front projection",
            "axis": "+Y",
            "confidence": 0.84,
            "crop": {"x": 0.08, "y": 0.18, "w": 0.32, "h": 0.44},
        },
        {
            "id": "top",
            "label": "Top projection",
            "axis": "+Z",
            "confidence": 0.72,
            "crop": {"x": 0.08, "y": 0.05, "w": 0.32, "h": 0.14},
        },
        {
            "id": "right",
            "label": "Right projection",
            "axis": "+X",
            "confidence": 0.77,
            "crop": {"x": 0.42, "y": 0.18, "w": 0.25, "h": 0.44},
        },
    ]
    if include_3d:
        projections.append(
            {
                "id": "perspective",
                "label": "Perspective view",
                "axis": "isometric",
                "confidence": 0.62,
                "crop": {"x": 0.66, "y": 0.11, "w": 0.25, "h": 0.35},
            }
        )

    detected_projections = structure.get("projections") or projections
    detected_annotation_masks = structure.get("annotationMasks") or []
    detected_callouts = structure.get("callouts") or [
        {"label": "Diameter callout", "value": "diameter 24.00", "confidence": 0.70},
        {"label": "Linear dimension", "value": "72.00 +/- 0.10", "confidence": 0.73},
        {"label": "Surface finish", "value": "Ra 1.6", "confidence": 0.52},
    ]
    detected_gdt = structure.get("gdt") or [
        {
            "id": "gdt-1",
            "label": "Position tolerance",
            "value": "POS | diameter 0.10 | A | B",
            "confidence": 0.69,
            "crop": {"x": 0.19, "y": 0.61, "w": 0.16, "h": 0.07},
            "symbol": "position",
            "kind": "feature_control_frame",
        },
        {
            "id": "gdt-2",
            "label": "Flatness tolerance",
            "value": "FLAT | 0.05",
            "confidence": 0.64,
            "crop": {"x": 0.47, "y": 0.59, "w": 0.12, "h": 0.06},
            "symbol": "flatness",
            "kind": "feature_control_frame",
        },
    ]
    detected_vision_callouts = structure.get("visionCallouts") or []

    return {
        "filename": filename,
        "imageUrl": image_url,
        "model": "algorithmic-tools" if mode == "tools" else gemma_model,
        "runtimeMode": mode,
        "gemmaModel": gemma_model,
        "analysisStrategy": strategy,
        "status": "tool-analysis" if mode == "tools" else "gemma-pending",
        "border": border,
        "titleBlock": {
            "present": title_block.get("present", False),
            "crop": title_block["crop"],
            "confidence": title_block.get("confidence", 0.12),
            "candidate": title_block.get("candidate", {}),
            "fields": [
                {"label": "Drawing number", "value": "DWG-UNREAD", "confidence": 0.41},
                {"label": "Revision", "value": "A", "confidence": 0.58},
                {"label": "Scale", "value": "1:1", "confidence": 0.66},
                {"label": "Material", "value": "unknown", "confidence": 0.20},
            ],
        },
        "gdt": detected_gdt,
        "visionCallouts": detected_vision_callouts,
        "projections": detected_projections,
        "callouts": detected_callouts,
        "annotationMasks": detected_annotation_masks,
        "cad": {
            "strategy": "orthographic profile extraction with GD&T annotation pass",
            "stepFile": None,
            "confidence": 0.37,
        },
    }


def run(host: str, port: int) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), DashboardHandler)
    url_host = "localhost" if host in {"", "0.0.0.0"} else host
    print(f"Draw CAD dashboard serving at http://{url_host}:{port}", flush=True)
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the drawing-to-CAD development dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12080)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
