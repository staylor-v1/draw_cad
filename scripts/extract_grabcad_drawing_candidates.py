#!/usr/bin/env python3
"""Extract likely 2D drawing sheets from GrabCAD project archives."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training_data" / "gdt_grabcad"
RAW_DIR = DATA_DIR / "raw_archives"
OUT_DIR = DATA_DIR / "orthographic_2d_candidates"
MANIFEST_PATH = DATA_DIR / "manifest.json"
EXTRACTION_MANIFEST = DATA_DIR / "extraction_manifest.json"
REVIEW_METADATA_PATH = DATA_DIR / "candidate_review_metadata.json"
USE_LIST_PATH = DATA_DIR / "selected_training_candidates.txt"
REJECT_LIST_PATH = DATA_DIR / "rejected_training_candidates.txt"
DUPLICATE_LIST_PATH = DATA_DIR / "duplicate_training_candidates.txt"

DRAWING_EXTENSIONS = {
    ".pdf",
    ".dwg",
    ".dxf",
    ".idw",
    ".slddrw",
    ".drw",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}

SKIP_IMAGE_HINTS = {
    "render",
    "screenshot",
    "preview",
    "cover",
    "card",
    "thumbnail",
    "hqdefault",
}


@dataclass
class ExtractedCandidate:
    source_archive: str
    model_slug: str
    source_url: str | None
    original_member: str
    extracted_path: str
    extension: str
    size_bytes: int
    confidence: str
    disposition: str
    notes: list[str]


def slugify(text: str) -> str:
    text = text.replace("\\", "/").split("/")[-1]
    stem = Path(text).stem
    suffix = Path(text).suffix.lower()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"{safe}{suffix}" if suffix else safe


def archive_slug(archive_name: str, manifest: dict) -> str:
    for download in manifest.get("downloads", []):
        for file_name in download.get("downloadedFiles") or []:
            if file_name == archive_name and download.get("model"):
                return download["model"].rstrip("/").split("/")[-1]
    return archive_name.split(".snapshot.")[0]


def source_url_by_slug(manifest: dict) -> dict[str, str]:
    return {
        model.get("slug"): model.get("url")
        for model in manifest.get("models", [])
        if model.get("slug") and model.get("url")
    }


def load_review_dispositions() -> dict[str, str]:
    if not REVIEW_METADATA_PATH.exists():
        return {}
    try:
        review = json.loads(REVIEW_METADATA_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    dispositions = review.get("dispositions")
    if isinstance(dispositions, dict):
        return {
            path: disposition
            for path, disposition in dispositions.items()
            if disposition in {"use", "reject", "duplicate"}
        }
    return {}


def candidate_confidence(member_name: str, extension: str) -> tuple[str, list[str]]:
    lower = member_name.lower()
    notes: list[str] = []
    if any(hint in lower for hint in SKIP_IMAGE_HINTS):
        return "skip", ["image name looks like a web preview/rendering, not a drawing sheet"]
    if extension in {".pdf", ".dwg", ".dxf", ".idw", ".slddrw", ".drw"}:
        return "high", ["native drawing or drawing exchange format"]
    if re.search(r"\bid\s*\d+|\d{4,}|drawing|draft|blueprint|gd.?t|asme|y14", lower):
        return "high", ["image filename/title suggests a drawing sheet"]
    return "medium", ["raster image requires visual review"]


def extract_archive(
    archive_path: Path,
    manifest: dict,
    source_urls: dict[str, str],
    review_dispositions: dict[str, str],
) -> list[ExtractedCandidate]:
    model_slug = archive_slug(archive_path.name, manifest)
    extracted: list[ExtractedCandidate] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            extension = Path(info.filename).suffix.lower()
            if extension not in DRAWING_EXTENSIONS:
                continue
            confidence, notes = candidate_confidence(info.filename, extension)
            if confidence == "skip":
                continue
            destination_name = f"{model_slug}__{slugify(info.filename)}"
            destination = OUT_DIR / destination_name
            counter = 2
            while destination.exists():
                destination = OUT_DIR / f"{model_slug}__{counter:02d}__{slugify(info.filename)}"
                counter += 1
            with archive.open(info) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_path = str(destination.relative_to(ROOT))
            extracted.append(
                ExtractedCandidate(
                    source_archive=archive_path.name,
                    model_slug=model_slug,
                    source_url=source_urls.get(model_slug),
                    original_member=info.filename,
                    extracted_path=extracted_path,
                    extension=extension,
                    size_bytes=info.file_size,
                    confidence=confidence,
                    disposition=review_dispositions.get(extracted_path, "unreviewed"),
                    notes=notes,
                )
            )
    return extracted


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    source_urls = source_url_by_slug(manifest)
    review_dispositions = load_review_dispositions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_candidate in OUT_DIR.iterdir():
        if old_candidate.is_file():
            old_candidate.unlink()
    candidates: list[ExtractedCandidate] = []
    archives = sorted(RAW_DIR.glob("*.zip"))
    for archive in archives:
        candidates.extend(extract_archive(archive, manifest, source_urls, review_dispositions))
    use_paths = sorted(item.extracted_path for item in candidates if item.disposition == "use")
    reject_paths = sorted(item.extracted_path for item in candidates if item.disposition == "reject")
    duplicate_paths = sorted(item.extracted_path for item in candidates if item.disposition == "duplicate")
    summary = {
        "source_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "raw_archive_dir": str(RAW_DIR.relative_to(ROOT)),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "archive_count": len(archives),
        "candidate_count": len(candidates),
        "confidence_counts": {
            level: sum(1 for item in candidates if item.confidence == level)
            for level in ["high", "medium", "low"]
        },
        "disposition_counts": {
            level: sum(1 for item in candidates if item.disposition == level)
            for level in ["use", "reject", "duplicate", "unreviewed"]
        },
        "candidates": [asdict(item) for item in candidates],
    }
    EXTRACTION_MANIFEST.write_text(json.dumps(summary, indent=2) + "\n")
    USE_LIST_PATH.write_text("\n".join(use_paths) + ("\n" if use_paths else ""))
    REJECT_LIST_PATH.write_text("\n".join(reject_paths) + ("\n" if reject_paths else ""))
    DUPLICATE_LIST_PATH.write_text("\n".join(duplicate_paths) + ("\n" if duplicate_paths else ""))
    print(json.dumps({k: summary[k] for k in ["archive_count", "candidate_count", "confidence_counts", "disposition_counts", "output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
