"""Supervised and detected drawing callout helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


ANNOTATION_DIR = Path("training_data/gdt_annotations")


def load_callout_fixture(image_path: str | Path) -> list[dict]:
    """Load curated callout annotations when a teacher fixture exists."""

    path = Path(image_path)
    fixture = ANNOTATION_DIR / f"{path.stem}_callouts.yaml"
    if not fixture.exists():
        return []
    data = yaml.safe_load(fixture.read_text(encoding="utf-8")) or {}
    return [
        {
            "id": item["id"],
            "view": item["view"],
            "kind": item["kind"],
            "label": item.get("text", item["kind"]),
            "value": item.get("text", ""),
            "confidence": 1.0,
            "crop": item["crop"],
            "note": item.get("note", ""),
            "source": "teacher_fixture",
        }
        for item in data.get("callouts", [])
    ]


def load_projection_fixture(image_path: str | Path) -> list[dict]:
    path = Path(image_path)
    fixture = ANNOTATION_DIR / f"{path.stem}_callouts.yaml"
    if not fixture.exists():
        return []
    data = yaml.safe_load(fixture.read_text(encoding="utf-8")) or {}
    return [
        {
            "id": item["id"],
            "label": item["label"],
            "axis": item.get("axis", "unassigned"),
            "confidence": 1.0,
            "crop": item["crop"],
            "segmentationMode": "teacher_fixture",
        }
        for item in data.get("projections", [])
    ]


def load_non_callout_fixture(image_path: str | Path) -> list[dict]:
    path = Path(image_path)
    fixture = ANNOTATION_DIR / f"{path.stem}_callouts.yaml"
    if not fixture.exists():
        return []
    data = yaml.safe_load(fixture.read_text(encoding="utf-8")) or {}
    return [
        {
            "id": item["id"],
            "kind": item["kind"],
            "crop": item["crop"],
            "note": item.get("note", ""),
            "source": "teacher_fixture",
        }
        for item in data.get("non_callout_regions", [])
    ]


def split_callouts_by_view(callouts: list[dict]) -> dict[str, list[dict]]:
    views: dict[str, list[dict]] = {}
    for item in callouts:
        views.setdefault(item.get("view", "unassigned"), []).append(item)
    return views
