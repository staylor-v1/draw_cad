"""Persist / load the precomputed training-data manifest (JSON)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.data_loader import TrainingDataIndex, TrainingPair
from src.training.ground_truth import StepGroundTruth
from src.utils.file_utils import load_json, save_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def save_manifest(index: TrainingDataIndex, path: str | Path) -> None:
    """Serialise the full index (pairs + ground truth + tiers) to JSON."""
    records: list[dict[str, Any]] = []
    for p in index.pairs:
        rec: dict[str, Any] = {
            "pair_id": p.pair_id,
            "part_id": p.part_id,
            "part_hash": p.part_hash,
            "view_number": p.view_number,
            "svg_path": str(p.svg_path),
            "step_path": str(p.step_path),
            "png_path": str(p.png_path) if p.png_path else None,
            "tier": p.tier,
            "tags": p.tags,
        }
        if p.ground_truth is not None:
            rec["ground_truth"] = p.ground_truth.model_dump()
        else:
            rec["ground_truth"] = None
        records.append(rec)

    payload = {
        "version": 1,
        "count": len(records),
        "pairs": records,
    }
    save_json(payload, path)
    logger.info("manifest_saved", path=str(path), count=len(records))


def load_manifest(
    path: str | Path,
    root_dir: str | Path | None = None,
) -> TrainingDataIndex:
    """Load a manifest and reconstitute a ``TrainingDataIndex``.

    If *root_dir* is given, relative paths stored in the manifest are
    resolved against it; otherwise they are used as-is.
    """
    data = load_json(path)
    pairs: list[TrainingPair] = []

    root = Path(root_dir) if root_dir else None

    for rec in data.get("pairs", []):
        gt = None
        if rec.get("ground_truth"):
            gt = StepGroundTruth.model_validate(rec["ground_truth"])

        svg_path = Path(rec["svg_path"])
        step_path = Path(rec["step_path"])
        png_path = Path(rec["png_path"]) if rec.get("png_path") else None

        # Re-derive absolute paths when a root is supplied and paths are relative
        if root:
            if not svg_path.is_absolute():
                svg_path = root / svg_path
            if not step_path.is_absolute():
                step_path = root / step_path
            if png_path and not png_path.is_absolute():
                png_path = root / png_path

        pairs.append(
            TrainingPair(
                pair_id=rec["pair_id"],
                part_id=rec["part_id"],
                part_hash=rec["part_hash"],
                view_number=rec["view_number"],
                svg_path=svg_path,
                step_path=step_path,
                png_path=png_path,
                ground_truth=gt,
                tier=rec.get("tier"),
                tags=rec.get("tags", []),
            )
        )

    index = TrainingDataIndex(pairs=pairs, root_dir=root)
    logger.info("manifest_loaded", path=str(path), count=len(pairs))
    return index
