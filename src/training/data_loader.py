"""Training data index: scans and indexes the 2,981 SVG/STEP pairs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.training.ground_truth import StepGroundTruth
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Filename pattern: {ProjectID}_{Hash}_{ViewNumber}.{ext}
_FILENAME_RE = re.compile(r"^(\d+)_([0-9a-f]+)_(\d+)$")


@dataclass
class TrainingPair:
    """A single SVG/STEP training pair."""

    pair_id: str  # basename without extension, e.g. "37605_e35cc4df_0007"
    part_id: str  # project ID portion
    part_hash: str  # hash portion
    view_number: str  # view number portion
    svg_path: Path
    step_path: Path
    png_path: Optional[Path] = None
    ground_truth: Optional[StepGroundTruth] = None
    tier: Optional[int] = None
    tags: list[str] = field(default_factory=list)


def parse_pair_id(basename: str) -> tuple[str, str, str] | None:
    """Parse a training-data basename into (part_id, part_hash, view_number).

    Returns ``None`` if the name does not match the expected pattern.
    """
    m = _FILENAME_RE.match(basename)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


class TrainingDataIndex:
    """Index of all training pairs, with lookups by tier/id/tags."""

    def __init__(
        self,
        pairs: list[TrainingPair] | None = None,
        root_dir: Path | None = None,
    ):
        self.pairs: list[TrainingPair] = pairs or []
        self.root_dir = root_dir
        self._by_id: dict[str, TrainingPair] = {p.pair_id: p for p in self.pairs}

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_directory(
        cls,
        root_dir: str | Path,
        svg_subdir: str = "drawings_svg",
        step_subdir: str = "shapes_step",
        png_subdir: str = "rendered_png",
    ) -> TrainingDataIndex:
        """Scan directories and build an index by matching basenames."""
        root = Path(root_dir)
        svg_dir = root / svg_subdir
        step_dir = root / step_subdir
        png_dir = root / png_subdir

        svg_files = {p.stem: p for p in svg_dir.glob("*.svg")} if svg_dir.is_dir() else {}
        step_files = {p.stem: p for p in step_dir.glob("*.step")} if step_dir.is_dir() else {}
        png_files = {p.stem: p for p in png_dir.glob("*.png")} if png_dir.is_dir() else {}

        matched_basenames = sorted(set(svg_files) & set(step_files))

        pairs: list[TrainingPair] = []
        for bn in matched_basenames:
            parsed = parse_pair_id(bn)
            if parsed is None:
                logger.debug("skipping_unrecognised_filename", basename=bn)
                continue
            part_id, part_hash, view_number = parsed
            pairs.append(
                TrainingPair(
                    pair_id=bn,
                    part_id=part_id,
                    part_hash=part_hash,
                    view_number=view_number,
                    svg_path=svg_files[bn],
                    step_path=step_files[bn],
                    png_path=png_files.get(bn),
                )
            )

        logger.info(
            "training_index_built",
            total_svg=len(svg_files),
            total_step=len(step_files),
            matched=len(pairs),
        )
        return cls(pairs=pairs, root_dir=root)

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    def get_by_id(self, pair_id: str) -> TrainingPair | None:
        return self._by_id.get(pair_id)

    def get_by_tier(self, tier: int) -> list[TrainingPair]:
        return [p for p in self.pairs if p.tier == tier]

    def get_by_tiers(self, tiers: list[int]) -> list[TrainingPair]:
        tier_set = set(tiers)
        return [p for p in self.pairs if p.tier in tier_set]

    def get_by_tags(self, tags: list[str]) -> list[TrainingPair]:
        tag_set = set(tags)
        return [p for p in self.pairs if tag_set & set(p.tags)]

    @property
    def size(self) -> int:
        return len(self.pairs)

    def rebuild_lookup(self) -> None:
        """Rebuild the internal lookup dict after mutation."""
        self._by_id = {p.pair_id: p for p in self.pairs}
