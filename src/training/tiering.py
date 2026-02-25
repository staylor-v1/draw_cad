"""Auto-classify training pairs into difficulty tiers."""
from __future__ import annotations

from enum import IntEnum

from src.training.data_loader import TrainingDataIndex
from src.training.ground_truth import StepGroundTruth
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DifficultyTier(IntEnum):
    """Difficulty tier for a training pair."""

    SIMPLE = 1
    MEDIUM = 2
    COMPLEX = 3


def classify_tier(gt: StepGroundTruth) -> DifficultyTier:
    """Classify a ground-truth record into a difficulty tier.

    Tier 1 (Simple):  face_count <= 12 AND solid_count == 1 AND edge_count <= 24
        (plates, blocks, simple cylinders)
    Tier 2 (Medium):  face_count <= 40 AND solid_count <= 2 AND edge_count <= 80
        (brackets with holes, stepped parts)
    Tier 3 (Complex): everything else
        (multi-body, many features)

    Secondary heuristics promote to a higher tier when geometry
    suggests additional complexity.
    """
    face = gt.face_count
    edge = gt.edge_count
    solid = gt.solid_count

    # Base tier assignment
    if face <= 12 and solid <= 1 and edge <= 24:
        tier = DifficultyTier.SIMPLE
    elif face <= 40 and solid <= 2 and edge <= 80:
        tier = DifficultyTier.MEDIUM
    else:
        tier = DifficultyTier.COMPLEX

    # Secondary heuristic: volume-to-bbox ratio (low => cutouts => complex)
    if gt.volume and gt.bbox_extents and len(gt.bbox_extents) == 3:
        dx, dy, dz = gt.bbox_extents
        bbox_vol = dx * dy * dz
        if bbox_vol > 0:
            fill_ratio = gt.volume / bbox_vol
            if fill_ratio < 0.15 and tier < DifficultyTier.COMPLEX:
                tier = DifficultyTier(tier + 1)

    # Secondary heuristic: high edge-to-face ratio (many fillets/chamfers)
    if face > 0 and edge / face > 4.0 and tier < DifficultyTier.COMPLEX:
        tier = DifficultyTier(tier + 1)

    return tier


def assign_tiers(index: TrainingDataIndex) -> None:
    """Classify every pair in the index that has ground truth."""
    for pair in index.pairs:
        if pair.ground_truth is not None:
            pair.tier = int(classify_tier(pair.ground_truth))


def compute_tier_distribution(index: TrainingDataIndex) -> dict[int, int]:
    """Return a histogram ``{tier: count}`` for calibration."""
    dist: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for pair in index.pairs:
        if pair.tier is not None:
            dist[pair.tier] = dist.get(pair.tier, 0) + 1
    return dist
