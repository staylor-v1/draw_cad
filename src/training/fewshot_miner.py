"""Auto-mine few-shot examples from successful Loop 2 runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.training.data_loader import TrainingDataIndex, TrainingPair
from src.training.ground_truth import StepGroundTruth
from src.schemas.evaluation_result import EvaluationMetrics
from src.utils.file_utils import save_yaml, load_yaml
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FewShotMiner:
    """Records successful pipeline runs as reusable few-shot examples."""

    def __init__(
        self,
        index: TrainingDataIndex,
        storage_dir: str | Path = "prompts/fewshot_examples/mined",
    ):
        self.index = index
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._recorded_ids: set[str] = set()
        self._load_existing()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_successful_run(
        self,
        pair: TrainingPair,
        code: str,
        metrics: EvaluationMetrics,
        reconciled_description: str = "",
    ) -> bool:
        """Record a successful run if it meets quality thresholds.

        Criteria:
          - composite_score > 0.7
          - geometry_valid == 1.0

        Returns True if the example was recorded.
        """
        if pair.pair_id in self._recorded_ids:
            logger.debug("fewshot_miner_duplicate", pair_id=pair.pair_id)
            return False

        if metrics.composite_score <= 0.7:
            return False
        if metrics.geometry_valid < 1.0:
            return False

        tags = self._derive_tags(pair)
        markdown = self._generate_markdown(pair, code, tags, reconciled_description)

        # Write example file
        example_path = self.storage_dir / f"{pair.pair_id}.md"
        example_path.write_text(markdown)

        self._recorded_ids.add(pair.pair_id)
        self._update_index(pair.pair_id, tags, str(example_path))

        logger.info("fewshot_mined", pair_id=pair.pair_id, tags=tags)
        return True

    @property
    def mined_count(self) -> int:
        return len(self._recorded_ids)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_existing(self) -> None:
        """Discover already-mined examples."""
        index_path = self.storage_dir / "index.yaml"
        if index_path.exists():
            data = load_yaml(str(index_path))
            for entry in data.get("examples", []):
                self._recorded_ids.add(entry.get("name", ""))

    def _update_index(self, pair_id: str, tags: list[str], path: str) -> None:
        """Append an entry to the mined index.yaml."""
        index_path = self.storage_dir / "index.yaml"
        if index_path.exists():
            data = load_yaml(str(index_path))
        else:
            data = {"examples": []}

        data["examples"].append({
            "name": pair_id,
            "path": path,
            "tags": tags,
        })
        save_yaml(data, str(index_path))

    def _derive_tags(self, pair: TrainingPair) -> list[str]:
        """Derive descriptive tags from ground truth."""
        tags: list[str] = []
        gt = pair.ground_truth
        if gt is None:
            return tags

        # Complexity tag
        if gt.face_count <= 12:
            tags.append("simple")
        elif gt.face_count <= 40:
            tags.append("medium")
        else:
            tags.append("complex")

        # Solid count
        if gt.solid_count > 1:
            tags.append("multi-body")

        # Cylindrical features heuristic: many edges relative to faces
        if gt.face_count > 0 and gt.edge_count / gt.face_count > 3.0:
            tags.append("revolve")

        # Fill ratio (cutouts)
        if gt.volume and gt.bbox_extents and len(gt.bbox_extents) == 3:
            dx, dy, dz = gt.bbox_extents
            bbox_vol = dx * dy * dz
            if bbox_vol > 0 and gt.volume / bbox_vol < 0.3:
                tags.append("cutouts")

        return tags

    def _generate_markdown(
        self,
        pair: TrainingPair,
        code: str,
        tags: list[str],
        reconciled_description: str,
    ) -> str:
        """Generate a markdown few-shot example."""
        gt = pair.ground_truth
        lines = [
            f"# Example: {pair.pair_id}",
            "",
            f"**Tags**: {', '.join(tags)}",
            "",
        ]

        if gt:
            lines.extend([
                "## Reference geometry",
                f"- Faces: {gt.face_count}, Edges: {gt.edge_count}",
                f"- Volume: {gt.volume:.2f}" if gt.volume else "",
                f"- Bounding box extents: {gt.bbox_extents}" if gt.bbox_extents else "",
                "",
            ])

        if reconciled_description:
            lines.extend([
                "## Description",
                reconciled_description,
                "",
            ])

        lines.extend([
            "## build123d code",
            "```python",
            code.strip(),
            "```",
        ])

        return "\n".join(line for line in lines if line is not None)
