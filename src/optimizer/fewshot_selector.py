"""Few-shot example management and selection for Loop 2."""
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_prompt, load_yaml
from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.training.ground_truth import StepGroundTruth

logger = get_logger(__name__)


class FewShotExample:
    """A single few-shot example."""

    def __init__(self, name: str, path: str, tags: list[str] | None = None, content: str = ""):
        self.name = name
        self.path = path
        self.tags = tags or []
        self.content = content
        self.success_count = 0
        self.failure_count = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


class FewShotSelector:
    """Manages and selects few-shot examples for the reasoning stage."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.examples: list[FewShotExample] = []
        self._load_examples()

    def _load_examples(self) -> None:
        """Load examples from the configured index, then auto-mined examples."""
        try:
            index = load_yaml(self.config.prompts.fewshot_index)
        except FileNotFoundError:
            logger.debug("fewshot_index_not_found")
            index = {"examples": []}

        for entry in index.get("examples", []):
            try:
                content = load_prompt(entry["path"])
                example = FewShotExample(
                    name=entry.get("name", Path(entry["path"]).stem),
                    path=entry["path"],
                    tags=entry.get("tags", []),
                    content=content,
                )
                self.examples.append(example)
            except (FileNotFoundError, KeyError) as e:
                logger.debug("fewshot_example_load_error", path=entry.get("path"), error=str(e))

        # Load auto-mined examples (lower initial priority)
        self.load_mined_examples(Path("prompts/fewshot_examples/mined"))
        logger.info("fewshot_examples_loaded", count=len(self.examples))

    def load_mined_examples(self, mined_dir: Path) -> None:
        """Discover and load auto-mined few-shot examples.

        Mined examples are appended after hand-written ones so they
        have lower priority with the ``fixed`` selection strategy.
        """
        index_path = mined_dir / "index.yaml"
        if not index_path.exists():
            return

        existing_names = {ex.name for ex in self.examples}

        try:
            data = load_yaml(str(index_path))
        except Exception:
            return

        for entry in data.get("examples", []):
            name = entry.get("name", "")
            if name in existing_names:
                continue
            try:
                content = load_prompt(entry["path"])
                example = FewShotExample(
                    name=name,
                    path=entry["path"],
                    tags=entry.get("tags", []),
                    content=content,
                )
                self.examples.append(example)
                existing_names.add(name)
            except (FileNotFoundError, KeyError):
                pass

        logger.debug("mined_examples_loaded", dir=str(mined_dir))

    def select(
        self,
        count: int = 3,
        strategy: str = "fixed",
        target_tags: list[str] | None = None,
    ) -> list[str]:
        """Select few-shot examples.
        
        Args:
            count: Number of examples to select.
            strategy: Selection strategy (fixed, random, coverage, failure_targeted).
            target_tags: Optional tags to target for coverage strategy.
        
        Returns:
            List of example content strings.
        """
        if not self.examples:
            return []

        count = min(count, len(self.examples))

        if strategy == "fixed":
            selected = self.examples[:count]
        elif strategy == "random":
            selected = random.sample(self.examples, count)
        elif strategy == "coverage":
            selected = self._select_coverage(count, target_tags)
        elif strategy == "failure_targeted":
            selected = self._select_failure_targeted(count)
        elif strategy == "similarity":
            selected = self._select_similarity(count, target_tags)
        else:
            selected = self.examples[:count]

        logger.info("fewshot_selected", count=len(selected), strategy=strategy)
        return [ex.content for ex in selected]

    def update_stats(self, example_indices: list[int], success: bool) -> None:
        """Update success/failure stats for selected examples."""
        for idx in example_indices:
            if 0 <= idx < len(self.examples):
                if success:
                    self.examples[idx].success_count += 1
                else:
                    self.examples[idx].failure_count += 1

    def add_example(self, name: str, content: str, tags: list[str] | None = None) -> None:
        """Add a new few-shot example."""
        example = FewShotExample(name=name, path="", tags=tags or [], content=content)
        self.examples.append(example)
        logger.info("fewshot_example_added", name=name)

    def remove_example(self, name: str) -> bool:
        """Remove an example by name."""
        for i, ex in enumerate(self.examples):
            if ex.name == name:
                self.examples.pop(i)
                logger.info("fewshot_example_removed", name=name)
                return True
        return False

    def _select_coverage(
        self, count: int, target_tags: list[str] | None
    ) -> list[FewShotExample]:
        """Select examples for maximum tag coverage."""
        if not target_tags:
            return self.examples[:count]

        # Greedy set cover
        selected = []
        covered_tags: set[str] = set()
        remaining = list(self.examples)

        while len(selected) < count and remaining:
            best = max(
                remaining,
                key=lambda ex: len(set(ex.tags) - covered_tags),
            )
            selected.append(best)
            covered_tags.update(best.tags)
            remaining.remove(best)

        return selected

    def _select_failure_targeted(self, count: int) -> list[FewShotExample]:
        """Select examples with highest success rates (proven to help)."""
        sorted_examples = sorted(
            self.examples,
            key=lambda ex: ex.success_rate,
            reverse=True,
        )
        return sorted_examples[:count]

    def _select_similarity(
        self,
        count: int,
        target_tags: list[str] | None,
    ) -> list[FewShotExample]:
        """Select examples whose tags overlap most with the target tags.

        When *reference_ground_truth* is available on the class, this
        can be extended to compare numeric StepGroundTruth features;
        for now it falls back to tag-based overlap which is populated
        from ground-truth properties by ``FewShotMiner``.
        """
        if not target_tags:
            return self.examples[:count]

        target_set = set(target_tags)

        def _score(ex: FewShotExample) -> float:
            overlap = len(target_set & set(ex.tags))
            return overlap + ex.success_rate * 0.1

        ranked = sorted(self.examples, key=_score, reverse=True)
        return ranked[:count]
