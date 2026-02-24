"""Few-shot example management and selection for Loop 2."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_prompt, load_yaml
from src.utils.logging_config import get_logger

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
        """Load examples from the configured index."""
        try:
            index = load_yaml(self.config.prompts.fewshot_index)
        except FileNotFoundError:
            logger.debug("fewshot_index_not_found")
            return

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

        logger.info("fewshot_examples_loaded", count=len(self.examples))

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
