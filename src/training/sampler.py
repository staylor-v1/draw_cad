"""Smart subset selection for benchmark sampling."""
from __future__ import annotations

import random
from enum import Enum
from typing import Optional

from src.training.data_loader import TrainingDataIndex, TrainingPair
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SamplingStrategy(str, Enum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    CURRICULUM = "curriculum"
    FAILURE_TARGETED = "failure_targeted"


class BenchmarkSampler:
    """Selects a representative subset of training pairs for evaluation."""

    def __init__(
        self,
        index: TrainingDataIndex,
        sample_size: int = 20,
        strategy: SamplingStrategy | str = SamplingStrategy.STRATIFIED,
        sentinel_ids: list[str] | None = None,
        sentinel_count: int = 5,
    ):
        self.index = index
        self.sample_size = sample_size
        self.strategy = SamplingStrategy(strategy) if isinstance(strategy, str) else strategy
        self._sentinel_ids = sentinel_ids
        self._sentinel_count = sentinel_count

        # Failure scores: pair_id -> last composite score
        self._failure_scores: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def sample(
        self,
        tier_filter: list[int] | None = None,
        rng: random.Random | None = None,
    ) -> list[TrainingPair]:
        """Select a subset of pairs according to the current strategy.

        Args:
            tier_filter: If given, restrict the pool to these tiers.
            rng: Optional seeded Random instance for reproducibility.

        Returns:
            List of selected TrainingPair objects.
        """
        rng = rng or random.Random()
        pool = self._filtered_pool(tier_filter)

        if not pool:
            logger.warning("sampler_empty_pool")
            return []

        sentinels = self._pick_sentinels(pool)
        sentinel_set = {s.pair_id for s in sentinels}
        remaining_pool = [p for p in pool if p.pair_id not in sentinel_set]
        remaining_n = max(0, self.sample_size - len(sentinels))

        if self.strategy == SamplingStrategy.RANDOM:
            chosen = self._sample_random(remaining_pool, remaining_n, rng)
        elif self.strategy == SamplingStrategy.STRATIFIED:
            chosen = self._sample_stratified(remaining_pool, remaining_n, rng)
        elif self.strategy == SamplingStrategy.FAILURE_TARGETED:
            chosen = self._sample_failure_targeted(remaining_pool, remaining_n, rng)
        else:
            # CURRICULUM delegates to the CurriculumScheduler; here we fallback
            chosen = self._sample_stratified(remaining_pool, remaining_n, rng)

        result = sentinels + chosen
        logger.info(
            "sample_selected",
            strategy=self.strategy.value,
            total=len(result),
            sentinels=len(sentinels),
        )
        return result

    def record_scores(self, scores: dict[str, float]) -> None:
        """Record per-pair scores for failure-targeted sampling."""
        self._failure_scores.update(scores)

    # ------------------------------------------------------------------ #
    # Internal strategies
    # ------------------------------------------------------------------ #

    def _filtered_pool(self, tier_filter: list[int] | None) -> list[TrainingPair]:
        if tier_filter:
            return self.index.get_by_tiers(tier_filter)
        return list(self.index.pairs)

    def _pick_sentinels(self, pool: list[TrainingPair]) -> list[TrainingPair]:
        """Return the fixed sentinel set for stable tracking."""
        if self._sentinel_ids:
            sentinels = [
                p for p in pool if p.pair_id in set(self._sentinel_ids)
            ]
            return sentinels[: self._sentinel_count]

        # Auto-pick: one per tier, deterministic by sorted pair_id
        by_tier: dict[int, list[TrainingPair]] = {}
        for p in pool:
            if p.tier is not None:
                by_tier.setdefault(p.tier, []).append(p)

        sentinels: list[TrainingPair] = []
        for tier in sorted(by_tier):
            candidates = sorted(by_tier[tier], key=lambda p: p.pair_id)
            needed = max(1, self._sentinel_count // max(len(by_tier), 1))
            sentinels.extend(candidates[:needed])
        return sentinels[: self._sentinel_count]

    def _sample_random(
        self, pool: list[TrainingPair], n: int, rng: random.Random
    ) -> list[TrainingPair]:
        n = min(n, len(pool))
        return rng.sample(pool, n) if n > 0 else []

    def _sample_stratified(
        self, pool: list[TrainingPair], n: int, rng: random.Random
    ) -> list[TrainingPair]:
        """Proportional sampling from each tier."""
        by_tier: dict[int, list[TrainingPair]] = {}
        for p in pool:
            by_tier.setdefault(p.tier if p.tier is not None else 0, []).append(p)

        total_pool = len(pool)
        if total_pool == 0:
            return []

        result: list[TrainingPair] = []
        for tier in sorted(by_tier):
            tier_pool = by_tier[tier]
            tier_n = max(1, round(n * len(tier_pool) / total_pool))
            tier_n = min(tier_n, len(tier_pool))
            result.extend(rng.sample(tier_pool, tier_n))

        # Trim or pad to exact size
        if len(result) > n:
            result = rng.sample(result, n)
        elif len(result) < n:
            extras = [p for p in pool if p not in result]
            needed = min(n - len(result), len(extras))
            if needed > 0:
                result.extend(rng.sample(extras, needed))

        return result

    def _sample_failure_targeted(
        self, pool: list[TrainingPair], n: int, rng: random.Random
    ) -> list[TrainingPair]:
        """50% from pairs with previous score < 0.5, 50% random."""
        half = n // 2

        # Low-scoring pairs
        low_scoring = [
            p for p in pool
            if self._failure_scores.get(p.pair_id, 1.0) < 0.5
        ]
        if low_scoring:
            failure_sample = rng.sample(low_scoring, min(half, len(low_scoring)))
        else:
            failure_sample = []

        # Random from remainder
        chosen_ids = {p.pair_id for p in failure_sample}
        remaining = [p for p in pool if p.pair_id not in chosen_ids]
        random_n = n - len(failure_sample)
        random_sample = rng.sample(remaining, min(random_n, len(remaining))) if remaining else []

        return failure_sample + random_sample
