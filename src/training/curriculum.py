"""Curriculum scheduler: phased difficulty progression for Loop 2."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.training.data_loader import TrainingDataIndex, TrainingPair
from src.training.sampler import BenchmarkSampler, SamplingStrategy
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CurriculumPhase:
    """Definition of a single curriculum phase."""

    name: str
    tiers: list[int]
    sample_size: int = 20
    min_score_to_advance: float = 0.5
    max_iterations: int = 10


class CurriculumScheduler:
    """Manages phased difficulty progression during optimisation.

    Default phases:
        - foundation:    tiers=[1],     advance at score > 0.6 or 5 iterations
        - intermediate:  tiers=[1,2],   advance at score > 0.5 or 8 iterations
        - advanced:      tiers=[1,2,3], no advancement (final)
    """

    def __init__(
        self,
        index: TrainingDataIndex,
        phases: list[CurriculumPhase] | None = None,
        sentinel_ids: list[str] | None = None,
        sentinel_count: int = 5,
    ):
        self.index = index
        self.phases = phases or self._default_phases()
        self._current_phase_idx = 0
        self._phase_iteration = 0
        self._sentinel_ids = sentinel_ids
        self._sentinel_count = sentinel_count

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._current_phase_idx]

    @property
    def phase_name(self) -> str:
        return self.current_phase.name

    @property
    def phase_iteration(self) -> int:
        return self._phase_iteration

    def get_current_sample(self, iteration: int) -> list[TrainingPair]:
        """Get a sample of pairs for the current phase and iteration."""
        phase = self.current_phase

        sampler = BenchmarkSampler(
            index=self.index,
            sample_size=phase.sample_size,
            strategy=SamplingStrategy.STRATIFIED,
            sentinel_ids=self._sentinel_ids,
            sentinel_count=self._sentinel_count,
        )

        pairs = sampler.sample(tier_filter=phase.tiers)
        self._phase_iteration += 1

        logger.info(
            "curriculum_sample",
            phase=phase.name,
            phase_iter=self._phase_iteration,
            global_iter=iteration,
            tiers=phase.tiers,
            sample_size=len(pairs),
        )
        return pairs

    def should_advance(self, score: float) -> bool:
        """Check whether the current phase should advance."""
        if self._current_phase_idx >= len(self.phases) - 1:
            return False  # Already at final phase

        phase = self.current_phase
        if score >= phase.min_score_to_advance:
            return True
        if self._phase_iteration >= phase.max_iterations:
            return True
        return False

    def advance(self) -> bool:
        """Advance to the next phase.

        Returns True if advanced, False if already at the last phase.
        """
        if self._current_phase_idx >= len(self.phases) - 1:
            return False

        old_phase = self.current_phase.name
        self._current_phase_idx += 1
        self._phase_iteration = 0

        logger.info(
            "curriculum_phase_advanced",
            from_phase=old_phase,
            to_phase=self.current_phase.name,
        )
        return True

    def reset(self) -> None:
        """Reset to the first phase."""
        self._current_phase_idx = 0
        self._phase_iteration = 0

    # ------------------------------------------------------------------ #
    # Defaults
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_phases() -> list[CurriculumPhase]:
        return [
            CurriculumPhase(
                name="foundation",
                tiers=[1],
                sample_size=15,
                min_score_to_advance=0.6,
                max_iterations=5,
            ),
            CurriculumPhase(
                name="intermediate",
                tiers=[1, 2],
                sample_size=20,
                min_score_to_advance=0.5,
                max_iterations=8,
            ),
            CurriculumPhase(
                name="advanced",
                tiers=[1, 2, 3],
                sample_size=25,
                min_score_to_advance=0.4,
                max_iterations=10,
            ),
        ]
