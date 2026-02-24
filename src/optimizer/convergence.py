"""Convergence detection for Loop 2 meta-optimizer."""
from __future__ import annotations

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConvergenceDetector:
    """Detects when the optimization loop has converged."""

    def __init__(self, threshold: float = 0.01, patience: int = 3):
        self.threshold = threshold
        self.patience = patience
        self.scores: list[float] = []

    def update(self, score: float) -> None:
        """Record a new score."""
        self.scores.append(score)

    def has_converged(self) -> tuple[bool, str]:
        """Check if optimization has converged.
        
        Returns:
            Tuple of (converged, reason).
        """
        if len(self.scores) < self.patience + 1:
            return False, "Not enough data points"

        # Check if improvement is below threshold for patience consecutive iterations
        recent = self.scores[-self.patience:]
        baseline = self.scores[-(self.patience + 1)]

        max_improvement = max(abs(s - baseline) for s in recent)

        if max_improvement < self.threshold:
            reason = (
                f"Improvement < {self.threshold} for {self.patience} "
                f"consecutive iterations (max improvement: {max_improvement:.4f})"
            )
            logger.info("convergence_detected", reason=reason)
            return True, reason

        # Check for score degradation
        if len(self.scores) >= 5:
            recent_avg = sum(self.scores[-3:]) / 3
            earlier_avg = sum(self.scores[-6:-3]) / 3 if len(self.scores) >= 6 else self.scores[0]
            if recent_avg < earlier_avg - self.threshold:
                reason = f"Score degradation detected: {recent_avg:.4f} < {earlier_avg:.4f}"
                logger.info("convergence_degradation", reason=reason)
                return True, reason

        return False, ""

    def reset(self) -> None:
        """Reset the convergence detector."""
        self.scores.clear()

    @property
    def best_score(self) -> float:
        """Get the best score seen so far."""
        return max(self.scores) if self.scores else 0.0

    @property
    def latest_score(self) -> float:
        """Get the most recent score."""
        return self.scores[-1] if self.scores else 0.0

    @property
    def improvement_trend(self) -> float:
        """Get the average improvement per iteration."""
        if len(self.scores) < 2:
            return 0.0
        improvements = [
            self.scores[i] - self.scores[i - 1]
            for i in range(1, len(self.scores))
        ]
        return sum(improvements) / len(improvements)
