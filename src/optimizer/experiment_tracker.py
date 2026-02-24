"""JSON-file-based experiment tracking for Loop 2."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.schemas.experiment import ExperimentRecord, OptimizationHistory
from src.utils.file_utils import load_json, save_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Tracks experiment records to JSON files."""

    def __init__(self, experiments_dir: str | Path = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._history: Optional[OptimizationHistory] = None

    def start_run(self, run_id: str) -> OptimizationHistory:
        """Start a new optimization run."""
        self._history = OptimizationHistory(run_id=run_id)
        self._save_history()
        logger.info("experiment_run_started", run_id=run_id)
        return self._history

    def load_run(self, run_id: str) -> OptimizationHistory:
        """Load an existing optimization run."""
        path = self.experiments_dir / f"run_{run_id}.json"
        if path.exists():
            data = load_json(path)
            self._history = OptimizationHistory.model_validate(data)
        else:
            self._history = OptimizationHistory(run_id=run_id)
        return self._history

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Record a single experiment result."""
        if self._history is None:
            raise RuntimeError("No active optimization run. Call start_run first.")

        self._history.add_experiment(record)
        self._save_history()

        # Also save individual experiment record
        exp_path = self.experiments_dir / f"exp_{record.experiment_id}.json"
        save_json(record.model_dump(), exp_path)

        logger.info(
            "experiment_recorded",
            experiment_id=record.experiment_id,
            iteration=record.iteration,
            score=record.composite_score,
        )

    def complete_run(self, converged: bool = False, reason: str = "") -> None:
        """Mark the current run as complete."""
        if self._history is None:
            return
        self._history.completed_at = datetime.utcnow().isoformat()
        self._history.converged = converged
        self._history.convergence_reason = reason
        self._save_history()
        logger.info(
            "experiment_run_complete",
            run_id=self._history.run_id,
            converged=converged,
            best_score=self._history.best_composite_score,
        )

    def get_history(self) -> Optional[OptimizationHistory]:
        """Get the current optimization history."""
        return self._history

    def get_best_experiment(self) -> Optional[ExperimentRecord]:
        """Get the best-performing experiment from the current run."""
        if self._history is None or not self._history.experiments:
            return None
        return max(self._history.experiments, key=lambda e: e.composite_score)

    def list_runs(self) -> list[str]:
        """List all available run IDs."""
        runs = []
        for path in self.experiments_dir.glob("run_*.json"):
            run_id = path.stem.replace("run_", "")
            runs.append(run_id)
        return sorted(runs)

    def _save_history(self) -> None:
        """Save the current history to disk."""
        if self._history is None:
            return
        path = self.experiments_dir / f"run_{self._history.run_id}.json"
        save_json(self._history.model_dump(), path)
