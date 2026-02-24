"""Experiment tracking schemas for Loop 2 meta-optimizer."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ConfigDelta(BaseModel):
    """A description of what changed between configs."""
    parameter_changes: dict[str, Any] = Field(default_factory=dict)
    prompt_patches: list[str] = Field(default_factory=list)
    fewshot_changes: list[str] = Field(default_factory=list)
    description: str = ""


class ExperimentRecord(BaseModel):
    """A single experiment record from Loop 2."""
    experiment_id: str
    iteration: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Config snapshot
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    config_delta: Optional[ConfigDelta] = None

    # Results
    composite_score: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    case_scores: dict[str, float] = Field(default_factory=dict)  # case_id -> score

    # Analysis
    failure_categories: dict[str, int] = Field(default_factory=dict)
    improvement_over_baseline: float = 0.0

    # Status
    status: str = "pending"  # pending, running, completed, failed
    error: str = ""
    duration_seconds: float = 0.0


class OptimizationHistory(BaseModel):
    """Complete history of a Loop 2 optimization run."""
    run_id: str
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    experiments: list[ExperimentRecord] = Field(default_factory=list)
    best_experiment_id: Optional[str] = None
    best_composite_score: float = 0.0
    converged: bool = False
    convergence_reason: str = ""

    def add_experiment(self, record: ExperimentRecord) -> None:
        """Add an experiment record and update best tracking."""
        self.experiments.append(record)
        if record.composite_score > self.best_composite_score:
            self.best_composite_score = record.composite_score
            self.best_experiment_id = record.experiment_id
