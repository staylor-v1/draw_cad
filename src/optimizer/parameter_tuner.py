"""Parameter tuning for Loop 2 optimization."""
from __future__ import annotations

import itertools
from typing import Any, Optional

from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_yaml
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ParameterCandidate:
    """A candidate parameter configuration."""

    def __init__(self, params: dict[str, Any], description: str = ""):
        self.params = params
        self.description = description
        self.score: Optional[float] = None

    def apply_to_config(self, config: PipelineConfig) -> PipelineConfig:
        """Apply these parameters to a pipeline config, returning a new config."""
        new_config = config.deep_copy()

        for key, value in self.params.items():
            if key == "temperature":
                new_config.models.reasoning.temperature = value
            elif key == "vision_temperature":
                new_config.models.vision.temperature = value
            elif key == "max_retries":
                new_config.pipeline.max_retries = value
            elif key == "ocr_enabled":
                new_config.pipeline.ocr_enabled = value
            elif key == "execution_timeout":
                new_config.pipeline.execution_timeout = value
            elif key == "max_tokens":
                new_config.models.reasoning.max_tokens = value

        return new_config


class ParameterTuner:
    """Grid/Bayesian parameter search for pipeline configuration."""

    def __init__(self, optimizer_config_path: str = "config/optimizer.yaml"):
        try:
            self.search_config = load_yaml(optimizer_config_path)
        except FileNotFoundError:
            self.search_config = {}
        self._candidates: list[ParameterCandidate] = []

    def generate_grid_candidates(self, max_candidates: int = 10) -> list[ParameterCandidate]:
        """Generate parameter candidates using grid search."""
        param_search = self.search_config.get("parameter_search", {})

        # Build parameter grid
        param_grid: dict[str, list] = {}

        for param_name, spec in param_search.items():
            if "values" in spec:
                param_grid[param_name] = spec["values"]
            elif "min" in spec and "max" in spec:
                step = spec.get("step", 0.1)
                values = []
                v = spec["min"]
                while v <= spec["max"]:
                    values.append(round(v, 4))
                    v += step
                param_grid[param_name] = values

        if not param_grid:
            return []

        # Generate combinations (limited)
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]

        candidates = []
        for combo in itertools.islice(itertools.product(*value_lists), max_candidates * 10):
            params = dict(zip(keys, combo))
            desc = ", ".join(f"{k}={v}" for k, v in params.items())
            candidates.append(ParameterCandidate(params=params, description=desc))

        # Sample if too many
        if len(candidates) > max_candidates:
            import random
            candidates = random.sample(candidates, max_candidates)

        self._candidates = candidates
        logger.info("grid_candidates_generated", count=len(candidates))
        return candidates

    def generate_targeted_candidates(
        self,
        base_config: PipelineConfig,
        failure_patterns: list[str],
    ) -> list[ParameterCandidate]:
        """Generate candidates targeted at specific failure patterns."""
        candidates = []

        for pattern in failure_patterns:
            pattern_lower = pattern.lower()

            if "timeout" in pattern_lower:
                candidates.append(ParameterCandidate(
                    params={"execution_timeout": 120},
                    description="Increase timeout for timeout failures",
                ))

            if "syntax" in pattern_lower or "type_error" in pattern_lower:
                candidates.append(ParameterCandidate(
                    params={"temperature": 0.1},
                    description="Lower temperature for syntax errors",
                ))

            if "boolean" in pattern_lower or "geometry" in pattern_lower:
                candidates.append(ParameterCandidate(
                    params={"max_retries": 5, "temperature": 0.15},
                    description="More retries + lower temp for geometry errors",
                ))

        logger.info("targeted_candidates_generated", count=len(candidates))
        return candidates

    def rank_candidates(self) -> list[ParameterCandidate]:
        """Rank candidates by their scores (after evaluation)."""
        scored = [c for c in self._candidates if c.score is not None]
        return sorted(scored, key=lambda c: c.score or 0, reverse=True)
