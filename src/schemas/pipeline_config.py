"""Pydantic configuration schemas for the drawing-to-CAD pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""
    name: str
    backend: str = "ollama"
    temperature: float = 0.2
    max_tokens: int = 4096


class InferenceBackendConfig(BaseModel):
    """Configuration for an inference backend."""
    base_url: str = "http://localhost:11434"
    timeout: int = 120


class InferenceConfig(BaseModel):
    """All inference backend configs."""
    ollama: InferenceBackendConfig = Field(default_factory=InferenceBackendConfig)
    vllm: InferenceBackendConfig = Field(default_factory=lambda: InferenceBackendConfig(base_url="http://localhost:8000"))
    llamacpp: InferenceBackendConfig = Field(default_factory=lambda: InferenceBackendConfig(base_url="http://localhost:8080"))


class ModelsConfig(BaseModel):
    """Model selection config."""
    vision: ModelConfig = Field(default_factory=lambda: ModelConfig(name="llama-3.2-11b-Vision", temperature=0.1))
    reasoning: ModelConfig = Field(default_factory=lambda: ModelConfig(name="gpt-oss-120b"))


class EvaluationWeights(BaseModel):
    """Weights for composite evaluation score."""
    dimension_fidelity: float = 0.30
    feature_recall: float = 0.25
    bounding_box_iou: float = 0.15
    volume_ratio: float = 0.10
    geometry_valid: float = 0.10
    feature_precision: float = 0.05
    retry_efficiency: float = 0.05


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    weights: EvaluationWeights = Field(default_factory=EvaluationWeights)
    dimension_tolerance_mm: float = 0.5
    volume_tolerance_ratio: float = 0.05


class PromptsConfig(BaseModel):
    """Paths to prompt templates."""
    system_prompt: str = "prompts/system_prompt.md"
    vision_prompt: str = "prompts/vision_prompt.md"
    retry_prompt: str = "prompts/retry_prompt.md"
    validation_prompt: str = "prompts/validation_prompt.md"
    reconciliation_prompt: str = "prompts/reconciliation_prompt.md"
    fewshot_index: str = "prompts/fewshot_examples/index.yaml"


class PipelineStageConfig(BaseModel):
    """Configuration for the pipeline stages."""
    max_retries: int = 3
    retry_backoff_factor: float = 1.5
    execution_timeout: int = 60
    ocr_enabled: bool = True
    default_unit: str = "mm"
    default_tolerance: float = 0.1


class OutputConfig(BaseModel):
    """Output configuration."""
    default_step_path: str = "output.step"
    experiments_dir: str = "experiments"


class PipelineConfig(BaseModel):
    """Root configuration for the entire pipeline."""
    pipeline: PipelineStageConfig = Field(default_factory=PipelineStageConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def deep_copy(self) -> PipelineConfig:
        """Return an independent copy of this config."""
        return PipelineConfig.model_validate(self.model_dump())
