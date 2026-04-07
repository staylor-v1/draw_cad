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


class TotalViewDataConfig(BaseModel):
    """Configuration for the zipped Total_view_data dataset."""

    root_dir: str = "/home/me/Downloads/Total_view_data"
    default_dataset: str = "ABC"
    preferred_views: list[str] = Field(
        default_factory=lambda: ["f", "r", "t"]
    )
    require_complete_triplet: bool = True
    cache_dir: str = "experiments/total_view_data/cache"
    svg_archives: dict[str, str] = Field(
        default_factory=lambda: {
            "ABC": "/home/me/Downloads/Total_view_data/SVG/ABC_data_SVG.zip",
            "CSG": "/home/me/Downloads/Total_view_data/SVG/CSG_data_SVG.zip",
        }
    )
    png_archives: dict[str, str] = Field(
        default_factory=lambda: {
            "ABC": "/home/me/Downloads/Total_view_data/PNG/ABC_data_PNG.zip",
            "CSG": "/home/me/Downloads/Total_view_data/PNG/CSG_data_PNG.zip",
        }
    )

    def get_svg_archive(self, dataset: str | None = None) -> Path:
        """Return the SVG archive path for a configured dataset name."""
        key = dataset or self.default_dataset
        return Path(self.svg_archives[key])

    def get_png_archive(self, dataset: str | None = None) -> Path | None:
        """Return the PNG archive path when one is configured."""
        key = dataset or self.default_dataset
        path = self.png_archives.get(key)
        return Path(path) if path else None


class OrthographicReconstructionConfig(BaseModel):
    """Deterministic orthographic reconstruction settings."""

    enabled: bool = True
    prefer_svg: bool = True
    raster_max_dimension_px: int = 768
    raster_padding_px: int = 10
    contour_simplify_tolerance_px: float = 2.5
    dilation_iterations: int = 1
    min_component_area_ratio: float = 0.05
    profile_component_area_ratio: float = 0.005
    visible_stroke_colors: list[str] = Field(
        default_factory=lambda: ["black", "#000000"]
    )
    hidden_feature_enabled: bool = True
    hidden_feature_circle_min_points: int = 20
    hidden_feature_circle_aspect_ratio_max: float = 1.1
    hidden_feature_circle_fit_error_max: float = 0.03
    hidden_feature_circle_min_coverage: float = 0.9
    hidden_feature_circle_min_radius: float = 0.6
    hidden_feature_match_center_tolerance: float = 0.5
    hidden_feature_match_radius_tolerance: float = 0.5
    hidden_feature_segment_merge_tolerance: float = 0.3
    hidden_feature_segment_join_gap: float = 0.2
    hidden_feature_z_cluster_tolerance: float = 0.5
    hidden_feature_min_depth: float = 0.5
    candidate_search_enabled: bool = True
    axisymmetric_enabled: bool = True
    axisymmetric_top_circularity_min: float = 0.9
    axisymmetric_top_aspect_ratio_max: float = 1.1
    axisymmetric_center_tolerance_ratio: float = 0.04
    axisymmetric_profile_tolerance_ratio: float = 0.08
    axisymmetric_profile_samples: int = 96


class ReprojectionConfig(BaseModel):
    """Rasterized reprojection benchmark settings."""

    raster_size_px: int = 640
    raster_padding_px: int = 20
    line_width_px: int = 2
    curve_samples: int = 48
    hidden_line_subtract_dilation_px: int = 2
    match_tolerance_px: int = 2
    overlay_enabled: bool = True


class OutputConfig(BaseModel):
    """Output configuration."""
    default_step_path: str = "output.step"
    experiments_dir: str = "experiments"
    total_view_data_output_dir: str = "experiments/total_view_data/steps"
    total_view_data_summary_path: str = "experiments/total_view_data/summary.json"
    total_view_data_benchmark_dir: str = "experiments/total_view_data/benchmark"
    total_view_data_benchmark_summary_path: str = (
        "experiments/total_view_data/benchmark_summary.json"
    )


class PipelineConfig(BaseModel):
    """Root configuration for the entire pipeline."""
    pipeline: PipelineStageConfig = Field(default_factory=PipelineStageConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    total_view_data: TotalViewDataConfig = Field(default_factory=TotalViewDataConfig)
    orthographic_reconstruction: OrthographicReconstructionConfig = Field(
        default_factory=OrthographicReconstructionConfig
    )
    reprojection: ReprojectionConfig = Field(default_factory=ReprojectionConfig)
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
