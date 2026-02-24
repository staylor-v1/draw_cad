"""Multi-view reconciliation: merge multiple 2D views into a single 3D description."""
from __future__ import annotations

from typing import Optional

from src.inference.base import BaseLLMClient, ChatMessage
from src.schemas.geometry import (
    Dimension,
    EnrichedGeometry,
    Feature,
    GeometryData,
    ReconciledGeometry,
    ViewData,
)
from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_prompt
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Mapping from view names to primary axes
VIEW_AXIS_MAP = {
    "Top": {"primary": "XY", "depth": "Z"},
    "Front": {"primary": "XZ", "depth": "Y"},
    "Right": {"primary": "YZ", "depth": "X"},
    "Left": {"primary": "YZ", "depth": "X"},
    "Bottom": {"primary": "XY", "depth": "Z"},
    "Rear": {"primary": "XZ", "depth": "Y"},
}

# Common dimension label to axis mapping heuristics
LABEL_AXIS_HINTS = {
    "length": "X",
    "width": "Y",
    "height": "Z",
    "thickness": "Z",
    "depth": "Y",
    "diameter": None,  # could be any axis
    "radius": None,
}


class ReconciliationStage:
    """Reconciles multi-view 2D geometry into a single 3D description."""

    def __init__(self, config: PipelineConfig, llm_client: BaseLLMClient | None = None):
        self.config = config
        self.llm_client = llm_client

    def run(self, enriched: EnrichedGeometry) -> ReconciledGeometry:
        """Reconcile enriched geometry into a 3D description."""
        logger.info("reconciliation_start")
        geometry = enriched.geometry

        # Build view data
        view_data = self._build_view_data(geometry)

        # Map dimensions to overall 3D dimensions
        overall = self._map_overall_dimensions(geometry.dimensions)

        # Merge features, deduplicating across views
        features = self._merge_features(geometry.features)

        # Supplement dimensions from OCR
        for ocr_dim in enriched.ocr_dimensions:
            if not self._dimension_already_present(ocr_dim, geometry.dimensions):
                geometry.dimensions.append(ocr_dim)

        # Detect conflicts
        conflicts = self._detect_conflicts(geometry.dimensions)

        # If we have an LLM client and conflicts, use it to resolve
        if conflicts and self.llm_client is not None:
            resolved = self._llm_resolve_conflicts(enriched, conflicts)
            if resolved:
                conflicts = resolved.conflicts
                if resolved.overall_dimensions:
                    overall.update(resolved.overall_dimensions)

        # Extract notes
        notes = []
        if geometry.notes:
            notes.append(geometry.notes)
        for region in enriched.text_regions:
            if region.category == "note":
                notes.append(region.text)

        result = ReconciledGeometry(
            overall_dimensions=overall,
            features=features,
            default_unit=self.config.pipeline.default_unit,
            default_tolerance=self.config.pipeline.default_tolerance,
            notes=notes,
            view_data=view_data,
            conflicts=conflicts,
        )
        logger.info(
            "reconciliation_complete",
            overall_dims=len(overall),
            features=len(features),
            conflicts=len(conflicts),
        )
        return result

    def _build_view_data(self, geometry: GeometryData) -> list[ViewData]:
        """Organize dimensions and features by view."""
        views = {}
        for view_name in geometry.views:
            views[view_name] = ViewData(view_type=view_name)

        for dim in geometry.dimensions:
            if dim.view and dim.view in views:
                views[dim.view].dimensions.append(dim)
            elif geometry.views:
                # Assign to first view if no view specified
                views[geometry.views[0]].dimensions.append(dim)

        for feat in geometry.features:
            if feat.view and feat.view in views:
                views[feat.view].features.append(feat)
            elif geometry.views:
                views[geometry.views[0]].features.append(feat)

        return list(views.values())

    def _map_overall_dimensions(self, dimensions: list[Dimension]) -> dict[str, float]:
        """Map labeled dimensions to overall 3D dimensions."""
        overall: dict[str, float] = {}
        for dim in dimensions:
            label_lower = dim.label.lower()
            for key, axis in LABEL_AXIS_HINTS.items():
                if key in label_lower and dim.value > 0:
                    overall[dim.label.lower()] = dim.value
                    break
        return overall

    def _merge_features(self, features: list[Feature]) -> list[Feature]:
        """Deduplicate features from multiple views."""
        seen: dict[str, Feature] = {}
        for feat in features:
            key = f"{feat.type}:{feat.description}".lower()
            if key not in seen:
                seen[key] = feat
            else:
                # Keep higher confidence
                if feat.confidence > seen[key].confidence:
                    seen[key] = feat
        return list(seen.values())

    def _dimension_already_present(
        self, new_dim: Dimension, existing: list[Dimension], tolerance: float = 0.5
    ) -> bool:
        """Check if a dimension value is already captured."""
        for dim in existing:
            if abs(dim.value - new_dim.value) < tolerance:
                return True
        return False

    def _detect_conflicts(self, dimensions: list[Dimension]) -> list[str]:
        """Detect conflicting dimensions (same label, different values)."""
        conflicts = []
        by_label: dict[str, list[Dimension]] = {}
        for dim in dimensions:
            label = dim.label.lower()
            by_label.setdefault(label, []).append(dim)

        for label, dims in by_label.items():
            if len(dims) > 1:
                values = [d.value for d in dims]
                if max(values) - min(values) > 0.5:
                    conflicts.append(
                        f"Conflicting values for '{label}': {values}"
                    )
        return conflicts

    def _llm_resolve_conflicts(
        self, enriched: EnrichedGeometry, conflicts: list[str]
    ) -> Optional[ReconciledGeometry]:
        """Use LLM to help resolve conflicts between views."""
        try:
            prompt_text = load_prompt(self.config.prompts.reconciliation_prompt)
        except FileNotFoundError:
            logger.warning("reconciliation_prompt_not_found")
            return None

        geometry_desc = (
            f"Views: {enriched.geometry.views}\n"
            f"Dimensions: {[d.model_dump() for d in enriched.geometry.dimensions]}\n"
            f"Features: {[f.model_dump() for f in enriched.geometry.features]}\n"
            f"Conflicts: {conflicts}"
        )

        messages = [
            ChatMessage(role="system", content=prompt_text),
            ChatMessage(role="user", content=geometry_desc),
        ]

        try:
            model_cfg = self.config.models.reasoning
            response = self.llm_client.chat(
                messages=messages,
                model=model_cfg.name,
                temperature=model_cfg.temperature,
                max_tokens=model_cfg.max_tokens,
            )
            # Parse response - in practice the LLM returns structured resolution
            logger.info("llm_conflict_resolution_complete")
            return None  # For now, return None; LLM response parsing can be added later
        except Exception as e:
            logger.warning("llm_conflict_resolution_failed", error=str(e))
            return None
