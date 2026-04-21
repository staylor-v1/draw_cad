"""Geometry data schemas for the drawing-to-CAD pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class UnitType(str, Enum):
    MM = "mm"
    INCH = "inch"
    CM = "cm"


class ProjectionPlane(str, Enum):
    TOP = "Top"
    FRONT = "Front"
    RIGHT = "Right"
    LEFT = "Left"
    BOTTOM = "Bottom"
    REAR = "Rear"
    ISOMETRIC = "Isometric"
    SECTION = "Section"


class Dimension(BaseModel):
    """A single dimension extracted from a drawing."""
    label: str
    value: float
    unit: str = "mm"
    tolerance: Optional[str] = None
    count: Optional[int] = None
    view: Optional[str] = None
    confidence: float = 1.0


class Feature(BaseModel):
    """A geometric feature identified in a drawing."""
    type: str  # e.g., "Through Hole", "Fillet", "Chamfer", "Slot"
    description: str
    dimensions: list[Dimension] = Field(default_factory=list)
    view: Optional[str] = None
    confidence: float = 1.0


class TextRegion(BaseModel):
    """A text region detected by OCR."""
    text: str
    bbox: list[float] = Field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float = 1.0
    category: str = "unknown"  # "dimension", "note", "title", "tolerance", "label"


class ViewData(BaseModel):
    """Data extracted from a single view of the drawing."""
    view_type: str  # ProjectionPlane value
    dimensions: list[Dimension] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GeometryData(BaseModel):
    """Raw geometry data extracted by the vision model from a drawing."""
    views: list[str] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    notes: str = ""
    raw_response: str = ""

    @classmethod
    def from_vision_dict(cls, data: dict[str, Any]) -> GeometryData:
        """Create from the vision model's raw JSON output."""
        dims = []
        for d in data.get("dimensions", []):
            dims.append(Dimension(
                label=d.get("label", ""),
                value=float(d.get("value", 0)),
                unit=d.get("unit", "mm"),
                tolerance=d.get("tolerance"),
                count=d.get("count"),
            ))
        feats = []
        for f in data.get("features", []):
            feats.append(Feature(
                type=f.get("type", ""),
                description=f.get("description", ""),
            ))
        return cls(
            views=data.get("views", []),
            dimensions=dims,
            features=feats,
            notes=data.get("notes", ""),
        )


class EnrichedGeometry(BaseModel):
    """Geometry data enriched with OCR text regions."""
    geometry: GeometryData
    text_regions: list[TextRegion] = Field(default_factory=list)
    ocr_dimensions: list[Dimension] = Field(default_factory=list)


class DrawingMetadata(BaseModel):
    """High-level metadata parsed from drawing title block text."""
    title: Optional[str] = None
    author: Optional[str] = None
    drawing_number: Optional[str] = None
    revision: Optional[str] = None
    scale: Optional[str] = None
    date: Optional[str] = None
    raw_fields: dict[str, str] = Field(default_factory=dict)


class GDTRemark(BaseModel):
    """A GD&T annotation extracted from OCR text."""
    text: str
    bbox: list[float] = Field(default_factory=list)
    confidence: float = 1.0


class DrawingTextRepresentation(BaseModel):
    """Text-forward representation of a drawing with vectorized geometry."""
    metadata: DrawingMetadata = Field(default_factory=DrawingMetadata)
    gdt_markups: list[GDTRemark] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    svg: str = ""
    masked_image_path: Optional[str] = None

    def to_text(self) -> str:
        """Serialize the representation into a compact text form."""
        lines: list[str] = ["# Drawing Representation"]
        lines.append("## Metadata")
        lines.append(f"- Title: {self.metadata.title or 'unknown'}")
        lines.append(f"- Author: {self.metadata.author or 'unknown'}")
        lines.append(f"- Drawing Number: {self.metadata.drawing_number or 'unknown'}")
        lines.append(f"- Revision: {self.metadata.revision or 'unknown'}")
        lines.append(f"- Scale: {self.metadata.scale or 'unknown'}")
        lines.append(f"- Date: {self.metadata.date or 'unknown'}")

        if self.metadata.raw_fields:
            lines.append("- Raw Fields:")
            for key, value in sorted(self.metadata.raw_fields.items()):
                lines.append(f"  - {key}: {value}")

        lines.append("## GD&T")
        if not self.gdt_markups:
            lines.append("- none detected")
        else:
            for mark in self.gdt_markups:
                lines.append(f"- {mark.text} @ {mark.bbox}")

        lines.append("## Dimensions")
        lines.extend(f"- {item}" for item in self.dimensions) if self.dimensions else lines.append("- none detected")

        lines.append("## Notes")
        lines.extend(f"- {item}" for item in self.notes) if self.notes else lines.append("- none detected")

        lines.append("## SVG")
        lines.append(self.svg)
        return "\n".join(lines)


class ReconciledGeometry(BaseModel):
    """Single 3D description reconciled from multiple 2D views."""
    overall_dimensions: dict[str, float] = Field(default_factory=dict)  # length, width, height
    features: list[Feature] = Field(default_factory=list)
    material: Optional[str] = None
    default_unit: str = "mm"
    default_tolerance: float = 0.1
    notes: list[str] = Field(default_factory=list)
    view_data: list[ViewData] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)  # Any unresolved conflicts

    def to_prompt_text(self) -> str:
        """Convert to a text description suitable for the reasoning prompt."""
        lines = ["## Reconciled 3D Geometry Description", ""]

        if self.overall_dimensions:
            lines.append("### Overall Dimensions")
            for name, val in self.overall_dimensions.items():
                lines.append(f"- {name}: {val} {self.default_unit}")
            lines.append("")

        if self.features:
            lines.append("### Features")
            for i, feat in enumerate(self.features, 1):
                lines.append(f"{i}. **{feat.type}**: {feat.description}")
                for d in feat.dimensions:
                    tol = f" ({d.tolerance})" if d.tolerance else ""
                    lines.append(f"   - {d.label}: {d.value} {d.unit}{tol}")
            lines.append("")

        if self.material:
            lines.append(f"### Material: {self.material}")
            lines.append("")

        if self.notes:
            lines.append("### Notes")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        if self.conflicts:
            lines.append("### Unresolved Conflicts (use best engineering judgment)")
            for c in self.conflicts:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)
