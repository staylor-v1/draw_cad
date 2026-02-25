"""Pydantic schema for precomputed STEP ground truth properties."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.tools.step_analyzer import StepProperties


class StepGroundTruth(BaseModel):
    """Precomputed ground truth properties from a reference STEP file.

    This captures the key geometric properties once so they can be
    serialised to manifest.json and reused across Loop 2 iterations
    without re-analysing the STEP file each time.
    """

    volume: Optional[float] = None
    surface_area: Optional[float] = None
    bounding_box: Optional[list[list[float]]] = None  # [[min_x,min_y,min_z],[max_x,max_y,max_z]]
    center_of_mass: Optional[list[float]] = None
    face_count: int = 0
    edge_count: int = 0
    vertex_count: int = 0
    solid_count: int = 0
    bbox_extents: list[float] = Field(
        default_factory=list,
        description="Sorted [dx, dy, dz] bounding-box extents (ascending).",
    )
    is_valid: bool = False


def step_properties_to_ground_truth(props: StepProperties) -> StepGroundTruth:
    """Convert a ``StepProperties`` result into a ``StepGroundTruth`` record."""
    bbox_extents: list[float] = []
    if props.bounding_box and len(props.bounding_box) == 2:
        mn, mx = props.bounding_box
        extents = [abs(mx[i] - mn[i]) for i in range(3)]
        bbox_extents = sorted(extents)

    return StepGroundTruth(
        volume=props.volume,
        surface_area=props.surface_area,
        bounding_box=props.bounding_box,
        center_of_mass=props.center_of_mass,
        face_count=props.face_count,
        edge_count=props.edge_count,
        vertex_count=props.vertex_count,
        solid_count=props.solid_count,
        bbox_extents=bbox_extents,
        is_valid=props.is_valid,
    )
