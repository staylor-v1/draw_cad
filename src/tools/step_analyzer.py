"""STEP file property extraction for evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StepProperties:
    """Properties extracted from a STEP file."""
    is_valid: bool = False
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    bounding_box: Optional[list[list[float]]] = None  # [[min],[max]]
    center_of_mass: Optional[list[float]] = None
    face_count: int = 0
    edge_count: int = 0
    vertex_count: int = 0
    solid_count: int = 0
    errors: list[str] = field(default_factory=list)


def analyze_step_file(step_path: str | Path) -> StepProperties:
    """Extract geometric properties from a STEP file.
    
    Uses OCP (OpenCASCADE) for precise B-Rep analysis when available,
    falls back to trimesh for mesh-based approximation.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        return StepProperties(errors=[f"File not found: {step_path}"])

    # Try OCP-based analysis first
    props = _analyze_with_ocp(step_path)
    if props.is_valid:
        return props

    # Fallback to trimesh
    props = _analyze_with_trimesh(step_path)
    return props


def _analyze_with_ocp(step_path: Path) -> StepProperties:
    """Analyze STEP file using OCP (OpenCASCADE Python bindings)."""
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID

        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        if status != 1:  # IFSelect_RetDone
            return StepProperties(errors=["Failed to read STEP file with OCP"])

        reader.TransferRoots()
        shape = reader.OneShape()

        # Volume and surface area
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        volume = props.Mass()

        sprops = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, sprops)
        surface_area = sprops.Mass()

        # Center of mass
        com = props.CentreOfMass()
        center_of_mass = [com.X(), com.Y(), com.Z()]

        # Bounding box
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        bounding_box = [[xmin, ymin, zmin], [xmax, ymax, zmax]]

        # Topology counts
        def count_topo(shape, topo_type):
            explorer = TopExp_Explorer(shape, topo_type)
            count = 0
            while explorer.More():
                count += 1
                explorer.Next()
            return count

        result = StepProperties(
            is_valid=True,
            volume=volume,
            surface_area=surface_area,
            bounding_box=bounding_box,
            center_of_mass=center_of_mass,
            face_count=count_topo(shape, TopAbs_FACE),
            edge_count=count_topo(shape, TopAbs_EDGE),
            vertex_count=count_topo(shape, TopAbs_VERTEX),
            solid_count=count_topo(shape, TopAbs_SOLID),
        )
        logger.info("ocp_analysis_complete", volume=volume, faces=result.face_count)
        return result

    except ImportError:
        logger.debug("ocp_not_available")
        return StepProperties(errors=["OCP not available"])
    except Exception as e:
        logger.error("ocp_analysis_error", error=str(e))
        return StepProperties(errors=[f"OCP error: {e}"])


def _analyze_with_trimesh(step_path: Path) -> StepProperties:
    """Fallback STEP analysis using trimesh."""
    try:
        import trimesh

        mesh = trimesh.load(str(step_path))
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return StepProperties(errors=["Empty geometry"])
            combined = trimesh.util.concatenate(list(mesh.geometry.values()))
        elif isinstance(mesh, trimesh.Trimesh):
            combined = mesh
        else:
            return StepProperties(errors=[f"Unexpected type: {type(mesh).__name__}"])

        bounds = combined.bounds.tolist() if combined.bounds is not None else None
        return StepProperties(
            is_valid=True,
            volume=float(combined.volume) if combined.is_watertight else None,
            bounding_box=bounds,
            face_count=len(combined.faces),
            edge_count=len(combined.edges_unique),
            vertex_count=len(combined.vertices),
        )
    except ImportError:
        return StepProperties(errors=["Neither OCP nor trimesh available"])
    except Exception as e:
        return StepProperties(errors=[f"trimesh error: {e}"])
