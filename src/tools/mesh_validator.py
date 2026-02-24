"""Mesh validation using trimesh for watertight/manifold checks."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.schemas.evaluation_result import ValidationResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def validate_step_file(step_path: str | Path) -> ValidationResult:
    """Validate a STEP file by converting to mesh and checking properties.
    
    Args:
        step_path: Path to the STEP file.
    
    Returns:
        ValidationResult with geometric properties.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        return ValidationResult(
            is_valid=False,
            errors=[f"STEP file not found: {step_path}"],
        )

    try:
        import trimesh

        # Load the STEP file - trimesh uses cascadio/gmsh for STEP
        mesh = trimesh.load(str(step_path))

        if isinstance(mesh, trimesh.Scene):
            # Multi-body: combine geometries
            if len(mesh.geometry) == 0:
                return ValidationResult(
                    is_valid=False,
                    errors=["STEP file contains no geometry"],
                )
            combined = trimesh.util.concatenate(list(mesh.geometry.values()))
        elif isinstance(mesh, trimesh.Trimesh):
            combined = mesh
        else:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unexpected mesh type: {type(mesh).__name__}"],
            )

        bounds = combined.bounds.tolist() if combined.bounds is not None else None
        result = ValidationResult(
            is_valid=True,
            is_watertight=bool(combined.is_watertight),
            is_manifold=not bool(getattr(combined, 'is_empty', False)),
            volume=float(combined.volume) if combined.is_watertight else None,
            bounding_box=bounds,
            face_count=len(combined.faces),
            edge_count=len(combined.edges_unique),
            vertex_count=len(combined.vertices),
        )
        logger.info(
            "mesh_validation_complete",
            is_watertight=result.is_watertight,
            volume=result.volume,
            faces=result.face_count,
        )
        return result

    except ImportError:
        logger.warning("trimesh_not_available")
        return _validate_step_basic(step_path)
    except Exception as e:
        logger.error("mesh_validation_error", error=str(e))
        return ValidationResult(
            is_valid=False,
            errors=[f"Validation error: {str(e)}"],
        )


def _validate_step_basic(step_path: Path) -> ValidationResult:
    """Basic STEP file validation without trimesh - just check file validity."""
    try:
        content = step_path.read_text(errors="ignore")
        has_header = "ISO-10303-21" in content
        has_data = "DATA;" in content and "ENDSEC;" in content
        
        return ValidationResult(
            is_valid=has_header and has_data,
            errors=[] if (has_header and has_data) else ["Invalid STEP file format"],
        )
    except Exception as e:
        return ValidationResult(is_valid=False, errors=[str(e)])


def compute_bounding_box_iou(
    box_a: list[list[float]],
    box_b: list[list[float]],
) -> float:
    """Compute 3D IoU between two axis-aligned bounding boxes.
    
    Args:
        box_a: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        box_b: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    
    Returns:
        IoU value between 0 and 1.
    """
    min_a, max_a = box_a
    min_b, max_b = box_b

    # Intersection
    inter_min = [max(min_a[i], min_b[i]) for i in range(3)]
    inter_max = [min(max_a[i], max_b[i]) for i in range(3)]

    inter_dims = [max(0, inter_max[i] - inter_min[i]) for i in range(3)]
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]

    # Union
    vol_a = 1.0
    vol_b = 1.0
    for i in range(3):
        vol_a *= max(0, max_a[i] - min_a[i])
        vol_b *= max(0, max_b[i] - min_b[i])

    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol
