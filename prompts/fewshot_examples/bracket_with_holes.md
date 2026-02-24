## Example: L-Shaped Bracket with Mounting Holes

### Drawing Description:
An L-shaped bracket. Base: 100mm × 60mm × 8mm. Vertical wall: 60mm × 50mm × 8mm rising from one edge. Two ∅6mm mounting holes in the base, centered 15mm from the short edges and centered on the width.

### build123d Code:
```python
from build123d import *

# Parameters
base_length = 100
base_width = 60
base_thickness = 8
wall_height = 50
wall_thickness = 8
hole_diameter = 6
hole_offset = 15

# Construction
with BuildPart() as part:
    # Base plate
    with BuildSketch(Plane.XY):
        Rectangle(base_length, base_width)
    Extrude(amount=base_thickness)

    # Vertical wall on one edge
    wall_face = part.faces().sort_by(Axis.Z)[-1]
    with BuildSketch(wall_face):
        with Locations([(base_length/2 - wall_thickness/2, 0)]):
            Rectangle(wall_thickness, base_width)
    Extrude(amount=wall_height)

    # Mounting holes in base
    base_top = part.faces().filter_by(Axis.Z).sort_by(Axis.Z)[0]
    with BuildSketch(Plane.XY.offset(base_thickness)):
        for x_pos in [-base_length/2 + hole_offset, base_length/2 - hole_offset - wall_thickness]:
            with Locations([(x_pos, 0)]):
                Circle(hole_diameter / 2)
    Extrude(amount=-base_thickness, mode=Mode.SUBTRACT)

# Export
part.part.export_step("output.step")
```
