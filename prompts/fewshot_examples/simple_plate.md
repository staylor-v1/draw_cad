## Example: Simple Rectangular Plate with Mounting Holes

### Drawing Description:
A rectangular plate 120mm × 80mm × 15mm with 4 mounting holes (M8, ∅8mm through-holes) positioned 10mm from each edge.

### build123d Code:
```python
from build123d import *

# Parameters
length = 120
width = 80
thickness = 15
hole_diameter = 8
edge_offset = 10

# Construction
with BuildPart() as part:
    # Base plate
    with BuildSketch(Plane.XY):
        Rectangle(length, width)
    Extrude(amount=thickness)

    # Mounting holes - 4 corners
    hole_positions = [
        (length/2 - edge_offset, width/2 - edge_offset),
        (-length/2 + edge_offset, width/2 - edge_offset),
        (length/2 - edge_offset, -width/2 + edge_offset),
        (-length/2 + edge_offset, -width/2 + edge_offset),
    ]
    with BuildSketch(part.faces().sort_by(Axis.Z)[-1]):
        for x, y in hole_positions:
            with Locations([(x, y)]):
                Circle(hole_diameter / 2)
    Extrude(amount=-thickness, mode=Mode.SUBTRACT)

# Export
part.part.export_step("output.step")
```
