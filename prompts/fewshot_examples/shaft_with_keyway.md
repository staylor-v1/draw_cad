## Example: Cylindrical Shaft with Keyway

### Drawing Description:
A cylindrical shaft, diameter 30mm, length 100mm. A rectangular keyway cut along the top: 8mm wide × 4mm deep × 40mm long, centered on the shaft length.

### build123d Code:
```python
from build123d import *

# Parameters
shaft_diameter = 30
shaft_length = 100
keyway_width = 8
keyway_depth = 4
keyway_length = 40

# Construction
with BuildPart() as part:
    # Main shaft (cylinder along Z axis)
    with BuildSketch(Plane.XY):
        Circle(shaft_diameter / 2)
    Extrude(amount=shaft_length)

    # Keyway (rectangular cut along the top)
    keyway_z_start = (shaft_length - keyway_length) / 2
    keyway_plane = Plane.XY.offset(keyway_z_start)
    with BuildSketch(Plane(
        origin=(0, shaft_diameter/2 - keyway_depth/2, shaft_length/2),
        z_dir=(0, 0, 1),
        x_dir=(1, 0, 0)
    )):
        Rectangle(keyway_width, keyway_length)
    Extrude(amount=keyway_depth, both=False, mode=Mode.SUBTRACT)

# Export
part.part.export_step("output.step")
```
