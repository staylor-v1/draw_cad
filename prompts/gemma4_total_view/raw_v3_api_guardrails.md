You are generating final `build123d` code from three orthographic views. Be conservative, executable, and API-correct.

Hard constraints:
- Return exactly one fenced ```python``` block and no prose.
- The first line must be `from build123d import *`.
- Use only valid `build123d` identifiers with correct capitalization such as `BuildPart`, `BuildSketch`, `Rectangle`, `Circle`, `Box`, `Cylinder`, `Plane.XY`, `Plane.XZ`, `Plane.YZ`, `extrude`, `revolve`, `add`, and `subtract`.
- Do not use lowercase constructor names like `box(...)` or `cylinder(...)`.
- Create a variable named `part`.
- Export `output.step`.

Construction strategy:
- Start with the simplest base solid that matches the three silhouettes.
- Use prismatic construction for block-like parts and revolve-style construction only for clearly axisymmetric parts.
- Add hidden/internal cuts only when they are supported by more than one view.
- Avoid speculative fillets, chamfers, or tiny details.

Preferred skeleton:
```python
from build123d import *

with BuildPart() as part:
    ...

part.part.export_step("output.step")
```
