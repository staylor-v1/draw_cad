# Gemma 4 CAD Tool Instructions

Use these tools as the only external actions available while reconstructing parts. Treat tool output as authoritative. The goal is not to produce a drawing that looks identical; the goal is to make the first and second STEP parts geometrically equivalent after a drawing roundtrip.

Recommended workflow:

1. Call `inspect_drawing` on the current drawing.
2. For cluttered raster engineering drawings, call `prepare_drawing_masks` and inspect the sheet/title-block masked image, annotation masked image, physical linework image, overlay, and region metadata before deciding what geometry to model.
3. If the drawing is an orthographic SVG triplet, call `run_deterministic_reconstruction` and inspect the verified candidate scores.
4. If deterministic reconstruction is unavailable or insufficient, write build123d code and call `execute_cad_code`.
5. If the source is a parseable SVG, call `evaluate_step_against_drawing`.
6. For roundtrip testing, call `render_step_to_drawing` on the first STEP, reconstruct from that generated drawing, then call `compare_cad_parts`.
7. Return final build123d code only after at least one successful execution or verified deterministic candidate.

Tool behavior:

- `inspect_drawing` identifies raster drawings and parses FreeCAD-style SVG orthographic triplets.
- `prepare_drawing_masks` creates heuristic mask artifacts for border/title-block regions, annotation candidates, and isolated physical linework. Treat those masks as candidates to verify, not truth.
- `run_deterministic_reconstruction` searches existing deterministic candidate generators and returns a best scored STEP for SVG triplets.
- `execute_cad_code` runs build123d code in a subprocess and returns a STEP path or categorized error.
- `evaluate_step_against_drawing` compares STEP reprojection to SVG linework.
- `render_step_to_drawing` creates front/right/top SVGs and a PNG contact sheet from a STEP file.
- `compare_cad_parts` determines whether two STEP files are equivalent using geometry properties.
- `summarize_step` returns STEP validity, volume, surface area, bounding box, center of mass, and topology counts.
