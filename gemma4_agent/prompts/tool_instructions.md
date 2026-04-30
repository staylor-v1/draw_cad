# Gemma 4 CAD Tool Instructions

Use these tools as the only external actions available while reconstructing parts. Treat tool output as authoritative. The goal is not to produce a drawing that looks identical; the goal is to make the first and second STEP parts geometrically equivalent after a drawing roundtrip.

Recommended workflow:

1. Call `inspect_drawing` on the current drawing.
2. For cluttered raster engineering drawings, call `prepare_drawing_masks` and inspect the sheet/title-block masked image, annotation masked image, physical linework image, overlay, view frame metadata, callout candidates, and region metadata.
3. Call `segment_drawing_views` after masking to crop each inferred view separately and inspect the first-angle/third-angle role hypotheses before deciding what geometry to model.
4. If the drawing is an orthographic SVG triplet, call `run_deterministic_reconstruction` and inspect the verified candidate scores.
5. If raster drawing evidence clearly names a common feature family such as a connecting rod/link or C-shaped bracket/support, call `build_feature_template_cad` before writing freehand code. Refine the template dimensions if the rendered contact sheet is closer than a plain envelope but still missing details.
6. If `cad_construction_strategies` is present, select the primary construction strategy before writing code. Use the recommended strategy list to decide between revolved section profile, sketch/extrude/cut plate, additive/subtractive prismatic stock, sweep/loft thin-wall, template replay/refinement, or direct roundtrip reconstruction.
7. If deterministic/template reconstruction is unavailable or insufficient, write build123d code and call `execute_cad_code`.
8. If the source is a parseable SVG, call `evaluate_step_against_drawing`.
9. For roundtrip testing, call `render_step_to_drawing` on the first STEP, reconstruct from that generated drawing, then call `compare_cad_parts`.
10. Return final build123d code only after at least one successful execution or verified deterministic/template candidate.

Tool behavior:

- `inspect_drawing` identifies raster drawings and parses FreeCAD-style SVG orthographic triplets.
- `prepare_drawing_masks` creates heuristic mask artifacts for border/title-block regions, annotation candidates, isolated physical linework, inferred view frames, and GD&T/callout anchor candidates. Treat those masks as candidates to verify, not truth. All mask artifacts preserve the original `image_px` coordinate frame. Each region includes original-image bbox/center fields, and each callout candidate includes a `view_frame_id`, `target_endpoint_image_px`, and `target_endpoint_view_norm` so CAD feature planning can reference the original drawing evidence.
- `segment_drawing_views` crops each inferred view from the original, masked, and physical-linework images. Each crop includes a transform back to `image_px`, nearby annotation/callout IDs, and projection role hypotheses. Use explicit labels and projection symbols if legible. If labels are missing, compare the first-angle and third-angle hypotheses: third-angle places top/right/left views above/right/left of the front view; first-angle places them below/left/right of the front view.
- `run_deterministic_reconstruction` searches existing deterministic candidate generators and returns a best scored STEP for SVG triplets.
- `build_feature_template_cad` creates and executes parameterized CAD templates for common mechanical feature families. Use it to avoid collapsing a recognized connecting rod, link, C-bracket, closet-rod support, or curved support into a plain slab.
- `cad_construction_strategies` is harness context, not a tool. It provides a strategy catalogue and ranked recommendations. For source drawings with section diameters/counterbores/chamfers, prefer a revolved profile first; for constant-thickness plates, sketch the silhouette and extrude; for blocks, build from stock and subtract pockets/holes; for curved supports, sweep/loft along a path; for pass-two generated drawings, replay template candidates and compare before custom code.
- `execute_cad_code` runs build123d code in a subprocess and returns a STEP path or categorized error.
- `evaluate_step_against_drawing` compares STEP reprojection to SVG linework.
- `render_step_to_drawing` creates front/right/top SVGs and a PNG contact sheet from a STEP file.
- `compare_cad_parts` determines whether two STEP files are equivalent using geometry properties.
- `summarize_step` returns STEP validity, volume, surface area, bounding box, center of mass, and topology counts.
