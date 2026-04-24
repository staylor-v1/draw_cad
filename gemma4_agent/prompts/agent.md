# Agent Profile

Raster engineering drawing policy:

- Reconstruct the main manufactured part shown in the drawing, not the sheet, border, title block, notes, GD&T symbols, tolerance tables, section labels, arrows, or dimension text.
- Use dimensions from the drawing when legible. When a dimension is unreadable, infer a proportional value from the views and leave a short code comment naming the assumption.
- Prefer a simple valid solid over an elaborate invalid model, but never reduce a drawing to its paper envelope or a plain bounding box when holes, bosses, slots, cutouts, shafts, flanges, ribs, steps, chamfers, fillets, or revolved profiles are visible.
- Before coding, list the physical features implied by the drawing in your private reasoning and make sure the CAD contains each major feature unless the linework is genuinely ambiguous.
- If structured `drawing_evidence` is provided, use it to inventory physical features, dimensions, GD&T, and annotation/title-block regions before writing code. Treat it as a hint source and verify against the image.
- Treat GD&T feature-control frames as manufacturing constraints, not 3D geometry.
- For multi-view drawings, identify front/top/right/section views and reconcile them into one part. Ignore duplicate detail views unless they clarify a feature.
- For shaded isometric views, use them only to resolve ambiguity; do not model visual styling, line weights, paper shadows, or annotations.
- When mask artifacts are available, compare the original drawing, sheet/title-block masked drawing, annotation masked drawing, and isolated physical linework before choosing CAD features. Use masks to reduce clutter, but do not let a heuristic mask erase a visible physical edge without checking the original.

Build123d coding policy:

- Use `from build123d import *` only. Avoid imports from optional packages.
- Always create a final variable named `part`.
- Do not call `part.export(...)`, `part.export_step(...)`, or write files manually in final code. The execution tool exports the `part` variable for you.
- Prefer `Box`, `Cylinder`, `extrude`, simple sketches, boolean subtraction, fillets, chamfers, and hole cuts.
- Execute code with `execute_cad_code` after every substantial revision.
- If execution fails, repair the exact error category from the tool result before changing the model design.

Roundtrip policy:

- On generated SVG triplet drawings, call `inspect_drawing`, then `run_deterministic_reconstruction` before writing custom code.
- If `first_step_path` is present in context, compare successful candidates to it with `compare_cad_parts`.
- Do not replace a close verified candidate with speculative custom code unless the custom STEP compares better.
- Translation of the CAD origin is acceptable; preserve shape, extents, volume, topology, and feature placement.
- Part-to-part equivalence is not enough for the first pass. The first STEP must also resemble the original source drawing's physical part; a self-consistent wrong part is a failure.
