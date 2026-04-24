# Agentic GD&T Drawing-To-CAD Harness Plan

## Goal

Develop an agentic harness that uses an LLM for reasoning and image analysis of
GD&T-marked engineering drawings, then reconstructs the main manufactured part as
build123d CAD and STEP. The harness should combine LLM judgment with specialized
tools: deterministic CAD reconstruction, STEP analysis, drawing reprojection, geometry
comparison, and optional local detector/VLM backends such as YOLO OBB, Florence-2, and
Donut.

The central design principle is that extracted evidence is a hint source, not truth.
Gemma 4, or whichever reasoning model is driving the harness, must understand its
options, verify tool outputs against the drawing, and keep annotation semantics
separate from physical CAD geometry.

## Current Harness Capabilities

- `Gemma4RoundTripAgent` orchestrates drawing -> CAD -> STEP -> drawing -> CAD -> STEP
  and compares the two STEP files for geometric stability.
- `toolbox.py` exposes agent-callable tools:
  - `inspect_drawing`
  - `run_deterministic_reconstruction`
  - `execute_cad_code`
  - `evaluate_step_against_drawing`
  - `render_step_to_drawing`
  - `compare_cad_parts`
  - `summarize_step`
- `extractors.py` can provide structured drawing evidence from:
  - `heuristic`: local raster/layout hints
  - `gemma4`: local Ollama vision reasoning
  - `florence2`: optional local fine-tuned Florence-2 adapter
  - `yolo_donut`: optional YOLO OBB detector plus Donut adapter point
- `training.py` and the GD&T scripts run prompt/evidence experiments and source
  fidelity checks against the original drawing.
- The current pass/fail gate distinguishes roundtrip stability from source fidelity:
  a self-consistent wrong part is still a failure.

## Reasoning Agent Options

The harness should make these choices visible to the reasoning agent:

- Use `inspect_drawing` first to understand whether the source is a raster drawing,
  parseable orthographic SVG triplet, or generated SVG layout.
- Use evidence extractors before CAD reconstruction when the input is a real raster
  engineering drawing.
- Use detector/VLM evidence to identify and mask non-geometry regions, not to blindly
  add geometry.
- Use deterministic reconstruction for parseable SVG triplets and generated drawings.
- Use `execute_cad_code` for custom build123d programs and repair tool errors before
  changing the modeling strategy.
- Use `render_step_to_drawing` and source-fidelity judging to check whether the first
  STEP still resembles the original physical part.
- Use `compare_cad_parts` to choose among successful pass-2 candidates.

Other models may be tools. Gemma 4 can remain the coordinating reasoning model while
YOLO, Florence-2, Donut, OCR, curve tracing, and deterministic reconstruction act as
specialized perception or geometry tools.

## Proposed Agentic Workflow

This is the current working hypothesis, not a fixed prescription. It should evolve
through experiments.

1. **Sheet Triage**

   Identify the page border, title block, revision table, notes, material block,
   general tolerances, projection symbols, scale text, and other drawing-description
   regions. These are important context, but they are not CAD solids.

   Candidate tools:
   - Gemma 4 structured evidence extraction
   - YOLO OBB title-block/note/table detectors
   - Florence-2 or Donut document-structure parsing
   - local raster heuristics for border density and sheet aspect ratio

   Output:
   - structured regions with labels and confidence
   - a masked image where border/title-block/description regions are removed
   - preserved text evidence for dimensions, material, scale, and tolerances

2. **GD&T And Annotation Triage**

   Recognize GD&T feature-control frames, datum tags, dimension text, arrows,
   extension lines, centerlines, section labels, detail labels, and callout leader
   lines. Determine which physical feature each callout is associated with whenever
   possible.

   Candidate tools:
   - YOLO OBB for GD&T symbols, dimensions, datum tags, and callout regions
   - Florence-2/Donut fine-tuning for structured annotation JSON
   - Gemma 4 to reason over ambiguous associations
   - OCR for legible dimensions and labels

   Output:
   - structured GD&T and dimension evidence
   - association graph from annotations to drawing entities or feature regions
   - a second masked image where annotations/callouts are removed or dimmed

3. **Physical Linework Isolation**

   Work from the masked drawing that contains mainly the 2D projections of the part.
   Preserve physical visible edges, hidden edges, section outlines, centerlines, and
   relevant construction cues. Suppress sheet furniture and annotation clutter.

   Candidate tools:
   - thresholding and connected-component analysis
   - line/arc/circle detection
   - curve tracing
   - view segmentation
   - Gemma 4 review of masked images

   Output:
   - per-view linework masks
   - candidate edge primitives: lines, arcs, circles, ellipses, splines
   - uncertainty regions where physical linework and annotations overlap

4. **View Identification And Reconciliation**

   Determine which views are present: front, top, right, section, detail, isometric,
   auxiliary, or repeated view. Reconcile view scale, alignment, hidden lines, section
   cuts, and projected feature positions into a single 3D interpretation.

   Candidate tools:
   - deterministic SVG triplet parser where available
   - view-layout heuristics
   - Gemma 4 multimodal reasoning
   - geometric consistency checks against candidate extents

   Output:
   - selected canonical views
   - view-to-axis mapping
   - scale and unit assumptions
   - rejected decorative/detail-only regions

5. **Feature Fitting And Geometry Recovery**

   Fit geometric primitives from isolated linework and annotation evidence. Use
   dimensions and GD&T constraints to inform CAD operations without turning annotations
   into solids.

   Candidate tools:
   - circle and arc fitting for holes, bores, bosses, flanges, and revolved profiles
   - curve tracing for profiles and slots
   - visual-hull or profile-extrude candidates
   - axisymmetric reconstruction for turned parts
   - section-view reasoning for internal bores and steps

   Output:
   - feature inventory: holes, slots, pockets, bosses, shafts, flanges, ribs, steps,
     chamfers, fillets, threads, and revolved profiles
   - ordered CAD operation plan
   - assumptions and unresolved ambiguities

6. **CAD Construction**

   Generate build123d code from the feature inventory and operation plan. Prefer simple
   valid operations over speculative complexity, but do not collapse visible features
   into a plain envelope.

   Candidate tools:
   - `execute_cad_code`
   - deterministic candidate generation for SVG-derived views
   - code repair for common build123d idiom errors
   - `summarize_step` for sanity checks

   Output:
   - valid STEP file
   - source code with `part` as the final object
   - explicit assumptions in short comments

7. **Verification And Selection**

   Reproject candidate STEP files into contact sheets and compare both to the source
   drawing and to other candidate STEP files. A candidate must pass both source
   fidelity and roundtrip stability.

   Candidate tools:
   - `render_step_to_drawing`
   - source-fidelity VLM judge
   - `compare_cad_parts`
   - line-mask comparison for parseable SVGs

   Output:
   - ranked candidates
   - failure reasons
   - prompt/tool feedback for the next iteration

## Model And Tool Development Tracks

### Gemma 4 Coordinator

- Improve prompts so Gemma inventories physical features before coding.
- Make Gemma explicitly choose among evidence sources and CAD tools.
- Teach Gemma to request masked/intermediate images when available.
- Preserve uncertainty rather than hallucinating dimensions or associations.

### YOLO OBB Detection

- Fine-tune detectors for title blocks, notes, GD&T frames, datum tags, dimensions,
  arrows, leader lines, section labels, and other annotation categories.
- Add callout-line categories and oriented bounding boxes where possible.
- Evaluate precision, recall, mAP, and downstream source-fidelity impact.
- Treat low-confidence detections as mask candidates, not authoritative deletions.

### Florence-2 / Donut Structured Parsing

- Fine-tune or plug in local weights for structured engineering drawing extraction.
- Parse cropped annotation regions into JSON: symbol type, text, datum references,
  nominal values, tolerances, and likely target feature.
- Track hallucination rate separately from recall.

### Masking And Image Products

- Store intermediate images in each case directory:
  - original
  - sheet/title-block masked
  - annotation/GD&T masked
  - isolated physical linework
  - per-view crops
- Keep masks reversible and auditable so failures can be diagnosed visually.

### Geometry Tools

- Add robust raster linework extraction and curve tracing.
- Add feature-fitting helpers for circles, slots, symmetric profiles, and revolved
  sections.
- Connect fitted primitives to build123d operation templates.
- Reuse deterministic SVG reconstruction where it already works.

## Near-Term Milestones

1. Add explicit mask-producing tools to the Gemma tool schema.
2. Persist structured evidence, masks, and per-view crops for every GD&T case.
3. Improve Gemma evidence extraction so all eight current GD&T cases produce useful
   physical feature, dimension, GD&T, annotation, and title-block hints.
4. Train or supply stronger local YOLO/Florence-2/Donut weights and compare them against
   Gemma-only extraction.
5. Add callout association data structures: annotation region -> leader line ->
   target feature region.
6. Add curve/primitive fitting over masked physical linework.
7. Make first-pass CAD use extracted feature plans before falling back to generic
   envelopes.
8. Gate success on source fidelity, no fallback geometry, and CAD roundtrip stability.

## Implementation Status

- 2026-04-24: Added `prepare_drawing_masks` as an agent-callable tool. It writes
  original, sheet/title-block masked, annotation masked, isolated physical linework,
  overlay, and JSON region metadata artifacts. The first implementation is heuristic
  and intentionally auditable; Gemma should verify the masks against the original
  drawing before trusting them.

## Experiment Discipline

- Every experiment should record config, model paths, prompts, masks, evidence JSON,
  CAD code, STEP files, contact sheets, and pass/fail criteria.
- Compare extractors by downstream CAD quality, not only extraction metrics.
- Keep detector/VLM failures visible in `uncertainties`; do not silently drop failed
  evidence.
- Prefer small, inspectable runs over long opaque runs until the pipeline stabilizes.
- Promote prompt/profile changes only when they improve measured source fidelity.

## Open Questions

- What annotation taxonomy should the detector use for callout lines and association
  targets?
- Should masking remove annotations completely, fade them, or provide separate layers
  for Gemma to compare?
- How should section views contribute to hidden/internal CAD features?
- What minimum detector quality is needed before YOLO/Donut evidence helps more than it
  distracts?
- Which features are best recovered by deterministic geometry fitting versus LLM-led
  reasoning?
