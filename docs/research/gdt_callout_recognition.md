# GD&T Callout Recognition

Researched: 2026-04-24

## Purpose

GD&T callouts must be recognized as annotation artifacts before projection
reconstruction. They are essential for metadata/tolerance extraction, but their
lines and symbols should be masked out of projection imagery used for geometric
reconstruction.

## What To Detect

High-priority targets:

- Feature control frames: rectangular boxes divided into cells.
- Datum feature symbols and datum letters.
- Boxed basic dimensions and boxed annotation callouts when they visually
  contaminate projection geometry.
- Leader arrows attached to GD&T frames, when feasible.

Feature control frames usually contain:

- geometric characteristic symbol
- tolerance zone shape, such as diameter when applicable
- tolerance value
- optional material condition modifier such as MMC or LMC
- datum references, ordered primary, secondary, tertiary

Common symbols/classes to train:

- Form: straightness, flatness, circularity, cylindricity
- Orientation: parallelism, perpendicularity, angularity
- Location: position, concentricity/coaxiality, symmetry
- Profile: profile of a line, profile of a surface
- Runout: circular runout, total runout
- Modifiers/references: datum feature, MMC, LMC, RFS/default, projected
  tolerance zone, common tolerance zone

## System Design

Use multiple layers:

1. **Fast deterministic detector**: line morphology finds ruled rectangular
   callout frames and boxed annotations. This gives immediate masks for the
   dashboard and reconstruction cleanup.
2. **Vision classifier**: YOLO, Florence-2, or another fine-tuned detector should
   classify symbols and callout types inside candidate boxes.
3. **Gemma 4 supervisor**: Gemma should review uncertain candidates, read the
   frame contents, and decide whether a boxed annotation is true GD&T or only a
   boxed dimension.
4. **User feedback loop**: missed or incorrect callouts should produce copied
   feedback with filename, crop, proposed kind/symbol, and `issue=<issue>`.

## Dashboard Rule

Projection views should display GD&T/callout masks overlaid or blanked out. The
masked view makes missed callouts obvious and prepares a cleaner line drawing for
CAD reconstruction.

## Current Implementation

- Detector: `src/segmentation/gdt.py`
- Output field: `analysis["gdt"]`
- Projection panes use `analysis["gdt"][*].crop` as masks when rendering.
- Current callouts are classified as `feature_control_frame` or
  `boxed_annotation_candidate`.
- Curated general callout fixtures live in `training_data/gdt_annotations/`.
- Gemma 4 tool-agent runner: `scripts/run_gemma4_callout_agent.py`.

The current detector is high-recall and intentionally conservative about symbol
classification. It does not yet claim reliable symbol OCR.

## `simple1.webp` Teacher Fixture

`training_data/gdt/simple1.webp` is a cropped, reduced-complexity development
image. It has no visible border or title block. It has two visible projections:

- front view on the left
- side view on the right

Expected visible callout counts:

- 10 front-view callouts
- 3 side-view callouts

The bottom `C` reference appears to call out cropped/missing context; the
corresponding projection is not visible. This is valuable because Gemma 4 must
identify the visible callout while also saying the referenced view/context is
outside the crop.

The supervised annotation file is:

```text
training_data/gdt_annotations/simple1_callouts.yaml
```

The tool-agent run:

```bash
.venv/bin/python scripts/run_gemma4_callout_agent.py training_data/gdt/simple1.webp
```

produced a valid JSON result with exactly 10 front-view and 3 side-view
callouts. The scored output is stored at:

```text
experiments/gemma4_callout_agent/simple1_callout_agent.json
```

## Sources

- GD&T Basics, "Feature Control Frame":
  https://www.gdandtbasics.com/feature-control-frame/
- GD&T Basics, "GD&T Symbols Explained":
  https://www.gdandtbasics.com/gdt-symbols/
- KEYENCE, "Feature Control Frame":
  https://www.keyence.com/ss/products/measure-sys/gd-and-t/basic/tolerance-entry-frame.jsp
- Machining Doctor, "Feature Control Frame":
  https://www.machiningdoctor.com/glossary/feature-control-frame/
