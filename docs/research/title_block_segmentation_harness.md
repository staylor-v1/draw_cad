# Title Block Segmentation Harness

Created: 2026-04-24

## Purpose

The dashboard feedback `filename=flange1.png | area=title-block-mask | proposed=title block isolated`
showed that a fixed lower-right crop can land on a random portion of the title
block. The project now has a deterministic supervisor tool that proposes a
title-block crop before Gemma 4 interprets title-block contents.

This gives Gemma 4 a better teaching loop:

1. Codex/tooling proposes a crop from line/table structure.
2. Gemma 4 can inspect the crop and extracted fields.
3. The user can click any bad crop or field and produce structured correction
   text with filename, area, proposed value, crop, and `issue=<issue>`.
4. Future runs can compare new candidates against the harness review artifacts.

## Tooling

- Segmentation module: `src/segmentation/title_block.py`
- Harness runner: `scripts/run_titleblock_harness.py`
- Focused tests: `tests/test_title_block_segmentation.py`
- Dashboard API wiring: `web_ui.py`
- Review artifacts: `experiments/titleblock_harness/`

Run the harness:

```bash
make run-titleblock-harness
```

Open the generated review page:

```text
experiments/titleblock_harness/index.html
```

## Detection Strategy

The current detector is intentionally transparent and deterministic:

- Binarize dark drawing pixels.
- Search the lower half of the sheet, where title blocks commonly appear.
- Find horizontal ruling rows that indicate a boxed table.
- Group the bottom table rows.
- Find vertical ruling columns within that bottom band.
- Score the candidate by horizontal/vertical line evidence, dark density,
  crop proportions, and bottom/right placement.
- Fall back to a low-confidence lower-right prior only when ruling evidence is
  weak or absent.

This is not meant to replace Gemma 4. It is a teacher tool that gives Gemma 4 a
plausible crop and confidence rather than forcing it to reason from the whole
drawing every time.

## Current Corpus

The harness default corpus includes:

- `training_data/gdt/*`
- `training_data/title_block_examples/*`
- `benchmarks/drawings/*/drawing.*`
- `test_result/cases/*/original.*`

The benchmark drawing PNG files currently appear to be invalid placeholders and
are skipped by the harness instead of aborting the run.

## Added External Examples

Additional examples are listed in:

```text
training_data/title_block_examples/manifest.yaml
```

The most useful positive example is:

- `iso_10628_symbols_sheet_title_block.png`: a Wikimedia ISO symbol sheet with
  visible border zones, legend, and a lower-right title block. It is derived
  from a CC BY-SA 4.0 SVG, with attribution recorded in the manifest.

The other added examples are useful negative/border cases:

- `iso_a2_title_block_template.png`: CC0 border template; title block is
  off-canvas in the source SVG.
- `iso_sectioned_part_title_block.png`: public-domain mechanical drawing without
  a visible title block.
- `simple_mechanical_drawing.png`: public-domain simple mechanical drawing
  without a visible title block.

## Current Results

On the local corpus run, the harness wrote 26 review records. Five invalid
benchmark placeholders were skipped. For `flange1.png`, the title-block crop is:

```text
crop=41,77,56,20
```

That crop covers the full title-block table rather than the previous fixed crop:

```text
crop=60,70,34,22
```

## Next Improvements

- Add confirmed ground-truth boxes for high-value examples.
- Convert user feedback strings into a JSONL correction set.
- Add a Gemma 4 prompt mode that asks the model to judge candidate crops before
  OCR or metadata extraction.
- Split adjacent legend/revision tables from title blocks when they share ruling
  lines.
- Track negative examples explicitly so "no visible title block" becomes a
  valid supervised outcome.

## Projection Mask Lesson From `flange2.png`

`flange2.png` has no sheet border and no title block. This must be treated as a
valid detection result, not as a failed crop. Gemma 4 should recognize that this
image is already cropped to drawing content and should not mask off border or
title-block regions.

It also demonstrates why projection segmentation cannot be represented as only a
bounding box. The left section view and right circular view can be separated as
drawing regions, but their callouts and leaders intrude into each other's
possible rectangular review boxes. The harness therefore marks projection
candidates with:

```text
segmentationMode=component_mask
```

The rectangular crop remains useful for human review, but the underlying
connected-component mask should be treated as the real segmentation artifact.
Gemma 4 should be taught to recognize this condition and avoid assuming that all
pixels inside the displayed rectangle belong to the projection.
