# Local VLM Research Notes

These notes capture the design choices behind the GD&T training loop and optional
drawing-evidence extractors.

## Fidelity Target

The target source-fidelity threshold is `0.99`. Early prompt and model iterations can
start lower through a curriculum threshold so the system still produces useful failure
records before it is good enough to satisfy the final gate.

## Florence-2 GD&T Extraction

Khan et al., "Fine-Tuning Vision-Language Model for Automated Engineering Drawing
Information Extraction" (`arXiv:2411.03707`) fine-tunes Florence-2-base for GD&T
extraction from 400 annotated engineering drawings. The relevant ideas for this repo are:

- Treat extraction as structured JSON, not free-form OCR text.
- Use multiple query formulations per image as data augmentation.
- Evaluate precision, recall, F1, and hallucination rate, because false extracted
  annotations can be as damaging as missed annotations.
- Fine-tuned small local VLMs can outperform stronger zero-shot general VLMs on narrow
  engineering drawing extraction tasks.

Implementation hook: `florence2` is an optional extractor backend. It requires a local
fine-tuned Florence-2 model path and never downloads weights implicitly.

## YOLOv11-OBB + Donut Structured Parsing

Khan et al., "Automated Parsing of Engineering Drawings for Structured Information
Extraction Using a Fine-tuned Document Understanding Transformer" (`arXiv:2505.01530`)
uses a two-stage local pipeline: YOLOv11-OBB detects annotation regions, then Donut
parses cropped regions into structured JSON. The cited categories are GD&T, general
tolerances, measures, materials, notes, radii, surface roughness, threads, and title
blocks. The paper reports that a single Donut parser generalized better than
category-specific parsers.

Implementation hook: `yolo_donut` is an optional extractor backend. It can run a local
YOLO OBB model to produce region evidence and crop detections. Donut parsing is kept as
an adapter point for a local fine-tuned model.

## Harness Implications

- The CAD agent should receive extracted evidence as hints, not as truth. It must still
  verify against the image.
- Annotation detection is useful even when it does not directly reconstruct geometry,
  because it prevents the CAD stage from turning borders, title blocks, GD&T frames, and
  notes into solids.
- Breakthroughs should be measured by source fidelity at the final `0.99` goal, not by
  part-to-part roundtrip alone.
