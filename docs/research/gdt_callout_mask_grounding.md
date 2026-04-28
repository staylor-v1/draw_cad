# GD&T Callout Mask Grounding

GD&T recognition has two related but different segmentation problems:

1. Identify the mark: a feature-control frame, datum feature, boxed basic
   dimension, diameter dimension, or other callout text/symbol region.
2. Ground the mark: find the leader, extension, arrow, dot, or dimension line
   that connects the mark to the feature on the drawing.

The first problem is naturally box-shaped because GD&T markings are usually
inside rectangular frames or text extents. The second problem is not box-shaped:
the line is a thin stroke, may be compound, may end in an arrow/dot, and may
cross part geometry. A broad bounding box around that line can hide real part
edges, which is exactly the failure mode we have seen in projection masks.

## Model Options Evaluated

### YOLO Box Detector

Status: current production path for mark detection.

Use it to find the callout mark, not to mask every connected line with a filled
rectangle. It is fast, local, and already tuned for the GD&T mark classes.

### YOLO11 Segmentation

Ultralytics YOLO11 supports instance segmentation variants and custom training.
This is the best next trainable mask model if we can create polygon masks for
`mark + leader line(s)` objects. It fits our current Ultralytics stack and GPU
workflow, but it needs real mask labels before it can outperform the hybrid
geometric baseline.

Source:

- https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-instance-segmentation

### SAM2

SAM2 is promptable image/video segmentation. Hugging Face documents point and
box prompts for segmenting objects of interest. SAM2 is promising as an
annotation assistant: prompt with the YOLO box, then add positive points along
the leader line and negative points on nearby part geometry. It is less
promising as an unattended detector because it is not class-aware and may
segment only the boxed mark unless explicitly prompted along the leader.

Sources:

- https://huggingface.co/docs/transformers/en/model_doc/sam2
- https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/

### Grounded SAM2

Grounded SAM2 combines an open-vocabulary detector with SAM2. It is useful for
bootstrapping annotation candidates from prompts such as `feature control
frame`, `datum label`, or `dimension callout`, but text grounding is unlikely
to reliably trace tiny leader lines on engineering drawings by itself.

Source:

- https://docs.autodistill.com/base_models/grounded-sam-2/

### RF-DETR Segmentation

RF-DETR Segmentation is a modern instance-segmentation candidate with an
Apache-2.0 implementation. It is a good later experiment once we have polygon
labels. Integration is heavier than YOLO-seg, so it should follow the first
mask-labelled YOLO baseline rather than precede it.

Source:

- https://rfdetr.roboflow.com/1.4.0/
- https://roboflow.com/model/rf-detr-segmentation

## Recommended Operating Model

Use a hybrid pipeline:

1. Detect the callout mark with the fine-tuned YOLO box model.
2. Extract vector line candidates with raster-to-vector/Hough tools.
3. Keep only line segments attached to the callout mark and leaving the mark
   box.
4. Score the terminal endpoint against projection crops and feature semantics.
5. Ask Gemma4 to choose among geometric candidates when semantics matter:
   diameter should point to a hole, flatness should point to a surface, datum
   features should attach to the datum feature, and section arrows should point
   through the section path.
6. Emit a per-pixel thin line mask, not a filled bounding rectangle, for the
   leader/dimension line.
7. Use SAM2 as an interactive teacher tool for difficult masks and use the
   accepted masks to train YOLO11-seg or RF-DETR-Seg.

This design lets the system preserve the strength of bounding boxes for the
mark while avoiding the major weakness of bounding boxes for leader lines.

## Harness Result

The current harness is `scripts/evaluate_gdt_mask_grounding.py`.

It evaluates:

- existing box/rectangle masking,
- vector-grounded thin line masks,
- YOLO11-seg as the recommended first trainable mask model,
- SAM2/Grounded SAM2 as annotation assistants,
- RF-DETR-Seg as a later trainable mask model,
- Gemma4 as the semantic arbitration layer.

Current smoke run:

- Images: 4
- Annotations: 47
- Grounded callouts: 31
- Grounded line segments: 58
- Thin line mask pixels / line bounding-box pixels: 0.0788 overall
- Thin line mask pixels / mark-box pixels: 0.2078 overall

The harness intentionally filters out segments that remain inside the mark box.
This lowers recall but avoids treating feature-control-frame borders and text
strokes as leader lines.

Artifacts:

- `experiments/gdt_mask_grounding/report.md`
- `experiments/gdt_mask_grounding/summary.json`
- `experiments/gdt_mask_grounding/*_grounding_overlay.png`
- `experiments/gdt_mask_grounding/*_line_mask.png`

## Next Step

Add a dashboard review mode for grounded callout objects:

- show mark box and traced line mask separately,
- allow approving/rejecting each candidate segment,
- allow clicking the true feature endpoint,
- write accepted masks as polygon labels.

Once 50-100 high-quality masks are accepted, train YOLO11-seg as the first
mask model and compare it against the hybrid vector baseline.
