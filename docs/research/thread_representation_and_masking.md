# Thread Representation and Projection Masking

Purpose: prevent threaded part geometry from being removed as if it were callout leader/dimension linework.

## Drawing Convention

Engineering drawings do not need to model every physical thread crest. Public teaching references describe threads as conventional or simplified symbolic linework, and detailed/alternative representations can show repeated straight lines for thread crests and roots.

Practical detector consequence:

- A single diagonal line touching a callout box may be an annotation leader.
- Three or more nearby parallel diagonal lines inside a projected part are more likely thread geometry or section/thread representation.
- Dense repeated horizontal/near-horizontal strokes inside a detected annotation candidate can also be rasterized thread texture, especially when Hough extraction breaks sloped thread marks into short horizontal spans.
- Broad unassigned projection crops must not suppress callout masks globally, but detected callout boxes that contain thread-like repeated line texture should be excluded.

## Current Implementation

`src/segmentation/masks.py` now computes thread-like zones from vector segments before creating annotation masks.

Thread-like zones are detected from non-teacher annotation candidates that contain:

- repeated diagonal segments with similar angle, or
- multiple distinct rows of short horizontal segments,
- enough total line density to suggest patterned part geometry instead of a single callout.

Any non-teacher annotation crop overlapping a thread-like zone is ignored, and vector annotation lines whose midpoint lies inside that zone are not sent to the projection mask renderer.

`threaded_cap.jpg` has a supervised teacher fixture in
`training_data/gdt_annotations/threaded_cap_callouts.yaml`:

- 2 confirmed projections: exterior profile and section A-A.
- 17 teacher callouts: 10 attached to the exterior profile and 7 attached to the section view.
- 1 negative region: the repeated internal thread texture in the section view.

The Gemma callout-agent harness uses fixture-provided view names, expected
counts, and negative regions instead of assuming a fixed front/side drawing. This
lets the same prompt shape train on both reduced examples like `simple1.webp`
and dense drawings like `threaded_cap.jpg`.

## Reference Images Added

Downloaded from McGill Engineering Design into `training_data/thread_examples/`:

- `mcgill_conventional_thread_representation.jpg`
- `mcgill_sectional_external_internal_threads.jpg`
- `mcgill_rolled_thread_representation.jpg`

These are used as vectorization regression inputs so the harness sees thread representations beyond `threaded_cap.jpg`.

## Sources

- McGill Engineering Design, "Dimensioning threaded fasteners": https://www.mcgill.ca/engineeringdesign/step-step-design-process/basics-graphics-communication/dimensioning-threaded-fasteners
- How Engineering Works, "How do you represent threads and threaded fasteners in a drawing?": https://www.howengineeringworks.com/questions/how-do-you-represent-threads-and-threaded-fasteners-in-a-drawing/
