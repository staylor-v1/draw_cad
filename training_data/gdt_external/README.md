# External GD&T Training Set

This directory is a provenance-rich seed set for GD&T detection and segmentation work.
It was compiled by `scripts/compile_external_gdt_training_set.py`.

## Contents

- Assets: 84
- Rendered images: 101
- NIST PDF drawings: 12
- Wikimedia Commons files: 72

## Source Policy

Included assets come from NIST public information/public-domain pages or Wikimedia
Commons files with explicit per-file license metadata. The manifest records the
source URL, local path, license label, license URL, SHA-256, and rendered derivatives
for every item.

This set is not yet hand-labelled for YOLO. It is ready for teacher review,
auto-proposal, and annotation. The NIST PMI pages are mostly 3D/model-based
PMI views, so they should not be used wholesale for 2D orthographic
drawing-to-CAD training. Use the generated split lists instead.

## View-Type Splits

Run `scripts/split_external_gdt_view_types.py` after rebuilding the dataset.
It writes:

- `splits/2d_orthographic_target.txt`: primary candidates for 2D drawing-to-CAD.
- `splits/3d_pmi_reference.txt`: 3D/model-based PMI views for GD&T recognition only.
- `splits/2d_reference.txt`: 2D symbol, callout, and small drawing references.
- `split_manifest.json`: per-rendered-image split metadata and source/license links.

Current split counts:

- 2D orthographic target: 6
- 3D PMI reference: 26
- 2D reference/symbol material: 69

Wikimedia assets are best used as symbol/reference material and synthetic
augmentation seeds unless promoted by manual review.

## Review-Only Leads

These sources looked useful during search, but were not downloaded because the
redistribution/training-data terms were not clear enough for an included corpus:

- [University of Illinois ME170 Engineering Drawing Notes](https://courses.grainger.illinois.edu/me170/fa2019/Engineering_Drawing_Notes_B.pdf): Useful GD&T teaching drawings, but redistribution/license terms were not explicit in the search result.
- [GD&T Basics feature control frame/datum examples](https://www.gdandtbasics.com/feature-control-frame/): High-quality examples for visual review; use as design reference unless permission is granted.
- [KEYENCE GD&T overview examples](https://www.keyence.com/ss/products/measure-sys/gd-and-t/basic/tolerance-entry-frame.jsp): High-quality commercial reference examples; not downloaded into training data without permission.
- [NASA internship final report GD&T discussion](https://ntrs.nasa.gov/api/citations/20180002822/downloads/20180002822.pdf): Public technical report with GD&T context, but it does not appear to be a dense drawing corpus.

## Rebuild

```bash
.venv/bin/python scripts/compile_external_gdt_training_set.py --clean
.venv/bin/python scripts/split_external_gdt_view_types.py
```
