# GrabCAD GD&T Candidate Drawings

Source tag page: https://grabcad.com/library/tag/gd-t

Collected through a user-authenticated temporary Chrome session. GrabCAD projects are user-contributed; verify each item's license and reuse terms before using outside local evaluation.

- Manifest: `manifest.json`
- Raw archives: `raw_archives/`
- Seeded source URLs: `seed_urls.txt`
- Extracted 2D candidates: `orthographic_2d_candidates/`
- Extraction manifest: `extraction_manifest.json`
- Review metadata: `candidate_review_metadata.json`
- Manual queue / source summary: `orthographic_download_queue.txt`

The collector prioritizes 2D orthographic drawing assets such as PDF, DWG, DXF, IDW, SLDDRW, DRW, and raster drawing sheets. Full project archives may include 3D CAD files; downstream extraction should keep only drawing-sheet candidates for the GD&T segmentation corpus.

## Current Corpus Snapshot

- 24 GrabCAD GD&T-tagged source pages were inspected.
- 21 archives were downloaded because their model pages exposed likely drawing-sheet filenames.
- 65 likely 2D drawing candidates were extracted.
- 53 candidates are high-confidence drawing formats or filename patterns.
- 12 candidates are medium-confidence raster drawings that need visual review.

Pages with no visible 2D drawing filenames were left in the manifest but skipped for archive download. Some projects may still contain drawings hidden inside folders, so those are worth revisiting manually later if we want maximum coverage.

## Review Dispositions

The dashboard review helper at `/training-review` saves one disposition per candidate:

- `use`: keep this file for model training.
- `reject`: exclude this file; use it as negative feedback for future source/file filtering.
- `duplicate`: exclude this file from training but keep it tied to the source as a duplicate example.
- unreviewed files remain available but are not training-ready.

Saved metadata includes filename, extracted path, source URL, source archive, original archive member, file extension, confidence, and disposition. The extractor preserves that metadata in `extraction_manifest.json` and writes path lists:

- `selected_training_candidates.txt`
- `rejected_training_candidates.txt`
- `duplicate_training_candidates.txt`

This gives future download/extraction passes concrete negative examples so image-heavy GrabCAD projects can be filtered more aggressively instead of treating every raster image as a useful drawing.

## Reproduction

Start a logged-in Chrome session with remote debugging enabled, then run:

```bash
node scripts/collect_grabcad_gdt_drawings.mjs --url-list training_data/gdt_grabcad/seed_urls.txt --only-with-file-candidates --download --settle-ms 3500
python3 scripts/extract_grabcad_drawing_candidates.py
```

The collector intentionally uses a slow settle delay. Fast repeated navigation caused GrabCAD/CloudFront to temporarily block the tag listing during collection.
