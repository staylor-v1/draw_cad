# Gemma 4 Roundtrip Test Results

This folder is a browseable copy of the earlier `training_data/gdt` roundtrip run.
Those artifacts should now be treated as a failed baseline, not evidence of drawing
understanding: the old acceptance criterion only checked `pass_1.step` against
`pass_2.step`, so fallback envelope geometry could pass by reproducing itself.

The corrected training loop in `scripts/tune_gemma4_gdt_roundtrip.py` rejects these
cases unless the first STEP also resembles the original mechanical drawing and no
fallback geometry was used.

Each case folder contains:

- `original.*`: source drawing image
- `pass_1.step`: CAD reconstructed from the original drawing
- `created_contact_sheet.png`: drawing generated from `pass_1.step`
- `created_triplet.svg`: generated front/right/top SVG drawing
- `created_front.svg`, `created_right.svg`, `created_top.svg`: individual generated views
- `pass_2.step`: CAD reconstructed from the generated drawing
- `metrics.txt`: comparison metrics
- `roundtrip_summary.json`: full per-case machine-readable result

Open `index.html` in a browser for a quick visual gallery.
