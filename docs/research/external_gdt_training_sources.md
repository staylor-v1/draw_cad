# External GD&T Training Sources

This note records the source pedigree for the external GD&T corpus in
`training_data/gdt_external`.

## Included Sources

### NIST MBE PMI Validation and Conformance Testing

The NIST MBE PMI project is the strongest source found for full-sheet GD&T
drawings. NIST describes the project as a conformance test system for CAD
software against ASME PMI standards, specifically GD&T. The project includes
2D drawing test cases, CAD models, STEP files, and reports. Its annotation
browser describes 421 PMI annotations and explicitly calls out dimensions,
geometric tolerances, datum features, datum targets, and multi-line feature
control frames.

Included files:

- `nist_pmi_ctc_01` through `nist_pmi_ctc_05`: Complex Test Cases.
- `nist_pmi_ftc_06` through `nist_pmi_ftc_11`: Fully Toleranced Test Cases.

Important caveat:

- Most NIST PMI rendered pages are 3D/model-based PMI or isometric views.
  They are valuable for GD&T mark recognition and callout grounding, but they
  are not primary 2D orthographic drawing-to-CAD training targets.
- The page-level split is recorded in
  `training_data/gdt_external/split_manifest.json`.

Why it matters:

- Dense, clean, high-resolution vector PDF drawings.
- Multiple pages with title blocks, notes, datum features, datum targets,
  basic dimensions, feature control frames, and projection/detail views.
- Purpose-built to test PMI/GD&T implementations, so it is more systematic
  than random internet examples.

Sources:

- https://pages.nist.gov/CAD-PMI-Testing/models.html
- https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/mbe-pmi-validation

### NIST Additive Manufacturing Test Artifact

The NIST additive manufacturing test artifact page includes a separate
engineering drawing with GD&T markings. NIST states that these files were
developed by federal employees in the course of official duties and are public
domain under 17 USC 105.

Included files:

- `nist_am_test_artifact_engineering_drawing`

Why it matters:

- Full-sheet engineering drawing with GD&T markings from a government source.
- Good validation image for title-block, border, projection, and GD&T callout
  segmentation together.

Sources:

- https://www.nist.gov/document/engineeringdrawingpdf
- https://www.nist.gov/el/intelligent-systems-division-73500/production-systems-group/nist-additive-manufacturing-test

### Wikimedia Commons GD&T and Geometric Tolerancing Categories

Wikimedia Commons provides symbol/reference material and smaller technical
drawing examples. These are less realistic than NIST full-sheet drawings, but
they are useful for symbol vocabulary, synthetic augmentation seeds, and
negative/positive callout recognition examples. The manifest records each
file's license label and license URL individually.

Included categories:

- `Category:Geometric dimensioning and tolerancing`
- `Category:Geometric tolerancing`

Why it matters:

- Broad symbol coverage: position, perpendicularity, flatness, profile,
  runout, MMC/LMC/projected tolerance zone modifiers, and related callout
  patterns.
- Per-file licensing metadata is available through Wikimedia Commons.

Source:

- https://commons.wikimedia.org/wiki/Category:Geometric_dimensioning_and_tolerancing

## Review-Only Leads

The search also found high-quality commercial and teaching references, but
they were not downloaded because their reuse terms were not clear enough for a
checked-in training corpus.

- University of Illinois ME170 engineering drawing notes:
  https://courses.grainger.illinois.edu/me170/fa2019/Engineering_Drawing_Notes_B.pdf
- GD&T Basics feature control frame examples:
  https://www.gdandtbasics.com/feature-control-frame/
- KEYENCE GD&T overview examples:
  https://www.keyence.com/ss/products/measure-sys/gd-and-t/basic/tolerance-entry-frame.jsp
- NASA internship report with GD&T discussion:
  https://ntrs.nasa.gov/api/citations/20180002822/downloads/20180002822.pdf

## Dataset Layout

- `raw/nist_pmi`: downloaded NIST PMI PDF drawings.
- `raw/nist_am`: downloaded NIST additive-manufacturing engineering drawing.
- `raw/wikimedia_commons`: downloaded Commons originals.
- `rendered/*`: PNG derivatives for detector/segmentation pipelines.
- `manifest.json`: source URL, page URL, license, checksum, and rendered paths.
- `split_manifest.json`: per-rendered-image 2D/3D/reference split metadata.
- `splits/2d_orthographic_target.txt`: primary 2D drawing-to-CAD candidates.
- `splits/3d_pmi_reference.txt`: 3D/model-based PMI references, excluded from
  2D reconstruction training.
- `splits/2d_reference.txt`: symbol/callout/drawing reference images.
- `README.md`: concise rebuild and usage notes.

## Next Training Step

This is a source corpus, not a labelled YOLO dataset yet. Recommended next
step is to use `splits/2d_orthographic_target.txt` for drawing-to-CAD
experiments, and use `splits/3d_pmi_reference.txt` only for GD&T mark/callout
recognition. The dashboard rectangle tool/Gemma worklist mode can then accept,
reject, and class-label feature control frames, datum features, basic
dimensions, linear/diameter dimensions, and thread/geometry negatives.
