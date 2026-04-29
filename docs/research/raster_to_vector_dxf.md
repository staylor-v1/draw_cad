# Raster-to-Vector and DXF Harness Notes

Purpose: test whether line-oriented vector evidence makes callout detection and projection masking more reliable than pixel morphology alone.

## Findings

- **VTracer** is a modern Rust/Python raster-to-vector library. Its PyPI package describes conversion from JPG/PNG to SVG, binary mode for line art, and a color pipeline for high-resolution scans. It is promising for future SVG-first experiments, but it was not installed in the local environment during this pass.
- **Potrace** is a mature bitmap tracing tool. The project page says it transforms bitmap inputs into vector formats including SVG, PDF, EPS, DXF, GeoJSON, and others. It is strongest for binarized black-and-white art. It was not installed locally.
- **AutoTrace** supports edge and centerline tracing, despeckling, color reduction, and many output formats including DXF according to the project README/search result. It was not installed locally.
- **OpenCV plus ezdxf** is available now. This is less of a general art vectorizer and more of a CAD evidence extractor: threshold the drawing, detect straight line segments and ruled rectangles, then write DXF `LINE`/`LWPOLYLINE` entities. For engineering drawings, that directness is useful because callout boxes, extension lines, projection outlines, and title-block ruling are mostly line primitives.

## Current Local Implementation

Implemented `src/vectorization/raster_to_dxf.py`:

- Otsu binary inverse thresholding for dark drawing primitives.
- Hough line extraction.
- Axis-aligned segment merging to reduce noisy duplicate lines.
- Ruled rectangle detection for feature-control-frame and boxed annotation candidates.
- DXF export through `ezdxf`.
- JSON and overlay review artifacts through `scripts/run_vector_harness.py`.

Initial comparison on the core development images:

| Image | Pixel-only GD&T candidates | Vector-augmented candidates | Vector-backed candidates |
| --- | ---: | ---: | ---: |
| `simple1.webp` | 0 | 3 | 3 |
| `flange1.png` | 0 | 16 | 16 |
| `flange2.png` | 6 | 26 | 21 |

The result is intentionally high-recall. These candidates are teacher-tool
evidence for masking/review, not final symbol classification.

The dashboard analysis payload now also includes `annotationMasks`, which
combines detected/teacher callout boxes with nearby vector line masks. Current
smoke counts:

| Image | Callout boxes | Total annotation masks | Vector line masks |
| --- | ---: | ---: | ---: |
| `simple1.webp` | 16 detected/teacher boxes | 75 | 59 |
| `flange2.png` | 26 detected boxes | 199 | 173 |

These masks should be reviewed visually because high recall can over-mask part
edges when a leader or dimension line touches the projection geometry.

The harness writes:

- `experiments/vector_harness/dxf/*.dxf`
- `experiments/vector_harness/json/*.json`
- `experiments/vector_harness/overlays/*.png`
- `experiments/vector_harness/index.html`

## How This Helps Gemma 4

Gemma should be offered vector evidence as a tool output, not only image crops. The useful primitives are:

- Long horizontal/vertical segments for projection outlines and dimension lines.
- Rectangular ruled frames for GD&T feature-control frames and boxed basic dimensions.
- Diagonal segments for leaders and section/view arrows.
- DXF layers separating horizontal, vertical, diagonal, and rectangle evidence.

For drawings like `flange2.png`, rectangular bounding boxes are ambiguous because left-view callouts can cross the right-view bounding region. Vector evidence lets the agent mask individual callout lines and frames instead of blanking a full rectangle around a projection.

## Sources

- VTracer PyPI: https://pypi.org/project/vtracer/
- Potrace project: https://potrace.sourceforge.net/
- ezdxf PyPI: https://pypi.org/project/ezdxf/
- AutoTrace repository/search result: https://github.com/autotrace/autotrace
