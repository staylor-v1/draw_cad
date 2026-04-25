# Source Pedigree for Drawing Format Decisions

Researched: 2026-04-24

## Purpose

This note records why the project currently treats drawing borders and title
blocks as separate recognition targets. It also ranks the sources used so future
work can decide when to keep, revise, or replace these assumptions.

## Source Hierarchy

1. **Standards bodies**
   - ASME Y14.1 establishes drawing sheet size and format practices.
   - ISO 7200 establishes data fields for title blocks and document headers.
   - These are the strongest pedigree sources, but the full standards are paid,
     so only public abstracts/product pages were used here.

2. **University engineering education references**
   - McGill Engineering Design provides public educational material on drawing
     format, borders, title block location, title block content, revision tables,
     scale, notes, and drawing titles.
   - This is useful because it is public, readable, and aligned with conventional
     engineering drawing practice.

3. **Public CAD-standard guidance**
   - St-5 CAD Standard gives practical drawing-sheet layout guidance, including
     margins and title block categories.
   - This is lower authority than ASME/ISO, but useful for implementation
     heuristics and visual expectations.

4. **Open-source vectorization tool documentation**
   - Potrace, VTracer, AutoTrace, and ezdxf documentation guide tool-selection
     experiments for turning pixel drawings into line/vector evidence.
   - These sources are implementation pedigree rather than drawing-standard
     pedigree; they explain why the current harness prefers local OpenCV plus
     ezdxf first while preserving a path to external vectorizers.

## Findings With Pedigree

### Separate border detection from title block extraction

Pedigree:

- McGill describes the inside border as enclosing the working area, including the
  title block and other tables.
- St-5 describes every drawing as having a title block and drawing area, then
  separately discusses drawing borders and margins.

Design consequence:

- The dashboard should render the border mask separately from title-block crops.
- The agent should store border masks separately from title-block OCR results.

### Treat border zone letters/numbers as low priority

Pedigree:

- McGill's revision-table discussion notes that revision records may include a
  zone location, which implies border zones can matter for revision references.
- However, the border itself is a sheet-format feature rather than part geometry.

Design consequence:

- Mask the border for now.
- Preserve correction hooks for cases where zone markings should be extracted.
- Do not use zone markings to drive projection or CAD reconstruction unless a
  later model explicitly links them to revisions or callouts.

### Use lower-right as a title-block prior, not a rule

Pedigree:

- McGill states that title blocks are normally in the lower-right corner.
- McGill also notes title block arrangement and size can vary by organization.

Design consequence:

- Start candidate search in the lower-right region.
- Allow alternate title-block locations in the segmentation data model and UI.
- Keep user feedback text able to describe "wrong title block crop" rather than
  only "wrong lower-right crop."

### Extract metadata fields before using them operationally

Pedigree:

- ISO 7200's public abstract says the standard defines field names, contents, and
  lengths to support document exchange and compatibility.
- ASME's public Y14.1 description emphasizes uniform location, readability,
  handling, filing, reproduction, and consistent revision/date information.
- McGill lists common mandatory and optional title-block fields.

Design consequence:

- Store raw field labels, normalized canonical labels, values, confidence, and
  crop provenance.
- The CAD pipeline should not rely on a field unless confidence is high or the
  user has confirmed it.

## Current Project Assumptions

- Border segmentation is a sheet-format cleanup/verification task.
- Title block segmentation is a metadata extraction task.
- Revision tables may need a separate segmentation class later.
- Raster-to-DXF conversion should expose line primitives as teacher-tool evidence
  before it is trusted as final CAD reconstruction.
- The dashboard should make segmentation mistakes easy to report with a copied
  text payload containing filename, region, proposed classification, crop, and
  `issue=<issue>`.

## Source Links

- ASME Y14.1 - Drawing Sheet Size and Format:
  https://www.asme.org/codes-standards/find-codes-standards/drawing-sheet-size-and-format
- ISO 7200:2004 - Data fields in title blocks and document headers:
  https://www.iso.org/standard/35446.html
- McGill Drawing Format and Elements:
  https://www.mcgill.ca/engineeringdesign/step-step-design-process/basics-graphics-communication/drawing-format-and-elements
- St-5 CAD Standard Drawing Layout, Borders and Margins:
  https://www.cad-standard.com/technical-drawing-basics/drawing-sheet-layout-boarders-margins
- VTracer Python binding:
  https://pypi.org/project/vtracer/
- Potrace:
  https://potrace.sourceforge.net/
- ezdxf:
  https://pypi.org/project/ezdxf/
- AutoTrace:
  https://github.com/autotrace/autotrace
