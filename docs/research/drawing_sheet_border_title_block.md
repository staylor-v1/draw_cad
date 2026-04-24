# Drawing Sheet Border and Title Block Recognition

Researched: 2026-04-24

## Summary

Engineering drawings normally separate the sheet into a drawing area, an inside
border, and metadata tables such as the title block and revision block. For our
dashboard and agent workflow, this supports treating border detection and title
block extraction as separate tasks:

1. **Border recognition** should locate and mask the sheet-edge frame.
2. **Title block recognition** should isolate the dense metadata table, usually
   near the lower-right corner of the drawing.
3. **Title block extraction** should then parse fields such as drawing number,
   title, scale, organization, revision, sheet number, drafter/checker/approval
   information, dates, material, general tolerances, and notes.

The current development dashboard follows this split by showing a full-sheet
border mask separately from a title-block crop.

## Border Knowledge

- The border is a format feature that encloses the working drawing area. McGill's
  engineering design notes describe the inside border as enclosing the working
  area, including the title block and other tables.
- Public drawing-format guidance commonly uses a larger binding-side margin and
  smaller margins on the remaining sides. The exact dimensions vary by standard,
  sheet size, and whether the drawing is printed or electronic.
- Some borders include zone letters and numbers around the sheet edge. These can
  help locate revisions or drawing references, but they do not usually describe
  part geometry.

## Product Implication

For the first recognition pass, mask the border as a sheet-format artifact and do
not interpret border subdivision letters/numbers. Keep the mask visible to the
user so they can flag cases where zone markings are important.

Current feedback text should preserve:

- filename
- area: `border-mask`
- proposed region/crop
- `issue=<issue>`

## Title Block Knowledge

- Title blocks record information necessary to identify and use the drawing.
- Location is not mathematically guaranteed, but lower-right placement is common
  in engineering drawings.
- The physical arrangement varies by company and discipline. This means the
  recognizer should avoid assuming a single fixed grid layout.
- Title blocks are usually made of a box containing smaller boxes. This supports
  a two-stage approach: first isolate the table region, then parse text and cells
  within that region.

## Common Title Block Fields

High-priority fields for extraction:

- drawing number or identification number
- drawing title or part description
- organization, owner, company, or design activity
- original scale
- revision index
- sheet number and total sheets
- issue/release date
- drafter, checker, approver, or signature fields
- general notes/specifications, including tolerances and finishes

Useful but not always present:

- material
- weight
- sheet size
- job/order/contract number
- CAGE/FSCM or similar organization code
- professional seal
- references to other documents
- language code
- document type/status

## Agent Segmentation Approach

Recommended segmentation phases:

1. Locate sheet extent.
2. Locate inside border lines and mask only the border band.
3. Locate title block candidate regions, with a lower-right prior but not a hard
   lower-right-only rule.
4. Locate revision table separately when it is adjacent to, above, or left of the
   title block.
5. Inside the title block, segment cells before OCR where image quality permits.
6. Normalize extracted labels into canonical field names while preserving raw OCR
   text and crop provenance.

## Dashboard Acceptance Criteria

The Border and Title Block tab should show:

- a border-only view or overlay showing what was masked
- a title-block-only crop
- extracted title-block fields with confidence values
- click targets that generate structured correction text

The tab should not imply border zones are semantically used unless the model has
explicitly extracted a revision-zone relationship.

## Sources

- McGill University, "Drawing Format and Elements":
  https://www.mcgill.ca/engineeringdesign/step-step-design-process/basics-graphics-communication/drawing-format-and-elements
- ASME, "Y14.1 - Drawing Sheet Size and Format":
  https://www.asme.org/codes-standards/find-codes-standards/drawing-sheet-size-and-format
- ISO, "ISO 7200:2004 Technical product documentation - Data fields in title
  blocks and document headers":
  https://www.iso.org/standard/35446.html
- St-5 CAD Standard, "Drawing Layout, Borders and Margins":
  https://www.cad-standard.com/technical-drawing-basics/drawing-sheet-layout-boarders-margins

