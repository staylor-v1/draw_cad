# Training Orthographic Case Study

This report compares the legacy deterministic candidate families (`visual_hull*`, `axisymmetric*`)
against the expanded candidate set that adds searched profile-extrusion plans from the training SVG triplets.

Closed-loop selection uses orthographic reprojection back into the source drawing views.

## Summary

- Cases evaluated: 3
- Enhanced best candidate matched or beat legacy on 3 cases
- Mean reprojection delta: 0.0586
- Selected-case reprojection delta: 0.1757

## Cases

| Case | Legacy best | Legacy reprojection | Enhanced best | Enhanced reprojection | Delta | Scale ratio |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 119452_114cbeb4_0000 | visual_hull_hidden | 0.3827 | profile_extrude_f | 0.5584 | 0.1757 | 18.95 |
| 51509_fd5fcb9c_0001 | visual_hull_hidden | 0.4736 | visual_hull_hidden | 0.4736 | 0.0000 | 10.03 |
| 26764_ab35159a_0002 | visual_hull_hidden | 0.4461 | visual_hull_hidden | 0.4461 | 0.0000 | 17.08 |

## Selected Case: 119452_114cbeb4_0000

- Legacy best: `visual_hull_hidden` with reprojection `0.3827`.
- Enhanced best: `profile_extrude_f` with reprojection `0.5584`.
- Visible-line F1 improved from `0.6231` to `0.7973`.
- Hidden-line F1 improved from `0.1422` to `0.3195`.

The key shape error in the legacy visual-hull candidate is that it preserves the outer envelope
but misses the through-profile void. The profile-extrusion candidate restores that void, which
shows up immediately in the reprojection overlays and the visible/hidden line scores.

Absolute STEP-vs-STEP metrics remain poor on these SVG-only training drawings because the raw
drawing coordinates are not already expressed in reference-model millimeters. The closed-loop
orthographic reprojection is therefore the primary supervision signal for technique selection here.

## Overlay Files

- Legacy overlays: `overlays/119452_114cbeb4_0000__visual_hull_hidden__*.png`
- Enhanced overlays: `overlays/119452_114cbeb4_0000__profile_extrude_f__*.png`
