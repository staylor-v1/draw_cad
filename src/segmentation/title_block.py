"""Title block and border segmentation for engineering drawing sheets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps

@dataclass(frozen=True)
class NormalizedBox:
    x: float
    y: float
    w: float
    h: float

    def clipped(self) -> "NormalizedBox":
        x0 = min(max(self.x, 0.0), 1.0)
        y0 = min(max(self.y, 0.0), 1.0)
        x1 = min(max(self.x + self.w, 0.0), 1.0)
        y1 = min(max(self.y + self.h, 0.0), 1.0)
        return NormalizedBox(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        box = self.clipped()
        left = int(round(box.x * width))
        top = int(round(box.y * height))
        right = int(round((box.x + box.w) * width))
        bottom = int(round((box.y + box.h) * height))
        return left, top, max(left + 1, right), max(top + 1, bottom)

    def to_dict(self) -> dict[str, float]:
        return asdict(self.clipped())


@dataclass(frozen=True)
class TitleBlockCandidate:
    crop: NormalizedBox
    confidence: float
    score: float
    method: str
    line_rows: int
    line_cols: int
    dark_density: float
    notes: tuple[str, ...]
    present: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["crop"] = self.crop.to_dict()
        data["notes"] = list(self.notes)
        return data


@dataclass(frozen=True)
class BorderSegmentation:
    confidence: float
    masks: tuple[NormalizedBox, ...]
    notes: tuple[str, ...]
    present: bool = True

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "confidence": self.confidence,
            "masks": [
                {"label": label, **box.to_dict()}
                for label, box in zip(("top border", "bottom border", "left border", "right border"), self.masks)
            ],
            "notes": list(self.notes),
        }


def analyze_drawing_structure(image_path: str | Path) -> dict:
    """Return title-block and border segmentation data for a drawing image."""

    from src.segmentation.callouts import load_callout_fixture, load_projection_fixture
    from src.segmentation.gdt import detect_gdt_callouts
    from src.segmentation.masks import build_annotation_masks

    image = Image.open(image_path)
    title_block = detect_title_block(image)
    border = estimate_border(image)
    if not border.present and title_block.crop.y + title_block.crop.h < 0.95:
        title_block = TitleBlockCandidate(
            crop=title_block.crop,
            confidence=min(title_block.confidence, 0.2),
            score=min(title_block.score, 0.2),
            method=title_block.method,
            line_rows=title_block.line_rows,
            line_cols=title_block.line_cols,
            dark_density=title_block.dark_density,
            notes=(
                *title_block.notes,
                "no title block detected because image is cropped and candidate does not reach sheet/title-block edge",
            ),
            present=False,
        )
    projection_fixture = load_projection_fixture(image_path)
    projections = projection_fixture or detect_projection_regions(image, title_block.crop if title_block.present else None)
    gdt = detect_gdt_callouts(image_path)
    callouts = load_callout_fixture(image_path)
    annotation_masks = build_annotation_masks(image_path, [*gdt, *callouts], projections)
    return {
        "titleBlock": {
            "present": title_block.present,
            "crop": title_block.crop.to_dict(),
            "confidence": title_block.confidence,
            "candidate": title_block.to_dict(),
        },
        "border": border.to_dict(),
        "projections": projections,
        "gdt": gdt,
        "callouts": callouts,
        "annotationMasks": annotation_masks,
    }


def detect_title_block(image: Image.Image) -> TitleBlockCandidate:
    """Detect the most likely title block using ruled-table structure.

    This is intentionally deterministic. Gemma 4 can use it as a teacher tool:
    it receives a best candidate, confidence, and reasons instead of a fixed
    lower-right crop.
    """

    gray = ImageOps.grayscale(image)
    arr = np.asarray(gray)
    height, width = arr.shape
    dark = _dark_mask(arr)

    search_y0 = int(height * 0.52)
    search_y1 = int(height * 0.985)
    search_x0 = int(width * 0.25)
    search_x1 = int(width * 0.99)
    roi = dark[search_y0:search_y1, search_x0:search_x1]

    if roi.size == 0:
        return _fallback_candidate(width, height, ("empty image region",))

    row_counts = roi.sum(axis=1)
    row_threshold = max(int((search_x1 - search_x0) * 0.16), int(np.percentile(row_counts, 96) * 0.72))
    row_indices = np.flatnonzero(row_counts >= row_threshold) + search_y0

    if len(row_indices) < 2:
        return _fallback_candidate(width, height, ("not enough lower-sheet horizontal ruling evidence",))

    row_groups = _group_close_indices(row_indices, max_gap=max(2, height // 420))
    row_centers = [int((start + end) / 2) for start, end in row_groups]
    bottom_cluster = _select_bottom_table_rows(row_centers, height)

    if len(bottom_cluster) < 2:
        return _fallback_candidate(width, height, ("could not form a lower title-block row cluster",))

    y_top = max(0, min(bottom_cluster) - max(3, height // 160))
    y_bottom = min(height, max(bottom_cluster) + max(5, height // 120))
    if y_bottom - y_top < height * 0.045:
        y_top = max(search_y0, y_bottom - int(height * 0.12))

    band = dark[y_top:y_bottom, search_x0:search_x1]
    col_counts = band.sum(axis=0)
    col_threshold = max(int((y_bottom - y_top) * 0.22), int(np.percentile(col_counts, 94) * 0.65))
    col_indices = np.flatnonzero(col_counts >= col_threshold) + search_x0

    if len(col_indices) >= 2:
        col_groups = _group_close_indices(col_indices, max_gap=max(2, width // 600))
        col_centers = [int((start + end) / 2) for start, end in col_groups]
        x_left, x_right = _select_title_block_columns(col_centers, width)
        if y_top / height > 0.86 and (x_right - x_left) / width > 0.55:
            right_table_cols = [col for col in col_centers if col >= width * 0.70]
            if len(right_table_cols) >= 2 and max(right_table_cols) - min(right_table_cols) >= width * 0.16:
                x_left, x_right = min(right_table_cols), max(right_table_cols)
    else:
        x_left, x_right = _horizontal_extent_from_rows(dark, bottom_cluster, search_x0, search_x1)

    pad_x = max(4, width // 280)
    pad_y = max(3, height // 260)
    x_left = max(0, x_left - pad_x)
    x_right = min(width, x_right + pad_x)
    y_top = max(0, y_top - pad_y)
    y_bottom = min(height, y_bottom + pad_y)

    if x_right - x_left < width * 0.16 or y_bottom - y_top < height * 0.04:
        return _fallback_candidate(width, height, ("candidate table was too small",))

    candidate_mask = dark[y_top:y_bottom, x_left:x_right]
    density = float(candidate_mask.mean())
    line_rows = _count_projection_peaks(candidate_mask.sum(axis=1), min_fraction=0.18, span=x_right - x_left)
    line_cols = _count_projection_peaks(candidate_mask.sum(axis=0), min_fraction=0.16, span=y_bottom - y_top)
    right_contact = x_right / width
    bottom_contact = y_bottom / height
    width_ratio = (x_right - x_left) / width
    height_ratio = (y_bottom - y_top) / height

    score = (
        min(line_rows / 6.0, 1.0) * 0.28
        + min(line_cols / 6.0, 1.0) * 0.24
        + _range_score(density, 0.025, 0.22) * 0.18
        + _range_score(width_ratio, 0.24, 0.72) * 0.12
        + _range_score(height_ratio, 0.07, 0.32) * 0.10
        + min(max((right_contact - 0.68) / 0.25, 0.0), 1.0) * 0.04
        + min(max((bottom_contact - 0.72) / 0.23, 0.0), 1.0) * 0.04
    )
    confidence = round(min(max(score, 0.05), 0.98), 3)

    notes = []
    if line_rows < 3:
        notes.append("weak horizontal title-block ruling evidence")
    if line_cols < 3:
        notes.append("weak vertical title-block ruling evidence")
    if bottom_contact < 0.78:
        notes.append("candidate is not strongly attached to lower sheet area")
    if not notes:
        notes.append("candidate selected from lower ruled-table cluster")

    present = line_rows >= 4 and line_cols >= 4 and density >= 0.075
    if not present:
        notes.append("no title block detected; lower ruled marks look like dimensions/callouts rather than a title table")
        confidence = min(confidence, 0.22)

    return TitleBlockCandidate(
        crop=NormalizedBox(x_left / width, y_top / height, (x_right - x_left) / width, (y_bottom - y_top) / height),
        confidence=confidence,
        score=round(score, 3),
        method="lower_ruled_table_projection_v1",
        line_rows=line_rows,
        line_cols=line_cols,
        dark_density=round(density, 4),
        notes=tuple(notes),
        present=present,
    )


def estimate_border(image: Image.Image) -> BorderSegmentation:
    """Estimate sheet border masks as edge bands.

    The current project treats border zones as format artifacts. The masks are
    deliberately broad and conservative so a supervisor can verify whether the
    border was separated from title-block/table content.
    """

    width, height = image.size
    gray = ImageOps.grayscale(image)
    dark = _dark_mask(np.asarray(gray))
    edge = max(4, int(min(width, height) * 0.06))
    continuity = (
        float(dark[:edge, :].sum(axis=1).max() / width),
        float(dark[-edge:, :].sum(axis=1).max() / width),
        float(dark[:, :edge].sum(axis=0).max() / height),
        float(dark[:, -edge:].sum(axis=0).max() / height),
    )
    strong_edges = sum(value >= 0.65 for value in continuity)
    present = strong_edges >= 3
    if not present:
        return BorderSegmentation(
            confidence=round(max(0.05, min(0.45, max(continuity))), 3),
            masks=(),
            notes=(
                "no sheet border detected; image appears cropped to drawing content",
                "do not mask border/title-block regions for this input unless another detector finds them",
            ),
            present=False,
        )
    masks = (
        NormalizedBox(0.0, 0.0, 1.0, edge / height),
        NormalizedBox(0.0, 1.0 - edge / height, 1.0, edge / height),
        NormalizedBox(0.0, 0.0, edge / width, 1.0),
        NormalizedBox(1.0 - edge / width, 0.0, edge / width, 1.0),
    )
    edge_density = float(
        np.concatenate(
            [
                dark[:edge, :].ravel(),
                dark[-edge:, :].ravel(),
                dark[:, :edge].ravel(),
                dark[:, -edge:].ravel(),
            ]
        ).mean()
    )
    confidence = round(min(0.95, max(0.25, edge_density * 8.0)), 3)
    return BorderSegmentation(
        confidence=confidence,
        masks=masks,
        notes=(
            "border zones are masked as sheet format artifacts",
            "zone letters/numbers are not interpreted in this pass",
        ),
        present=True,
    )


def detect_projection_regions(image: Image.Image, title_block: NormalizedBox | None = None, max_regions: int = 5) -> list[dict]:
    """Find nonblank drawing regions for projection review panes.

    The detector groups nearby dark drawing primitives after a coarse dilation,
    excluding the border and known title block. It is deliberately conservative:
    candidates should show useful content in the dashboard even before Gemma 4
    assigns semantic front/top/right labels.
    """

    gray = ImageOps.grayscale(image)
    arr = np.asarray(gray)
    height, width = arr.shape
    mask = _dark_mask(arr)

    edge = max(4, int(min(width, height) * 0.06))
    mask[:edge, :] = False
    mask[-edge:, :] = False
    mask[:, :edge] = False
    mask[:, -edge:] = False

    title_limit_y = height
    if title_block is not None:
        left, top, right, bottom = title_block.to_pixels(width, height)
        title_limit_y = top
        pad = max(6, int(min(width, height) * 0.012))
        mask[max(0, top - pad) : min(height, bottom + pad), max(0, left - pad) : min(width, right + pad)] = False

    scale = min(1.0, 520.0 / max(width, height))
    small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    small = Image.fromarray((mask * 255).astype("uint8"), mode="L").resize(small_size, Image.Resampling.NEAREST)
    dilation = max(5, int(min(small_size) * 0.016))
    if dilation % 2 == 0:
        dilation += 1
    small_mask = np.asarray(small.filter(ImageFilter.MaxFilter(dilation))) > 0

    components = _connected_components(small_mask)
    min_area = max(18, int(small_mask.size * 0.004))
    candidates = []
    for component in components:
        sx0, sy0, sx1, sy1, area = component
        if area < min_area:
            continue
        bw = sx1 - sx0
        bh = sy1 - sy0
        if bw < small_size[0] * 0.06 or bh < small_size[1] * 0.045:
            continue
        x0 = max(0, int(sx0 / scale) - int(width * 0.012))
        y0 = max(0, int(sy0 / scale) - int(height * 0.012))
        x1 = min(width, int(sx1 / scale) + int(width * 0.012))
        y1 = min(height, int(sy1 / scale) + int(height * 0.012))
        box = NormalizedBox(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height).clipped()
        if box.w * box.h < 0.01:
            continue
        candidates.append((area, box))

    merged = _merge_overlapping_boxes([box for _, box in sorted(candidates, key=lambda item: item[0], reverse=True)])
    expanded = [_expand_projection_box(box, width, height, title_limit_y) for box in merged]
    ordered = sorted(expanded, key=lambda box: (box.y, box.x))[:max_regions]
    return [
        {
            "id": f"projection-{index + 1}",
            "label": _projection_label(box),
            "axis": "unassigned",
            "confidence": 0.42,
            "crop": box.to_dict(),
            "segmentationMode": "component_mask",
            "maskNote": (
                "Rectangular crop is for review only; use the underlying connected-component mask "
                "when callouts cross another projection's bounding box."
            ),
        }
        for index, box in enumerate(ordered)
    ]


def crop_image(image_path: str | Path, box: NormalizedBox) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image.crop(box.to_pixels(*image.size))


def iter_default_title_block_images(root: str | Path = ".") -> Iterable[Path]:
    root = Path(root)
    patterns = [
        "training_data/gdt/*",
        "training_data/title_block_examples/*",
        "benchmarks/drawings/*/drawing.*",
        "test_result/cases/*/original.*",
    ]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.suffix.lower() not in exts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _dark_mask(arr: np.ndarray) -> np.ndarray:
    threshold = min(225, max(145, int(np.percentile(arr, 35) + 42)))
    return arr < threshold


def _fallback_candidate(width: int, height: int, notes: tuple[str, ...]) -> TitleBlockCandidate:
    return TitleBlockCandidate(
        crop=NormalizedBox(0.52, 0.72, 0.43, 0.22),
        confidence=0.12,
        score=0.12,
        method="lower_right_prior_fallback",
        line_rows=0,
        line_cols=0,
        dark_density=0.0,
        notes=notes,
        present=False,
    )


def _group_close_indices(indices: np.ndarray | list[int], max_gap: int) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    values = [int(v) for v in indices]
    groups = []
    start = prev = values[0]
    for value in values[1:]:
        if value - prev <= max_gap:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _select_bottom_table_rows(row_centers: list[int], height: int) -> list[int]:
    rows = [row for row in row_centers if row >= height * 0.55]
    if not rows:
        return []
    bottom = max(rows)
    selected = [bottom]
    max_gap = max(18, int(height * 0.045))
    for row in reversed(rows[:-1]):
        if selected[-1] - row <= max_gap:
            selected.append(row)
        elif len(selected) >= 3:
            break
    selected = sorted(selected)
    if len(selected) < 2:
        return rows[-4:]
    return selected


def _select_title_block_columns(col_centers: list[int], width: int) -> tuple[int, int]:
    cols = [col for col in col_centers if col >= width * 0.25]
    if len(cols) < 2:
        return int(width * 0.52), int(width * 0.95)
    right = max(cols)
    left_candidates = [col for col in cols if right - col >= width * 0.18]
    left = min(left_candidates) if left_candidates else min(cols)
    if right - left > width * 0.74:
        left = min((col for col in cols if col >= width * 0.38), default=left)
    return left, right


def _horizontal_extent_from_rows(dark: np.ndarray, rows: list[int], search_x0: int, search_x1: int) -> tuple[int, int]:
    xs: list[int] = []
    for row in rows:
        row_values = np.flatnonzero(dark[row, search_x0:search_x1]) + search_x0
        if len(row_values):
            xs.extend([int(row_values[0]), int(row_values[-1])])
    if len(xs) >= 2:
        return min(xs), max(xs)
    width = dark.shape[1]
    return int(width * 0.52), int(width * 0.95)


def _count_projection_peaks(values: np.ndarray, min_fraction: float, span: int) -> int:
    if values.size == 0:
        return 0
    threshold = max(int(span * min_fraction), int(np.percentile(values, 94) * 0.6))
    groups = _group_close_indices(np.flatnonzero(values >= threshold), max_gap=2)
    return len(groups)


def _range_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / low)
    return max(0.0, 1.0 - (value - high) / high)


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                x0 = min(x0, cx)
                x1 = max(x1, cx)
                y0 = min(y0, cy)
                y1 = max(y1, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            components.append((x0, y0, x1 + 1, y1 + 1, area))
    return components


def _merge_overlapping_boxes(boxes: list[NormalizedBox]) -> list[NormalizedBox]:
    merged: list[NormalizedBox] = []
    for box in boxes:
        current = box
        changed = True
        while changed:
            changed = False
            kept = []
            for other in merged:
                if _box_overlap(current, other) > 0.35:
                    current = _box_union(current, other)
                    changed = True
                else:
                    kept.append(other)
            merged = kept
        merged.append(current)
    return merged


def _box_overlap(a: NormalizedBox, b: NormalizedBox) -> float:
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x + b.w, b.y + b.h
    ix = max(0.0, min(ax1, bx1) - max(a.x, b.x))
    iy = max(0.0, min(ay1, by1) - max(a.y, b.y))
    inter = ix * iy
    smaller = max(1e-6, min(a.w * a.h, b.w * b.h))
    return inter / smaller


def _box_union(a: NormalizedBox, b: NormalizedBox) -> NormalizedBox:
    x0 = min(a.x, b.x)
    y0 = min(a.y, b.y)
    x1 = max(a.x + a.w, b.x + b.w)
    y1 = max(a.y + a.h, b.y + b.h)
    return NormalizedBox(x0, y0, x1 - x0, y1 - y0).clipped()


def _projection_label(box: NormalizedBox) -> str:
    vertical = "upper" if box.y < 0.33 else "middle" if box.y < 0.62 else "lower"
    horizontal = "left" if box.x < 0.33 else "center" if box.x < 0.62 else "right"
    return f"{vertical} {horizontal} projection candidate"


def _expand_projection_pixels(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    title_limit_y: int,
) -> tuple[int, int, int, int]:
    pad_x = int(width * 0.014)
    pad_top = int(height * 0.018)
    pad_bottom = int(height * 0.035)
    expanded_x0 = max(0, x0 - pad_x)
    expanded_y0 = max(0, y0 - pad_top)
    expanded_x1 = min(width, x1 + pad_x)
    bottom_limit = max(y1, title_limit_y - max(3, int(height * 0.006)))
    expanded_y1 = min(height, bottom_limit, y1 + pad_bottom)
    return expanded_x0, expanded_y0, expanded_x1, expanded_y1


def _expand_projection_box(box: NormalizedBox, width: int, height: int, title_limit_y: int) -> NormalizedBox:
    x0, y0, x1, y1 = box.to_pixels(width, height)
    expanded = _expand_projection_pixels(x0, y0, x1, y1, width, height, title_limit_y)
    left, top, right, bottom = expanded
    return NormalizedBox(left / width, top / height, (right - left) / width, (bottom - top) / height).clipped()
