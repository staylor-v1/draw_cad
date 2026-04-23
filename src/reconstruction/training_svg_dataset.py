"""Parse training-data SVG drawings into orthographic triplets."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.reconstruction.total_view_dataset import (
    OrthographicTriplet,
    SvgOrthographicView,
    SvgPolyline,
)


_COMMAND_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
)
_MATRIX_RE = re.compile(r"matrix\(([^)]+)\)")
_TRANSLATE_RE = re.compile(r"translate\(([^)]+)\)")
_STYLE_SPLIT_RE = re.compile(r"\s*;\s*")
_VIEW_SUFFIXES = ("f", "r", "t")
_VISIBLE_STROKES = frozenset({"black", "#000000"})
_CURVE_SAMPLES = 16
_NAMED_VIEW_GROUPS = frozenset({"front", "right", "top", "f", "r", "t", "view_f", "view_r", "view_t"})


@dataclass(frozen=True)
class _StrokeEntity:
    """One sampled stroked entity from a training SVG."""

    points: list[tuple[float, float]]
    stroke: str
    stroke_width: float
    group_id: str


def load_training_svg_triplet(
    svg_path: str | Path,
    view_suffixes: tuple[str, str, str] = _VIEW_SUFFIXES,
) -> OrthographicTriplet:
    """Load one training-data SVG drawing as an orthographic triplet."""
    path = Path(svg_path)
    return parse_training_svg_triplet(
        path.read_text(encoding="utf-8"),
        case_id=path.stem,
        source_path=path,
        view_suffixes=view_suffixes,
    )


def parse_training_svg_triplet(
    svg_text: str,
    case_id: str = "training_case",
    source_path: str | Path = "training_svg",
    view_suffixes: tuple[str, str, str] = _VIEW_SUFFIXES,
) -> OrthographicTriplet:
    """Parse one FreeCAD-style training SVG into front/right/top views."""
    root = ET.fromstring(svg_text)
    drawing_content = _find_drawing_content(root)
    entities = _collect_entities(drawing_content)
    groups = _group_entities(entities)
    selected_groups = _select_primary_views(groups)
    mapped_groups = _assign_view_suffixes(selected_groups, view_suffixes=view_suffixes)

    archive_path = Path(source_path)
    views: dict[str, SvgOrthographicView] = {}
    for suffix, group in mapped_groups.items():
        min_x, min_y, max_x, max_y = group["bbox"]
        views[suffix] = SvgOrthographicView(
            case_id=case_id,
            suffix=suffix,
            archive_path=archive_path,
            member_name=str(archive_path),
            view_box=(min_x, min_y, max_x - min_x, max_y - min_y),
            polylines=group["polylines"],
            svg_text=svg_text,
        )

    return OrthographicTriplet(case_id=case_id, views=views)


def _find_drawing_content(root: ET.Element) -> ET.Element:
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] == "g" and elem.attrib.get("id") == "DrawingContent":
            return elem
    return root


def _collect_entities(root: ET.Element) -> list[_StrokeEntity]:
    entities: list[_StrokeEntity] = []
    _walk_svg(
        root,
        matrix=_identity_matrix(),
        inherited_style={},
        group_id=None,
        sink=entities,
    )
    return entities


def _walk_svg(
    elem: ET.Element,
    matrix: tuple[float, float, float, float, float, float],
    inherited_style: dict[str, str],
    group_id: str | None,
    sink: list[_StrokeEntity],
) -> None:
    tag = elem.tag.rsplit("}", 1)[-1]
    style = _merge_style(inherited_style, elem.attrib)

    next_matrix = matrix
    next_group_id = group_id
    explicit_group_id = _normalized_view_group_id(elem.attrib.get("id", ""))
    if explicit_group_id is not None:
        next_group_id = explicit_group_id

    transform = elem.attrib.get("transform")
    if transform:
        next_matrix = _compose_matrices(matrix, _parse_transform(transform))
        if next_group_id is None:
            next_group_id = _group_id_from_matrix(next_matrix)

    if tag in {"path", "polyline"}:
        stroke = style.get("stroke", "").strip().lower()
        if stroke in {"", "none"}:
            return

        raw_points = (
            _parse_path_points(elem.attrib.get("d", ""))
            if tag == "path"
            else _parse_polyline_points(elem.attrib.get("points", ""))
        )
        if len(raw_points) < 2:
            return

        points = [_apply_matrix(next_matrix, point) for point in raw_points]
        dasharray = style.get("stroke-dasharray", "").strip().lower()
        normalized_stroke = "red" if dasharray not in {"", "none"} and stroke in _VISIBLE_STROKES else stroke
        sink.append(
            _StrokeEntity(
                points=points,
                stroke=normalized_stroke,
                stroke_width=float(style.get("stroke-width", "1.0") or "1.0"),
                group_id=next_group_id or "root",
            )
        )
        return

    for child in list(elem):
        _walk_svg(
            child,
            matrix=next_matrix,
            inherited_style=style,
            group_id=next_group_id,
            sink=sink,
        )


def _merge_style(inherited: dict[str, str], attrib: dict[str, str]) -> dict[str, str]:
    merged = dict(inherited)
    merged.update(_parse_style_attribute(attrib.get("style", "")))
    for key in ("stroke", "stroke-width", "stroke-dasharray", "fill"):
        if key in attrib:
            merged[key] = attrib[key]
    return merged


def _parse_style_attribute(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    parsed: dict[str, str] = {}
    for chunk in _STYLE_SPLIT_RE.split(raw.strip()):
        if not chunk or ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _group_entities(entities: list[_StrokeEntity]) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for entity in entities:
        group = groups.setdefault(
            entity.group_id,
            {
                "polylines": [],
                "bbox": [math.inf, math.inf, -math.inf, -math.inf],
                "stroke_count": 0,
            },
        )
        polylines = group["polylines"]
        assert isinstance(polylines, list)
        polylines.append(
            SvgPolyline(
                points=entity.points,
                stroke=entity.stroke,
                stroke_width=entity.stroke_width,
            )
        )
        bbox = group["bbox"]
        assert isinstance(bbox, list)
        xs = [point[0] for point in entity.points]
        ys = [point[1] for point in entity.points]
        bbox[0] = min(bbox[0], min(xs))
        bbox[1] = min(bbox[1], min(ys))
        bbox[2] = max(bbox[2], max(xs))
        bbox[3] = max(bbox[3], max(ys))
        group["stroke_count"] = int(group["stroke_count"]) + 1
    return groups


def _select_primary_views(groups: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for group_id, group in groups.items():
        bbox = group["bbox"]
        assert isinstance(bbox, list)
        min_x, min_y, max_x, max_y = bbox
        is_named_view = _view_suffix_from_group_id(group_id, _VIEW_SUFFIXES) is not None
        if not math.isfinite(min_x) or (int(group["stroke_count"]) < 2 and not is_named_view):
            continue
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            continue
        score = width * height
        candidates.append(
            {
                "group_id": group_id,
                "polylines": group["polylines"],
                "bbox": (min_x, min_y, max_x, max_y),
                "score": score,
                "center_x": (min_x + max_x) / 2.0,
                "center_y": (min_y + max_y) / 2.0,
            }
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    if len(candidates) < 3:
        raise ValueError("Could not identify three orthographic views from training SVG.")
    return candidates[:3]


def _assign_view_suffixes(
    groups: list[dict[str, object]],
    view_suffixes: tuple[str, str, str],
) -> dict[str, dict[str, object]]:
    explicit_groups: dict[str, dict[str, object]] = {}
    for group in groups:
        suffix = _view_suffix_from_group_id(str(group["group_id"]), view_suffixes)
        if suffix is not None:
            explicit_groups[suffix] = group
    if set(explicit_groups) == set(view_suffixes):
        return {suffix: explicit_groups[suffix] for suffix in view_suffixes}

    front_suffix, right_suffix, top_suffix = view_suffixes
    ranked_by_y = sorted(groups, key=lambda item: float(item["center_y"]))
    top_group = ranked_by_y[0]
    lower_groups = sorted(ranked_by_y[1:], key=lambda item: float(item["center_x"]))
    if len(lower_groups) != 2:
        raise ValueError("Expected exactly two lower orthographic views.")
    front_group, right_group = lower_groups
    return {
        front_suffix: front_group,
        right_suffix: right_group,
        top_suffix: top_group,
    }


def _parse_polyline_points(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in raw.split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points


def _parse_path_points(raw: str) -> list[tuple[float, float]]:
    tokens = _COMMAND_RE.findall(raw)
    if not tokens:
        return []

    points: list[tuple[float, float]] = []
    index = 0
    command = ""
    current = (0.0, 0.0)
    start = (0.0, 0.0)

    def read_number() -> float:
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
        if not command:
            raise ValueError(f"Path data is missing a command: {raw}")

        absolute = command.isupper()
        opcode = command.upper()

        if opcode == "M":
            x = read_number()
            y = read_number()
            current = (x, y) if absolute else (current[0] + x, current[1] + y)
            start = current
            points.append(current)
            command = "L" if absolute else "l"
        elif opcode == "L":
            x = read_number()
            y = read_number()
            current = (x, y) if absolute else (current[0] + x, current[1] + y)
            points.append(current)
        elif opcode == "H":
            x = read_number()
            current = (x, current[1]) if absolute else (current[0] + x, current[1])
            points.append(current)
        elif opcode == "V":
            y = read_number()
            current = (current[0], y) if absolute else (current[0], current[1] + y)
            points.append(current)
        elif opcode == "C":
            c1x = read_number()
            c1y = read_number()
            c2x = read_number()
            c2y = read_number()
            px = read_number()
            py = read_number()
            control_1 = (c1x, c1y) if absolute else (current[0] + c1x, current[1] + c1y)
            control_2 = (c2x, c2y) if absolute else (current[0] + c2x, current[1] + c2y)
            end = (px, py) if absolute else (current[0] + px, current[1] + py)
            for step in range(1, _CURVE_SAMPLES + 1):
                t = step / _CURVE_SAMPLES
                points.append(_sample_cubic_bezier(current, control_1, control_2, end, t))
            current = end
        elif opcode == "Z":
            if points and points[-1] != start:
                points.append(start)
            current = start
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")

        while index < len(tokens) and not tokens[index].isalpha():
            if command.upper() == "Z":
                break
            # Implicit command repetition.
            if opcode == "M":
                command = "L" if absolute else "l"
                opcode = "L"
            else:
                break

    return points


def _sample_cubic_bezier(
    start: tuple[float, float],
    control_1: tuple[float, float],
    control_2: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    one_minus_t = 1.0 - t
    x = (
        one_minus_t ** 3 * start[0]
        + 3.0 * one_minus_t * one_minus_t * t * control_1[0]
        + 3.0 * one_minus_t * t * t * control_2[0]
        + t ** 3 * end[0]
    )
    y = (
        one_minus_t ** 3 * start[1]
        + 3.0 * one_minus_t * one_minus_t * t * control_1[1]
        + 3.0 * one_minus_t * t * t * control_2[1]
        + t ** 3 * end[1]
    )
    return (x, y)


def _identity_matrix() -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _parse_transform(raw: str) -> tuple[float, float, float, float, float, float]:
    raw = raw.strip()
    matrix_match = _MATRIX_RE.fullmatch(raw)
    if matrix_match:
        values = _parse_transform_values(matrix_match.group(1))
        if len(values) != 6:
            raise ValueError(f"Unexpected SVG matrix transform: {raw}")
        return tuple(values)  # type: ignore[return-value]

    translate_match = _TRANSLATE_RE.fullmatch(raw)
    if translate_match:
        values = _parse_transform_values(translate_match.group(1))
        if len(values) == 1:
            values.append(0.0)
        if len(values) != 2:
            raise ValueError(f"Unexpected SVG translate transform: {raw}")
        return (1.0, 0.0, 0.0, 1.0, values[0], values[1])

    raise ValueError(f"Unsupported SVG transform: {raw}")


def _parse_transform_values(raw: str) -> list[float]:
    return [float(value) for value in re.split(r"[,\s]+", raw.strip()) if value]


def _compose_matrices(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_matrix(
    matrix: tuple[float, float, float, float, float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def _normalized_view_group_id(raw: str) -> str | None:
    normalized = raw.strip().lower()
    if normalized in _NAMED_VIEW_GROUPS:
        return normalized
    return None


def _view_suffix_from_group_id(
    group_id: str,
    view_suffixes: tuple[str, str, str],
) -> str | None:
    front_suffix, right_suffix, top_suffix = view_suffixes
    normalized = group_id.strip().lower()
    mapping = {
        "front": front_suffix,
        "f": front_suffix,
        "view_f": front_suffix,
        "right": right_suffix,
        "r": right_suffix,
        "view_r": right_suffix,
        "top": top_suffix,
        "t": top_suffix,
        "view_t": top_suffix,
    }
    return mapping.get(normalized)


def _group_id_from_matrix(matrix: tuple[float, float, float, float, float, float]) -> str:
    return ",".join(f"{value:.6f}" for value in matrix)
