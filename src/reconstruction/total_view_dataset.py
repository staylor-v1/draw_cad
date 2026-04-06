"""Dataset helpers for the zipped Total_view_data orthographic archives."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


_VIEWBOX_SPLIT_RE = re.compile(r"[,\s]+")


@dataclass(frozen=True)
class SvgPolyline:
    """A polyline entity extracted from a dataset SVG view."""

    points: list[tuple[float, float]]
    stroke: str
    stroke_width: float


@dataclass(frozen=True)
class SvgOrthographicView:
    """One SVG orthographic view from the archive."""

    case_id: str
    suffix: str
    archive_path: Path
    member_name: str
    view_box: tuple[float, float, float, float]
    polylines: list[SvgPolyline]
    svg_text: str


@dataclass(frozen=True)
class OrthographicTriplet:
    """A grouped f/r/t case from Total_view_data."""

    case_id: str
    views: dict[str, SvgOrthographicView]


class TotalViewArchive:
    """Read grouped orthographic views directly from the dataset zip file."""

    def __init__(self, svg_zip_path: str | Path):
        self.svg_zip_path = Path(svg_zip_path)
        self._members_by_case = self._build_index()

    def _build_index(self) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        with ZipFile(self.svg_zip_path) as zf:
            for member_name in zf.namelist():
                if member_name.endswith("/"):
                    continue
                stem = Path(member_name).stem
                if "_" not in stem:
                    continue
                case_id, suffix = stem.rsplit("_", 1)
                index.setdefault(case_id, {})[suffix] = member_name
        return index

    def available_views(self, case_id: str) -> list[str]:
        """Return sorted available suffixes for a case."""
        return sorted(self._members_by_case.get(case_id, {}))

    def case_ids(
        self,
        required_views: tuple[str, ...] = ("f", "r", "t"),
        require_complete: bool = True,
    ) -> list[str]:
        """List case ids, optionally filtering to complete orthographic triplets."""
        case_ids = sorted(self._members_by_case)
        if not require_complete:
            return case_ids

        required = set(required_views)
        return [
            case_id
            for case_id in case_ids
            if required.issubset(self._members_by_case[case_id])
        ]

    def load_view(self, case_id: str, suffix: str) -> SvgOrthographicView:
        """Load and parse one SVG view from the archive."""
        member_name = self._members_by_case[case_id][suffix]
        with ZipFile(self.svg_zip_path) as zf:
            svg_text = zf.read(member_name).decode("utf-8")
        view_box, polylines = parse_svg_view(svg_text)
        return SvgOrthographicView(
            case_id=case_id,
            suffix=suffix,
            archive_path=self.svg_zip_path,
            member_name=member_name,
            view_box=view_box,
            polylines=polylines,
            svg_text=svg_text,
        )

    def load_triplet(
        self,
        case_id: str,
        view_suffixes: tuple[str, str, str] = ("f", "r", "t"),
    ) -> OrthographicTriplet:
        """Load the configured orthographic triplet for a case."""
        views = {
            suffix: self.load_view(case_id, suffix)
            for suffix in view_suffixes
        }
        return OrthographicTriplet(case_id=case_id, views=views)


def parse_svg_view(svg_text: str) -> tuple[tuple[float, float, float, float], list[SvgPolyline]]:
    """Parse the polyline-only dataset SVG format."""
    root = ET.fromstring(svg_text)
    view_box = _parse_view_box(root.attrib["viewBox"])

    polylines: list[SvgPolyline] = []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag != "polyline":
            continue

        points_attr = elem.attrib.get("points", "").strip()
        points = _parse_polyline_points(points_attr)
        if len(points) < 2:
            continue

        polylines.append(
            SvgPolyline(
                points=points,
                stroke=elem.attrib.get("stroke", "").strip().lower(),
                stroke_width=float(elem.attrib.get("stroke-width", "0.7")),
            )
        )

    return view_box, polylines


def _parse_view_box(raw: str) -> tuple[float, float, float, float]:
    parts = [part for part in _VIEWBOX_SPLIT_RE.split(raw.strip()) if part]
    if len(parts) != 4:
        raise ValueError(f"Unexpected viewBox value: {raw}")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _parse_polyline_points(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in raw.split():
        x_str, y_str = token.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points
