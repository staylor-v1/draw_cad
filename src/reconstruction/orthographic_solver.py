"""Deterministic reconstruction of Total_view_data orthographic triplets."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

from src.reconstruction.total_view_dataset import OrthographicTriplet, SvgOrthographicView, SvgPolyline
from src.schemas.pipeline_config import OrthographicReconstructionConfig, PipelineConfig


@dataclass(frozen=True)
class OrthographicContour:
    """The extracted outer silhouette for one orthographic view."""

    suffix: str
    points: list[tuple[float, float]]
    extents: tuple[float, float]


@dataclass(frozen=True)
class CircularFeatureCandidate:
    """A circular hidden-line feature detected in the top view."""

    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class CylindricalCut:
    """A cylinder cut represented in consensus view coordinates."""

    center_x: float
    center_y: float
    radius: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class HorizontalSupportSegment:
    """A merged horizontal red segment in a side view."""

    center: float
    half_length: float
    z: float


@dataclass(frozen=True)
class OrthographicReconstructionResult:
    """Generated build123d program and supporting metadata."""

    case_id: str
    code: str
    contours: dict[str, OrthographicContour]
    consensus_extents: dict[str, float]
    hidden_cylinders: list[CylindricalCut]


@dataclass(frozen=True)
class ReconstructionCandidate:
    """One deterministic reconstruction candidate."""

    name: str
    result: OrthographicReconstructionResult


class OrthographicTripletReconstructor:
    """Build a deterministic solid from f/r/t SVG views."""

    def __init__(
        self,
        config: OrthographicReconstructionConfig | None = None,
        view_suffixes: tuple[str, str, str] = ("f", "r", "t"),
    ):
        self.config = config or OrthographicReconstructionConfig()
        self.view_suffixes = view_suffixes
        self.visible_stroke_colors = {
            value.strip().lower()
            for value in self.config.visible_stroke_colors
        }

    @classmethod
    def from_pipeline_config(cls, config: PipelineConfig) -> OrthographicTripletReconstructor:
        """Create a reconstructor from the repo-wide config object."""
        preferred = tuple(config.total_view_data.preferred_views)
        return cls(
            config=config.orthographic_reconstruction,
            view_suffixes=preferred,  # type: ignore[arg-type]
        )

    def generate_program(
        self,
        triplet: OrthographicTriplet,
    ) -> OrthographicReconstructionResult:
        """Generate build123d code for the best available deterministic solid."""
        front_suffix, right_suffix, top_suffix = self.view_suffixes
        front = self._extract_outer_contour(triplet.views[front_suffix])
        right = self._extract_outer_contour(triplet.views[right_suffix])
        top = self._extract_outer_contour(triplet.views[top_suffix])

        x_extent = (front.extents[0] + top.extents[0]) / 2.0
        y_extent = (top.extents[1] + right.extents[0]) / 2.0
        z_extent = (front.extents[1] + right.extents[1]) / 2.0

        consensus = {
            "x": float(x_extent),
            "y": float(y_extent),
            "z": float(z_extent),
        }

        axisymmetric_profile = self._infer_axisymmetric_profile(
            front=front,
            right=right,
            top=top,
            consensus_extents=consensus,
        )
        if axisymmetric_profile is not None:
            diameter = float(np.mean([
                front.extents[0],
                right.extents[0],
                top.extents[0],
                top.extents[1],
            ]))
            consensus["x"] = diameter
            consensus["y"] = diameter

        hidden_cylinders = self._infer_hidden_cylinders(triplet, consensus)

        if axisymmetric_profile is not None:
            code = self._build_revolve_code(
                profile_points=axisymmetric_profile,
                consensus_extents=consensus,
                hidden_cylinders=hidden_cylinders,
            )
        else:
            code = self._build_code(
                front_points=self._scale_points(front.points, front.extents, (consensus["x"], consensus["z"])),
                top_points=self._scale_points(top.points, top.extents, (consensus["x"], consensus["y"])),
                right_points=self._scale_points(right.points, right.extents, (consensus["y"], consensus["z"])),
                consensus_extents=consensus,
                hidden_cylinders=hidden_cylinders,
            )
        return OrthographicReconstructionResult(
            case_id=triplet.case_id,
            code=code,
            contours={
                front_suffix: front,
                right_suffix: right,
                top_suffix: top,
            },
            consensus_extents=consensus,
            hidden_cylinders=hidden_cylinders,
        )

    def generate_candidate_programs(
        self,
        triplet: OrthographicTriplet,
    ) -> list[ReconstructionCandidate]:
        """Generate a small set of deterministic candidate programs."""
        variants = [
            ("visual_hull_hidden", False, True),
            ("visual_hull_base", False, False),
            ("axisymmetric_hidden", True, True),
            ("axisymmetric_base", True, False),
        ]

        candidates: list[ReconstructionCandidate] = []
        seen_signatures: set[tuple[str, tuple[tuple[str, float], ...], int]] = set()
        for name, axisymmetric_enabled, hidden_feature_enabled in variants:
            config = self.config.model_copy(
                update={
                    "axisymmetric_enabled": axisymmetric_enabled,
                    "hidden_feature_enabled": hidden_feature_enabled,
                }
            )
            result = OrthographicTripletReconstructor(
                config=config,
                view_suffixes=self.view_suffixes,
            ).generate_program(triplet)
            signature = (
                result.code,
                tuple(sorted(result.consensus_extents.items())),
                len(result.hidden_cylinders),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            candidates.append(ReconstructionCandidate(name=name, result=result))
        return candidates

    def _extract_outer_contour(self, view: SvgOrthographicView) -> OrthographicContour:
        visible_polylines = [
            polyline
            for polyline in view.polylines
            if polyline.stroke in self.visible_stroke_colors
        ]
        if not visible_polylines:
            raise ValueError(f"No visible polylines found for {view.member_name}")

        min_x, min_y, width, height = view.view_box
        scale = self.config.raster_max_dimension_px / max(width, height)
        max_stroke_pixels = max(
            polyline.stroke_width * scale
            for polyline in visible_polylines
        )
        pad = (
            self.config.raster_padding_px
            + int(np.ceil(max_stroke_pixels))
            + self.config.dilation_iterations
            + 2
        )

        raster_width = int(round(width * scale)) + pad * 2
        raster_height = int(round(height * scale)) + pad * 2

        image = Image.new("L", (raster_width, raster_height), 0)
        draw = ImageDraw.Draw(image)
        for polyline in visible_polylines:
            pixel_points = [
                (
                    pad + (x - min_x) * scale,
                    pad + (min_y + height - y) * scale,
                )
                for x, y in polyline.points
            ]
            draw.line(pixel_points, fill=255, width=1, joint="curve")

        mask = np.array(image, dtype=bool)
        if self.config.dilation_iterations > 0:
            mask = ndimage.binary_dilation(mask, iterations=self.config.dilation_iterations)
        mask = ndimage.binary_fill_holes(mask)
        mask = self._largest_component(mask)

        contours = measure.find_contours(mask.astype(float), 0.5)
        if not contours:
            raise ValueError(f"Could not extract a contour for {view.member_name}")

        contour = max(contours, key=self._contour_area_rows_cols)
        contour = measure.approximate_polygon(
            contour,
            tolerance=self.config.contour_simplify_tolerance_px,
        )

        points = [
            (
                float(min_x + (col - pad) / scale),
                float(min_y + height - (row - pad) / scale),
            )
            for row, col in contour
        ]
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]

        min_point_x = min(x for x, _ in points)
        min_point_y = min(y for _, y in points)
        normalized = [
            (x - min_point_x, y - min_point_y)
            for x, y in points
        ]

        extents = (
            max(x for x, _ in normalized),
            max(y for _, y in normalized),
        )
        return OrthographicContour(
            suffix=view.suffix,
            points=normalized,
            extents=extents,
        )

    def _infer_axisymmetric_profile(
        self,
        front: OrthographicContour,
        right: OrthographicContour,
        top: OrthographicContour,
        consensus_extents: dict[str, float],
    ) -> list[tuple[float, float]] | None:
        if not self.config.axisymmetric_enabled:
            return None

        top_circularity = contour_circularity(top.points)
        top_aspect = max(top.extents) / max(min(top.extents), 1e-6)
        if top_circularity < self.config.axisymmetric_top_circularity_min:
            return None
        if top_aspect > self.config.axisymmetric_top_aspect_ratio_max:
            return None

        diameter = float(np.mean([
            front.extents[0],
            right.extents[0],
            top.extents[0],
            top.extents[1],
        ]))
        z_extent = float(consensus_extents["z"])
        center = diameter / 2.0

        front_points = self._scale_points(front.points, front.extents, (diameter, z_extent))
        right_points = self._scale_points(right.points, right.extents, (diameter, z_extent))
        z_samples = np.linspace(0.0, z_extent, self.config.axisymmetric_profile_samples)

        front_profile = self._sample_axisymmetric_profile(front_points, center, z_samples)
        right_profile = self._sample_axisymmetric_profile(right_points, center, z_samples)
        if len(front_profile) < max(8, len(z_samples) // 2):
            return None
        if len(right_profile) < max(8, len(z_samples) // 2):
            return None

        front_center_error = np.mean(
            [offset for _, _, offset in front_profile]
        ) / max(diameter, 1e-6)
        right_center_error = np.mean(
            [offset for _, _, offset in right_profile]
        ) / max(diameter, 1e-6)
        if front_center_error > self.config.axisymmetric_center_tolerance_ratio:
            return None
        if right_center_error > self.config.axisymmetric_center_tolerance_ratio:
            return None

        right_by_z = {round(z, 6): radius for z, radius, _ in right_profile}
        paired = [
            (z, front_radius, right_by_z[round(z, 6)])
            for z, front_radius, _ in front_profile
            if round(z, 6) in right_by_z
        ]
        if len(paired) < max(8, len(z_samples) // 2):
            return None

        profile_error = np.mean(
            [abs(front_radius - right_radius) for _, front_radius, right_radius in paired]
        ) / max(diameter, 1e-6)
        if profile_error > self.config.axisymmetric_profile_tolerance_ratio:
            return None

        combined_profile = [
            (radius, z)
            for z, front_radius, right_radius in paired
            for radius in [(front_radius + right_radius) / 2.0]
        ]
        simplified = simplify_radius_profile(combined_profile)
        if len(simplified) < 4:
            return None

        bottom = simplified[0][1]
        top_z = simplified[-1][1]
        profile_polygon = [(0.0, bottom), *simplified, (0.0, top_z)]
        return profile_polygon

    @staticmethod
    def _sample_axisymmetric_profile(
        points: list[tuple[float, float]],
        center: float,
        z_samples: np.ndarray,
    ) -> list[tuple[float, float, float]]:
        profile: list[tuple[float, float, float]] = []
        for z in z_samples:
            span = horizontal_span(points, float(z))
            if span is None:
                continue
            min_x, max_x = span
            midpoint = (min_x + max_x) / 2.0
            radius = max(max_x - center, center - min_x)
            if radius <= 0:
                continue
            profile.append((float(z), float(radius), abs(midpoint - center)))
        return profile

    def _infer_hidden_cylinders(
        self,
        triplet: OrthographicTriplet,
        consensus_extents: dict[str, float],
    ) -> list[CylindricalCut]:
        if not self.config.hidden_feature_enabled:
            return []

        front_suffix, right_suffix, top_suffix = self.view_suffixes
        top_view = triplet.views[top_suffix]
        front_view = triplet.views[front_suffix]
        right_view = triplet.views[right_suffix]

        circular_features = self._extract_top_view_circles(top_view, consensus_extents)
        if not circular_features:
            return []

        front_segments = self._extract_horizontal_support_segments(
            front_view,
            front_suffix,
            (consensus_extents["x"], consensus_extents["z"]),
        )
        right_segments = self._extract_horizontal_support_segments(
            right_view,
            right_suffix,
            (consensus_extents["y"], consensus_extents["z"]),
        )

        cuts: list[CylindricalCut] = []
        for candidate in circular_features:
            z_supports = [
                segment.z
                for segment in front_segments
                if abs(segment.center - candidate.center_x)
                <= self.config.hidden_feature_match_center_tolerance
                and abs(segment.half_length - candidate.radius)
                <= self.config.hidden_feature_match_radius_tolerance
            ]
            z_supports.extend(
                segment.z
                for segment in right_segments
                if abs(segment.center - candidate.center_y)
                <= self.config.hidden_feature_match_center_tolerance
                and abs(segment.half_length - candidate.radius)
                <= self.config.hidden_feature_match_radius_tolerance
            )

            if not z_supports:
                continue

            clustered_supports = cluster_values(
                sorted(z_supports),
                self.config.hidden_feature_z_cluster_tolerance,
            )
            support_levels = [sum(group) / len(group) for group in clustered_supports]

            if len(support_levels) >= 2:
                z_min = min(support_levels)
                z_max = max(support_levels)
            else:
                z_cap = support_levels[0]
                z_min, z_max = (
                    (z_cap, consensus_extents["z"])
                    if consensus_extents["z"] - z_cap < z_cap
                    else (0.0, z_cap)
                )

            if z_max - z_min < self.config.hidden_feature_min_depth:
                continue

            cuts.append(
                CylindricalCut(
                    center_x=candidate.center_x,
                    center_y=candidate.center_y,
                    radius=candidate.radius,
                    z_min=float(z_min),
                    z_max=float(z_max),
                )
            )

        return dedupe_cylindrical_cuts(cuts)

    def _extract_top_view_circles(
        self,
        view: SvgOrthographicView,
        consensus_extents: dict[str, float],
    ) -> list[CircularFeatureCandidate]:
        points_x_extent = consensus_extents["x"]
        points_y_extent = consensus_extents["y"]

        candidates: list[CircularFeatureCandidate] = []
        for polyline in view.polylines:
            if polyline.stroke != "red":
                continue

            points = self._scaled_polyline_points(
                view=view,
                suffix=view.suffix,
                polyline=polyline,
                target_extents=(points_x_extent, points_y_extent),
            )
            if len(points) < self.config.hidden_feature_circle_min_points:
                continue

            arr = np.array(points)
            bbox_width = float(np.ptp(arr[:, 0]))
            bbox_height = float(np.ptp(arr[:, 1]))
            if min(bbox_width, bbox_height) <= 0:
                continue
            if (
                max(bbox_width, bbox_height) / min(bbox_width, bbox_height)
                > self.config.hidden_feature_circle_aspect_ratio_max
            ):
                continue

            center_x, center_y, radius, fit_error = fit_circle(arr)
            if radius < self.config.hidden_feature_circle_min_radius:
                continue
            if fit_error > self.config.hidden_feature_circle_fit_error_max:
                continue

            path_length = polyline_path_length(points)
            coverage = path_length / (2 * math.pi * radius) if radius else 0.0
            if coverage < self.config.hidden_feature_circle_min_coverage:
                continue

            candidates.append(
                CircularFeatureCandidate(
                    center_x=float(center_x),
                    center_y=float(center_y),
                    radius=float(radius),
                )
            )

        return dedupe_circular_candidates(candidates)

    def _extract_horizontal_support_segments(
        self,
        view: SvgOrthographicView,
        suffix: str,
        target_extents: tuple[float, float],
    ) -> list[HorizontalSupportSegment]:
        raw_segments: list[tuple[float, float, float]] = []
        for polyline in view.polylines:
            if polyline.stroke != "red":
                continue

            points = self._scaled_polyline_points(
                view=view,
                suffix=suffix,
                polyline=polyline,
                target_extents=target_extents,
            )
            if len(points) < 2 or not polyline_is_axis_aligned(points):
                continue

            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            if max(ys) - min(ys) < 0.05 and max(xs) - min(xs) > 0.3:
                raw_segments.append((ys[0], min(xs), max(xs)))

        raw_segments.sort()
        merged: list[tuple[float, float, float]] = []
        for z_level, min_x, max_x in raw_segments:
            if (
                merged
                and abs(merged[-1][0] - z_level) <= self.config.hidden_feature_segment_merge_tolerance
                and min_x <= merged[-1][2] + self.config.hidden_feature_segment_join_gap
            ):
                previous_z, previous_min_x, previous_max_x = merged[-1]
                merged[-1] = (
                    previous_z,
                    min(previous_min_x, min_x),
                    max(previous_max_x, max_x),
                )
            else:
                merged.append((z_level, min_x, max_x))

        return [
            HorizontalSupportSegment(
                center=(min_x + max_x) / 2.0,
                half_length=(max_x - min_x) / 2.0,
                z=z_level,
            )
            for z_level, min_x, max_x in merged
        ]

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        labels, count = ndimage.label(mask)
        if count <= 1:
            return mask

        component_sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
        largest_label = int(np.argmax(component_sizes)) + 1
        return labels == largest_label

    @staticmethod
    def _contour_area_rows_cols(contour: np.ndarray) -> float:
        points = [(float(col), float(row)) for row, col in contour]
        return polygon_area(points)

    @staticmethod
    def _scale_points(
        points: list[tuple[float, float]],
        source_extents: tuple[float, float],
        target_extents: tuple[float, float],
    ) -> list[tuple[float, float]]:
        src_x, src_y = source_extents
        dst_x, dst_y = target_extents
        scale_x = dst_x / src_x if src_x else 1.0
        scale_y = dst_y / src_y if src_y else 1.0
        return [
            (round(x * scale_x, 6), round(y * scale_y, 6))
            for x, y in points
        ]

    def _scaled_polyline_points(
        self,
        view: SvgOrthographicView,
        suffix: str,
        polyline: SvgPolyline,
        target_extents: tuple[float, float],
    ) -> list[tuple[float, float]]:
        min_x, min_y, width, height = view.view_box
        target_x, target_y = target_extents
        scale_x = target_x / width if width else 1.0
        scale_y = target_y / height if height else 1.0
        return [
            (
                (x - min_x) * scale_x,
                (y - min_y) * scale_y,
            )
            for x, y in polyline.points
        ]

    @staticmethod
    def _build_code(
        front_points: list[tuple[float, float]],
        top_points: list[tuple[float, float]],
        right_points: list[tuple[float, float]],
        consensus_extents: dict[str, float],
        hidden_cylinders: list[CylindricalCut],
    ) -> str:
        front_points_text = format_points("front_pts", front_points)
        top_points_text = format_points("top_pts", top_points)
        right_points_text = format_points("right_pts", right_points)
        hidden_cylinders_text = format_hidden_cylinders(hidden_cylinders)

        lines = [
            "from build123d import *",
            "",
            "# Deterministic visual-hull reconstruction from orthographic silhouettes.",
            f"size_x = {consensus_extents['x']:.6f}",
            f"size_y = {consensus_extents['y']:.6f}",
            f"size_z = {consensus_extents['z']:.6f}",
            "",
            front_points_text,
            "",
            top_points_text,
            "",
            right_points_text,
            "",
            hidden_cylinders_text,
            "",
            "def prism_from_profile(points, plane, amount):",
            "    with BuildSketch(plane) as sketch:",
            "        Polygon(*points)",
            "    return extrude(sketch.sketch.face(), amount=amount)",
            "",
            "def map_bbox_value(value, source_max, bb_min, bb_size):",
            "    return bb_min + (value / source_max) * bb_size if source_max else bb_min",
            "",
            "front = Pos(size_x / 2.0, size_y, size_z / 2.0) * prism_from_profile(",
            "    front_pts, Plane.XZ, amount=size_y",
            ")",
            "top = Pos(size_x / 2.0, size_y / 2.0, 0.0) * prism_from_profile(",
            "    top_pts, Plane.XY, amount=size_z",
            ")",
            "right = Pos(0.0, size_y / 2.0, size_z / 2.0) * prism_from_profile(",
            "    right_pts, Plane.YZ, amount=size_x",
            ")",
            "",
            "part = (front & top) & right",
        ]

        if hidden_cylinders:
            lines.extend([
                "",
                "# Conservative hidden-feature carving from corroborated red-line circles.",
                "_bb = part.bounding_box()",
                "_bbox_size_x = _bb.max.X - _bb.min.X",
                "_bbox_size_y = _bb.max.Y - _bb.min.Y",
                "_bbox_size_z = _bb.max.Z - _bb.min.Z",
                "_xy_scale = ((_bbox_size_x / size_x) + (_bbox_size_y / size_y)) / 2.0",
                "for _cut in hidden_cylinders:",
                "    _center_x = map_bbox_value(_cut['center_x'], size_x, _bb.min.X, _bbox_size_x)",
                "    _center_y = map_bbox_value(_cut['center_y'], size_y, _bb.min.Y, _bbox_size_y)",
                "    _z_min = map_bbox_value(_cut['z_min'], size_z, _bb.min.Z, _bbox_size_z)",
                "    _z_max = map_bbox_value(_cut['z_max'], size_z, _bb.min.Z, _bbox_size_z)",
                "    _radius = _cut['radius'] * _xy_scale",
                "    if _z_max - _z_min <= 0 or _radius <= 0:",
                "        continue",
                "    _cutter = Pos(_center_x, _center_y, _z_min) * Cylinder(",
                "        radius=_radius,",
                "        height=_z_max - _z_min,",
                "        align=(Align.CENTER, Align.CENTER, Align.MIN),",
                "    )",
                "    part = part - _cutter",
            ])

        return "\n".join(lines)

    @staticmethod
    def _build_revolve_code(
        profile_points: list[tuple[float, float]],
        consensus_extents: dict[str, float],
        hidden_cylinders: list[CylindricalCut],
    ) -> str:
        profile_points_text = format_points("profile_pts", profile_points)
        hidden_cylinders_text = format_hidden_cylinders(hidden_cylinders)

        lines = [
            "from build123d import *",
            "",
            "# Deterministic axisymmetric reconstruction from orthographic silhouettes.",
            f"size_x = {consensus_extents['x']:.6f}",
            f"size_y = {consensus_extents['y']:.6f}",
            f"size_z = {consensus_extents['z']:.6f}",
            "",
            profile_points_text,
            "",
            hidden_cylinders_text,
            "",
            "with BuildLine(Plane.XZ) as profile_line:",
            "    Polyline(*profile_pts, close=True)",
            "profile_face = make_face(profile_line.line)",
            "part = Pos(size_x / 2.0, size_y / 2.0, 0.0) * revolve(",
            "    profile_face,",
            "    axis=Axis.Z,",
            ")",
        ]

        if hidden_cylinders:
            lines.extend([
                "",
                "# Conservative hidden-feature carving from corroborated red-line circles.",
                "_bb = part.bounding_box()",
                "_bbox_size_x = _bb.max.X - _bb.min.X",
                "_bbox_size_y = _bb.max.Y - _bb.min.Y",
                "_bbox_size_z = _bb.max.Z - _bb.min.Z",
                "_xy_scale = ((_bbox_size_x / size_x) + (_bbox_size_y / size_y)) / 2.0",
                "for _cut in hidden_cylinders:",
                "    _center_x = map_bbox_value(_cut['center_x'], size_x, _bb.min.X, _bbox_size_x)",
                "    _center_y = map_bbox_value(_cut['center_y'], size_y, _bb.min.Y, _bbox_size_y)",
                "    _z_min = map_bbox_value(_cut['z_min'], size_z, _bb.min.Z, _bbox_size_z)",
                "    _z_max = map_bbox_value(_cut['z_max'], size_z, _bb.min.Z, _bbox_size_z)",
                "    _radius = _cut['radius'] * _xy_scale",
                "    if _z_max - _z_min <= 0 or _radius <= 0:",
                "        continue",
                "    _cutter = Pos(_center_x, _center_y, _z_min) * Cylinder(",
                "        radius=_radius,",
                "        height=_z_max - _z_min,",
                "        align=(Align.CENTER, Align.CENTER, Align.MIN),",
                "    )",
                "    part = part - _cutter",
            ])

        return "\n".join([
            *lines[:8],
            "def map_bbox_value(value, source_max, bb_min, bb_size):",
            "    return bb_min + (value / source_max) * bb_size if source_max else bb_min",
            "",
            *lines[8:],
        ])


def format_points(name: str, points: list[tuple[float, float]]) -> str:
    """Render a stable Python point-list literal for generated code."""
    lines = [f"{name} = ["]
    for x, y in points:
        lines.append(f"    ({x:.6f}, {y:.6f}),")
    lines.append("]")
    return "\n".join(lines)


def format_hidden_cylinders(hidden_cylinders: list[CylindricalCut]) -> str:
    """Render hidden cylinder metadata into generated Python code."""
    lines = ["hidden_cylinders = ["]
    for cut in hidden_cylinders:
        lines.append(
            "    {"
            f"'center_x': {cut.center_x:.6f}, "
            f"'center_y': {cut.center_y:.6f}, "
            f"'radius': {cut.radius:.6f}, "
            f"'z_min': {cut.z_min:.6f}, "
            f"'z_max': {cut.z_max:.6f}"
            "},"
        )
    lines.append("]")
    return "\n".join(lines)


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Compute polygon area via the shoelace formula."""
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def contour_circularity(points: list[tuple[float, float]]) -> float:
    """Return the isoperimetric circularity of a closed contour."""
    perimeter = sum(
        math.dist(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )
    if perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * polygon_area(points) / (perimeter * perimeter))


def horizontal_span(
    points: list[tuple[float, float]],
    y_value: float,
    tolerance: float = 1e-6,
) -> tuple[float, float] | None:
    """Find the horizontal span of a closed contour at a given y value."""
    intersections: list[float] = []
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        if abs(y2 - y1) <= tolerance:
            if abs(y_value - y1) <= tolerance:
                intersections.extend([x1, x2])
            continue

        lower = min(y1, y2)
        upper = max(y1, y2)
        if y_value < lower - tolerance or y_value > upper + tolerance:
            continue

        t = (y_value - y1) / (y2 - y1)
        if -tolerance <= t <= 1.0 + tolerance:
            intersections.append(x1 + (x2 - x1) * t)

    unique = sorted(dedupe_numeric(intersections, tolerance=1e-5))
    if len(unique) < 2:
        return None
    return unique[0], unique[-1]


def polyline_path_length(points: list[tuple[float, float]]) -> float:
    """Compute cumulative polyline length."""
    return sum(
        math.dist(points[index], points[index + 1])
        for index in range(len(points) - 1)
    )


def fit_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares circle fit returning center, radius, and mean radial error."""
    system = np.column_stack([2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))])
    target = np.sum(points * points, axis=1)
    center_x, center_y, offset = np.linalg.lstsq(system, target, rcond=None)[0]
    radius = math.sqrt(max(offset + center_x * center_x + center_y * center_y, 0.0))
    radial_error = np.mean(
        np.abs(np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2) - radius)
    )
    return float(center_x), float(center_y), float(radius), float(radial_error)


def polyline_is_axis_aligned(points: list[tuple[float, float]]) -> bool:
    """Return True when all segments are horizontal or vertical."""
    return all(
        abs(points[index + 1][0] - points[index][0]) < 1e-6
        or abs(points[index + 1][1] - points[index][1]) < 1e-6
        for index in range(len(points) - 1)
    )


def cluster_values(values: list[float], tolerance: float) -> list[list[float]]:
    """Cluster sorted numeric values by adjacency tolerance."""
    clusters: list[list[float]] = []
    for value in values:
        if clusters and abs(clusters[-1][-1] - value) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def dedupe_circular_candidates(
    candidates: list[CircularFeatureCandidate],
) -> list[CircularFeatureCandidate]:
    """Remove near-duplicate circular candidates."""
    deduped: list[CircularFeatureCandidate] = []
    for candidate in sorted(candidates, key=lambda item: -item.radius):
        if any(
            math.dist(
                (candidate.center_x, candidate.center_y),
                (existing.center_x, existing.center_y),
            ) < 0.2
            and abs(candidate.radius - existing.radius) < 0.2
            for existing in deduped
        ):
            continue
        deduped.append(candidate)
    return deduped


def dedupe_cylindrical_cuts(cuts: list[CylindricalCut]) -> list[CylindricalCut]:
    """Remove near-duplicate cylindrical cuts."""
    deduped: list[CylindricalCut] = []
    for cut in cuts:
        if any(
            math.dist((cut.center_x, cut.center_y), (existing.center_x, existing.center_y)) < 0.2
            and abs(cut.radius - existing.radius) < 0.2
            and abs(cut.z_min - existing.z_min) < 0.5
            and abs(cut.z_max - existing.z_max) < 0.5
            for existing in deduped
        ):
            continue
        deduped.append(cut)
    return deduped


def dedupe_numeric(values: list[float], tolerance: float) -> list[float]:
    """Remove near-duplicate numeric values from a sorted sequence."""
    deduped: list[float] = []
    for value in sorted(values):
        if deduped and abs(deduped[-1] - value) <= tolerance:
            continue
        deduped.append(value)
    return deduped


def simplify_radius_profile(
    profile: list[tuple[float, float]],
    tolerance: float = 1e-3,
) -> list[tuple[float, float]]:
    """Drop near-collinear samples from a radius-vs-height profile."""
    if len(profile) <= 2:
        return profile

    simplified = [profile[0]]
    for index in range(1, len(profile) - 1):
        previous = simplified[-1]
        current = profile[index]
        next_point = profile[index + 1]
        area = abs(
            (current[0] - previous[0]) * (next_point[1] - previous[1])
            - (next_point[0] - previous[0]) * (current[1] - previous[1])
        )
        if area <= tolerance:
            continue
        simplified.append(current)
    simplified.append(profile[-1])
    return simplified
