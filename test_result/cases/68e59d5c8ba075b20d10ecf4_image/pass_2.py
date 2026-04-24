from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 137.668359
size_y = 106.328125
size_z = 13.175521

front_pts = [
    (137.668359, 0.000000),
    (0.268359, 0.000000),
    (0.000000, 12.908449),
    (137.668359, 13.175521),
]

top_pts = [
    (137.668359, 0.000000),
    (0.268359, 0.000000),
    (0.000000, 106.060071),
    (137.668359, 106.328125),
]

right_pts = [
    (106.328125, 0.000000),
    (0.207267, 0.000000),
    (0.000000, 12.967486),
    (106.328125, 13.175521),
]

hidden_cylinders = [
]

def prism_from_profile(points, plane, amount):
    with BuildSketch(plane) as sketch:
        Polygon(*points)
    return extrude(sketch.sketch.face(), amount=amount)

def map_bbox_value(value, source_max, bb_min, bb_size):
    return bb_min + (value / source_max) * bb_size if source_max else bb_min

front = Pos(size_x / 2.0, size_y, size_z / 2.0) * prism_from_profile(
    front_pts, Plane.XZ, amount=size_y
)
top = Pos(size_x / 2.0, size_y / 2.0, 0.0) * prism_from_profile(
    top_pts, Plane.XY, amount=size_z
)
right = Pos(0.0, size_y / 2.0, size_z / 2.0) * prism_from_profile(
    right_pts, Plane.YZ, amount=size_x
)

part = (front & top) & right