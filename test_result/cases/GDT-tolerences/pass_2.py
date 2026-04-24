from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 75.146484
size_y = 49.454004
size_z = 6.125326

front_pts = [
    (75.146484, 0.000000),
    (0.146484, 0.000000),
    (0.000000, 5.979484),
    (75.146484, 6.125326),
]

top_pts = [
    (75.146484, 0.000000),
    (0.146484, 0.000000),
    (0.000000, 49.307690),
    (75.146484, 49.454004),
]

right_pts = [
    (49.454004, 0.000000),
    (0.096402, 0.000000),
    (0.000000, 6.028610),
    (49.454004, 6.125326),
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