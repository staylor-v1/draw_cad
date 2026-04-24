from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 85.366406
size_y = 60.178125
size_z = 7.427344

front_pts = [
    (85.366406, 0.000000),
    (0.166406, 0.000000),
    (0.000000, 7.261060),
    (85.366406, 7.427344),
]

top_pts = [
    (85.366406, 0.000000),
    (0.166406, 0.000000),
    (0.000000, 60.011887),
    (85.366406, 60.178125),
]

right_pts = [
    (60.178125, 0.000000),
    (0.117306, 0.000000),
    (0.000000, 7.310070),
    (60.178125, 7.427344),
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