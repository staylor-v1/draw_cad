from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 102.600000
size_y = 72.670703
size_z = 9.011198

front_pts = [
    (102.600000, 0.000000),
    (0.200000, 0.000000),
    (0.000000, 8.812421),
    (102.600000, 9.011198),
]

top_pts = [
    (102.600000, 0.000000),
    (0.200000, 0.000000),
    (0.000000, 72.471058),
    (102.600000, 72.670703),
]

right_pts = [
    (72.670703, 0.000000),
    (0.141658, 0.000000),
    (0.000000, 8.868916),
    (72.670703, 9.011198),
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