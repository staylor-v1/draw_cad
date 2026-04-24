from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 76.950000
size_y = 51.400000
size_z = 6.366667

front_pts = [
    (76.950000, 0.000000),
    (0.150000, 0.000000),
    (0.000000, 6.217448),
    (76.950000, 6.366667),
]

top_pts = [
    (76.950000, 0.000000),
    (0.150000, 0.000000),
    (0.000000, 51.250291),
    (76.950000, 51.400000),
]

right_pts = [
    (51.400000, 0.000000),
    (0.100195, 0.000000),
    (0.000000, 6.266140),
    (51.400000, 6.366667),
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