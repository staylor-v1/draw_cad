from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 76.950000
size_y = 54.302832
size_z = 6.696029

front_pts = [
    (76.950000, 0.000000),
    (0.150000, 0.000000),
    (0.000000, 6.546118),
    (76.950000, 6.696029),
]

top_pts = [
    (76.950000, 0.000000),
    (0.150000, 0.000000),
    (0.000000, 54.153100),
    (76.950000, 54.302832),
]

right_pts = [
    (54.302832, 0.000000),
    (0.105853, 0.000000),
    (0.000000, 6.590302),
    (54.302832, 6.696029),
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