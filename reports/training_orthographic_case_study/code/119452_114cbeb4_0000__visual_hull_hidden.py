from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 482.812500
size_y = 962.812500
size_z = 482.812500

front_pts = [
    (467.470209, 0.000000),
    (15.342291, 0.000000),
    (8.453915, 1.878648),
    (2.504864, 7.201483),
    (0.000000, 12.837427),
    (0.000000, 470.601289),
    (6.575268, 480.307636),
    (12.211211, 482.812500),
    (469.975073, 482.812500),
    (479.681420, 476.237232),
    (482.812500, 467.470209),
    (482.812500, 15.968507),
    (479.055204, 6.575268),
    (473.106153, 1.252432),
]

top_pts = [
    (480.941134, 0.000000),
    (0.000000, 1.873176),
    (1.871366, 962.812500),
    (482.812500, 960.939324),
]

right_pts = [
    (962.812500, 0.000000),
    (1.876827, 0.000000),
    (0.000000, 480.941134),
    (962.812500, 482.812500),
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