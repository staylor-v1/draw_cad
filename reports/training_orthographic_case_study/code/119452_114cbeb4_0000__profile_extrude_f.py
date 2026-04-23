from build123d import *

# Deterministic profile-extrusion reconstruction from the f view.
size_x = 482.812500
size_y = 962.812500
size_z = 482.812500

outer_pts = [
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

inner_profile_sets = [
    [
        (452.441026, 29.432150),
        (28.805934, 30.371474),
        (29.745258, 454.006566),
        (453.380350, 453.067242),
    ],
]

hidden_cylinders = [
]

def prism_from_profile(points, plane, amount):
    with BuildSketch(plane) as sketch:
        Polygon(*points)
    return extrude(sketch.sketch.face(), amount=amount)

def map_bbox_value(value, source_max, bb_min, bb_size):
    return bb_min + (value / source_max) * bb_size if source_max else bb_min

outer = Pos(size_x / 2.0, size_y, size_z / 2.0) * prism_from_profile(
    outer_pts, Plane.XZ, amount=962.812500
)
part = outer
for _points in inner_profile_sets:
    _inner = Pos(size_x / 2.0, size_y, size_z / 2.0) * prism_from_profile(
        _points, Plane.XZ, amount=962.812500
    )
    part = part - _inner