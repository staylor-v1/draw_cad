from build123d import *

# Deterministic profile-extrusion reconstruction from the f view.
size_x = 993.216935
size_y = 950.271469
size_z = 54.223891

outer_pts = [
    (993.216935, 0.000000),
    (1.936096, 0.000000),
    (0.000000, 52.287324),
    (993.216935, 54.223891),
]

inner_profile_sets = [
    [
        (294.983369, 52.287324),
        (1.936096, 52.287324),
        (1.936096, 1.440012),
        (294.983369, 1.440012),
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
    outer_pts, Plane.XZ, amount=950.271469
)
part = outer
for _points in inner_profile_sets:
    _inner = Pos(size_x / 2.0, size_y, size_z / 2.0) * prism_from_profile(
        _points, Plane.XZ, amount=950.271469
    )
    part = part - _inner