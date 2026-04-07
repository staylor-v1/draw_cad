from build123d import *

# Deterministic profile-extrusion reconstruction from the t view.
size_x = 871.298927
size_y = 767.881281
size_z = 20.155385

outer_pts = [
    (822.140038, 0.000000),
    (48.028800, 0.000000),
    (26.557101, 6.775423),
    (7.910626, 24.278599),
    (0.000000, 42.346394),
    (3.390268, 73.965035),
    (387.620664, 740.214971),
    (411.917587, 763.364332),
    (424.348570, 767.881281),
    (445.820268, 767.881281),
    (463.901699, 759.976621),
    (482.548173, 740.214971),
    (864.518391, 77.352747),
    (871.298927, 53.638766),
    (866.778570, 32.183259),
    (851.522363, 12.421609),
]

inner_profile_sets = [
    [
        (708.001008, 85.822025),
        (163.297918, 85.822025),
        (147.476667, 93.726686),
        (136.740818, 106.712913),
        (134.480639, 128.168420),
        (142.391264, 145.106978),
        (402.311826, 595.672612),
        (418.698122, 612.046551),
        (429.999017, 615.434262),
        (448.080447, 613.175788),
        (466.726922, 597.931085),
        (732.297931, 137.202318),
        (734.558110, 110.100625),
        (721.562082, 92.597448),
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

outer = Pos(size_x / 2.0, size_y / 2.0, 0.0) * prism_from_profile(
    outer_pts, Plane.XY, amount=20.155385
)
part = outer
for _points in inner_profile_sets:
    _inner = Pos(size_x / 2.0, size_y / 2.0, 0.0) * prism_from_profile(
        _points, Plane.XY, amount=20.155385
    )
    part = part - _inner