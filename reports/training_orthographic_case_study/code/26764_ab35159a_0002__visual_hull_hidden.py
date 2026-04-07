from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 871.298927
size_y = 767.881281
size_z = 20.155385

front_pts = [
    (871.298927, 0.000000),
    (1.698438, 0.000000),
    (0.000000, 18.475770),
    (871.298927, 20.155385),
]

top_pts = [
    (822.140038, 0.000000),
    (48.028799, 0.000000),
    (26.557101, 6.775423),
    (7.910626, 24.278599),
    (0.000000, 42.346394),
    (3.390268, 73.965035),
    (387.620664, 740.214970),
    (411.917586, 763.364332),
    (424.348569, 767.881281),
    (445.820268, 767.881281),
    (463.901698, 759.976621),
    (482.548173, 740.214970),
    (864.518390, 77.352747),
    (871.298927, 53.638766),
    (866.778569, 32.183260),
    (851.522362, 12.421609),
]

right_pts = [
    (767.881281, 0.000000),
    (1.498792, 0.000000),
    (0.000000, 18.643732),
    (767.881281, 20.155385),
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