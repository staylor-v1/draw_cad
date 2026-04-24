from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 144.281250
size_y = 102.093164
size_z = 12.671224

front_pts = [
    (144.281250, 0.000000),
    (0.281250, 0.000000),
    (0.000000, 12.391712),
    (144.281250, 12.671224),
]

top_pts = [
    (144.281250, 0.000000),
    (0.281250, 0.000000),
    (0.000000, 101.812174),
    (144.281250, 102.093164),
]

right_pts = [
    (102.093164, 0.000000),
    (0.199012, 0.000000),
    (0.000000, 12.471152),
    (102.093164, 12.671224),
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