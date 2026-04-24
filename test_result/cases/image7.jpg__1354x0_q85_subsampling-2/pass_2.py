from build123d import *

# Deterministic visual-hull reconstruction from orthographic silhouettes.
size_x = 135.664453
size_y = 72.162142
size_z = 8.942643

front_pts = [
    (135.664453, 0.000000),
    (0.264453, 0.000000),
    (0.000000, 8.679624),
    (135.664453, 8.942643),
]

top_pts = [
    (135.664453, 0.000000),
    (0.264453, 0.000000),
    (0.000000, 71.898134),
    (135.664453, 72.162142),
]

right_pts = [
    (72.162142, 0.000000),
    (0.140667, 0.000000),
    (0.000000, 8.801444),
    (72.162142, 8.942643),
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