import numpy as np

from macromol_dataframe import (
        Coord, Frame, make_coord_frame_from_rotation_vector,
)
from math import pi

def sample_frame(
        rng: np.random.Generator,
        origin: Coord,
) -> Frame:
    angle_rad = 2 * pi * rng.uniform()
    rot_vec_rad = sample_uniform_unit_vector(rng) * angle_rad
    return make_coord_frame_from_rotation_vector(origin, rot_vec_rad)

def sample_uniform_unit_vector(rng: np.random.Generator) -> Coord:
    # https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe

    # I chose the rejection sampling approach rather than the Gaussian approach 
    # because (i) I'd need the while loop either way to check for a null vector 
    # and (ii) I understand why it works.  The Gaussian approach would be ≈2x 
    # faster, though.

    while True:
        v = rng.uniform(-1, 1, size=3)
        m = np.linalg.norm(v)
        if 0 < m < 1:
            return v / m

def sample_coord_from_cube(
        rng: np.random.Generator,
        center: Coord,
        size: float,
) -> Coord:
    s2 = size / 2
    return center + rng.uniform(-s2, s2, size=3)

