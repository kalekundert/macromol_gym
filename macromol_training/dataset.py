import torch
import macromol_voxelize as mmvox
import polars as pl
import numpy as np

from .database_io import (
        open_db, select_metadatum, select_split, select_neighbors,
        select_zone_center_A, select_zone_neighbors, select_zone_atoms,
)
from macromol_dataframe import (
        Atoms, Coords, Coord, Frame, transform_atom_coords,
        make_coord_frame, make_coord_frame_from_rotation_vector,
)
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from dataclasses import dataclass
from reprfunc import repr_from_init
from itertools import product
from pathlib import Path
from math import radians

from typing import Any, Callable, Optional
from numpy.typing import NDArray

@dataclass
class NeighborParams:
    direction_candidates: Coords
    distance_A: float
    noise_max_distance_A: float
    noise_max_angle_deg: float

@dataclass
class ImageParams:
    grid: mmvox.Grid
    atom_radius_A: Optional[float]
    element_channels: list[str]
    ligand_channel: bool


class NeighborDataset(Dataset):
    
    def __init__(
            self,
            *,
            db_path: Path,
            split: str,
            neighbor_params: NeighborParams,
            input_from_atoms: Callable[[Atoms], Any],
    ):
        # Don't store a connection to the database in the constructor.  This 
        # method is called in the parent process, after which the instantiated 
        # dataset object is sent to the worker process.  If the worker process 
        # was forked, this would cause weird deadlock/race condition problems!
        # If the worker process was spawned, this would require pickling the 
        # connection, which isn't possible.
        self.db_path = db_path
        self.db = None

        db = open_db(db_path)
        self.zone_ids = select_split(db, split)

        self.neighbor_params = neighbor_params
        self.input_from_atoms = input_from_atoms

    def __len__(self):
        return len(self.zone_ids)

    def __getitem__(self, i):
        if self.db is None:
            self.db = open_db(self.db_path)

        zone_id, frame_ia, frame_ab, b = get_neighboring_frames(
                self.db, i,
                zone_ids=self.zone_ids,
                neighbor_params=self.neighbor_params,
                db_cache=self.db_cache,
        )

        atoms_i = select_zone_atoms(self.db, zone_id)
        atoms_a = transform_atom_coords(atoms_i, frame_ia)
        atoms_b = transform_atom_coords(atoms_a, frame_ab)

        input_a = self.input_from_atoms(atoms_a)
        input_b = self.input_from_atoms(atoms_b)
        input_ab = np.stack([input_a, input_b])

        return torch.from_numpy(input_ab).float(), torch.tensor(b)

class CnnNeighborDataset(NeighborDataset):

    def __init__(
            self,
            db_path: Path,
            split: str,
            neighbor_params: NeighborParams,
            img_params: ImageParams,
    ):
        # This class is slightly opinionated about how images should be 
        # created.  This allows it to provide a simple---but not fully 
        # general---API for common image parameters.  If you need to do 
        # something beyond the scope of this API, use `NeighborDataset` 
        # directly.

        def input_from_atoms(atoms):
            atoms = mmvox.set_atom_radius_A(atoms, img_params.atom_radius_A)
            mmvox_img_params = mmvox.ImageParams(
                    channels=(
                        len(img_params.element_channels)
                        + img_params.ligand_channel
                    ),
                    grid=img_params.grid,
                    assign_channels=assign_channels,
            )
            return mmvox.image_from_atoms(atoms, mmvox_img_params)

        def assign_channels(atoms):
            channels = img_params.element_channels
            atoms = mmvox.set_atom_channels_by_element(atoms, channels)

            if img_params.ligand_channel:
                atoms = add_ligand_channel(atoms, len(channels))

            return atoms

        super().__init__(
                db_path=db_path,
                split=split,
                neighbor_params=neighbor_params,
                input_from_atoms=input_from_atoms,
        )

class InfiniteSampler:
    """
    Yield indices that continue incrementing between epochs.

    This sampler is meant to be used with an infinite map-style dataset, i.e. a 
    dataset that accepts integer indices of any size.  The sampler accepts an 
    "epoch size" parameter, but this really just determines how often "end of 
    epoch" tasks, like running the validation set or saving checkpoints, are 
    performed.  It doesn't have to relate to the number of training examples, 
    which may be infinite.
    """

    def __init__(
            self,
            epoch_size: int,
            *,
            start_index: int = 0,
            curr_epoch: int = 0,
            shuffle: bool = False,
            rng_factory: Callable[[int], np.random.Generator] = np.random.default_rng,
    ):
        self.start_index = start_index
        self.epoch_size = epoch_size
        self.curr_epoch = curr_epoch
        self.shuffle = shuffle
        self.rng_factory = rng_factory

    def __iter__(self):
        n = self.epoch_size
        i = self.start_index + n * self.curr_epoch
        indices = range(i, i+n)

        if self.shuffle:
            rng = self.rng_factory(self.curr_epoch)
            indices = rng.permutation(indices)

        yield from indices

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch: int):
        self.curr_epoch = epoch

    __repr__ = repr_from_init

def add_ligand_channel(atoms, channel):
    ligand_channel = (
            pl.when(pl.col('is_polymer'))
            .then([channel])
            .otherwise([])
    )
    return atoms.with_columns(
            channels=pl.col('channels').list.concat(ligand_channel)
    )

def get_neighboring_frames(db, i, zone_ids, neighbor_params, db_cache):
    # Nomenclature
    # ============
    # home:
    #   The "first" region (i.e. the one that will appear in index 0 of the 
    #   input tensor) of macromolecular structure shown to the model.  This 
    #   region is centered somewhere in the zone corresponding to index *i*.
    #   
    # neighbor:
    #   The "second" region of macromolecular structure shown to the model.  
    #   The goal of the model will be to predict which of a small number of 
    #   possible translations relates home to neighbor.
    #
    # Coordinate frames
    # =================
    # i: The frame for any coordinates in the database, include zone centers 
    #    and atomic coordinates.  This function does not use the atomic 
    #    coordinates itself, but it's assumed that the caller will.  Note that 
    #    this frame has nothing to do with the index argument *i*.
    #
    # a: The frame where the home is located at the origin, and the neighbor is 
    #    offset in one of a small number of possible directions.  Because the 
    #    neighbor is also constrained to be near one of the points identified 
    #    in the database as having sufficient density, "a" will usually be 
    #    rotated relative to "i" (such that the chosen direction points towards 
    #    a region of high density).
    #
    # b: The frame where the second region is at the origin and the first 
    #    region is offset in the opposite of the direction mentioned above.  
    #    This is just a translation relative to "a".
    #
    # c: A frame that has been randomly translated and rotated, by a small 
    #    amount, relative to "b".  This is the frame that will actually be used 
    #    to generate the second region.  It's just meant to add a little noise 
    #    and prevent the model from keying in on exact distances.

    # If *i* continues to increment between epochs, then we will sample 
    # different rotations/translations in each epoch.  Otherwise, we won't.

    rng = np.random.default_rng(i)

    zone_id = zone_ids[i % len(zone_ids)]
    zone_size_A = cache_zone_size_A(db, db_cache)
    zone_center_A = select_zone_center_A(db, zone_id)
    zone_neighbor_indices = select_zone_neighbors(db, zone_id)

    home_origin_i = _sample_coord_from_cube(rng, zone_center_A, zone_size_A)

    neighbor_directions_i = cache_neighbor_directions(db, db_cache)
    neighbor_direction_i = _sample_uniform_unit_vector_in_neighborhood(
            rng,
            neighbor_directions_i,
            cache_pairwise_rotation_matrices(db_cache, neighbor_directions_i),
            zone_neighbor_indices,
    )

    neighbor_label = rng.integers(len(neighbor_params.direction_candidates))
    neighbor_direction_a = neighbor_params.direction_candidates[neighbor_label]

    frame_ia, frame_ab = _make_neighboring_frames(
            home_origin_i,
            neighbor_params.distance_A,
            neighbor_direction_i,
            neighbor_direction_a,
    )
    frame_bc = _sample_noise_frame(
            rng,
            neighbor_params.noise_max_distance_A,
            neighbor_params.noise_max_angle_deg,
    )
    frame_ac = frame_bc @ frame_ab

    return zone_id, frame_ia, frame_ac, neighbor_label

def cache_zone_size_A(db, db_cache):
    return _load_from_cache(
            db_cache, 'zone_size_A',
            lambda: select_metadatum(db, 'zone_size_A'),
    )

def cache_neighbor_directions(db, db_cache):
    return _load_from_cache(
            db_cache, 'neighbor_directions',
            lambda: _require_unit_vectors(select_neighbors(db)),
    )

def cache_pairwise_rotation_matrices(db_cache, neighbor_directions):
    return _load_from_cache(
            db_cache, 'pairwise_rotation_matrices',
            lambda: _precalculate_pairwise_rotation_matrices(neighbor_directions)
    )

def _load_from_cache(cache: dict, key: str, value_factory: Callable[[], Any]):
    try:
        return cache[key]
    except KeyError:
        value = cache[key] = value_factory()
        return value

def _make_neighboring_frames(
        home_origin_i: Coord,
        neighbor_distance: float,
        neighbor_direction_i: Coord,
        neighbor_direction_a: Coord,
) -> tuple[Frame, Frame]:
    rotation_ia = _align_vectors(
            neighbor_direction_i,
            neighbor_direction_a,
    )
    frame_ia = make_coord_frame(home_origin_i, rotation_ia)

    neighbor_origin_a = neighbor_direction_a * neighbor_distance
    frame_ab = make_coord_frame(neighbor_origin_a)

    return frame_ia, frame_ab

def _sample_noise_frame(
        rng: np.random.Generator,
        max_distance_A: float,
        max_angle_deg: float,
) -> Frame:
    distance_A = rng.uniform() * max_distance_A
    origin_A = _sample_uniform_unit_vector(rng) * distance_A
    angle_rad = radians(rng.uniform() * max_angle_deg)
    rot_vec_rad = _sample_uniform_unit_vector(rng) * angle_rad
    return make_coord_frame_from_rotation_vector(origin_A, rot_vec_rad)

def _sample_uniform_unit_vector_in_neighborhood(
        rng: np.random.Generator,
        neighbors: Coords,
        pairwise_rotation_matrices: NDArray,
        valid_neighbor_indices: list[int],
) -> Coord:
    x = _sample_uniform_unit_vector(rng)
    d = np.linalg.norm(neighbors - x, axis=1)
    i = np.argmin(d)
    j = rng.choice(valid_neighbor_indices)
    return pairwise_rotation_matrices[i,j] @ x

def _sample_uniform_unit_vector(rng: np.random.Generator) -> Coord:
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

def _sample_coord_from_cube(
        rng: np.random.Generator,
        center: Coord,
        size: float,
) -> Coord:
    s2 = size / 2
    return center + rng.uniform(-s2, s2, size=3)

def _precalculate_pairwise_rotation_matrices(neighbors: Coords) -> NDArray:
    n = len(neighbors)
    R = np.full((n,n,3,3), np.nan)

    for i, j in product(range(n), repeat=2):
        Rij = _align_vectors(neighbors[j], neighbors[i])
        R[i,j] = Rij.as_matrix()

    assert not np.isnan(R).any()
    return R

def _require_unit_vectors(v: Coords) -> Coords:
    return v / np.linalg.norm(v, axis=1)[...,np.newaxis]

def _align_vectors(a, b):
    from math import atan2

    # Copied from `scipy/spatial/transform/_rotation.pyx` to work around 
    # anti-parallel vector bug, see scipy/scipy#20660 for more info.

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.shape != (3,):
        raise ValueError("Expected input `a` to have shape (3,), "
                         "got {}".format(a.shape))
    if b.shape != (3,):
        raise ValueError("Expected input `b` to have shape (3,), "
                         "got {}".format(b.shape))

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        raise ValueError("Cannot align zero length vectors")
    a /= a_norm
    b /= b_norm

    # We first find the minimum angle rotation between the primary
    # vectors.
    cross = np.cross(b, a)
    cross_norm = np.linalg.norm(cross)
    theta = atan2(cross_norm, np.dot(a, b))
    tolerance = 1e-3  # tolerance for small angle approximation (rad)
    R_flip = Rotation.identity()
    if (np.pi - theta) < tolerance:
        # Near pi radians, the Taylor series approximation of x/sin(x)
        # diverges, so for numerical stability we flip pi and then
        # rotate back by the small angle pi - theta
        if cross_norm == 0:
            # For anti-parallel vectors, cross = [0, 0, 0] so we need to
            # manually set an arbitrary orthogonal axis of rotation
            i = np.argmin(np.abs(a))
            r = np.zeros(3)
            r[i - 1], r[i - 2] = a[i - 2], -a[i - 1]
        else:
            r = cross  # Shortest angle orthogonal axis of rotation
        R_flip = Rotation.from_rotvec(r / np.linalg.norm(r) * np.pi)
        theta = np.pi - theta
        cross = -cross
    if abs(theta) < tolerance:
        # Small angle Taylor series approximation for numerical stability
        theta2 = theta * theta
        r = cross * (1 + theta2 / 6 + theta2 * theta2 * 7 / 360)
    else:
        r = cross * theta / np.sin(theta)

    return Rotation.from_rotvec(r) * R_flip

