import numpy as np
import sqlite3

from .images import image_from_atoms, ImageParams
from .random import sample_frame, sample_coord_from_cube
from .database_io import (
        select_cached_metadatum, select_zone_center_A, select_zone_atoms,
)
from macromol_dataframe import transform_atom_coords

from typing import TypeAlias, Callable, Any

MakeSampleFunc: TypeAlias = Callable[
        [sqlite3.Connection, dict[str, Any], np.random.Generator, int],
        Any,
]

def zone_id_from_index(i, zone_ids):
    # If *i* continues to increment between epochs, then we will sample 
    # different rotations/translations in each epoch.  Otherwise, we won't.
    zone_id = zone_ids[i % len(zone_ids)]
    rng = np.random.default_rng(i)
    return zone_id, rng

def make_unprocessed_sample(db, db_cache, rng, zone_id):
    return dict(rng=rng, zone_id=zone_id)

def make_unsupervised_sample(db, db_cache, rng, zone_id):
    atoms_i = select_zone_atoms(db, zone_id)
    zone_center_A = select_zone_center_A(db, zone_id)
    zone_size_A = select_cached_metadatum(db, db_cache, 'zone_size_A')

    origin_i = sample_coord_from_cube(rng, zone_center_A, zone_size_A)
    frame_ia = sample_frame(rng, origin_i)

    return dict(
            rng=rng,
            zone_id=zone_id,
            atoms_i=atoms_i,
            frame_ia=frame_ia,
    )

def make_unsupervised_image_sample(
        db, db_cache, rng, zone_id,
        *,
        img_params: ImageParams,
):
    x = make_unsupervised_sample(db, db_cache, rng, zone_id)

    atoms_a = transform_atom_coords(x['atoms_i'], x['frame_ia'])
    img, img_atoms_a = image_from_atoms(atoms_a, img_params)

    return dict(
            **x,
            atoms_a=atoms_a,
            image_atoms_a=img_atoms_a,
            image=img,
    )

