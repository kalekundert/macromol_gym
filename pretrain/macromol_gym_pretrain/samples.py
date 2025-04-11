from .database_io import select_zone_atoms
from .neighbors import NeighborParams, get_neighboring_frames
from macromol_gym_unsupervised import (
        MakeSampleArgs, ImageParams, image_from_atoms,
)
from macromol_dataframe import transform_atom_coords

def make_neighbor_sample(
        sample: MakeSampleArgs,
        *,
        neighbor_params: NeighborParams,
):
    frame_ia, frame_ab, b = get_neighboring_frames(
            db=sample.db, db_cache=sample.db_cache,
            rng=sample.rng,
            zone_id=sample.zone_id,
            neighbor_params=neighbor_params,
    )
    atoms_i = select_zone_atoms(sample.db, sample.zone_id)
    atoms_a = transform_atom_coords(atoms_i, frame_ia)
    atoms_b = transform_atom_coords(atoms_a, frame_ab)

    return dict(
            i=sample.i,
            zone_id=sample.zone_id,
            rng=sample.rng,
            frame_ia=frame_ia,
            frame_ab=frame_ab,
            atoms_i=atoms_i,
            atoms_a=atoms_a,
            atoms_b=atoms_b,
            b=b,
    )

def make_neighbor_image_sample(
        sample: MakeSampleArgs,
        *,
        img_params: ImageParams,
        neighbor_params: NeighborParams,
):
    x = make_neighbor_sample(
            sample,
            neighbor_params=neighbor_params,
    )

    img_a, img_atoms_a = image_from_atoms(x["atoms_a"], img_params)
    img_b, img_atoms_b = image_from_atoms(x["atoms_b"], img_params)

    return dict(
            **x,
            img_a=img_a,
            img_b=img_b,
            img_atoms_a=img_atoms_a,
            img_atoms_b=img_atoms_b,
    )

