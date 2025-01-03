from .database_io import select_zone_atoms
from .neighbors import NeighborParams, get_neighboring_frames
from macromol_gym_unsupervised import ImageParams, image_from_atoms
from macromol_dataframe import transform_atom_coords

def make_neighbor_sample(
        db, db_cache, rng, zone_id,
        *,
        neighbor_params: NeighborParams,
):
    frame_ia, frame_ab, b = get_neighboring_frames(
            db=db, db_cache=db_cache,
            rng=rng,
            zone_id=zone_id,
            neighbor_params=neighbor_params,
    )
    atoms_i = select_zone_atoms(db, zone_id)
    atoms_a = transform_atom_coords(atoms_i, frame_ia)
    atoms_b = transform_atom_coords(atoms_a, frame_ab)

    return dict(
            rng=rng,
            zone_id=zone_id,
            frame_ia=frame_ia,
            frame_ab=frame_ab,
            atoms_i=atoms_i,
            atoms_a=atoms_a,
            atoms_b=atoms_b,
            b=b,
    )

def make_neighbor_image_sample(
        db, db_cache, rng, zone_id,
        *,
        img_params: ImageParams,
        neighbor_params: NeighborParams,
):
    x = make_neighbor_sample(
            db, db_cache, rng, zone_id,
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

