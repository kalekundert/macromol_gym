import macromol_gym_pretrain as mmgp
import polars as pl
import numpy as np

def make_db(path=':memory:', *, split='train'):
    db = mmgp.open_db(path, mode='rwc')
    with db:
        mmgp.init_db(db)

        zone_size_A = 10
        neighbors_i = mmgp.icosahedron_faces() * 30

        mmgp.upsert_metadata(db, {
            'zone_size_A': zone_size_A,
            'neighbor_count_threshold': 1,
            'polymer_labels': [],
            'cath_labels': [],
        })
        mmgp.insert_neighbors(db, neighbors_i)

        struct_ids = [
                mmgp.insert_structure(db, '1abc', model_id='1'),
                mmgp.insert_structure(db, '2abc', model_id='1'),
        ]
        assembly_ids = [
                mmgp.insert_assembly(
                    db, struct_ids[0], '1',
                    atoms=pl.DataFrame([
                        dict(element='C', x=0, y=0, z=0, is_polymer=True),
                    ]),
                ),
                mmgp.insert_assembly(
                    db, struct_ids[1], '1',
                    atoms=pl.DataFrame([
                        dict(element='N', x=15, y=15, z=15, is_polymer=True),
                    ]),
                ),
        ]
        zone_centers_A = [
                # Separate by more than 10Å, so that the zones can't be confused.
                np.array([0, 0, 0]),
                np.array([15, 15, 15]),
        ]
        zone_ids = [
                mmgp.insert_zone(
                    db,
                    assembly_ids[0],
                    center_A=zone_centers_A[0],
                    neighbor_ids=[0],
                ),
                mmgp.insert_zone(
                    db,
                    assembly_ids[1],
                    center_A=zone_centers_A[1],
                    neighbor_ids=[10, 11, 12, 13, 14],
                ),
        ]

        mmgp.update_splits(db, {'1abc': split, '2abc': split})

    return db, zone_ids, zone_centers_A, zone_size_A

