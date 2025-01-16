import macromol_gym_unsupervised as mmgu
import polars as pl
import numpy as np

def make_db(path=':memory:', *, split='train'):
    db = mmgu.open_db(path, mode='rwc')
    with db:
        mmgu.init_db(db)

        zone_size_A = 2

        mmgu.upsert_metadata(db, {
            'zone_size_A': zone_size_A,
            'neighbor_count_threshold': 0,
            'polymer_labels': [],
            'cath_labels': [],

        })

        # Insert two structures, each with two atoms separated by 10Å.  A zone 
        # is centered on each atom.

        struct_ids = [
                mmgu.insert_structure(db, '1abc', model_id='1'),
                mmgu.insert_structure(db, '2abc', model_id='1'),
        ]
        assembly_ids = [
                mmgu.insert_assembly(
                    db, struct_ids[0], '1',
                    atoms=pl.DataFrame([
                        dict(element='C', x=0, y=0, z=0),
                        dict(element='C', x=10, y=0, z=0),
                    ]),
                ),
                mmgu.insert_assembly(
                    db, struct_ids[1], '1',
                    atoms=pl.DataFrame([
                        dict(element='C', x=0, y=0, z=0),
                        dict(element='C', x=0, y=10, z=0),
                    ]),
                ),
        ]
        zone_centers_A = [
                np.array([ 0,  0,  0]),
                np.array([10,  0,  0]),
                np.array([ 0,  0,  0]),
                np.array([ 0, 10,  0]),
        ]
        zone_ids = [
                mmgu.insert_zone(
                    db,
                    assembly_ids[0],
                    center_A=zone_centers_A[0],
                    neighbor_ids=[],
                ),
                mmgu.insert_zone(
                    db,
                    assembly_ids[0],
                    center_A=zone_centers_A[1],
                    neighbor_ids=[],
                ),
                mmgu.insert_zone(
                    db,
                    assembly_ids[1],
                    center_A=zone_centers_A[2],
                    neighbor_ids=[],
                ),
                mmgu.insert_zone(
                    db,
                    assembly_ids[1],
                    center_A=zone_centers_A[3],
                    neighbor_ids=[],
                ),
        ]

        mmgu.update_splits(db, {'1abc': split, '2abc': split})

    return db, zone_ids, zone_centers_A, zone_size_A

