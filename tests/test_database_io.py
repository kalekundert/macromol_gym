import macromol_training as mmt
import polars as pl
import numpy as np
import pytest

from pytest import approx
from pytest_unordered import unordered

def test_metadata():
    db = mmt.open_db(':memory:')
    mmt.init_db(db)

    with db:
        mmt.upsert_metadata(db, {
            'a': 1,
            'b': 2,
            'c': 'x',
        })

    assert mmt.select_metadatum(db, 'a') == 1
    assert mmt.select_metadatum(db, 'b') == 2
    assert mmt.select_metadatum(db, 'c') == 'x'
    assert mmt.select_metadata(db, ['a', 'c']) == {
            'a': 1,
            'c': 'x',
    }

    with db:
        mmt.delete_metadatum(db, 'a')
    
    with pytest.raises(KeyError):
        mmt.select_metadatum(db, 'a')

def test_zone():
    db = mmt.open_db(':memory:')
    mmt.init_db(db)

    with db:
        mmt.insert_neighbors(
                db, 
                np.array([
                    [ 1,  0,  0],
                    [-1,  0,  0],
                    [ 0,  1,  0],
                    [ 0, -1,  0],
                    [ 0,  0,  1],
                    [ 0,  0, -1],
                ]),
        )
        struct_ids = [
                mmt.insert_structure(db, '1abc', model_id='1'),
                mmt.insert_structure(db, '2abc', model_id='1'),
        ]
        assembly_ids = [
                mmt.insert_assembly(
                    db, struct_ids[0],
                    pdb_id='2',
                    atoms=pl.DataFrame([
                        dict(element='C'),
                    ]),
                ),
                mmt.insert_assembly(
                    db, struct_ids[1],
                    pdb_id='3',
                    atoms=pl.DataFrame([
                        dict(element='N'),
                    ]),
                ),
        ]
        zone_ids = [
                mmt.insert_zone(
                    db, assembly_ids[0],
                    center_A=np.array([1, 2, 3]),
                    neighbor_ids=[1,3,5],
                    subchains=[('A', 0)],
                    subchain_pairs=[],
                ),
                mmt.insert_zone(
                    db, assembly_ids[1],
                    center_A=np.array([4, 5, 6]),
                    neighbor_ids=[1],
                    subchains=[],
                    subchain_pairs=[(('A', 0), ('B', 0))],
                ),
        ]

    assert mmt.select_zone_ids(db) == unordered(zone_ids)

    center_A, atoms = mmt.select_zone_atoms(db, zone_ids[0])
    subchains, subchain_pairs = mmt.select_zone_subchains(db, zone_ids[0])
    neighbor_ids = mmt.select_zone_neighbors(db, zone_ids[0])
    pdb_ids = mmt.select_zone_pdb_ids(db, zone_ids[0])

    assert center_A == approx([1, 2, 3])
    assert atoms.to_dicts() == [dict(element='C')]
    assert subchains == [('A', 0)]
    assert subchain_pairs == []
    assert neighbor_ids == [1,3,5]
    assert pdb_ids == {
            'struct_pdb_id': '1abc',
            'model_pdb_id': '1',
            'assembly_pdb_id': '2',
            'zone_center_A': approx([1, 2, 3]),
    }
    
    center_A, atoms = mmt.select_zone_atoms(db, zone_ids[1])
    subchains, subchain_pairs = mmt.select_zone_subchains(db, zone_ids[1])
    neighbor_ids = mmt.select_zone_neighbors(db, zone_ids[1])
    pdb_ids = mmt.select_zone_pdb_ids(db, zone_ids[1])

    assert center_A == approx([4, 5, 6])
    assert atoms.to_dicts() == [dict(element='N')]
    assert subchains == []
    assert subchain_pairs == [(('A', 0), ('B', 0))]
    assert neighbor_ids == [1]
    assert pdb_ids == {
            'struct_pdb_id': '2abc',
            'model_pdb_id': '1',
            'assembly_pdb_id': '3',
            'zone_center_A': approx([4, 5, 6]),
    }

def test_splits():
    db = mmt.open_db(':memory:')
    mmt.init_db(db)

    with db:
        struct_ids = [
                mmt.insert_structure(db, '1abc', model_id='1'),
                mmt.insert_structure(db, '2abc', model_id='1'),
                mmt.insert_structure(db, '3abc', model_id='1'),
        ]

        # Insert one assembly and one zone per structure---the simplest thing 
        # possible.  The goal is just to make sure that the right zones get 
        # selected for each split.
        assembly_ids = [
                mmt.insert_assembly(db, struct_id, '1', pl.DataFrame())
                for struct_id in struct_ids
        ]
        zone_ids = [
                mmt.insert_zone(
                    db, assembly_id,
                    center_A=np.zeros(3),
                    neighbor_ids=[],
                )
                for assembly_id in assembly_ids
        ]

        mmt.update_splits(db, {
                '1abc': 'train',
                '2abc': 'val',
                '3abc': 'train',
        })

    assert mmt.select_structures(db) == ['1abc', '2abc', '3abc']

    assert list(mmt.select_split(db, 'train')) == [zone_ids[0], zone_ids[2]]
    assert list(mmt.select_split(db, 'val')) == [zone_ids[1]]

    with db:
        mmt.delete_splits(db)
        mmt.update_splits(db, {
                '1abc': 'train',
                '2abc': 'train',
                '3abc': 'val',
        })

    assert list(mmt.select_split(db, 'train')) == [zone_ids[0], zone_ids[1]]
    assert list(mmt.select_split(db, 'val')) == [zone_ids[2]]
