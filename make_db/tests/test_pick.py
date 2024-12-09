import macromol_gym as mmg
import macromol_census as mmc
import macromol_dataframe as mmdf
import parametrize_from_file as pff
import polars as pl
import polars.testing
import numpy as np
import re

from macromol_dataframe.testing import dataframe, coord, coords
from pytest import approx
from pytest_unordered import unordered
from pytest_tmp_files import make_files
from pathlib import Path

CATH_LABELS_SCHEMA = {
        'pdb_id': str,
        'chain_id': str,
        'domain_id': int,
        'seq_ids': list[int],
        'cath_label': int,
}

def atoms(params):
    if re.search(r'.cif(\.gz)?$', params):
        cif_path = Path(__file__).parent / 'pdb' / params
        return mmdf.read_biological_assembly(
                cif_path,
                model_id='1',
                assembly_id='1',
        )

    return dataframe(
            params,
            exprs={
                'oob': pl.col('out_of_bounds') == '1',
                'is_polymer': pl.col('is_polymer') == '1',
            },
            dtypes={
                'sym': int,
                'resi': int,
                'x': float,
                'y': float,
                'z': float,
                'occ': float,
                'oob': bool,
                'is_polymer': bool,
            },
            col_aliases={
                'sym': 'symmetry_mate',
                'subchain': 'subchain_id',
                'resn': 'comp_id',
                'resi': 'seq_id',
                'atom': 'atom_id',
                'e': 'element',
                'occ': 'occupancy',
                'oob': 'out_of_bounds',
            },
    )

def assert_index_exists(db, table, col):
    cur = db.execute('''\
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index' AND tbl_name=?
    ''', [table])
    rows = cur.fetchall()
    assert len(rows) == 1

    sql, = rows[0]
    assert f'({col})' in sql


@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            expected=atoms,
        )
)
def test_prune_nonbiological_residues(atoms, blacklist, expected):
    actual = mmg.prune_nonbiological_residues(atoms, blacklist)
    pl.testing.assert_frame_equal(actual, expected)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            centers_A=coords,
            radius_A=float,
            boundary_depth_A=float,
            expected=atoms,
        )
)
def test_prune_distant_atoms(atoms, centers_A, radius_A, boundary_depth_A, expected):
    kd_tree = mmg.make_kd_tree(atoms)
    actual = mmg.prune_distant_atoms(
            atoms, 
            kd_tree,
            centers_A=centers_A,
            radius_A=radius_A,
            boundary_depth_A=boundary_depth_A,
    )
    pl.testing.assert_frame_equal(actual, expected, check_row_order=False)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            expected=dataframe(
                dtypes={'occupancy': float},
                col_aliases={'subchain': 'subchain_id', 'occ': 'occupancy'},
            ),
        ),
)
def test_count_atoms(atoms, expected):
    actual = mmg.count_atoms(atoms, ['subchain_id'])
    pl.testing.assert_frame_equal(
            actual, expected,
            check_exact=False,
            check_row_order=False,
    )

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            center_A=coord,
            radius_A=float,
            solo_fraction_of_zone=float,
            solo_fraction_of_subchain=float,
            pair_fraction_of_zone=float,
            pair_fraction_of_subchain=float,
        ),
)
def test_find_zone_subchains(
            atoms,
            center_A,
            radius_A,
            solo_fraction_of_zone,
            solo_fraction_of_subchain,
            pair_fraction_of_zone,
            pair_fraction_of_subchain,
            subchains,
            subchain_pairs,
):
    kd_tree = mmg.make_kd_tree(atoms)
    asym_counts = mmg.count_atoms(atoms, ['subchain_id'])
    actual_subchains, actual_subchain_pairs = mmg.find_zone_subchains(
            atoms,
            kd_tree,
            asym_counts=asym_counts,
            center_A=center_A,
            radius_A=radius_A,
            solo_fraction_of_zone=solo_fraction_of_zone,
            solo_fraction_of_subchain=solo_fraction_of_subchain,
            pair_fraction_of_zone=pair_fraction_of_zone,
            pair_fraction_of_subchain=pair_fraction_of_subchain,
    )

    def normalize(subchains):
        return [(str(i), int(n)) for i, n in subchains]

    def normalize_pairs(subchain_pairs):
        return [
                ((str(i1), int(n1)), (str(i2), int(n2)))
                for (i1, n1), (i2, n2) in subchain_pairs
        ]

    expected_subchains = normalize(subchains)
    expected_subchain_pairs = normalize_pairs(subchain_pairs)

    assert actual_subchains == unordered(expected_subchains)
    assert actual_subchain_pairs == unordered(expected_subchain_pairs)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            center_A=coord,
            radius_A=float,
            expected=eval,
        ),
)
def test_check_elements(atoms, center_A, radius_A, whitelist, expected):
    actual = mmg.check_elements(atoms, whitelist)
    assert actual == expected

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            centers_A=coords,
            radius_A=float,
            min_density_atoms_nm3=float,
            expected=lambda xs: [int(x) for x in xs],
        ),
)
def test_find_zone_neighbors(atoms, centers_A, radius_A, min_density_atoms_nm3, expected):
    from test_density import calc_density_atoms_nm3

    kd_tree = mmg.make_kd_tree(atoms)

    actual = mmg.find_zone_neighbors(
            # The real program uses an approximate method of calculating 
            # densities.  But exact densities are much easier to reason about, 
            # for the purposes of testing.
            lambda coords_A: np.array([
                calc_density_atoms_nm3(atoms, kd_tree, coord_A, radius_A)
                for coord_A in coords_A
            ]),
            center_A=np.zeros(3),
            offsets_A=centers_A,
            min_density_atoms_nm3=min_density_atoms_nm3,
    )
    assert actual == expected

    # This function uses numpy under the hood, so it's possible that it would 
    # return numpy integers.  However, these are not understood to be integers 
    # by SQLite, which causes problems down the road.
    assert all(type(x) is int for x in actual)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            spacing_A=float,
            expected=coords,
        ),
)
def test_calc_zone_centers_A(atoms, spacing_A, expected):
    actual = mmg.calc_zone_centers_A(atoms, spacing_A)

    def normalize_coords(coords):
        return sorted(tuple(x) for x in coords)

    actual = normalize_coords(actual)
    expected = normalize_coords(expected)

    assert actual == approx(expected)

def test_annotate_polymers():
    asym_atoms = pl.DataFrame([
        dict(entity_id='1'),
        dict(entity_id='2'),
        dict(entity_id='3'),
        dict(entity_id='4'),
        dict(entity_id='5'),
        dict(entity_id='6'),
        dict(entity_id='7'),
        dict(entity_id='8'),
        dict(entity_id='9'),
        dict(entity_id='10'),
    ])
    entities = pl.DataFrame([
        dict(id='1', type='polymer'),
        dict(id='2', type='polymer'),
        dict(id='3', type='polymer'),
        dict(id='4', type='polymer'),
        dict(id='5', type='polymer'),
        dict(id='6', type='polymer'),
        dict(id='7', type='polymer'),
        dict(id='8', type='polymer'),
        dict(id='9', type='non-polymer'),
        dict(id='10', type='water'),
    ])
    polymers = pl.DataFrame([
        dict(entity_id='1', type='polypeptide(L)'),
        dict(entity_id='2', type='polypeptide(D)'),
        dict(entity_id='3', type='polyribonucleotide'),
        dict(entity_id='4', type='polydeoxyribonucleotide'),
        dict(entity_id='5', type='polydeoxyribonucleotide/polyribonucleotide hybrid'),
        dict(entity_id='6', type='cyclic-pseudo-peptide'),
        dict(entity_id='7', type='peptide nucleic acid'),
        dict(entity_id='8', type='other'),
    ])
    labels = mmg.get_polymer_labels()

    actual_atoms = mmg.annotate_polymers(
            asym_atoms,
            entities, 
            polymers,
            labels,
    )
    expected_atoms = pl.DataFrame([
        dict(entity_id='1',  is_polymer=True,  polymer_label=0),
        dict(entity_id='2',  is_polymer=True,  polymer_label=None),
        dict(entity_id='3',  is_polymer=True,  polymer_label=2),
        dict(entity_id='4',  is_polymer=True,  polymer_label=1),
        dict(entity_id='5',  is_polymer=True,  polymer_label=None),
        dict(entity_id='6',  is_polymer=True,  polymer_label=None),
        dict(entity_id='7',  is_polymer=True,  polymer_label=None),
        dict(entity_id='8',  is_polymer=True,  polymer_label=None),
        dict(entity_id='9',  is_polymer=False, polymer_label=None),
        dict(entity_id='10', is_polymer=False, polymer_label=None),
    ])
    pl.testing.assert_frame_equal(
            actual_atoms,
            expected_atoms,
            check_dtypes=False,
    )

def test_annotate_domains():
    asym_atoms = pl.DataFrame([
        dict(chain_id='A', seq_id=1),
        dict(chain_id='A', seq_id=2),
        dict(chain_id='A', seq_id=3),
        dict(chain_id='B', seq_id=1),
        dict(chain_id='B', seq_id=2),
        dict(chain_id='B', seq_id=3),
    ])
    cath_labels = pl.DataFrame([
            dict(chain_id='A', seq_ids=[1, 2], cath_label=0),
            dict(chain_id='B', seq_ids=[2, 3], cath_label=1),
    ])

    actual_atoms = mmg.annotate_domains(asym_atoms, cath_labels)
    expected_atoms = pl.DataFrame([
        dict(chain_id='A', seq_id=1, cath_label=0),
        dict(chain_id='A', seq_id=2, cath_label=0),
        dict(chain_id='A', seq_id=3, cath_label=None),
        dict(chain_id='B', seq_id=1, cath_label=None),
        dict(chain_id='B', seq_id=2, cath_label=1),
        dict(chain_id='B', seq_id=3, cath_label=1),
    ])
    pl.testing.assert_frame_equal(
            actual_atoms,
            expected_atoms,
            check_dtypes=False,
    )

def test_pick_polymer_labels():
    db = mmg.open_db(':memory:', mode='rwc')
    mmg.init_db(db)
    mmg.pick_polymer_labels(db)

    assert mmg.select_metadatum(db, 'polymer_labels') == [
            'polypeptide(L)',
            'polydeoxyribonucleotide',
            'polyribonucleotide',
    ]

def test_pick_cath_labels():
    db = mmg.open_db(':memory:', mode='rwc')
    mmg.init_db(db)

    # Not using all of the columns that would be present in the real dataframe; 
    # just those that are necessary for creating labels.
    cath_domains = pl.DataFrame([
            dict(pdb_id='1xyz', c=1, a=20, t=5, h=10),
            dict(pdb_id='2xyz', c=1, a=20, t=5, h=10),

            dict(pdb_id='3xyz', c=1, a=10, t=8, h=10),
            dict(pdb_id='4xyz', c=1, a=10, t=8, h=20),
            dict(pdb_id='5xyz', c=1, a=10, t=10, h=10),

            # Only one cluster member; too few to include
            dict(pdb_id='6xyz', c=2, a=10, t=10, h=10),
    ])

    actual_labels = mmg.pick_cath_labels(db, cath_domains, 2)
    expected_labels = pl.DataFrame([
            dict(pdb_id='1xyz', cath_label=1),
            dict(pdb_id='2xyz', cath_label=1),

            dict(pdb_id='3xyz', cath_label=0),
            dict(pdb_id='4xyz', cath_label=0),
            dict(pdb_id='5xyz', cath_label=0),
    ])

    pl.testing.assert_frame_equal(
            actual_labels,
            expected_labels,
            check_dtypes=False,
    )
    assert mmg.select_metadatum(db, 'cath_labels') == ['1.10', '1.20']

def test_pick_training_zones_1ypi_2ypi_3ypi(tmp_path):
    # Prepare an in-memory census database containing 1ypi (apo TIM), 2ypi 
    # (holo TIM), and 3ypi (holo TIM).  The idea is to make a training database 
    # that (i) only includes the parts of the first holo structure that aren't 
    # in the apo structure, e.g. the ligand binding interactions, and (ii) 
    # doesn't include any of the second holo structure.
    #
    # Some details that need to be accounted for:
    # - The protein chains must all be clustered together.
    # - The ligands in the holo structures must be clustered together.  Note 
    #   that these are actually different ligands, and in my real training 
    #   database they wouldn't be clustered together.
    # - 1ypi must be ranked ahead of 2ypi, which must be ranked ahead of 3ypi.

    pdb_dir = Path(__file__).parent / 'pdb'
    pdb_paths = [
            pdb_dir / '3ypi.cif.gz',
            pdb_dir / '1ypi.cif.gz',
            pdb_dir / '2ypi.cif.gz',
    ]

    # Structures:
    # ┌────┬────────┐
    # │ id │ pdb_id │
    # ├────┼────────┤
    # │  1 │ 3ypi   │
    # │  2 │ 1ypi   │
    # │  3 │ 2ypi   │
    # └────┴────────┘
    #
    # Entities:
    # ┌────┬───────────┬────────┬───────────────────────────┐
    # │ id │ struct_id │ pdb_id │ description               │
    # ├────┼───────────┼────────┼───────────────────────────┤
    # │  1 │         1 │ 1      │ TIM                       │
    # │  2 │         1 │ 2      │ non-hydrolyzable ligand   │
    # │  3 │         2 │ 1      │ TIM                       │
    # │  4 │         2 │ 2      │ water                     │
    # │  5 │         3 │ 1      │ TIM                       │
    # │  6 │         3 │ 2      │ 2-phosphoglycolate        │
    # │  7 │         3 │ 3      │ water                     │
    # └────┴───────────┴────────┴───────────────────────────┘

    census_db = mmc.open_db(':memory:')
    mmc.init_db(census_db)
    mmc.ingest_structures(census_db, pdb_paths)
    mmc.update_structure_ranks(
            census_db,
            pl.DataFrame([
                dict(struct_id=2, rank=1),
                dict(struct_id=3, rank=2),
                dict(struct_id=1, rank=3),
            ]),
    )
    mmc.insert_assembly_ranks(
            census_db,
            pl.DataFrame([
                dict(assembly_id=1, rank=1),
                dict(assembly_id=2, rank=1),
                dict(assembly_id=3, rank=1),
            ]),
    )
    mmc.insert_entity_clusters(
            census_db,
            pl.DataFrame([
                dict(cluster_id=1, entity_id=1),
                dict(cluster_id=1, entity_id=3),
                dict(cluster_id=1, entity_id=5),
                dict(cluster_id=2, entity_id=2),
                dict(cluster_id=2, entity_id=6),
            ]),
            'protein and ligand clusters',
    )

    # Finished setting up the census database, now we can pick the dataset:
    train_db = mmg.open_db(':memory:', mode='rwc')
    mmg.init_db(train_db)

    make_files(tmp_path, {
        'cath-classification-data/cath-domain-list.txt': '''\
1ypiA00     3    20    20    70     1     4     1     8     1   247 1.900
1ypiB00     3    20    20    70     1     4     1     8     2   247 1.900
2ypiA00     3    20    20    70     1     4     1     8     5   247 2.500
2ypiB00     3    20    20    70     1     4     1     8     6   247 2.500
3ypiA00     3    20    20    70     1     4     1     9     1   247 2.800
3ypiB00     3    20    20    70     1     4     1     9     2   247 2.800
''',
        'cath-classification-data/cath-domain-boundaries-seqreschopping.txt': '''\
1ypiA00	1-247
1ypiB00	1-247
2ypiA00	1-247
2ypiB00	1-247
3ypiA00	1-247
3ypiB00	1-247
'''
    })

    config = mmg.Config(
            census_md5=None,

            zone_size_A=10,

            density_check_radius_A=15,
            density_check_voxel_size_A=2,
            density_check_min_atoms_nm3=35,
            density_check_max_atoms_nm3=70,

            subchain_check_radius_A=8,
            subchain_check_fraction_of_zone=0.75,
            subchain_check_fraction_of_subchain=0.75,
            subchain_pair_check_fraction_of_zone=0.25,
            subchain_pair_check_fraction_of_subchain=0.75,

            neighbor_geometry='icosahedron and dodecahedron faces',
            neighbor_distance_A=30,
            neighbor_count_threshold=1,

            allowed_elements=['C', 'N', 'O', 'S', 'P'],
            nonbiological_residues=[],

            atom_inclusion_radius_A=75,
            atom_inclusion_boundary_depth_A=3,

            cath_md5={},
            cath_min_domains=7000,
    )
    polymer_labels = mmg.get_polymer_labels()
    cath_labels = pl.DataFrame([
        dict(pdb_id='1ypi', chain_id='A', domain_id=0, seq_ids=inclusive_range(1, 247), cath_label=6),
        dict(pdb_id='1ypi', chain_id='B', domain_id=0, seq_ids=inclusive_range(1, 247), cath_label=6),
        dict(pdb_id='2ypi', chain_id='A', domain_id=0, seq_ids=inclusive_range(1, 247), cath_label=6),
        dict(pdb_id='2ypi', chain_id='B', domain_id=0, seq_ids=inclusive_range(1, 247), cath_label=6),
        dict(pdb_id='3ypi', chain_id='A', domain_id=0, seq_ids=inclusive_range(1, 247), cath_label=6),
        dict(pdb_id='3ypi', chain_id='B', domain_id=0, seq_ids=inclusive_range(1, 247), cath_label=6),
    ])

    mmg.pick_training_zones(
            train_db,
            census_db,
            config=config,
            polymer_labels=polymer_labels,
            cath_labels=cath_labels,
            get_mmcif_path=lambda pdb_id: pdb_dir / f'{pdb_id}.cif.gz',
    )

    mmg.show(train_db, 'SELECT * FROM structure')
    mmg.show(train_db, 'SELECT * FROM assembly')
    mmg.show(train_db, '''\
            SELECT 
                zone.id,
                zone.assembly_id,
                zone.center_A,
                subchain.pdb_id AS solo,
                subchain_pair.pdb_id_1 AS pair_1,
                subchain_pair.pdb_id_2 AS pair_2
            FROM zone
            LEFT JOIN subchain on zone.id = subchain.zone_id
            LEFT JOIN subchain_pair ON zone.id = subchain_pair.zone_id
    ''')

    # This isn't the most stringent check, but I can't really think of anything 
    # else that would be robust to small changes in the code.
    assert mmg.select_structures(train_db) == ['1ypi', '2ypi']

    # Check that each atom gets the right polymer/CATH labels.  We can take 
    # advantage of the fact that for all of the structures in this test, 
    # subchains A & B are the only proteins.  The other subchains are either 
    # water or small molecules.
    for atoms, in train_db.execute('SELECT atoms FROM assembly').fetchall():
        atoms = (
                atoms
                .with_columns(
                    expected_polymer_label=(
                        pl.when(pl.col('subchain_id').is_in(['A', 'B']))
                        .then(0)
                        .otherwise(None)
                    ),
                    expected_cath_label=(
                        pl.when(pl.col('subchain_id').is_in(['A', 'B']))
                        .then(6)
                        .otherwise(None)
                    ),
                )
        )
        pl.testing.assert_series_equal(
                atoms['polymer_label'],
                atoms['expected_polymer_label'],
                check_names=False,
                check_dtypes=False,
        )
        pl.testing.assert_series_equal(
                atoms['cath_label'],
                atoms['expected_cath_label'],
                check_names=False,
                check_dtypes=False,
        )

    assert_index_exists(train_db, 'zone_neighbor', 'zone_id')

def test_pick_training_zones_7spt_1c58():
    # 7spt is a structure of a glucose transporter.  It contains a number of 
    # non-specific ligands (specifically 1-oleoyl-R-glycerol, a.k.a. OLC) which 
    # should be ignored.

    # 1c58 is a structure of amylose, which is composed entirely of glucose.  
    # Glucose is on my list of non-biological ligands, which means this 
    # structure ends up with no atoms (when it comes to calculating densities).  
    # This structure should be ignored, but shouldn't cause any problems.

    pdb_dir = Path(__file__).parent / 'pdb'
    pdb_paths = [
            pdb_dir / '7spt.cif.gz',
            pdb_dir / '1c58.cif.gz',
    ]

    census_db = mmc.open_db(':memory:')
    mmc.init_db(census_db)
    mmc.ingest_structures(census_db, pdb_paths)
    mmc.insert_assembly_ranks(
            census_db,
            pl.DataFrame([
                dict(assembly_id=1, rank=1),
                dict(assembly_id=2, rank=2),
            ]),
    )
    mmc.insert_nonspecific_ligands(
            census_db,
            pl.DataFrame({'pdb_comp_id': ['OLC']}),
    )

    # Finished setting up the census database, now we can pick the dataset:
    train_db = mmg.open_db(':memory:', mode='rwc')
    mmg.init_db(train_db)

    config = mmg.Config(
            census_md5=None,

            zone_size_A=10,

            density_check_radius_A=15,
            density_check_voxel_size_A=2,
            density_check_min_atoms_nm3=35,
            density_check_max_atoms_nm3=70,

            subchain_check_radius_A=8,
            subchain_check_fraction_of_zone=0.75,
            subchain_check_fraction_of_subchain=0.75,
            subchain_pair_check_fraction_of_zone=0.25,
            subchain_pair_check_fraction_of_subchain=0.75,

            neighbor_geometry='icosahedron and dodecahedron faces',
            neighbor_distance_A=30,
            neighbor_count_threshold=1,

            allowed_elements=['C', 'N', 'O', 'S', 'P'],
            nonbiological_residues=['GLC'],

            atom_inclusion_radius_A=75,
            atom_inclusion_boundary_depth_A=3,

            cath_md5={},
            cath_min_domains=7000,
    )
    mmg.pick_training_zones(
            train_db,
            census_db,
            config=config,
            polymer_labels=[],
            cath_labels=pl.DataFrame([], CATH_LABELS_SCHEMA),
            get_mmcif_path=lambda pdb_id: pdb_dir / f'{pdb_id}.cif.gz',
    )

    mmg.show(train_db, 'SELECT * FROM structure')
    mmg.show(train_db, 'SELECT * FROM assembly')
    mmg.show(train_db, '''\
            SELECT 
                zone.id,
                zone.assembly_id,
                zone.center_A,
                subchain.pdb_id AS solo,
                subchain_pair.pdb_id_1 AS pair_1,
                subchain_pair.pdb_id_2 AS pair_2
            FROM zone
            LEFT JOIN subchain on zone.id = subchain.zone_id
            LEFT JOIN subchain_pair ON zone.id = subchain_pair.zone_id
    ''')

    assert mmg.select_structures(train_db) == ['7spt']

    # Subchains B, C, and D are the lipids, and should be ignored.
    subchains = mmg.select_dataframe(
            train_db,
            'SELECT pdb_id FROM subchain',
    )
    assert set(subchains['pdb_id']) == {'A', 'E'}

    # The structure should contain at least one zone with both the protein and 
    # the glucose ligand.
    subchain_pairs = mmg.select_dataframe(
            train_db,
            'SELECT pdb_id_1, pdb_id_2 FROM subchain_pair',
    )
    assert set(subchain_pairs.iter_rows()) == {('A', 'E')}

    assert_index_exists(train_db, 'zone_neighbor', 'zone_id')

def test_load_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_files(tmp_path, {
        'census.duckdb': 'version 1',
        'cath_domain_list.txt': '1oaiA00     1    10     8    10',
        'cath_domain_boundaries.txt': '1oaiA00	1-59',
        'nonbio_residues': 'GOL\nPEG',
    })

    db = mmg.open_db(':memory:', mode='rwc')
    mmg.init_db(db)

    params = dict(
        zone_size_A='10',
        density_check_radius_A='15',
        density_check_voxel_size_A='2',
        density_check_min_atoms_nm3='40',
        density_check_max_atoms_nm3='70',
        subchain_check_radius_A='8',
        subchain_check_fraction_of_zone='0.75',
        subchain_check_fraction_of_subchain='0.75',
        subchain_pair_check_fraction_of_zone='0.25',
        subchain_pair_check_fraction_of_subchain='0.75',
        neighbor_geometry='icosahedron faces',
        neighbor_distance_A='30',
        neighbor_count_threshold='1',
        allowed_elements=['C', 'N', 'O', 'S', 'SE', 'P'],
        nonbiological_residues='nonbio_residues',
        atom_inclusion_radius_A='76',
        atom_inclusion_boundary_depth_A='3',
        cath_min_domains=7000,
    )
    census_path = 'census.duckdb'
    cath_paths = {
            'domain-list.txt': 'cath_domain_list.txt',
            'domain-boundaries.txt': 'cath_domain_boundaries.txt',
    }

    with db:
        config_1 = mmg.load_config(db, census_path, cath_paths, params)

    assert config_1.census_md5 == 'db3ec040e20dfc657dab510aeab74759'
    assert config_1.zone_size_A == 10
    assert config_1.density_check_radius_A == 15
    assert config_1.density_check_voxel_size_A == 2
    assert config_1.density_check_min_atoms_nm3 == 40
    assert config_1.density_check_max_atoms_nm3 == 70
    assert config_1.subchain_check_radius_A == 8
    assert config_1.subchain_check_fraction_of_zone == approx(0.75)
    assert config_1.subchain_check_fraction_of_subchain == approx(0.75)
    assert config_1.subchain_pair_check_fraction_of_zone == approx(0.25)
    assert config_1.subchain_pair_check_fraction_of_subchain == approx(0.75)
    assert config_1.neighbor_geometry == 'icosahedron faces'
    assert config_1.neighbor_distance_A == 30
    assert config_1.neighbor_count_threshold == 1
    assert config_1.allowed_elements == ['C', 'N', 'O', 'S', 'SE', 'P']
    assert config_1.nonbiological_residues == ['GOL', 'PEG']
    assert config_1.atom_inclusion_radius_A == 76
    assert config_1.atom_inclusion_boundary_depth_A == 3

    assert mmg.select_metadatum(db, 'census_md5') == 'db3ec040e20dfc657dab510aeab74759'
    assert mmg.select_metadatum(db, 'zone_size_A') == 10
    assert mmg.select_metadatum(db, 'density_check_radius_A') == 15
    assert mmg.select_metadatum(db, 'density_check_voxel_size_A') == 2
    assert mmg.select_metadatum(db, 'density_check_min_atoms_nm3') == 40
    assert mmg.select_metadatum(db, 'density_check_max_atoms_nm3') == 70
    assert mmg.select_metadatum(db, 'subchain_check_radius_A') == 8
    assert mmg.select_metadatum(db, 'subchain_check_fraction_of_zone') == approx(0.75)
    assert mmg.select_metadatum(db, 'subchain_check_fraction_of_subchain') == approx(0.75)
    assert mmg.select_metadatum(db, 'subchain_pair_check_fraction_of_zone') == approx(0.25)
    assert mmg.select_metadatum(db, 'subchain_pair_check_fraction_of_subchain') == approx(0.75)
    assert mmg.select_metadatum(db, 'neighbor_geometry') == 'icosahedron faces'
    assert mmg.select_metadatum(db, 'neighbor_distance_A') == 30
    assert mmg.select_metadatum(db, 'neighbor_count_threshold') == 1
    assert mmg.select_metadatum(db, 'atom_inclusion_radius_A') == 76
    assert mmg.select_metadatum(db, 'atom_inclusion_boundary_depth_A') == 3
    assert mmg.select_metadatum(db, 'cath_md5') == {
            'domain-list.txt': 'c12d622a3d928a74b834c6b74aa320ee',
            'domain-boundaries.txt': '2dc2593e5e1cf1721003040016178351',
    }
    assert mmg.select_metadatum(db, 'cath_min_domains') == 7000

    with db:
        config_2 = mmg.load_config(db, census_path, cath_paths, params)

    assert config_1 == config_2

@pff.parametrize(
        schema=[
            pff.defaults(pre_config=None),
            pff.cast(error=pff.error),
        ],
        indirect=['tmp_files'],
)
def test_load_config_err(tmp_files, pre_config, config, error, monkeypatch):
    monkeypatch.chdir(tmp_files)

    cath_paths = {
            'domain-list.txt': 'cath_domain_list.txt',
            'domain-boundaries.txt': 'cath_domain_boundaries.txt',
    }

    db = mmg.open_db(':memory:', mode='rwc')
    mmg.init_db(db)

    if pre_config:
        with db:
            mmg.load_config(db, tmp_files / 'pre_mmc_pdb.duckdb', cath_paths, pre_config)

    with error:
        with db:
            mmg.load_config(db, tmp_files / 'mmc_pdb.duckdb', cath_paths, config)

def inclusive_range(start, stop):
    return list(range(start, stop+1))
