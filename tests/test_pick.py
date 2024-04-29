import macromol_training as mmt
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


@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            expected=atoms,
        )
)
def test_prune_nonbiological_residues(atoms, blacklist, expected):
    actual = mmt.prune_nonbiological_residues(atoms, blacklist)
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
    kd_tree = mmt.make_kd_tree(atoms)
    actual = mmt.prune_distant_atoms(
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
    actual = mmt.count_atoms(atoms, ['subchain_id'])
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
    kd_tree = mmt.make_kd_tree(atoms)
    asym_counts = mmt.count_atoms(atoms, ['subchain_id'])
    actual_subchains, actual_subchain_pairs = mmt.find_zone_subchains(
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
    actual = mmt.check_elements(atoms, whitelist)
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

    kd_tree = mmt.make_kd_tree(atoms)

    actual = mmt.find_zone_neighbors(
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

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            spacing_A=float,
            expected=coords,
        ),
)
def test_calc_zone_centers_A(atoms, spacing_A, expected):
    actual = mmt.calc_zone_centers_A(atoms, spacing_A)

    def normalize_coords(coords):
        return sorted(tuple(x) for x in coords)

    actual = normalize_coords(actual)
    expected = normalize_coords(expected)

    assert actual == approx(expected)

def test_pick_training_zones_1ypi_2ypi_3ypi():
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
    train_db = mmt.open_db(':memory:')
    mmt.init_db(train_db)

    config = mmt.Config(
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
    )
    mmt.pick_training_zones(
            train_db,
            census_db,
            config=config,
            get_mmcif_path=lambda pdb_id: pdb_dir / f'{pdb_id}.cif.gz',
    )

    mmt.show(train_db, 'SELECT * FROM structure')
    mmt.show(train_db, 'SELECT * FROM assembly')
    mmt.show(train_db, '''\
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
    assert mmt.select_structures(train_db) == ['1ypi', '2ypi']

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
    train_db = mmt.open_db(':memory:')
    mmt.init_db(train_db)

    config = mmt.Config(
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
    )
    mmt.pick_training_zones(
            train_db,
            census_db,
            config=config,
            get_mmcif_path=lambda pdb_id: pdb_dir / f'{pdb_id}.cif.gz',
    )

    mmt.show(train_db, 'SELECT * FROM structure')
    mmt.show(train_db, 'SELECT * FROM assembly')
    mmt.show(train_db, '''\
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

    assert mmt.select_structures(train_db) == ['7spt']

    # Subchains B, C, and D are the lipids, and should be ignored.
    subchains = mmt.select_dataframe(
            train_db,
            'SELECT pdb_id FROM subchain',
    )
    assert set(subchains['pdb_id']) == {'A', 'E'}

    # The structure should contain at least one zone with both the protein and 
    # the glucose ligand.
    subchain_pairs = mmt.select_dataframe(
            train_db,
            'SELECT pdb_id_1, pdb_id_2 FROM subchain_pair',
    )
    assert set(subchain_pairs.iter_rows()) == {('A', 'E')}

def test_load_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_files(tmp_path, {
        'census_db': 'version 1',
        'nonbio_residues': 'GOL\nPEG',
    })

    db = mmt.open_db(':memory:')
    mmt.init_db(db)

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
    )

    with db:
        config_1 = mmt.load_config(db, 'census_db', params)

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

    assert mmt.select_metadatum(db, 'census_md5') == 'db3ec040e20dfc657dab510aeab74759'
    assert mmt.select_metadatum(db, 'zone_size_A') == 10
    assert mmt.select_metadatum(db, 'density_check_radius_A') == 15
    assert mmt.select_metadatum(db, 'density_check_voxel_size_A') == 2
    assert mmt.select_metadatum(db, 'density_check_min_atoms_nm3') == 40
    assert mmt.select_metadatum(db, 'density_check_max_atoms_nm3') == 70
    assert mmt.select_metadatum(db, 'subchain_check_radius_A') == 8
    assert mmt.select_metadatum(db, 'subchain_check_fraction_of_zone') == approx(0.75)
    assert mmt.select_metadatum(db, 'subchain_check_fraction_of_subchain') == approx(0.75)
    assert mmt.select_metadatum(db, 'subchain_pair_check_fraction_of_zone') == approx(0.25)
    assert mmt.select_metadatum(db, 'subchain_pair_check_fraction_of_subchain') == approx(0.75)
    assert mmt.select_metadatum(db, 'neighbor_geometry') == 'icosahedron faces'
    assert mmt.select_metadatum(db, 'neighbor_distance_A') == 30
    assert mmt.select_metadatum(db, 'neighbor_count_threshold') == 1
    assert mmt.select_metadatum(db, 'atom_inclusion_radius_A') == 76
    assert mmt.select_metadatum(db, 'atom_inclusion_boundary_depth_A') == 3

    with db:
        config_2 = mmt.load_config(db, 'census_db', params)

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

    db = mmt.open_db(':memory:')
    mmt.init_db(db)

    if pre_config:
        with db:
            mmt.load_config(db, tmp_files / 'pre_mmc_pdb.duckdb', pre_config)

    with error:
        with db:
            mmt.load_config(db, tmp_files / 'mmc_pdb.duckdb', config)


