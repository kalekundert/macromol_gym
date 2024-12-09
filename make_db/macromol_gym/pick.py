"""
Usage:
    mmt_pick_training_examples <db> <census> <config> [-c <dir>]

Arguments:
    <db>
        The path to a database initialized by `mmt_init`.

    <census>
        The path to the completed database produced by `macromol_census`.  This 
        database identifies all of the biological assemblies in the PDB that 
        are sufficiently unique and high-quality to be worth including in the 
        training data.

    <config>
        A NestedText file specifying the following parameters:

        - `zone_size_A`

        - `density_check_radius_A`
        - `density_check_voxel_size_A`
        - `density_check_min_atoms_nm3`
        - `density_check_max_atoms_nm3`

        - `subchain_check_radius_A`
        - `subchain_check_fraction_of_zone`
        - `subchain_check_fraction_of_subchain`
        - `subchain_pair_check_fraction_of_zone`
        - `subchain_pair_check_fraction_of_subchain`

        - `neighbor_geometry`
        - `neighbor_distance_A`
        - `neighbor_count_threshold`

        - `allowed_elements`:
          A list of atomic elements that are allowed to be included in the 
          database.

        - `nonbiological_residues`:
          The path to a file containing a list of residue types to ignore when 
          calculating densities.  The file should be a text file with one 
          residue name on each line.  Most residue names are 3 letters.

          These residues are not removed from the structures stored in the 
          database.  They just don't count for the various density thresholds 
          that are checked.  The idea is that regions of structure with more 
          "non-biological" atoms aren't as likely to be added to the database, 
          but if they are added, they won't be missing any atoms.

        - `atom_inclusion_radius_A`
        - `atom_inclusion_boundary_depth_A`

Options:
    -c --cath <dir>
        The path to a directory containing a copy of the CATH-Plus database.  
        This directory must contain at least the following two files:

            cath-classification-data/cath-domain-list.txt
            cath-classification-data/cath-domain-boundaries-seqreschopping.txt

        Both of these files may optionally have a version number inserted after 
        the file name and before the `.txt`.  If no path is specified, the 
        `$CATH_PLUS` environment variable will be used instead.  An error will 
        occur if no CATH directory can be found.
"""

import polars as pl
import numpy as np
import periodictable
import nestedtext as nt
import pickle
import os

from .database_io import (
        open_db,
        insert_metadata, upsert_metadata, delete_metadatum, select_metadata, select_metadatum,
        insert_structure, insert_assembly, insert_zone,
        insert_neighbors, select_neighbors,
)
from .cath import load_cath_domains
from .density import make_density_interpolator
from .geometry import (
        tetrahedron_faces,
        cube_faces,
        octahedron_faces,
        dodecahedron_faces,
        icosahedron_faces,
)
from macromol_dataframe import (
        Atoms, Coord, Coords3,
        read_mmcif, get_pdb_path, select_model, make_biological_assembly,
        prune_hydrogen, prune_water, get_atom_coords
)
from macromol_census import (
        open_db as open_census_db, visit_assemblies, Visitor, Candidate,
        tquiet,
)
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from icosphere import icosphere
from pipeline_func import f
from collections import defaultdict
from itertools import product, combinations
from more_itertools import one, flatten
from tqdm import tqdm
from math import ceil
from pathlib import Path
from hashlib import md5
from dataclasses import dataclass, fields, asdict
from typing import Callable, Any

@dataclass
class Config:
    census_md5: str

    zone_size_A: float

    density_check_radius_A: float
    density_check_voxel_size_A: float
    density_check_min_atoms_nm3: float
    density_check_max_atoms_nm3: float

    subchain_check_radius_A: float
    subchain_check_fraction_of_zone: float
    subchain_check_fraction_of_subchain: float
    subchain_pair_check_fraction_of_zone: float
    subchain_pair_check_fraction_of_subchain: float

    neighbor_geometry: str
    neighbor_distance_A: float
    neighbor_count_threshold: int

    allowed_elements: list[str]
    nonbiological_residues: list[str]

    atom_inclusion_radius_A: float
    atom_inclusion_boundary_depth_A: float

    cath_md5: dict[str, str]
    cath_min_domains: int

def main():
    import docopt

    args = docopt.docopt(__doc__)

    train_db = open_db(args['<db>'], mode='rw')
    census_db = open_census_db(census_path := args['<census>'], read_only=True)
    cath_dir = Path(args['--cath'] or os.environ['CATH_PLUS'])
    cath_domains, cath_paths = load_cath_domains(cath_dir)
    pdb_dir = Path(os.environ['PDB_MMCIF'])

    config = load_config(
            db=train_db,
            census_path=census_path,
            cath_paths=cath_paths,
            config=nt.load(args['<config>']),
    )

    try:
        polymer_labels = pick_polymer_labels(train_db)
        cath_labels = pick_cath_labels(
                train_db,
                cath_domains,
                config.cath_min_domains,
        )
        pick_training_zones(
                train_db,
                census_db,
                config=config,
                polymer_labels=polymer_labels,
                cath_labels=cath_labels,
                get_mmcif_path=lambda pdb_id: get_pdb_path(pdb_dir, pdb_id),
                progress_factory=tqdm,
        )
    except KeyboardInterrupt:
        pass

def load_config(db, census_path: str, cath_paths: Path, config: dict[str, Any]):
    db_config = _select_config(db)
    user_config = _make_config(config, census_path, cath_paths)

    if db_config:
        if db_config != user_config:
            db_dict = asdict(db_config)
            user_dict = asdict(user_config)
            diff_keys = [
                    k for k in db_dict
                    if db_dict[k] != user_dict[k]
            ]
            raise ValueError(f"the following parameters have changed since the last run: {diff_keys}\nTo use different parameters, create a new database.")
    else:
        _insert_config(db, user_config)

    return user_config

def pick_polymer_labels(db):
    labels = get_polymer_labels()
    meta = select_metadata(db, ['polymer_labels'])

    if not meta:
        insert_metadata(db, {'polymer_labels': labels})
    elif meta['polymer_labels'] != labels:
        raise ValueError(f"the polymer labels have changed since the last run:\nprevious: {meta['polymer_labels']}\ncurrent: {labels}\nTo use different labels, create a new database.")

    return labels

def get_polymer_labels():
    # See Experiment #105.  These are the only kinds of polymers that are 
    # present in the PDB in significant quantities.
    return [
            'polypeptide(L)',
            'polydeoxyribonucleotide',
            'polyribonucleotide',
    ]

def pick_cath_labels(db, cath_domains, min_domains: int):
    labels = (
            cath_domains
            .group_by(['c', 'a'])
            .len()
            .filter(pl.col('len') >= min_domains)
            .sort('c', 'a')
            .with_row_index('cath_label')
            .select('c', 'a', 'cath_label')
    )

    label_strs = [f'{c}.{a}' for c, a, _ in labels.iter_rows()]
    meta = select_metadata(db, ['cath_labels'])

    if not meta:
        insert_metadata(db, {'cath_labels': label_strs})
    elif meta['cath_labels'] != label_strs:
        raise ValueError(f"the CATH labels have changed since the last run:\nprevious: {meta['cath_labels']}\ncurrent: {label_strs}\nTo use different labels, create a new database.")

    return (
            cath_domains
            .join(
                labels,
                on=['c', 'a'],
                how='inner',
                coalesce=True,
            )
            .drop('c', 'a', 't', 'h', strict=False)
    )

def pick_training_zones(
        db, census, *,
        config: Config,
        get_mmcif_path: Callable[[str], Path],
        polymer_labels: list[str],
        cath_labels: pl.DataFrame,
        progress_factory=tquiet,
):
    cath_labels = _partition_df(cath_labels, ['pdb_id'])

    if config.neighbor_count_threshold:
        neighbor_centers_A = load_neighbors(
                db,
                config.neighbor_geometry, 
                config.neighbor_distance_A,
        )

    class ZoneVisitor(Visitor):

        def __init__(self, structure):
            self.structure = structure
            self.mmcif = mmcif = read_mmcif(get_mmcif_path(structure.pdb_id))
            self.asym_atoms = (
                    mmcif.asym_atoms
                    | f(select_model, structure.model_pdb_ids[0])
                    | f(prune_water)
                    | f(prune_hydrogen)
                    | f(annotate_polymers, mmcif.entities, mmcif.polymers, polymer_labels)
                    | f(annotate_domains, cath_labels[structure.pdb_id])
            )
            self.subchain_counts = count_atoms(
                    self.asym_atoms,
                    ['subchain_id'],
            )

        def propose(self, assembly):
            self.assembly = assembly
            self.atoms = atoms = make_biological_assembly(
                    self.asym_atoms,
                    self.mmcif.assembly_gen,
                    self.mmcif.oper_map,
                    assembly.pdb_id,
            )
            self.kd_tree = kd_tree = make_kd_tree(atoms)

            # These two structures have millions of atoms, and biological 
            # assemblies that leave a large amount of space between each 
            # asymmetric unit.  Together, these issues cause the density 
            # estimator to use an extreme amount of memory.  Because I think 
            # the empty space is a sign that these structures are erroneous, I 
            # decided to exclude these structures from the dataset.
            if self.structure.pdb_id in ['8h2i', '5zz8']:
                return

            if not check_elements(atoms, config.allowed_elements):
                return

            calc_density_atoms_nm3 = make_density_interpolator(
                    prune_nonbiological_residues(
                        atoms,
                        config.nonbiological_residues,
                    ),
                    config.density_check_radius_A,
                    config.density_check_voxel_size_A

                    # Hard-coded hack to handle massive structures.
                    if self.atoms.height < 1e7 else 5
            )

            # If any part of the structure exceeds the density threshold, take 
            # this to mean that the structure probably has one molecule 
            # superimposed on another, and should be excluded from the dataset.
            max_atoms_nm3 = np.max(calc_density_atoms_nm3.values)
            if max_atoms_nm3 > config.density_check_max_atoms_nm3:
                return

            zone_centers_A = calc_zone_centers_A(atoms, config.zone_size_A)
            zone_densities_atoms_nm3 = calc_density_atoms_nm3(zone_centers_A)
            dense_enough = zone_densities_atoms_nm3 > \
                    config.density_check_min_atoms_nm3

            for center_A in zone_centers_A[dense_enough]:
                if not config.neighbor_count_threshold:
                    neighbor_ids = []
                else:
                    neighbor_ids = find_zone_neighbors(
                            calc_density_atoms_nm3,
                            center_A=center_A,
                            offsets_A=neighbor_centers_A,
                            min_density_atoms_nm3=config.density_check_min_atoms_nm3,
                    )
                    if len(neighbor_ids) < config.neighbor_count_threshold:
                        continue

                subchains, subchain_pairs = find_zone_subchains(
                        atoms,
                        kd_tree,
                        asym_counts=self.subchain_counts.filter(
                            pl.col('subchain_id')
                            .is_in(assembly.subchain_pdb_ids),
                        ),
                        center_A=center_A,
                        radius_A=config.subchain_check_radius_A,
                        solo_fraction_of_zone=config.subchain_check_fraction_of_zone,
                        solo_fraction_of_subchain=config.subchain_check_fraction_of_subchain,
                        pair_fraction_of_zone=config.subchain_pair_check_fraction_of_zone,
                        pair_fraction_of_subchain=config.subchain_pair_check_fraction_of_subchain,
                )
                if subchains or subchain_pairs:
                    yield ZoneCandidate(
                            subchains=subchains,
                            subchain_pairs=subchain_pairs,
                            center_A=center_A,
                            neighbor_ids=neighbor_ids,
                    )

        def accept(self, candidates, memento):
            with db:
                if candidates:
                    struct_id = insert_structure(
                            db, self.structure.pdb_id,
                            model_id=self.structure.model_pdb_ids[0],
                    )
                    relevant_atoms = prune_distant_atoms(
                            self.atoms,
                            self.kd_tree,
                            centers_A=np.array([x.center_A for x in candidates]),
                            radius_A=config.atom_inclusion_radius_A,
                            boundary_depth_A=config.atom_inclusion_boundary_depth_A,
                    )
                    assembly_id = insert_assembly(
                            db, struct_id,
                            self.assembly.pdb_id,
                            relevant_atoms,
                    )

                for candidate in candidates:
                    insert_zone(
                            db, assembly_id,
                            center_A=candidate.center_A,
                            neighbor_ids=candidate.neighbor_ids,
                            subchains=candidate.subchains,
                            subchain_pairs=candidate.subchain_pairs,
                    )

                _save_memento(db, memento)

            # These variables should always be filled in by `propose()` before 
            # they're used here.  Delete them so that we don't mistakenly use 
            # old data if `accept()` is called twice in a row for some reason.
            del self.assembly
            del self.atoms
            del self.kd_tree

    @dataclass(kw_only=True)
    class ZoneCandidate(Candidate):
        center_A: Coord
        neighbor_ids: list[int]

    visit_assemblies(
            census,
            ZoneVisitor,
            memento=_load_memento(db),
            progress_factory=progress_factory,
    )

    with db:
        _create_indices(db)
        _delete_memento(db)

def load_neighbors(db, geometry, distance_A):
    neighbors = select_neighbors(db)

    if neighbors.size > 0:
        return neighbors

    centers_A = find_neighbor_centers_A(geometry, distance_A)
    insert_neighbors(db, centers_A)
    return centers_A

def find_neighbor_centers_A(geometry: str, distance_A: float):
    geometries = {
            'tetrahedron faces': tetrahedron_faces,
            'cube faces': cube_faces,
            'octahedron faces': octahedron_faces,
            'octahedron and cube faces': lambda:
                np.vstack([octahedron_faces(), cube_faces()]),
            'dodecahedron faces': dodecahedron_faces,
            'icosahedron faces': icosahedron_faces,
            'icosahedron and dodecahedron faces': lambda:
                np.vstack([icosahedron_faces(), dodecahedron_faces()]),
            'icosphere 2': lambda: icosphere(2)[0],
            'icosphere 3': lambda: icosphere(3)[0],
    }
    return geometries[geometry]() * distance_A

def annotate_polymers(asym_atoms, entities, polymers, labels):
    is_polymer = entities.select(
            entity_id='id',
            is_polymer=pl.col('type') == 'polymer',
    )
    polymer_labels = polymers.select(
            entity_id='entity_id',
            polymer_label=pl.col('type').replace_strict(
                labels,
                pl.int_range(len(labels)),
                default=None,
            ),
    )
    return (
            asym_atoms
            .join(is_polymer, on='entity_id', how='left', coalesce=True)
            .join(polymer_labels, on='entity_id', how='left', coalesce=True)
            .with_columns(
                pl.col('is_polymer').fill_null(False),
            )
    )

def annotate_domains(asym_atoms, cath_labels):
    asym_atoms.write_parquet('asym_atoms.parquet')
    cath_labels.write_parquet('cath_labels.parquet')

    cath_labels = (
            cath_labels
            .explode('seq_ids')
            .select(
                pl.col('chain_id'),
                pl.col('seq_ids').alias('seq_id'),
                pl.col('cath_label'),
            )
            .filter(
                # If the same residue is in multiple domains, arbitrarily 
                # assign it to whichever one comes first.  See Experiment #106.
                pl.struct('chain_id', 'seq_id').is_first_distinct()
            )
    )
    cath_labels.write_parquet('cath_labels_2.parquet')
    annotated_asym_atoms = asym_atoms.join(
            cath_labels,
            on=['chain_id', 'seq_id'],
            how='left',
            coalesce=True,
    )
    annotated_asym_atoms.write_parquet('annotated_asym_atoms.parquet')

    # Make sure the left join didn't accidentally duplicate any atoms.
    assert len(annotated_asym_atoms) == len(asym_atoms)

    return annotated_asym_atoms

def check_elements(
        atoms: Atoms,
        whitelist: float,
):
    if not whitelist:
        return True

    whitelist = [e.upper() for e in whitelist]
    return (
            atoms
            .filter(~pl.col('element').str.to_uppercase().is_in(whitelist))
            .is_empty()
    )

def calc_zone_centers_A(atoms: Atoms, spacing_A: float):
    """
    Choose coordinates for each zone.

    Small proteins are sensitive to this process, because only the most buried 
    regions will have enough atoms to satisfy the density requirements.  This 
    algorithm tries to maximize the chances of including these proteins in the 
    dataset by putting one zone in the exact center of the structure, and 
    aligning the rest to the principal components of the atom coordinates.
    """

    # The variable names in this function use suffixes to identify which frame 
    # coordinates belong to:
    #
    # - `_i`: The input coordinate frame, i.e. the coordinates that are present 
    #   in the input data frame.
    #
    # - `_p`: The coordinate frame defined by a principal components analysis 
    #   (PCA) of the input coordinates.  In this frame, the x-axis is the 
    #   direction of most variation.
    #
    # Note that all coordinates are in units of angstroms.  Normally this is 
    # indicated by the suffix `_A`, but here it's more important to use the 
    # suffix to keep track of the coordinate frame.

    coords_i = get_atom_coords(atoms)

    pca = PCA()
    pca.fit(coords_i)
    frame_ip = pca.components_

    coords_p = coords_i @ frame_ip.T

    def get_axis_p(i):
        low_p = coords_p[:,i].min()
        mid_p = coords_p[:,i].mean()
        high_p = coords_p[:,i].max()

        # We can subtract half the spacing from either end because the 
        # coordinates we're calculating are the centers of the zones.  So any 
        # atoms within half the spacing of the last zone coordinate will still 
        # fall in that zone.
        span_high = high_p - mid_p - (spacing_A / 2)
        span_low =   mid_p - low_p - (spacing_A / 2)

        steps_high = int(ceil(span_high / spacing_A))
        steps_low = int(ceil(span_low / spacing_A))

        start_p = mid_p - (steps_low * spacing_A)

        for i in range(steps_low + steps_high + 1):
            yield start_p + i * spacing_A

    zones_p = np.vstack(
            list(product(
                get_axis_p(0),
                get_axis_p(1),
                get_axis_p(2),
            ))
    )

    return zones_p @ frame_ip

def find_zone_neighbors(
        calc_density_atoms_nm3,
        *,
        center_A: Coord,
        offsets_A: Coords3,
        min_density_atoms_nm3: float,
):
    atoms_nm3 = calc_density_atoms_nm3(center_A + offsets_A)
    indices = np.arange(len(offsets_A))[atoms_nm3 > min_density_atoms_nm3]
    return indices.tolist()

def find_zone_subchains(
        atoms: Atoms,
        kd_tree: KDTree,
        *,
        asym_counts: pl.DataFrame,
        center_A: Coord,
        radius_A: float,
        solo_fraction_of_zone: float,
        solo_fraction_of_subchain: float,
        pair_fraction_of_zone: float,
        pair_fraction_of_subchain: float,
        ):
    """
    Return the subchains and subchain pairs that are present in the specified 
    volume.

    In order for a subchain to be considered "present", it must either comprise 
    a certain fraction of all the atoms in the zone (``*_fraction_of_zone``) or 
    all the atoms in that subchain (``*_fraction_of_subchain``).

    Subchains are identified as ``(subchain_id, symmetry_mate)`` tuples.  This 
    ensure that each subchain is uniquely identified, even if multiple 
    symmetric copies are present.
    """
    atoms = select_nearby_atoms(atoms, kd_tree, center_A, radius_A)
    atom_counts = (
            count_atoms(atoms, ['subchain_id', 'symmetry_mate'])
            .join(asym_counts, on='subchain_id', suffix='_asym')
            .sort('subchain_id', 'symmetry_mate')
            .with_columns(
                fraction_of_zone=pl.col('occupancy') / pl.col('occupancy').sum(),
                fraction_of_subchain=pl.col('occupancy') / pl.col('occupancy_asym'),
            )
    )
    solo_subchains = (
            atom_counts
            .filter(
                    (pl.col('fraction_of_zone') >= solo_fraction_of_zone) |
                    (pl.col('fraction_of_subchain') >= solo_fraction_of_subchain)
            )
            .select('subchain_id', 'symmetry_mate')
            .rows()
    )
    pair_subchains = (
            atom_counts
            .filter(
                (pl.col('fraction_of_zone') >= pair_fraction_of_zone) |
                (pl.col('fraction_of_subchain') >= pair_fraction_of_subchain)
            )
            .select('subchain_id', 'symmetry_mate')
            .rows()
    )
    return solo_subchains, list(combinations(pair_subchains, r=2))

def select_nearby_atoms(atoms, kd_tree, center_A, radius_A):
    center_A = center_A.reshape(1, 3)
    i = one(kd_tree.query_radius(center_A, radius_A))
    return atoms[i]

def count_atoms(atoms, group_by):
    return (
            atoms
            .group_by(*group_by)
            .agg(pl.col('occupancy').sum())
    )

def prune_nonbiological_residues(
        atoms: Atoms,
        blacklist: list[str],
):
    non_bio = (
            pl.col('comp_id').is_in(blacklist)
            & ~pl.col('is_polymer')
    )
    return atoms.filter(~non_bio)

def prune_distant_atoms(
        atoms: Atoms,
        kd_tree: KDTree,
        *,
        centers_A: Coords3,
        radius_A: float,
        boundary_depth_A: float,
):

    def query_radius(radius_A):
        indices = kd_tree.query_radius(centers_A, radius_A)
        return np.array(sorted(set(flatten(indices))))

    i_include = query_radius(radius_A)
    i_boundary = query_radius(radius_A + boundary_depth_A)

    out_of_bounds = np.full(atoms.height, True)
    out_of_bounds[i_include] = False

    atoms = atoms.with_columns(out_of_bounds=out_of_bounds)
    return atoms[i_boundary]

def make_kd_tree(atoms):
    return KDTree(get_atom_coords(atoms))


def _make_config(config, census_path, cath_paths):

    def validator(f):
        def wrapper(key):
            if key not in config:
                raise ValueError(f"{key!r} not specified")
            return f(key, config[key])
        return wrapper

    @validator
    def string(key, value):
        return value

    @validator
    def integer(key, value):
        try:
            return int(value)
        except:
            raise ValueError(f"{key!r} must be an integer, not {value!r}")

    @validator
    def quantity(key, value):
        try:
            return float(value)
        except:
            raise ValueError(f"{key!r} must be a number, not {value!r}")

    @validator
    def fraction(key, value):
        x = quantity(key)
        if not 0 <= x <= 1:
            raise ValueError(f"{key!r} must be between 0 and 1, not {x}")
        return x

    @validator
    def elements(key, value):
        if not isinstance(value, list):
            raise ValueError(f"{key!r} must be a list of elements, not {value!r}")

        user_elements = [x.upper() for x in value]
        known_elements = {e.symbol.upper() for e in periodictable.elements}
        unknown_elements = set(user_elements) - known_elements

        if unknown_elements:
            raise ValueError(f"{key!r} must be a list of elements, not {unknown_elements!r}")

        return user_elements

    @validator
    def residues(key, value):
        if value == '-':
            return []

        path = Path(value)
        if not path.exists():
            raise ValueError(f"{key!r} must be a path, but the following doesn't exist: {path}")

        residues = [
                x.strip()
                for x in path.read_text().splitlines()
        ]

        if too_long := [x for x in residues if len(x) > 5]:
            raise ValueError(f"{key!r} must be a list of residue names, but the following are too long: {too_long}")

        return residues

    validators = dict(
            zone_size_A=quantity,

            density_check_radius_A=quantity,
            density_check_voxel_size_A=quantity,
            density_check_min_atoms_nm3=quantity,
            density_check_max_atoms_nm3=quantity,

            subchain_check_radius_A=quantity,
            subchain_check_fraction_of_zone=fraction,
            subchain_check_fraction_of_subchain=fraction,
            subchain_pair_check_fraction_of_zone=fraction,
            subchain_pair_check_fraction_of_subchain=fraction,

            neighbor_geometry=string,
            neighbor_distance_A=quantity,
            neighbor_count_threshold=integer,

            allowed_elements=elements,
            nonbiological_residues=residues,

            atom_inclusion_radius_A=quantity,
            atom_inclusion_boundary_depth_A=quantity,

            cath_min_domains=integer,
    )

    neighbor_keys = {
            'neighbor_geometry',
            'neighbor_distance_A',
            'neighbor_count_threshold',
    }
    if not set(config) & neighbor_keys:
        config |= dict(
            neighbor_geometry='none',
            neighbor_distance_A=0,
            neighbor_count_threshold=0,
        )

    kwargs = {k: v(k) for k, v in validators.items()}

    if unexpected_keys := set(config) - set(kwargs):
        raise ValueError(f"the following keys aren't recognized: {unexpected_keys!r}")

    kwargs['census_md5'] = _md5_hash(census_path)
    kwargs['cath_md5'] = {
            k: _md5_hash(v)
            for k, v in cath_paths.items()
    }

    return Config(**kwargs)

def _select_config(db):
    keys = [x.name for x in fields(Config)]
    config = select_metadata(db, keys)

    if not config:
        return None

    if missing_keys := set(keys) - set(config):
        raise ValueError(f"failed to load config from database.\nThe following required parameters were not found: {missing_keys!r}")

    return Config(**config)

def _insert_config(db, config):
    insert_metadata(db, asdict(config))

def _md5_hash(db_path: Path | str):
    hash = md5()
    hash.update(Path(db_path).read_bytes())
    return hash.hexdigest()

def _save_memento(db, memento):
    upsert_metadata(db, {'census_memento': pickle.dumps(memento)})

def _load_memento(db):
    try:
        memento = select_metadatum(db, 'census_memento')
    except KeyError:
        return None
    else:
        return pickle.loads(memento)

def _delete_memento(db):
    delete_metadatum(db, 'census_memento')

def _create_indices(db):
    db.execute('''\
            CREATE INDEX zone_neighbor_zone_id
            ON zone_neighbor (zone_id)
    ''')

def _partition_df(df, cols):
    return defaultdict(
            lambda: pl.DataFrame([], df.schema),
            _flatten_keys(df.partition_by(cols, as_dict=True))
    )

def _flatten_keys(d):
    return {one(k): v for k, v in d.items()}

if __name__ == '__main__':
    main()
