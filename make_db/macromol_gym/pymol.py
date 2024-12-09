import pymol
import numpy as np
import macromol_dataframe as mmdf
import re

from .database_io import open_db, select_zone_pdb_ids, select_zone_center_A
from .pick import calc_zone_centers_A, find_neighbor_centers_A
from contextlib import contextmanager
from pymol import cmd
from math import pi

def show_zone(db_path, zone_id):
    db = open_db(db_path)
    zone_id = int(zone_id)

    pdb_ids = select_zone_pdb_ids(db, zone_id)
    center_A = select_zone_center_A(db, zone_id)

    cmd.set('assembly', pdb_ids['assembly_pdb_id'])
    cmd.fetch(pdb_ids['struct_pdb_id'])
    cmd.pseudoatom('zone', pos=list(center_A))

pymol.cmd.extend('show_zone', show_zone)

def _sum_occ(sele):
    with sele_or_pseudoatom(sele) as sele:
        counter = {'q': 0}
        cmd.iterate(sele, 'counter["q"] += q', space=locals())
        return counter['q']

def sum_occ(sele):
    print(_sum_occ(sele), 'atoms')

pymol.cmd.extend('sum_occ', sum_occ)
cmd.auto_arg[0]['sum_occ'] = cmd.auto_arg[0]['zoom']

def _density(center, radius_A):
    with sele_or_pseudoatom(center) as sele:
        radius_nm = float(radius_A) / 10
        volume_nm3 = 4/3 * pi * radius_nm**3
        occupancy = _sum_occ(f'all within {radius_A} of ({sele})')
        return occupancy / volume_nm3

def density(center, radius_A):
    cmd.iterate_state(0, center, 'print(f"{x:.3f} {y:.3f} {z:.3f}")')
    print(_density(center, radius_A), 'atoms/nm^3')

pymol.cmd.extend('density', density)
cmd.auto_arg[0]['density'] = cmd.auto_arg[0]['zoom']

def show_centers(spacing_A=10, density_target_atoms_nm3=35, density_radius_A=15, sele='all', state=0):
    cmd.delete('zone_centers')

    spacing_A = float(spacing_A)
    density_target_atoms_nm3 = float(density_target_atoms_nm3)
    density_radius_A = float(density_radius_A)

    atoms = mmdf.from_pymol(sele, state)
    zone_centers_A = calc_zone_centers_A(atoms, spacing_A)

    centers_above_density_target = 0

    for i, center_A in enumerate(zone_centers_A):
        density_atoms_nm3 = _density(center_A, density_radius_A)

        if density_atoms_nm3 >= density_target_atoms_nm3:
            centers_above_density_target += 1
            color = 'yellow'
            print(i, center_A)
        else:
            shade = 10 * int(10 * density_atoms_nm3 / density_target_atoms_nm3)
            color = f'gray{shade}'

        cmd.pseudoatom(
                'zone_centers',
                pos=repr(center_A.tolist()),
                resi=i,
                color=color,
        )

    print(f"Centers with >{density_target_atoms_nm3} atoms/nm^3: {centers_above_density_target}")

pymol.cmd.extend('show_centers', show_centers)

def show_neighbors(sele='sele', geometry='icosahedron faces', distance_A=30, density_target_atoms_nm3=35, density_radius_A=15, state=0):
    cmd.delete('neighbor_centers')

    distance_A = float(distance_A)
    density_target_atoms_nm3 = float(density_target_atoms_nm3)
    density_radius_A = float(density_radius_A)

    neighbors = find_neighbor_centers_A(geometry, distance_A)

    center_A = np.zeros(3)
    cmd.iterate_state(state, sele, 'center_A[:] = (x, y, z)', space=locals())

    neighbors_above_density_target = 0

    for i, offset_A in enumerate(neighbors):
        neighbor_A = center_A + offset_A
        density_atoms_nm3 = _density(neighbor_A, density_radius_A)

        if density_atoms_nm3 >= density_target_atoms_nm3:
            neighbors_above_density_target += 1
            color = 'yellow'
            print(i, neighbor_A)
        else:
            shade = 10 * int(10 * density_atoms_nm3 / density_target_atoms_nm3)
            color = f'gray{shade}'

        cmd.pseudoatom(
                'neighbor_centers',
                pos=repr(neighbor_A.tolist()),
                resi=i,
                color=color,
        )

    print(f"Neighbors with >{density_target_atoms_nm3} atoms/nm^3: {neighbors_above_density_target}")

pymol.cmd.extend('show_neighbors', show_neighbors)
cmd.auto_arg[0]['show_neighbors'] = cmd.auto_arg[0]['zoom']

@contextmanager
def sele_or_pseudoatom(sele_or_xyz):
    if isinstance(sele_or_xyz, str):
        sele = sele_or_xyz
        yield sele
    else:
        xyz = sele_or_xyz
        with tmp_pseudoatom(xyz) as sele:
            yield sele

@contextmanager
def tmp_pseudoatom(xyz):
    name = '__xyz'
    names = cmd.get_names('all')

    if name in names:
        pattern = fr'^{name}(\d+)$'
        i = 1 + max(
                (m.group(1) for x in names if (m := re.match(pattern, x))),
                default=0,
        )
        name = f'{name}{i}'

    cmd.pseudoatom(name, pos=list(xyz))

    try:
        yield name
    finally:
        cmd.delete(name)



