import macromol_gym as mmg
import macromol_dataframe as mmdf
import parametrize_from_file as pff
import numpy as np
import pytest

from test_pick import atoms
from macromol_dataframe.testing import coords, vector
from pytest import approx
from pathlib import Path
from math import pi

MOCK_PDB = Path(__file__).parent / 'pdb'

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            radius_A=float,
            voxel_size_A=float,
            coords_A=coords,
            expected_atoms_nm3=vector,
            allowed_err_atoms_nm3=float,
        )
)
def test_density_manual(
        atoms,
        radius_A,
        voxel_size_A,
        coords_A,
        expected_atoms_nm3,
        allowed_err_atoms_nm3,
):
    calc_density_atoms_nm3 = mmg.make_density_interpolator(
            atoms, radius_A, voxel_size_A,
    )
    assert calc_density_atoms_nm3(coords_A) == approx(
            expected_atoms_nm3,
            abs=allowed_err_atoms_nm3,
    )

@pytest.mark.parametrize('mmcif_path', MOCK_PDB.glob('*.cif.gz'))
def test_density_auto(mmcif_path):
    atoms = mmdf.read_asymmetric_unit(mmcif_path)
    kd_tree = mmg.make_kd_tree(atoms)
    coords_A = mmg.calc_zone_centers_A(atoms, spacing_A=10)
    radius_A = 15

    expected = np.array([
            calc_density_atoms_nm3(atoms, kd_tree, coord_A, radius_A)
            for coord_A in coords_A
    ])

    interp_density_atoms_nm3 = mmg.make_density_interpolator(
            atoms, radius_A, voxel_size_A=2,
    )

    # See expt #55.  With 2Å voxels, the error was usually below 0.5 atoms/nm³, 
    # and almost always below 2 atoms/nm³.
    assert interp_density_atoms_nm3(coords_A) == approx(expected, abs=2)


def calc_density_atoms_nm3(atoms, kd_tree, center_A, radius_A):
    atoms = mmg.select_nearby_atoms(
            atoms,
            kd_tree,
            center_A,
            radius_A,
    )
    volume_nm3 = calc_sphere_volume_nm3(radius_A)
    return atoms['occupancy'].sum() / volume_nm3

def calc_sphere_volume_nm3(radius_A):
    return 4/3 * pi * (radius_A / 10)**3

