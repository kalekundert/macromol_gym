import macromol_voxelize as mmvox
import polars as pl
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
from functools import cache
from more_itertools import one, unique_everseen as unique
from math import ceil

def make_density_interpolator(atoms, radius_A, voxel_size_A):
    if atoms.is_empty():
        return RegularGridInterpolator(
                points=([0], [0], [0]),
                values=[[[0]]],
                bounds_error=False,
                fill_value=0,
        )

    volume_nm3 = (voxel_size_A / 10)**3

    group_densities = (
            atoms
            .with_columns(
                (pl.col('x', 'y', 'z') / voxel_size_A)
                .floor()
                .cast(int)
            )
            .group_by(['x', 'y', 'z'])
            .agg(
                atoms_per_nm3=pl.col('occupancy').sum() / volume_nm3
            )
    )

    voxels = group_densities.select('x', 'y', 'z').to_numpy()
    min_voxel = np.min(voxels, axis=0)
    max_voxel = np.max(voxels, axis=0)
    indices = voxels - min_voxel

    voxel_densities = np.zeros(max_voxel - min_voxel + 1)
    voxel_densities[tuple(indices.T)] = group_densities['atoms_per_nm3']

    sphere_kernel = make_sphere_kernel(radius_A, voxel_size_A)
    sphere_densities = convolve(voxel_densities, sphere_kernel)

    # Account for the fact that the convolution will increase the size of the 
    # image.
    pad = (one(unique(sphere_kernel.shape)) - 1) // 2
    min_voxel -= pad
    max_voxel += pad

    dist_to_voxel_center = 0.5
    grid_points = [
            voxel_size_A * (np.arange(a, b) + dist_to_voxel_center)
            for a, b in zip(min_voxel, max_voxel + 1)
    ]

    return RegularGridInterpolator(
            points=grid_points,
            values=sphere_densities,
            bounds_error=False,
            fill_value=0,
    )
    
@cache
def make_sphere_kernel(radius_A, voxel_size_A):
    # Force the kernel to have an odd number of voxels in each dimension.  This 
    # ensures that the center of the sphere falls in the center of a voxel (as 
    # opposed to an edge or a corner).
    length_voxels = ceil(2 * radius_A / voxel_size_A)
    if length_voxels % 2 == 0:
        length_voxels += 1
    
    # This is kind-of an abuse of `mmvox`, since we're voxelizing a giant 
    # sphere (here referred to as a "pseudoatom") and not atoms.  But it still 
    # works, even though the function names don't exactly make sense.
    img_params = mmvox.ImageParams(
            channels=1,
            grid=mmvox.Grid(
                length_voxels=length_voxels,
                resolution_A=voxel_size_A,
                center_A=[0, 0, 0],
            ),
    )
    pseudoatom = pl.DataFrame([
            dict(x=0, y=0, z=0, radius_A=radius_A, occupancy=1, channels=[0]),
    ])
    return mmvox.image_from_atoms(pseudoatom, img_params)[0]

