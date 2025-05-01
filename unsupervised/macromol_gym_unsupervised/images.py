import macromol_voxelize as mmvox
import numpy as np

from pipeline_func import f
from dataclasses import dataclass

from typing import Optional
from numpy.typing import ArrayLike

@dataclass(kw_only=True)
class ImageParams:
    grid: mmvox.Grid
    atom_radius_A: Optional[float] = None
    element_channels: list[str]
    ligand_channel: bool = False
    normalize_mean: ArrayLike = 0
    normalize_std: ArrayLike = 1

    def resolve_atom_radius_A(self):
        if self.atom_radius_A is None:
            return self.grid.resolution_A / 2
        else:
            return self.atom_radius_A

def image_from_atoms(atoms, img_params):
    atom_radius_A = img_params.resolve_atom_radius_A()

    mmvox_img_params = mmvox.ImageParams(
            channels=(
                len(img_params.element_channels) + img_params.ligand_channel
            ),
            grid=img_params.grid,
            max_radius_A=atom_radius_A,
    )

    img_atoms = (
            atoms
            | f(mmvox.discard_atoms_outside_image, mmvox_img_params)
            | f(mmvox.set_atom_channels_by_element, img_params.element_channels)
            | f(mmvox.set_atom_radius_A, atom_radius_A)
    )

    if img_params.ligand_channel:
        img_atoms = mmvox.add_atom_channel_by_expr(
                img_atoms,
                expr='is_polymer',
                channel=len(img_params.element_channels),
        )

    img = mmvox.image_from_all_atoms(img_atoms, mmvox_img_params)

    normalize_image_in_place(
            img,
            img_params.normalize_mean,
            img_params.normalize_std,
    )

    return img, img_atoms

def normalize_image_in_place(img, mean, std):
    # I haven't actually done any benchmarking, but this post [1] suggests that 
    # in-place operations are ≈2-3x faster for arrays with >10K elements.  For 
    # reference, a 21x21x21 image with 6 channels would have 55K voxels.
    #
    # [1]: https://stackoverflow.com/questions/57024802/numpy-in-place-operation-performance

    if mean != 0:
        img -= np.asarray(mean).reshape(-1, 1, 1, 1)
    if std != 1:
        img /= np.asarray(std).reshape(-1, 1, 1, 1)


