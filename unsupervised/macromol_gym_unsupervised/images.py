import macromol_voxelize as mmvox
import numpy as np

from dataclasses import dataclass
from numpy.typing import ArrayLike

@dataclass
class ImageParams:
    grid: mmvox.Grid
    atom_radius_A: float
    element_channels: list[str]
    ligand_channel: bool = False
    normalize_mean: ArrayLike = 0
    normalize_std: ArrayLike = 1

def image_from_atoms(atoms, img_params):

    def assign_channels(atoms):
        channels = img_params.element_channels
        atoms = mmvox.set_atom_channels_by_element(atoms, channels)
        atoms = mmvox.set_atom_radius_A(atoms, img_params.atom_radius_A)

        if img_params.ligand_channel:
            atoms = mmvox.add_atom_channel_by_expr(
                    atoms,
                    expr='is_polymer',
                    channel=len(channels),
            )

        return atoms

    mmvox_img_params = mmvox.ImageParams(
            channels=(
                len(img_params.element_channels) + img_params.ligand_channel
            ),
            grid=img_params.grid,
            process_filtered_atoms=assign_channels,
            max_radius_A=img_params.atom_radius_A,
    )
    img = mmvox.image_from_atoms(atoms, mmvox_img_params)

    _normalize_image(img, img_params.normalize_mean, img_params.normalize_std)

    return img

def _normalize_image(img, mean, std):
    # I haven't actually done any benchmarking, but this post [1] suggests that 
    # in-place operations are ≈2-3x faster for arrays with >10K elements.  For 
    # reference, a 21x21x21 image with 6 channels would have 55K voxels.
    #
    # [1]: https://stackoverflow.com/questions/57024802/numpy-in-place-operation-performance

    if mean != 0:
        img -= np.asarray(mean).reshape(-1, 1, 1, 1)
    if std != 1:
        img /= np.asarray(std).reshape(-1, 1, 1, 1)


