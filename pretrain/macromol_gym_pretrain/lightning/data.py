import macromol_voxelize as mmvox

from macromol_gym_unsupervised.lightning import (
        MacromolDataModule, MakeSampleFuncs, require_split_dict,
)
from .. import ImageParams, NeighborParams, cube_faces
from functools import partial

from typing import Optional
from numpy.typing import ArrayLike
from pathlib import Path

class MacromolNeighborImageDataModule(MacromolDataModule):

    def __init__(
            self,
            db_path: Path,
            *,

            # Neighbor parameters:
            direction_candidates: ArrayLike = cube_faces(),
            neighbor_padding_A: Optional[float] = None,
            neighbor_distance_A: Optional[float] = None,
            noise_max_distance_A: float,
            noise_max_angle_deg: float,
            add_noise_during_validation: bool = False,

            # Image parameters:
            image_length_voxels: int,
            image_resolution_A: float,
            atom_radius_A: Optional[float] = None,
            element_channels: list[str],
            normalize_mean: ArrayLike = 0,
            normalize_std: ArrayLike = 1,

            # Dataset parameters:
            make_sample: MakeSampleFuncs = None,
            max_difficulty: float = 1,

            # Data loader parameters:
            batch_size: int,
            train_epoch_size: Optional[int] = None,
            val_epoch_size: Optional[int] = None,
            test_epoch_size: Optional[int] = None,
            identical_epochs: bool = False,
            num_workers: Optional[int] = None,
    ):
        grid = mmvox.Grid(
                length_voxels=image_length_voxels,
                resolution_A=image_resolution_A,
        )

        if neighbor_padding_A and neighbor_distance_A:
            raise ValueError("must not specify both `neighbor_padding_A` and `neighbor_distance_A`")
        elif neighbor_padding_A is None and neighbor_distance_A is None:
            raise ValueError("must specify either `neighbor_padding_A` or `neighbor_distance_A`")
        elif neighbor_padding_A is not None:
            neighbor_distance_A = grid.length_A + neighbor_padding_A

        def make_sample_for_split(split, noise):
            return partial(
                    require_split_dict(make_sample)[split],
                    img_params=ImageParams(
                        grid=grid,
                        atom_radius_A=atom_radius_A,
                        element_channels=element_channels,
                        normalize_mean=normalize_mean,
                        normalize_std=normalize_std,
                    ),
                    neighbor_params=NeighborParams(
                        direction_candidates=direction_candidates,
                        distance_A=neighbor_distance_A,
                        noise_max_distance_A=noise_max_distance_A if noise else 0,
                        noise_max_angle_deg=noise_max_angle_deg if noise else 0,
                    ),
            )

        super().__init__(
                db_path=db_path,
                make_sample={
                    'train': make_sample_for_split('train', noise=True),
                    'val': make_sample_for_split('val', noise=add_noise_during_validation),
                    'test': make_sample_for_split('test', noise=add_noise_during_validation),
                },
                max_difficulty=max_difficulty,
                batch_size=batch_size,
                train_epoch_size=train_epoch_size,
                val_epoch_size=val_epoch_size,
                test_epoch_size=test_epoch_size,
                identical_epochs=identical_epochs,
                num_workers=num_workers,
        )

