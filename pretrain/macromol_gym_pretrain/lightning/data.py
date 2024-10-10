import lightning as L
import macromol_voxelize as mmvox
import logging

from ..torch.data import CnnNeighborDataset, NeighborParams, ImageParams
from ..torch.infinite_sampler import InfiniteSampler
from ..utils import get_num_workers
from ..geometry import cube_faces
from torch.utils.data import DataLoader

from typing import Optional
from numpy.typing import ArrayLike

log = logging.getLogger('macromol_gym_pretrain')

class CnnNeighborDataModule(L.LightningDataModule):

    def __init__(
            self,
            db_path,
            *,

            # Neighbor parameters
            neighbor_padding_A: Optional[float] = None,
            neighbor_distance_A: Optional[float] = None,
            noise_max_distance_A: float,
            noise_max_angle_deg: float,

            # Image parameters
            direction_candidates: ArrayLike = cube_faces(),
            grid_length_voxels: int,
            grid_resolution_A: float,
            atom_radius_A: Optional[float] = None,
            element_channels: list[str],
            ligand_channel: bool,
            normalize_mean: ArrayLike = 0,
            normalize_std: ArrayLike = 1,

            # Curriculum parameters
            max_difficulty: float = 1,

            # Data loader parameters
            batch_size: int,
            train_epoch_size: Optional[int] = None,
            val_epoch_size: Optional[int] = None,
            test_epoch_size: Optional[int] = None,
            identical_epochs: bool = False,
            num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        grid = mmvox.Grid(
                length_voxels=grid_length_voxels,
                resolution_A=grid_resolution_A,
        )

        if neighbor_padding_A and neighbor_distance_A:
            raise ValueError("must not specify both `neighbor_padding_A` and `neighbor_distance_A`")
        elif neighbor_padding_A is None and neighbor_distance_A is None:
            raise ValueError("must specify either `neighbor_padding_A` or `neighbor_distance_A`")
        elif neighbor_padding_A is not None:
            neighbor_distance_A = grid.length_A + neighbor_padding_A

        if atom_radius_A is None:
            atom_radius_A = grid_resolution_A / 2

        datasets = {
                split: CnnNeighborDataset(
                    db_path=db_path,
                    split=split,
                    max_difficulty=max_difficulty if split == 'train' else 1,
                    neighbor_params=NeighborParams(
                        direction_candidates=direction_candidates,
                        distance_A=neighbor_distance_A,
                        noise_max_distance_A=noise_max_distance_A if split == 'train' else 0,
                        noise_max_angle_deg=noise_max_angle_deg if split == 'train' else 0,
                    ),
                    img_params=ImageParams(
                        grid=grid,
                        atom_radius_A=atom_radius_A,
                        element_channels=element_channels,
                        ligand_channel=ligand_channel,
                        normalize_mean=normalize_mean,
                        normalize_std=normalize_std,
                    ),
                )
                for split in ['train', 'val', 'test']
        }
        num_workers = get_num_workers(num_workers)

        def make_dataloader(split, epoch_size):
            log.info("configure dataloader: split=%s num_workers=%d", split, num_workers)

            sampler = InfiniteSampler(
                    epoch_size or len(datasets[split]),
                    shuffle=True,
                    shuffle_size=len(datasets[split]),
                    increment_across_epochs=(
                        (split == 'train') and (not identical_epochs)
                    ),
            )

            return DataLoader(
                    dataset=datasets[split],
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    
                    # For some reason I don't understand, my worker processes
                    # get killed by SIGABRT if I use the 'fork' context.  The
                    # behavior is very sensitive to all sorts of small changes
                    # in the code (e.g. `debug()` calls), which makes me think
                    # it's some sort of race condition.
                    multiprocessing_context='spawn' if num_workers else None,

                    pin_memory=True,
                    drop_last=True,
            )

        self._train_dataloader = make_dataloader('train', train_epoch_size)
        self._val_dataloader = make_dataloader('val', val_epoch_size)
        self._test_dataloader = make_dataloader('test', test_epoch_size)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader



