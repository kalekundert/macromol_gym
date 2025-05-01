import lightning as L
import macromol_voxelize as mmvox
import logging

from ..torch.data import MacromolDataset
from ..images import ImageParams
from ..samples import MakeSampleFunc
from ..utils import get_num_workers
from torch.utils.data import DataLoader
from torch_deterministic import InfiniteSampler, collate_rngs
from functools import partial

from pathlib import Path
from typing import Optional, Callable
from numpy.typing import ArrayLike

log = logging.getLogger('macromol_gym')

class MacromolDataModule(L.LightningDataModule):

    def __init__(
            self,
            db_path: Path,
            *,

            # Dataset parameters:
            make_sample: Optional[MakeSampleFunc] = None,
            max_difficulty: float = 1,
            truncate_dataset: Optional[int] = None,

            # Data loader parameters:
            batch_size: int,
            train_epoch_size: Optional[int] = None,
            val_epoch_size: Optional[int] = None,
            test_epoch_size: Optional[int] = None,
            identical_epochs: bool = False,
            num_workers: Optional[int] = None,
            collate_fn: Optional[Callable] = None,
    ):
        super().__init__()

        num_workers = get_num_workers(num_workers)

        def make_dataloader(split, epoch_size):
            dataset = MacromolDataset(
                db_path=db_path,
                split=split,
                make_sample=make_sample,
                max_difficulty=max_difficulty if split == 'train' else 1,
                truncate_dataset=truncate_dataset,
            )

            split_size = len(dataset)
            if epoch_size is None:
                epoch_size = split_size
            if callable(epoch_size):
                epoch_size = epoch_size(split_size)

            log.info(
                    "configure dataloader: split=%s split_size=%d epoch_size=%d batch_size=%d num_workers=%d",
                    split, split_size, epoch_size, batch_size, num_workers,
            )

            sampler = InfiniteSampler(
                    epoch_size,
                    shuffle=True,
                    shuffle_size=split_size,
                    increment_across_epochs=(
                        (split == 'train') and (not identical_epochs)
                    ),
            )

            return DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collate_fn or collate_rngs,

                    # For some reason I don't understand, my worker processes 
                    # get killed by SIGABRT if I use the default 'fork' 
                    # context.  The behavior is very sensitive to all sorts of 
                    # small changes in the code (e.g. `debug()` calls), which 
                    # makes me think it's some sort of race condition.
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

class MacromolImageDataModule(MacromolDataModule):

    def __init__(
            self,
            db_path: Path,
            *,

            # Image parameters:
            image_length_voxels: int,
            image_resolution_A: float,
            atom_radius_A: Optional[float] = None,
            element_channels: list[str],
            normalize_mean: ArrayLike = 0,
            normalize_std: ArrayLike = 1,

            # Dataset parameters:
            make_sample: Optional[MakeSampleFunc] = None,
            max_difficulty: float = 1,
            truncate_dataset: Optional[int] = None,

            # Data loader parameters:
            batch_size: int,
            train_epoch_size: Optional[int] = None,
            val_epoch_size: Optional[int] = None,
            test_epoch_size: Optional[int] = None,
            identical_epochs: bool = False,
            num_workers: Optional[int] = None,
            collate_fn: Optional[Callable] = None,
    ):
        super().__init__(
                db_path=db_path,
                make_sample=partial(
                    make_sample,
                    img_params=ImageParams(
                        grid=mmvox.Grid(
                            length_voxels=image_length_voxels,
                            resolution_A=image_resolution_A,
                        ),
                        atom_radius_A=atom_radius_A,
                        element_channels=element_channels,
                        normalize_mean=normalize_mean,
                        normalize_std=normalize_std,
                    ),
                ),
                max_difficulty=max_difficulty,
                truncate_dataset=truncate_dataset,
                batch_size=batch_size,
                train_epoch_size=train_epoch_size,
                val_epoch_size=val_epoch_size,
                test_epoch_size=test_epoch_size,
                identical_epochs=identical_epochs,
                num_workers=num_workers,
                collate_fn=collate_fn,
        )

