import torch
import numpy as np

from ..database_io import (
        open_db, select_split, select_zone_atoms, select_curriculum,
)
from ..dataset import (
        NeighborParams, ImageParams,
        get_neighboring_frames, image_from_atoms, log
)
from macromol_dataframe import Atoms, transform_atom_coords
from torch.utils.data import Dataset
from reprfunc import repr_from_init
from functools import partial
from pathlib import Path

from typing import Any, Callable, Optional

class NeighborDataset(Dataset):
    
    def __init__(
            self,
            *,
            db_path: Path,
            split: str,
            neighbor_params: NeighborParams,
            input_from_atoms: Callable[[Atoms], Any],
            max_difficulty: float = 1,
    ):
        # Don't store a connection to the database in the constructor.  The 
        # constructor runs in the parent process, after which the instantiated 
        # dataset object is sent to the worker process.  If the worker process 
        # was forked, this would cause weird deadlock/race condition problems!
        # If the worker process was spawned, this would require pickling the 
        # connection, which isn't possible.
        self.db_path = db_path
        self.db = None

        db = open_db(db_path)
        self.zone_ids = select_split(db, split)

        if max_difficulty < 1:
            n = len(self.zone_ids)
            self.zone_ids = _filter_zones_by_curriculum(
                    self.zone_ids,
                    select_curriculum(db, max_difficulty),
            )
            log.info("remove difficult training examples: split=%s max_difficulty=%s num_examples_before_filter=%d num_examples_after_filter=%d", split, max_difficulty, n, len(self.zone_ids))

        self.neighbor_params = neighbor_params
        self.input_from_atoms = input_from_atoms

    def __len__(self):
        return len(self.zone_ids)

    def __getitem__(self, i):
        if self.db is None:
            self.db = open_db(self.db_path)
            self.db_cache = {}

        zone_id, frame_ia, frame_ab, b = get_neighboring_frames(
                self.db, i,
                zone_ids=self.zone_ids,
                neighbor_params=self.neighbor_params,
                db_cache=self.db_cache,
        )

        atoms_i = select_zone_atoms(self.db, zone_id)
        atoms_a = transform_atom_coords(atoms_i, frame_ia)
        atoms_b = transform_atom_coords(atoms_a, frame_ab)

        input_a = self.input_from_atoms(atoms_a)
        input_b = self.input_from_atoms(atoms_b)
        input_ab = np.stack([input_a, input_b])

        return torch.from_numpy(input_ab).float(), torch.tensor(b)

class CnnNeighborDataset(NeighborDataset):

    def __init__(
            self,
            db_path: Path,
            split: str,
            neighbor_params: NeighborParams,
            img_params: ImageParams,
            max_difficulty: float = 1,
    ):
        # This class is slightly opinionated about how images should be 
        # created.  This allows it to provide a simple---but not fully 
        # general---API for common image parameters.  If you need to do 
        # something beyond the scope of this API, use `NeighborDataset` 
        # directly.
        super().__init__(
                db_path=db_path,
                split=split,
                neighbor_params=neighbor_params,
                input_from_atoms=partial(
                    image_from_atoms,
                    img_params=img_params,
                ),
                max_difficulty=max_difficulty,
        )

class InfiniteSampler:
    """
    Draw reproducible samples from an infinite map-style dataset, i.e. a 
    dataset that accepts integer indices of any size.

    Arguments:
        epoch_size:
            The number of examples to include in each epoch.  Note that, 
            because the dataset is assumed to have an infinite number of 
            examples, this parameter doesn't have to relate to the amount of 
            data in the dataset.  Instead, it usually just specifies how often 
            "end of epoch" tasks, like running the validation set or saving 
            checkpoints, are performed.

        start_epoch:
            The epoch number to base the random seed on, if *shuffle* is 
            enabled and *increment_across_epochs* is not.  Note also that if 
            the training environment doesn't call `set_epoch()` before every 
            epoch, which every sane training environment should, then this 
            setting will determine the random seed used for *shuffle* 
            regardless of *increment_across_epochs*.

        increment_across_epochs:
            If *False*, yield the same indices in the same order every epoch.  
            If *True*, yield new indices in every epoch, without skipping any.  
            This option is typically enabled for the training set, and disabled 
            for the validation and test sets.

        shuffle:
            If *True*, shuffle the indices within each epoch.  The shuffling is 
            guaranteed to be a deterministic function of the epoch number, as 
            set by `set_epoch()`.  This means that every training run will 
            visit the same examples in the same order.

        shuffle_size:
            The number of indices to consider when shuffling.  For example, 
            with a shuffle size of 5, the first 5 indices would be some 
            permutation of 0-4, the second 5 would be some permutation of 5-9, 
            and so on.  Note that this setting is independent of the epoch 
            size.  For example, with a shuffle size of 5 and an epoch size of 
            3, the first epoch would consist of three values between 0-4.  The 
            second epoch would begin with the two values between 0-4 that 
            weren't in the first epoch, then end with a value between 5-9.  The 
            third epoch would begin with the unused values between 5-9, and so 
            on.  That said, by default the shuffle size is the same as the 
            epoch size.

        rng_factory:
            A factory function that creates a random number generator from a 
            given integer seed.  This generator is only used to shuffle the 
            indices, and only then if *shuffle* is enabled.
    """

    def __init__(
            self,
            epoch_size: int,
            *,
            start_epoch: int = 0,
            increment_across_epochs: bool = True,
            shuffle: bool = False,
            shuffle_size: Optional[int] = None,
            rng_factory: Callable[[int], np.random.Generator] = np.random.default_rng,
    ):
        self.epoch_size = epoch_size
        self.curr_epoch = start_epoch
        self.increment_across_epochs = increment_across_epochs
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size or epoch_size
        self.rng_factory = rng_factory

    def __iter__(self):
        n = self.epoch_size
        i = n * self.curr_epoch

        if not self.shuffle:
            yield from range(i, i+n)
        else:
            yield from _iter_shuffled_indices(
                    self.rng_factory,
                    self.shuffle_size,
                    i, i+n,
            )

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch: int):
        if self.increment_across_epochs:
            self.curr_epoch = epoch

    __repr__ = repr_from_init

def _filter_zones_by_curriculum(zone_ids, curriculum):
    mask = np.isin(zone_ids, curriculum)
    return zone_ids[mask]

def _iter_shuffled_indices(rng_factory, n, i, j):
    while True:
        seed = i // n
        rng = rng_factory(seed)

        i0 = n * seed; i1 = i0 + n
        indices = rng.permutation(range(i0, i1))
        
        start = i - i0
        end = j - i0

        if end > n:
            yield from indices[start:]
            i = i1
        else:
            yield from indices[start:end]
            return


