import torch
import numpy as np

from ..database_io import (
        open_db, select_split, select_zone_atoms, select_curriculum,
)
from ..neighbors import NeighborParams, get_neighboring_frames
from ..images import ImageParams, image_from_atoms
from ..utils import log
from macromol_dataframe import Atoms, transform_atom_coords
from torch.utils.data import Dataset
from functools import partial
from pathlib import Path

from typing import Any, Callable

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

def _filter_zones_by_curriculum(zone_ids, curriculum):
    mask = np.isin(zone_ids, curriculum)
    return zone_ids[mask]

