import numpy as np
import torch

from ..images import image_from_atoms, ImageParams
from ..random import sample_frame, sample_coord_from_cube
from ..database_io import (
        open_db, select_split, select_metadatum,
        select_zone_center_A, select_zone_atoms,
)
from macromol_dataframe import transform_atom_coords
from torch.utils.data import Dataset
from pathlib import Path

class MacromolDataset(Dataset):

    def __init__(
            self,
            *,
            db_path: Path,
            split: str,
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
        self.zone_size_A = select_metadatum(db, 'zone_size_A')

    def __len__(self):
        return len(self.zone_ids)

    def __getitem__(self, i):
        if self.db is None:
            self.db = open_db(self.db_path)

        rng = np.random.default_rng(i)

        zone_id = self.zone_ids[i % len(self.zone_ids)]
        zone_center_A = select_zone_center_A(self.db, zone_id)

        atoms_i = select_zone_atoms(self.db, zone_id)
        origin_i = sample_coord_from_cube(rng, zone_center_A, self.zone_size_A)
        frame_ia = sample_frame(rng, origin_i)

        return dict(
                rng=rng,
                zone_id=zone_id,
                atoms_i=atoms_i,
                frame_ia=frame_ia,
        )

class MacromolImageDataset(MacromolDataset):

    def __init__(
            self,
            *,
            db_path: Path,
            split: str,
            img_params: ImageParams,
    ):
        # This class is slightly opinionated about how images should be 
        # created.  This allows it to provide a simple---but not fully 
        # general---API for common image parameters.  If you need to do 
        # something beyond the scope of this API, use `MacromolDataset` 
        # directly.
        super().__init__(db_path=db_path, split=split)
        self.img_params = img_params

    def __getitem__(self, i):
        x = super().__getitem__(i)

        atoms_a = transform_atom_coords(x['atoms_i'], x['frame_ia'])
        image = image_from_atoms(atoms_a, self.img_params)
        image = torch.from_numpy(image).float()

        return dict(
                **x,
                atoms_a=atoms_a,
                image=image,
        )

class MapDataset(Dataset):

    def __init__(self, dataset, func):
        self.dataset = dataset
        self.func = func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.func(self.dataset[i])
