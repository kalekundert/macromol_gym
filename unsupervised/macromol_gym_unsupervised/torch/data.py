import numpy as np
import logging

from ..samples import (
        MakeSampleFunc, zone_id_from_index, make_unprocessed_sample,
)
from ..database_io import (
        open_db, select_split, select_curriculum,
)
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path

log = logging.getLogger('macromol_gym')

class MacromolDataset(Dataset):

    def __init__(
            self,
            *,
            db_path: Path,
            split: str,
            make_sample: Optional[MakeSampleFunc] = None,
            max_difficulty: float = 1,
    ):
        # Don't store a connection to the database in the constructor.  The 
        # constructor runs in the parent process, after which the instantiated 
        # dataset object is sent to the worker process.  If the worker process 
        # was forked, this would cause weird deadlock/race condition problems!
        # If the worker process was spawned, this would require pickling the 
        # connection, which isn't possible.
        self.db = None
        self.db_path = db_path
        self.split = split
        self.make_sample = make_sample or make_unprocessed_sample

        db = open_db(db_path)
        self.zone_ids = select_split(db, split)

        if max_difficulty < 1:
            n = len(self.zone_ids)
            self.zone_ids = _filter_zones_by_curriculum(
                    self.zone_ids,
                    select_curriculum(db, max_difficulty),
            )
            log.info("remove difficult training examples: split=%s max_difficulty=%s num_examples_before_filter=%d num_examples_after_filter=%d", split, max_difficulty, n, len(self.zone_ids))

    def __repr__(self):
        return f'<{self.__class__.__name__} db={str(self.db_path)!r} split={self.split!r} len={len(self)}>'

    def __len__(self):
        return len(self.zone_ids)

    def __getitem__(self, i):
        if self.db is None:
            self.db = open_db(self.db_path)
            self.db_cache = {}

        zone_id, rng = zone_id_from_index(i, self.zone_ids)

        return self.make_sample(self.db, self.db_cache, rng, zone_id)

def _filter_zones_by_curriculum(zone_ids, curriculum):
    mask = np.isin(zone_ids, curriculum)
    return zone_ids[mask]

