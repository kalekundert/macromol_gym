"""
Rank the difficulty of each training example.

Usage:
    mmg_make_curriculum <db> <script> [-r <reps>] [-b <batch>] [-j <jobs>] [-c]

Arguments:
    <db>
        The path to a database of training examples.
        
    <script>
        A python script that defines the following functions:

            encode(polars.DataFrame) -> T
            collate(list[tuple[T, T]]) -> U
            predict(U) -> Iterable[Sequence[int]]

        The `encode()` function will be passed a dataframe of atom labels and 
        coordinates, and should return a representation of that data suitable 
        for making predictions.  For each training example, the `encode()` 
        function will be invoked twice.  Both invocations will be given all the 
        same atoms, but with the coordinates expressed in different frames.  
        The origin of each frame will be the center of the region made 
        available to the prediction algorithm.

        The `collate()` function should combine the results of multiple 
        invocations of `encode()` into a single minibatch.  The resulting 
        minibatch is the exact object that will be passed to `predict()`.  Note 
        that it is essential to maintain the order of the given list, otherwise 
        the labels (which are tracked externally to these functions) will not 
        be matched up with the right predictions.

        The `predict()` function should predict the probability that each 
        training example belongs to each possible label.  The outer collection 
        should contain one entry for each training example in the minibatch, 
        and each inner collection should contain one entry for each label.

        Note that extracting these three function requires executing the whole 
        script.  If the script contains any top-level code that you don't want 
        to execute at runtime, guard that code via `if __name__ == '__main__'`.

Options:
    -r --replicates <int>                           [default: 8]
        How many different training examples to sample from each "zone".

    -b --batch-size <int>                           [default: 64]
        How many training examples to evaluate in each minibatch.  Typically 
        the best value is the largest power of two that fits in VRAM.

    -j --num-workers <int>
        The number of data-loading subprocesses to run.  By default, this will 
        be determined either by the `$SLURM_JOB_CPUS_PER_NODE` environment 
        variable, or by the number of available processors.

    -c --copy-to-tmp
        Copy the database to /tmp before starting.  This can be faster, if the
        database is stored on a network filesystem.

Description:
    The purpose of this script is to allow hard-to-train models to start on 
    easier subsets of the data.  This is a concept called "curriculum 
    learning".  As each prediction is made, the average probability assigned to 
    the correct label is tallied and recorded in the database.  Future training 
    runs can consider this information when choosing training examples.
"""

class CurriculumDataset:

    def __init__(self, db_path, script_path):
        from ..database_io import open_db, select_split

        self.db_path = db_path
        self.script_path = script_path

        db = open_db(db_path)

        self.db = None
        self.zone_ids = select_split(db, 'train')

    def __init_subprocess__(self):
        from ..database_io import open_db

        # Perform any initialization that has to be deferred until we're in the 
        # data loader subprocess.

        self.db = open_db(self.db_path)
        self.db_cache = {}

        user_funcs = load_user_funcs(self.script_path)
        self.neighbor_params, self.array_from_atoms = \
                user_funcs['make_data_params']()

    def __len__(self):
        return len(self.zone_ids)

    def __getitem__(self, i):
        if self.db is None:
            self.__init_subprocess__()

        import numpy as np
        from ..dataset import get_neighboring_frames
        from ..database_io import select_zone_atoms
        from macromol_dataframe import transform_atom_coords

        zone_id, frame_ia, frame_ab, label = get_neighboring_frames(
                self.db, i,
                zone_ids=self.zone_ids,
                neighbor_params=self.neighbor_params,
                db_cache=self.db_cache,
        )

        atoms_i = select_zone_atoms(self.db, zone_id)
        atoms_a = transform_atom_coords(atoms_i, frame_ia)
        atoms_b = transform_atom_coords(atoms_a, frame_ab)

        input_a = self.array_from_atoms(atoms_a)
        input_b = self.array_from_atoms(atoms_b)
        input_ab = np.stack([input_a, input_b])

        return zone_id, i, input_ab, label

def load_user_funcs(path):
    user_globals = {}
    user_funcs = {}

    exec(path.read_text(), user_globals)

    for func in ['make_data_params', 'make_model']:
        if func not in user_globals:
            raise ValueError(f"`{path}` must define `{func}()`")
        else:
            user_funcs[func] = user_globals[func]

    return user_funcs

def calc_error_probability(labels, prediction_logits):
    import torch
    n = len(labels)
    prediction_prob = torch.softmax(prediction_logits, axis=1)
    return 1 - prediction_prob[torch.arange(n), labels]

def maybe_int(x):
    return None if x is None else int(x)

def main():
    try:
        import docopt
        args = docopt.docopt(__doc__)

        import logging
        logging.basicConfig()

        import torch
        import sys; sys.path.insert(0, '')

        from ..dataset import get_num_workers, copy_db_to_tmp
        from ..database_io import open_db, insert_curriculum, select_max_curriculum_seed
        from torch.utils.data import DataLoader
        from pathlib import Path
        from tqdm import tqdm

        db_path = Path(args['<db>'])
        script_path = Path(args['<script>'])

        with copy_db_to_tmp(
                db_path,
                noop=not args['--copy-to-tmp'],
                write=True,
        ) as db_path:

            db = open_db(db_path, mode='rw')
            user_funcs = load_user_funcs(script_path)
            model = user_funcs['make_model']()

            dataset = CurriculumDataset(db_path, script_path)
            begin_seed = select_max_curriculum_seed(db) + 1
            end_seed = int(args['--replicates']) * len(dataset)
            batch_size = int(args['--batch-size'])
            num_workers = get_num_workers(maybe_int(args['--num-workers']))

            data_loader = DataLoader(
                    dataset=dataset,
                    sampler=range(begin_seed, end_seed),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    multiprocessing_context='spawn',
            )

            if torch.cuda.is_available():
                gpu = torch.device("cuda:0")
                model = model.to(gpu)

            for zone_ids, random_seeds, x, y in tqdm(
                    data_loader,
                    initial=begin_seed // batch_size,
            ):
                if torch.cuda.is_available():
                    x = x.to(gpu)

                y_hat = model(x)
                difficulty = calc_error_probability(y, y_hat)

                with db:
                    insert_curriculum(
                            db,
                            zone_ids.tolist(),
                            random_seeds.tolist(),
                            difficulty.tolist(),
                    )
            
    except KeyboardInterrupt:
        print("canceled by user")
