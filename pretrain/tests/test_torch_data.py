import macromol_gym_pretrain.torch as mmgp
import macromol_gym_pretrain.torch.data as _mmgp
import macromol_voxelize as mmvox
import torch.testing
import numpy as np
import numpy.testing
import pickle

from param_helpers import make_db
from pipeline_func import f

def test_cnn_neighbor_dataset_pickle(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    db, *_ = make_db(db_path, split='train')

    dataset = mmgp.CnnNeighborDataset(
            db_path,
            split='train',
            neighbor_params=mmgp.NeighborParams(
                direction_candidates=mmgp.cube_faces(),
                distance_A=30,
                noise_max_distance_A=5,
                noise_max_angle_deg=10,
            ),
            img_params=mmgp.ImageParams(
                grid=mmvox.Grid(
                    length_voxels=24,
                    resolution_A=1,
                ),
                atom_radius_A=0.5,
                element_channels=[['C'], ['N'], ['O'], ['*']],
                ligand_channel=True,
            ),
    )
    dataset_pickle = (
            dataset
            | f(pickle.dumps)
            | f(pickle.loads)
    )

    img, b = dataset[0]
    img_pickle, b_pickle = dataset_pickle[0]

    torch.testing.assert_close(img, img_pickle)
    assert b == b_pickle

def test_filter_zones_by_curriculum():
    zone_ids = np.array([1, 2, 4, 5, 6])

    # Deliberately put the curriculum out of order.  The curriculum isn't 
    # guaranteed to be in any order in particular, and part of the point of 
    # this function is to maintain the order of the zone ids.
    #
    # The curriculum also includes `3`, which isn't one of the listed zone ids 
    # (presumably because it's not part of the training split).  This is a bit 
    # unrealistic, because the curriculum should be a subset of the training 
    # split, but it is possible for that constraint to be violated.
    curriculum = np.array([5, 4, 3, 2])

    np.testing.assert_equal(
            _mmgp._filter_zones_by_curriculum(zone_ids, curriculum),
            np.array([2, 4, 5]),
    )

