import macromol_gym_unsupervised.torch as mmgu
import macromol_gym_unsupervised.torch.data as _mmgu
import macromol_voxelize as mmvox
import torch.testing
import numpy as np
import numpy.testing
import pickle

from make_db import make_db
from torch.utils.data import DataLoader
from functools import partial
from pipeline_func import f
from pytest import approx

def test_macromol_dataset(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    db, *_ = make_db(db_path, split='train')

    def make_unsupervised_image_tensor(sample):
        x = mmgu.make_unsupervised_image_sample(
                sample,
                img_params=mmgu.ImageParams(
                    grid=mmvox.Grid(
                        length_voxels=5,
                        resolution_A=1,
                    ),
                    atom_radius_A=0.5,
                    element_channels=[['C'], ['N'], ['O'], ['*']],
                ),
        )
        return x['image']

    dataset = mmgu.MacromolDataset(
            db_path=db_path,
            split='train',
            make_sample=make_unsupervised_image_tensor,
    )

    # This is just the check that the dataset implements the API expected by 
    # the data loader.
    dataloader = DataLoader(dataset)

    assert len(dataset) == 4
    assert len(dataloader) == 4

    for x in dataloader:
        # There should be exactly one atom in each image.
        assert x.sum() == approx(1)
        assert x.shape == (1, 4, 5, 5, 5)

def test_macromol_dataset_pickle(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    db, *_ = make_db(db_path, split='train')

    dataset = mmgu.MacromolDataset(
            db_path=db_path,
            split='train',
            make_sample=partial(
                mmgu.make_unsupervised_image_sample,
                img_params=mmgu.ImageParams(
                    grid=mmvox.Grid(
                        length_voxels=5,
                        resolution_A=1,
                    ),
                    atom_radius_A=0.5,
                    element_channels=[['C'], ['N'], ['O'], ['*']],
                ),
            ),
    )
    dataset_pickle = (
            dataset
            | f(pickle.dumps)
            | f(pickle.loads)
    )

    img = dataset[0]['image']
    img_pickle = dataset_pickle[0]['image']

    torch.testing.assert_close(img, img_pickle)

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
            _mmgu._filter_zones_by_curriculum(zone_ids, curriculum),
            np.array([2, 4, 5]),
    )


