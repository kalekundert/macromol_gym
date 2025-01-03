import macromol_gym_unsupervised.lightning as mmgu

from make_db import make_db
from pytest import approx

def make_unsupervised_image_tensor(*args, **kwargs):
    return mmgu.make_unsupervised_image_sample(*args, **kwargs)['image']

def test_macromol_data_module(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    db, *_ = make_db(db_path, split='train')

    data = mmgu.MacromolImageDataModule(
            db_path=db_path,
            image_length_voxels=5,
            image_resolution_A=1,
            element_channels=[['C'], ['N'], ['O'], ['*']],
            make_sample=make_unsupervised_image_tensor,
            batch_size=2,
            num_workers=2,
    )

    # The mock database only has "train" data, so we can't test the other data 
    # loaders here.
    dataloader = data.train_dataloader()

    assert len(dataloader) == 2

    for x in dataloader:
        # There should be exactly one atom in each image, so two atoms in each 
        # minibatch.
        assert x.sum() == approx(2)
        assert x.shape == (2, 4, 5, 5, 5)

