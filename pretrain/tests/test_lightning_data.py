import macromol_gym_pretrain.lightning as mmgp
from make_db import make_db

def test_macromol_data_module(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    db, *_ = make_db(db_path, split='train')

    data = mmgp.MacromolNeighborImageDataModule(
            db_path=db_path,
            make_sample=mmgp.make_neighbor_image_tensors,

            image_length_voxels=5,
            image_resolution_A=1,
            element_channels=[['C'], ['N'], ['O'], ['*']],

            direction_candidates=mmgp.cube_faces(),
            neighbor_padding_A=1,
            noise_max_distance_A=0,
            noise_max_angle_deg=0,

            batch_size=2,
            num_workers=2,
    )
    dataloader = data.train_dataloader()

    assert len(dataloader) == 1

    for x, y in dataloader:
        assert x.shape == (2, 2, 4, 5, 5, 5)
        assert y.shape == (2,)


