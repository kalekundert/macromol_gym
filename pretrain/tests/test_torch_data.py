import macromol_gym_pretrain.torch as mmgp
import macromol_voxelize as mmvox

from make_db import make_db
from torch.utils.data import DataLoader
from functools import partial

def test_macromol_dataset(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    db, *_ = make_db(db_path, split='train')

    dataset = mmgp.MacromolDataset(
            db_path=db_path,
            split='train',
            make_sample=partial(
                mmgp.make_neighbor_image_tensors,
                img_params=mmgp.ImageParams(
                    grid=mmvox.Grid(
                        length_voxels=5,
                        resolution_A=1,
                    ),
                    atom_radius_A=0.5,
                    element_channels=[['C'], ['N'], ['O'], ['*']],
                ),
                neighbor_params=mmgp.NeighborParams(
                    direction_candidates=mmgp.cube_faces(),
                    distance_A=5,
                    noise_max_distance_A=0,
                    noise_max_angle_deg=0,
                )
            ),
    )

    # This is just the check that the dataset implements the API expected by 
    # the data loader.
    dataloader = DataLoader(dataset)

    assert len(dataset) == 2
    assert len(dataloader) == 2

    for x, y in dataloader:
        assert x.shape == (1, 2, 4, 5, 5, 5)
        assert y.shape == (1,)

