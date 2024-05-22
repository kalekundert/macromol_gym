import macromol_gym_pretrain.torch as mmgp
import macromol_voxelize as mmvox
import torch.testing
import parametrize_from_file as pff
import pickle

from param_helpers import make_db
from pipeline_func import f

with_py = pff.Namespace()
with_mmgp = pff.Namespace('from macromol_gym_pretrain.torch import *')

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

@pff.parametrize(
        schema=pff.cast(
            sampler=with_mmgp.eval,
            expected_len=with_py.eval,
            expected_iter=with_py.eval,
        ),
)
def test_infinite_sampler(sampler, expected_len, expected_iter):
    assert len(sampler) == expected_len

    for i, indices in enumerate(expected_iter):
        sampler.set_epoch(i)
        assert list(sampler) == list(indices)


