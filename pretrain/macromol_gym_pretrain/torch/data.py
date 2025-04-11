import torch
import numpy as np

from ..samples import MakeSampleArgs, make_neighbor_image_sample
from ..neighbors import NeighborParams
from macromol_gym_unsupervised import ImageParams

def make_neighbor_image_tensors(
        sample: MakeSampleArgs,
        *,
        img_params: ImageParams,
        neighbor_params: NeighborParams,
):
    x = make_neighbor_image_sample(
            sample,
            img_params=img_params,
            neighbor_params=neighbor_params,
    )
    img_ab = np.stack([x['img_a'], x['img_b']])

    return torch.from_numpy(img_ab).float(), torch.tensor(x['b'])

