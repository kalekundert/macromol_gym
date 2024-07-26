import macromol_gym_unsupervised.images as _mmgu
import parametrize_from_file as pff

from param_helpers import image, with_np
from pytest import approx

@pff.parametrize(
        schema=pff.cast(
            img=image,
            mean=with_np.eval,
            std=with_np.eval,
            expected=image,
        )
)
def test_normalize_image(img, mean, std, expected):
    _mmgu._normalize_image(img, mean, std)
    assert img == approx(expected)

