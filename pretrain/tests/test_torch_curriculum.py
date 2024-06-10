import torch
import macromol_gym_pretrain.torch.curriculum as _mmgp

from pytest import approx

def test_maybe_int():
    assert _mmgp.maybe_int(None) is None
    assert _mmgp.maybe_int('3') == 3

def test_calc_error_probability_2_class():
    y = torch.tensor([0, 1, 0, 1, 0])
    y_hat = torch.tensor([
        # logits      probabilities
        [2.0, 0.0], # 0.8808  0.1192
        [1.0, 0.0], # 0.7311  0.2689
        [0.0, 0.0], # 0.5000  0.5000
        [0.0, 1.0], # 0.2689  0.7311
        [0.0, 2.0], # 0.1192  0.8808
    ])
    p_err = _mmgp.calc_error_probability(y, y_hat)

    assert list(p_err) == approx(
            [0.1192, 0.7311, 0.5000, 0.2689, 0.8808],
            abs=1e-4,
    )

