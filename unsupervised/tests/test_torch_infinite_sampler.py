import parametrize_from_file as pff

with_py = pff.Namespace()
with_mmgu = pff.Namespace('from macromol_gym_unsupervised.torch import *')

@pff.parametrize(
        schema=pff.cast(
            sampler=with_mmgu.eval,
            expected_len=with_py.eval,
            expected_iter=with_py.eval,
        ),
)
def test_infinite_sampler(sampler, expected_len, expected_iter):
    assert len(sampler) == expected_len

    for i, indices in enumerate(expected_iter):
        sampler.set_epoch(i)
        assert list(sampler) == list(indices)

