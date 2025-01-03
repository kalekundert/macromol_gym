import numpy as np
from torch.utils.data import default_collate

class BatchGenerator:
    """
    Wrap a collection of NumPy pseudorandom-number generators (PRNGs) such that 
    samples can easily be drawn from all of them at once.

    Instantiate this class with a list of NumPy PRNGs.  Then, any method 
    invoked on this class will automatically be invoked on all of those PRNGs, 
    and the results will collated into a PyTorch tensor.  For example::

        >>> from atompaint.diffusion.data import BatchGenerator
        >>> bg = BatchGenerator([
        ...     np.random.default_rng(0),
        ...     np.random.default_rng(1),
        ... ])
        >>> bg.uniform()
        tensor([0.6370, 0.5118], dtype=torch.float64)

    This class is meant to facilitate the idea that all of the randomness in 
    each training step should come a PRNG seeded based on the index of the 
    corresponding training example.  This PRNG would be created by the dataset, 
    used to build the training example, then returned in case the training loop 
    itself requires any more randomness.

    The benefit of this approach is that it's very robust.  The randomness does 
    not depend on the number of data loader processes, and every training 
    example can be reproduced without having to replay the whole dataset or 
    constantly log the PRNG state.  However, it's worth noting that from the 
    point-of-view of trying to get the best possible distribution of random 
    numbers, this approach is suboptimal.  PRNGs are only designed to output 
    high-quality randomness if seeded once.  There's no guarantee that two 
    PRNGs with different seeds won't output correlated values.  In practice, 
    though, this doesn't seem to be a significant issue.

    The `collate_rngs()` function can be used to make PyTorch dataloaders 
    automatically wrap collections of NumPy PRNGs with this class.
    """

    def __init__(self, rngs):
        self._rngs = rngs

    def __repr__(self):
        return f'<{self.__class__.__name__} n={len(self._rngs)}>'

    def __len__(self):
        return len(self._rngs)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        def method_wrapper(*args, **kwargs):
            return default_collate([
                getattr(rng, name)(*args, **kwargs)
                for rng in self._rngs
            ])

        return method_wrapper

    def pin_memory(self):
        return self

def collate_rngs(x):
    from torch.utils.data._utils.collate import collate, default_collate_fn_map

    def collate_rng_fn(x, *, collate_fn_map=None):
        return BatchGenerator(x)

    collate_fn_map = {
            np.random.Generator: collate_rng_fn,
            **default_collate_fn_map,
    }
    return collate(x, collate_fn_map=collate_fn_map)


