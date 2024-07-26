import numpy as np

from reprfunc import repr_from_init
from typing import Callable, Optional

class InfiniteSampler:
    """
    Draw reproducible samples from an infinite map-style dataset, i.e. a 
    dataset that accepts integer indices of any size.

    Arguments:
        epoch_size:
            The number of examples to include in each epoch.  Note that, 
            because the dataset is assumed to have an infinite number of 
            examples, this parameter doesn't have to relate to the amount of 
            data in the dataset.  Instead, it usually just specifies how often 
            "end of epoch" tasks, like running the validation set or saving 
            checkpoints, are performed.

        start_epoch:
            The epoch number to base the random seed on, if *shuffle* is 
            enabled and *increment_across_epochs* is not.  Note also that if 
            the training environment doesn't call `set_epoch()` before every 
            epoch, which every sane training environment should, then this 
            setting will determine the random seed used for *shuffle* 
            regardless of *increment_across_epochs*.

        increment_across_epochs:
            If *False*, yield the same indices in the same order every epoch.  
            If *True*, yield new indices in every epoch, without skipping any.  
            This option is typically enabled for the training set, and disabled 
            for the validation and test sets.

        shuffle:
            If *True*, shuffle the indices within each epoch.  The shuffling is 
            guaranteed to be a deterministic function of the epoch number, as 
            set by `set_epoch()`.  This means that every training run will 
            visit the same examples in the same order.

        shuffle_size:
            The number of indices to consider when shuffling.  For example, 
            with a shuffle size of 5, the first 5 indices would be some 
            permutation of 0-4, the second 5 would be some permutation of 5-9, 
            and so on.  Note that this setting is independent of the epoch 
            size.  For example, with a shuffle size of 5 and an epoch size of 
            3, the first epoch would consist of three values between 0-4.  The 
            second epoch would begin with the two values between 0-4 that 
            weren't in the first epoch, then end with a value between 5-9.  The 
            third epoch would begin with the unused values between 5-9, and so 
            on.  That said, by default the shuffle size is the same as the 
            epoch size.

        rng_factory:
            A factory function that creates a random number generator from a 
            given integer seed.  This generator is only used to shuffle the 
            indices, and only then if *shuffle* is enabled.
    """

    def __init__(
            self,
            epoch_size: int,
            *,
            start_epoch: int = 0,
            increment_across_epochs: bool = True,
            shuffle: bool = False,
            shuffle_size: Optional[int] = None,
            rng_factory: Callable[[int], np.random.Generator] = np.random.default_rng,
    ):
        self.epoch_size = epoch_size
        self.curr_epoch = start_epoch
        self.increment_across_epochs = increment_across_epochs
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size or epoch_size
        self.rng_factory = rng_factory

    def __iter__(self):
        n = self.epoch_size
        i = n * self.curr_epoch

        if not self.shuffle:
            yield from range(i, i+n)
        else:
            yield from _iter_shuffled_indices(
                    self.rng_factory,
                    self.shuffle_size,
                    i, i+n,
            )

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch: int):
        if self.increment_across_epochs:
            self.curr_epoch = epoch

    __repr__ = repr_from_init

def _iter_shuffled_indices(rng_factory, n, i, j):
    while True:
        seed = i // n
        rng = rng_factory(seed)

        i0 = n * seed; i1 = i0 + n
        indices = rng.permutation(range(i0, i1))
        
        start = i - i0
        end = j - i0

        if end > n:
            yield from indices[start:]
            i = i1
        else:
            yield from indices[start:end]
            return
