import numpy as np

from more_itertools import grouper
from reprfunc import repr_from_init
from typing import Callable, Optional

class InfiniteSampler:
    """
    Draw reproducible samples from an infinite map-style dataset, i.e. a 
    dataset that accepts integer indices of any size.

    The typical reason to use this sampler is for deterministic data 
    augmentations.  Even for a finite dataset, there are typically an infinite 
    number of augmentations that can be applied.  Tying the specific choice of 
    augmentation to the index makes it easy to reproduce exact training 
    examples.  In contrast, if the augmentations are based on a pseudo-random 
    number generator seed set at the beginning of the training run, specific 
    training examples can't be reproduced without replaying the whole dataset 
    up to that point, and possibly taking into account factors such as the 
    number of dataloader processes.

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
            for the validation and test sets.  In order for this option to have 
            any effect, the training environment must call `set_epoch()` before 
            every epoch.

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

        world_size:
            The number of processes being used for distributed training.  It 
            should not usually be necessary to specify this argument.  If no 
            distributed context is detected, this will default to 1.  Otherwise 
            it will default to the world size indicated by the distributed 
            context.

        rank:
            The index number of the current process within the group of 
            processes being used for distributed training, counting from 0.  It 
            should not usually be necessary to specify this argument.  If no 
            distributed context is detected, this will default to 0.  Otherwise 
            it will default to the rank indicated by the distributed context.
            
        rng_factory:
            A factory function that creates a random number generator from a 
            given integer seed.  This generator is only used to shuffle the 
            indices, and only then if *shuffle* is enabled.

    This sampler supports distributed sampling.  Specifically, it automatically 
    detects when it's being used in a distributed context, and ensures that 
    each process is given a unique set of indices.  It also ensures that each 
    process is given the same number of indices, to avoid deadlock.

    If you are using the Lightning framework, be aware that its default 
    behavior is to wrap any custom sampler you use with a `DistributedSampler` 
    when doing distributed training.  Unfortunately, this wrapper is 
    implemented in such a way that the same samples will be used for each epoch
    (i.e. as if `increment_across_epochs` were always False), which largely 
    defeats the purpose of using `InfiniteSampler`.  Therefore, you must take 
    care to disable this wrapper via `Trainer(use_distributed_sampler=False)`.  
    As mentioned above, this sampler natively supports distributed contexts, 
    and so training will work as expected without the wrapper.
    """

    def __init__(
            self,
            epoch_size: int,
            *,
            start_epoch: int = 0,
            increment_across_epochs: bool = True,
            shuffle: bool = False,
            shuffle_size: Optional[int] = None,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            rng_factory: Callable[[int], np.random.Generator] = np.random.default_rng,
    ):
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            if world_size is None:
                world_size = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()

        self.epoch_size = epoch_size
        self.curr_epoch = start_epoch
        self.increment_across_epochs = increment_across_epochs
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size or epoch_size
        self.rng_factory = rng_factory
        self.world_size = world_size or 1
        self.rank = rank or 0

    def __iter__(self):
        n = self.epoch_size
        i = n * self.curr_epoch

        if not self.shuffle:
            indices = range(i, i+n)
        else:
            indices = _iter_shuffled_indices(
                    self.rng_factory,
                    self.shuffle_size,
                    i, i+n,
            )

        yield from _distribute(indices, self.rank, self.world_size)

    def __len__(self):
        return self.epoch_size // self.world_size

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

def _distribute(iterable, rank, world_size):
    # It's important to return the same number of items for each distributed 
    # process, otherwise the training will deadlock while the processes with 
    # more items wait for those with fewer to "catch up".

    for group in grouper(iterable, world_size, incomplete='ignore'):
        yield group[rank]

