import torch

from torch import nn
from torch.nn import Module
from itertools import pairwise

from typing import TypeAlias
from collections.abc import Iterable, Callable

MlpLayerFactory: TypeAlias = Callable[[int, int], Iterable[Module]]

class NeighborLocationPredictor(Module):
    """
    Predict the relative orientation of one set of atoms relative to another.
    """

    def __init__(
            self, *,
            pair_encoder: Module,
            pair_classifier: Module,
    ):
        super().__init__()
        self.pair_encoder = pair_encoder
        self.pair_classifier = pair_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            input:
                A tensor of dimension (B, 2, C, W, H, D) containing two regions 
                of a macromolecule that are related by an unknown transformation.
            
                B: minibatch size
                2: region index
                C: atom-type channels
                W: region width
                H: region height
                D: region depth

                Note that the regions will always be cubical, meaning that W, 
                H, and D will always be equal.

        Returns:
            A tensor of dimension (B, V) that describes the probability that 
            each minibatch member belongs to each possible view.  The values in 
            this tensor are unnormalized logits, suitable to be passed to the 
            softmax or cross-entropy loss functions.

            B: minibatch size
            V: number of possible views
        """
        latent = self.pair_encoder(x)
        return self.pair_classifier(latent)


class ViewPairEncoder(Module):

    def __init__(self, view_encoder: Module):
        super().__init__()
        self.view_encoder = view_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.view_encoder(x[:,0])
        y1 = self.view_encoder(x[:,1])
        return torch.cat([y0, y1], dim=1)

class ViewPairClassifier(Module):
    """
    A simple MLP classifier.

    The motivations behind this classifier are that:

    - As it is, the equivariant classifier doesn't account for all symmetries 
      of the problem.  Specifically, if the two input views are swapped, then 
      the output should change in a deterministic way, but this is not done.

    - I expect that a regular, non-equivariant MLP will be more expressive, 
      because it can use ordinary nonlinearities.  

    - Even when not enforced, the model can still learn a degree of 
      "equivariance" via data augmentation.  This might be enough to train the 
      model.
    """

    def __init__(
            self, *,
            channels: list[int],
            layer_factory: MlpLayerFactory,
    ):
        super().__init__()

        *channels, num_categories = channels

        layers = []
        for in_channels, out_channels in pairwise(channels):
            layers += list(layer_factory(in_channels, out_channels))

        self.mlp = torch.nn.Sequential(
                *layers,
                torch.nn.Linear(channels[-1], num_categories),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        return self.mlp(x)

def linear_relu_dropout_layer(
        in_channels: int,
        out_channels: int,
        *,
        drop_rate: float,
):
    assert 0 <= drop_rate <= 1
    yield nn.Linear(in_channels, out_channels)
    yield nn.ReLU()
    yield nn.Dropout(drop_rate)


