from enum import Enum, auto
from typing import Type

from .attribute_losses import CrossEntropy, UniformTargetKLDivergence
from .base import LossBase, WeightedMultitaskLoss
from .semihard_triplet_miner import SemiHardTripletMiner


class LOSS(Enum):
    CROSSENTROPY = auto()
    UNIFORM_KLDIVERGENCE = auto()
    SEMIHARD_MINED_TRIPLETS = auto()

    def __str__(self) -> str:
        return self.name


def get_loss(loss: LOSS, **kwargs) -> LossBase:
    if loss == LOSS.CROSSENTROPY:
        return CrossEntropy(**kwargs)
    elif loss == LOSS.UNIFORM_KLDIVERGENCE:
        return UniformTargetKLDivergence(**kwargs)
    elif loss == LOSS.SEMIHARD_MINED_TRIPLETS:
        return SemiHardTripletMiner(**kwargs)
    else:
        raise ValueError()


def get_multitask_loss(
    loss_f: LOSS, loss_a: LOSS, lambda_value: float, **kwargs
) -> WeightedMultitaskLoss:
    l_f = get_loss(loss_f, **kwargs)
    l_a = get_loss(loss_a, **kwargs)
    return WeightedMultitaskLoss(l_f, l_a, lambda_value)
