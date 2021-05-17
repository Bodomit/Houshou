from enum import Enum, auto
from typing import Type
import houshou.losses.semihard_triplet_miner as shm

from .semihard_kldiv import SHM_UniformKLDivergence


class LOSS(Enum):
    SEMIHARD_CROSSENTROPY = auto()
    SHM_UNIFORMKLDIVERGENCE = auto()

    def __str__(self) -> str:
        return self.name


def get_loss(loss: LOSS, **kwargs) -> shm.SemiHardTripletMiner:
    if loss == LOSS.SEMIHARD_CROSSENTROPY:
        return shm.SHM_CategoricalCrossEntropy(**kwargs)
    elif loss == LOSS.SHM_UNIFORMKLDIVERGENCE:
        return SHM_UniformKLDivergence(**kwargs)
    else:
        raise ValueError()
