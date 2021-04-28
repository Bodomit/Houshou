from enum import Enum, auto
from typing import Type
from houshou.losses.semihard_triplet_miner import SemiHardTripletMiner
import houshou.losses.semihard_triplet_miner as stm


class LOSS(Enum):
    SEMIHARD_CROSSENTROPY = auto()


def get_loss(loss: LOSS, **kwargs) -> SemiHardTripletMiner:
    if loss == LOSS.SEMIHARD_CROSSENTROPY:
        return stm.SHTWithCategoricalCrossEntropy(**kwargs)
    else:
        raise ValueError()
