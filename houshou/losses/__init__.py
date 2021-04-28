from enum import Enum, auto

import houshou.losses.semihard_triplet_miner as stm


class LOSS(Enum):
    SEMIHARD_CROSSENTROPY = auto()


def get_loss(loss: LOSS):
    if loss == LOSS.SEMIHARD_CROSSENTROPY:
        return stm.SHTWithCategoricalCrossEntropy
    else:
        raise ValueError()
