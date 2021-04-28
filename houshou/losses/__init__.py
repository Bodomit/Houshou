from enum import Enum, auto

import houshou.losses.semihard_triplet_miner as stm


class LOSS(Enum):
    SEMIHARD_CROSSENTROPY = auto()


