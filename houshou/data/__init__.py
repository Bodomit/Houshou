from enum import Enum, auto
from typing import Union

from torch.utils.data import Dataset

from .base import TripletsAttributeDataModule
from .celeba import CelebA
from .market_1501 import Market1501
from .vggface2 import VGGFace2


class DATASET(Enum):
    VGGFACE2 = auto()
    CELEBA = auto()
    LFW = auto()

    def __str__(self) -> str:
        return self.name


def get_dataset_module(dataset: DATASET, **kwargs) -> Union[CelebA, VGGFace2]:
    if dataset == DATASET.CELEBA:
        dataset_module = CelebA(**kwargs)
    elif dataset == DATASET.VGGFACE2:
        dataset_module = VGGFace2(**kwargs)
    else:
        raise ValueError()

    return dataset_module
