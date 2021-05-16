from enum import Enum, auto
from typing import Union

from torch.utils.data import Dataset
import pytorch_lightning as pl

from .celeba import CelebA
from .vggface2 import VGGFace2
from .samplers import TripletBatchRandomSampler


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
