from enum import Enum, auto

from torch.utils.data import Dataset

from .celeba import CelebA
from .vggface2 import VGGFace2
from .datasets import AttributeDataset
from .samplers import TripletBatchRandomSampler


class DATASET(Enum):
    VGGFACE2 = auto()
    CELEBA = auto()
    LFW = auto()


def get_dataset(dataset: DATASET, **kwargs) -> Dataset:
    if dataset == DATASET.CELEBA:
        dataset_ = CelebA(**kwargs)
    elif dataset == DATASET.VGGFACE2:
        dataset_ = VGGFace2(**kwargs)
    else:
        raise ValueError()

    dataset_ = AttributeDataset(dataset_, **kwargs)
    return dataset_
