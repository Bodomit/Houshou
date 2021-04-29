from enum import Enum, auto

from torch.utils.data import Dataset

from .celeba import CelebA
from .datasets import AttributeDataset


class DATASET(Enum):
    VGGFACE2 = auto()
    CELEBA = auto()
    LFW = auto()


def get_dataset(dataset: DATASET, **kwargs) -> Dataset:
    if dataset == DATASET.CELEBA:
        dataset_ = CelebA(**kwargs)
    else:
        raise ValueError()

    dataset_ = AttributeDataset(dataset_, **kwargs)
    return dataset_
