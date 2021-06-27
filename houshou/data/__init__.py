from enum import Enum, auto
from typing import Union

from torch.utils.data import Dataset

from .base import TripletsAttributeDataModule
from .celeba import CelebA
from .market_1501 import Market1501
from .rap2 import RAP2
from .vggface2 import VGGFace2
