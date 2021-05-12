import os
import glob
from functools import partial

from typing import Any, Callable, List, Optional, Set, Tuple
from numpy.random import sample

import pandas
import numpy as np

import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.utils import verify_str_arg


class VGGFace2(ImageFolder):
    def __init__(
        self,
        root: str,
        split: str,
        base_folder="vggface2_MTCNN",
        target_type: List[str] = ["identity", "attr"],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        valid_split=0.05,
        valid_split_seed=42,
        **kwargs,
    ):
        self.base_folder = base_folder
        self.target_type = target_type
        split_ = verify_str_arg(split.lower(), "split", ("train", "valid", "test"))
        real_split = "train" if split_ in ["train", "valid"] else "test"

        super(VGGFace2, self).__init__(
            os.path.join(root, self.base_folder, real_split),
            transform=transform,
            target_transform=target_transform,
        )

        if split_ in ["train", "valid"]:
            valid_classes = self.get_valid_set_classes(
                set(self.classes), valid_split, valid_split_seed
            )

            if split_ == "train":
                train_classes = set(self.classes) - valid_classes
                self.restrict_dataset(train_classes)
            else:
                self.restrict_dataset(valid_classes)

    @staticmethod
    def get_valid_set_classes(
        classes: Set[str], valid_split: float, valid_split_seed: int
    ) -> Set[str]:
        assert valid_split >= 0 and valid_split < 1.0
        n_valid_set_classes = int(len(classes) * valid_split)
        rng = np.random.default_rng(valid_split_seed)
        valid_classes = rng.choice(sorted(classes), size=n_valid_set_classes)
        return set(valid_classes)

    def restrict_dataset(self, classes_to_keep: Set[str]):
        self.classes = list(sorted(classes_to_keep))
        self.class_to_idx = {k: self.class_to_idx[k] for k in classes_to_keep}
        self.samples = self.make_dataset(self.root, self.class_to_idx, self.extensions)
        self.targets = [s[1] for s in self.samples]
