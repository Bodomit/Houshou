import os

from typing import Any, Callable, List, Optional, Set, Tuple

import pandas
import numpy as np

import torch
from torchvision.datasets import ImageFolder
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
        self.target_type = target_type
        self.target_transform = target_transform

        split_ = verify_str_arg(split.lower(), "split", ("train", "valid", "test"))
        real_split = "train" if split_ in ["train", "valid"] else "test"
        real_split_dir = os.path.join(root, base_folder, real_split)
        real_split_classes = set(
            [d.name for d in os.scandir(real_split_dir) if d.is_dir()]
        )

        # Read the attributes. Usecols is specified to reduce memory usage
        # and parsing time, and needs expanded if other attributes are used.
        attributes = pandas.read_csv(  # type: ignore
            os.path.join(root, base_folder, "MAAD_Face.csv"),
            index_col=0,
            usecols=["Filename", "Male"],
        )

        # Get the permitted classes to be used for the dataset.
        # Note is_valid_file() does not affect self.classes or self.class_to_idx
        # so these attributes will be the same for train and val... which is annoying.
        if split_ in ["train", "valid"]:
            valid_classes = self.get_valid_set_classes(
                real_split_classes, valid_split, valid_split_seed
            )
            train_classes = real_split_classes - valid_classes
            split_classes = valid_classes if split_ == "valid" else train_classes

        elif split_ == "test":
            split_classes = real_split_classes
        else:
            raise ValueError

        # Get the filenames with corresponding attributes.
        imgs_with_attrs = set(attributes.index.tolist())

        # Read the samples but only if they have corresponding attributes
        # and belong to the correct class for the split.
        def is_valid_file(path: str):
            return (
                os.path.basename(os.path.dirname(path)) in split_classes
                and os.path.relpath(path, real_split_dir) in imgs_with_attrs
            )

        super(VGGFace2, self).__init__(
            real_split_dir,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        # Remove any attribute lines that are not in the dataset samples.
        real_imgs = [os.path.relpath(s[0], start=real_split_dir) for s in self.samples]
        diff = set(attributes.index.values) - set(real_imgs)
        attributes = attributes.drop(list(diff), errors="ignore")

        # Sort the attributes to match the sample order.
        sort_order = attributes.index.sort_values()
        attributes = attributes.loc[sort_order]

        # Ensure the attribute file and samples are aligned.
        assert isinstance(attributes, pandas.DataFrame)
        for x, y in zip(real_imgs, attributes.index.tolist()):
            assert x == y

        self.identity = torch.as_tensor([s[1] for s in self.samples])
        self.attributes = torch.as_tensor(attributes.values)
        self.attr_names = list(attributes.columns)

        # One last sanity check
        assert len(self.identity) == len(self.attributes) == len(self.samples)

    @staticmethod
    def get_valid_set_classes(
        classes: Set[str], valid_split: float, valid_split_seed: int
    ) -> Set[str]:
        assert valid_split >= 0 and valid_split < 1.0
        n_valid_set_classes = int(len(classes) * valid_split)
        rng = np.random.default_rng(valid_split_seed)
        valid_classes = rng.choice(sorted(classes), size=n_valid_set_classes)
        return set(valid_classes)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, identity = super(VGGFace2, self).__getitem__(index)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attributes[index, :])
            elif t == "identity":
                # Reads identity from the separte tensor rather than the underlying
                # sample target. Messy but helps ensure everythign is correctly aligned.
                identity_ = self.identity[index]
                assert identity_.item() == identity
                target.append(identity_)
            else:
                raise ValueError('Target type "{}" is not recognized.'.format(t))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return x, target
