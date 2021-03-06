import os
from functools import partial
from typing import Any, Callable, Counter, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg

from .base import TripletsAttributeDataModule


class VGGFace2Dataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        identities: torch.Tensor,
        attributes: torch.Tensor,
        attribute_names: List[str],
        attributes_support: torch.Tensor,
        target_type: List[str],
        transform: transforms.Compose,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.identities = identities
        self.attributes = attributes
        self.attribute_names = attribute_names
        self.attributes_support = attributes_support
        self.target_type = target_type
        self.transform = transform

        self.classes = self.identities.unique()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x_path, identity = self.samples[index]

        x = PIL.Image.open(x_path)  # type: ignore

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attributes[index, :])
            elif t == "identity":
                # Using self.identities field rather than the identity from the sample,
                # as self.identities is specific to the dataset whereas the sample
                # identity covers both train and val (if stage == "fit"). Mess like.
                identity_ = self.identities[index]
                target.append(identity_)
            else:
                raise ValueError('Target type "{}" is not recognized.'.format(t))

        if self.transform is not None:
            x = self.transform(x)

        return x, tuple(target)


class VGGFace2(TripletsAttributeDataModule):
    def __init__(
        self,
        batch_size: int,
        attribute: List[str],
        buffer_size: Optional[int] = None,
        data_dir: str = "vggface2_MTCNN",
        valid_split: float = 0.05,
        valid_split_seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            data_dir,
            batch_size,
            buffer_size,
            attribute,
            **kwargs,
        )

        # Store attributes.
        self.valid_split = valid_split
        self.valid_split_seed = valid_split_seed

        # Define the transformations.
        common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),  # Reads images scales to [0, 1]
                transforms.Lambda(lambda x: x * 2 - 1),  # Change range to [-1, 1]
                transforms.Resize((160, 160)),
            ]
        )
        self.train_transforms = transforms.Compose(
            [common_transforms, transforms.RandomHorizontalFlip(p=0.5)]
        )
        self.val_transforms = common_transforms
        self.test_transforms = common_transforms

        self.dims = (3, 160, 160)

    def setup(self, stage: Optional[str]) -> None:

        # Read the attributes. Usecols is specified to reduce memory usage
        # and parsing time.
        attributes = pandas.read_csv(  # type: ignore
            os.path.join(self.data_dir, "MAAD_Face.csv"),
            index_col=0,
            usecols=["Filename"] + self.attribute,
        )
        assert isinstance(attributes, pandas.DataFrame)

        if stage is None or stage == "fit":
            self.train, self.valid = self._process_train_valid(attributes)

        if stage is None or stage == "test":
            self.test = self._process_test(attributes)

        super().setup(stage)

    def _process_train_valid(
        self, attributes: pandas.DataFrame
    ) -> Tuple[VGGFace2Dataset, VGGFace2Dataset]:
        real_split_dir = os.path.join(self.data_dir, "train")

        # Get the full training set.
        imgs_with_attrs: Set[str] = set(attributes.index.tolist())

        def is_valid_file(path: str):
            return os.path.relpath(path, real_split_dir) in imgs_with_attrs

        image_folder = ImageFolder(real_split_dir, is_valid_file=is_valid_file)

        # Get the classes for the validation set and mask.
        real_split_classes = set(
            [d.name for d in os.scandir(real_split_dir) if d.is_dir()]
        )
        val_classes = self.get_val_set_classes(
            real_split_classes, self.valid_split, self.valid_split_seed
        )
        val_classes_encoded = set([image_folder.class_to_idx[c] for c in val_classes])

        val_class_mask = [s[1] in val_classes_encoded for s in image_folder.samples]

        # Split into validation and training samples.
        val_samples: List[Tuple[str, int]] = []
        train_samples: List[Tuple[str, int]] = []
        for (
            s,
            cm,
        ) in zip(image_folder.samples, val_class_mask):
            if cm:
                val_samples.append(s)
            else:
                train_samples.append(s)

        # Construct the dataset objects proper.
        construct_fn = partial(
            self.construct_dataset,
            real_split_dir=real_split_dir,
            attributes=attributes,
            target_type=self.target_type,
        )
        train_dataset = construct_fn(
            samples=train_samples, transform=self.train_transforms
        )
        val_dataset = construct_fn(samples=val_samples, transform=self.val_transforms)

        return train_dataset, val_dataset

    def _process_test(self, attributes: pandas.DataFrame) -> VGGFace2Dataset:
        real_split_dir = os.path.join(self.data_dir, "test")

        # Get the full test set, removing those without attributes.
        imgs_with_attrs: Set[str] = set(attributes.index.tolist())

        def is_valid_file(path: str):
            return os.path.relpath(path, real_split_dir) in imgs_with_attrs

        image_folder = ImageFolder(real_split_dir, is_valid_file=is_valid_file)

        test_dataset = self.construct_dataset(
            real_split_dir,
            image_folder.samples,
            attributes,
            self.target_type,
            self.test_transforms,
        )

        return test_dataset

    def construct_dataset(
        self,
        real_split_dir: str,
        samples: List[Tuple[str, int]],
        attributes: pandas.DataFrame,
        target_type: List[str],
        transform: transforms.Compose,
    ) -> VGGFace2Dataset:

        # Remove any attribute lines that are not in the dataset samples.
        real_imgs = [os.path.relpath(s[0], start=real_split_dir) for s in samples]
        diff = set(attributes.index.values) - set(real_imgs)
        attributes = attributes.drop(list(diff), errors="ignore")

        # Sort the attributes to match the sample order.
        sort_order = attributes.index.sort_values()
        attributes = attributes.loc[sort_order]  # type: ignore

        # Ensure the attribute file and samples are aligned.
        assert isinstance(attributes, pandas.DataFrame)
        for x, y in zip(real_imgs, attributes.index.tolist()):
            assert x == y

        identities = torch.as_tensor([s[1] for s in samples])
        attributes_ = torch.as_tensor(attributes.values)
        attributes_ = (attributes_ + 1) // 2  # map from {-1, 1} to {0, 1}
        attr_names = list(attributes.columns)

        # Remap the identities to be contiguous (for classification training).
        unique_identities, identity_map = identities.unique(return_inverse=True)
        local_unique_identities = torch.arange(
            0, len(unique_identities), dtype=torch.int64
        )
        local_identities = local_unique_identities[identity_map]

        # Get the indexes for the specified columns.
        selected_attribute_indexs = self.get_indexes(attr_names, self.attribute)
        attributes_ = attributes_[:, selected_attribute_indexs]

        attributes_support = self.calc_attribute_support(attributes_)

        # One last sanity check
        assert len(local_identities) == len(attributes_) == len(samples)

        return VGGFace2Dataset(
            samples,
            local_identities,
            attributes_,
            attr_names,
            attributes_support,
            target_type,
            transform,
        )

    @staticmethod
    def get_indexes(attr_names: List[str], selected_attrs: List[str]):
        indexs: List[int] = []
        for selected_attr in selected_attrs:
            try:
                indexs.append(attr_names.index(selected_attr))
            except ValueError:
                raise ValueError(
                    f"Selected Attribute {selected_attr}"
                    + " not in attributes for dataset."
                )
        return indexs
