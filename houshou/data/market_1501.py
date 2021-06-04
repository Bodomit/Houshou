import os
from typing import Any, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas
import PIL
import scipy.io as spio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .base import TripletsAttributeDataModule


class Market1501(TripletsAttributeDataModule):
    def __init__(
        self,
        batch_size: int,
        buffer_size: Optional[int],
        attribute: List[str],
        data_dir: str = "Market-1501",
        valid_split: float = 0.05,
        valid_split_seed: int = 42,
        ext: str = "jpg",
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
        self.ext = ext

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

        raw_train_attributes, raw_test_attributes = self.read_attributes()

        if stage is None or stage == "fit":
            train_filenames = self.read_filenames("bounding_box_train")
            self.train, self.valid = self._process_train_valid(
                raw_train_attributes, train_filenames
            )

        if stage is None or stage == "test":
            test_filenames = self.read_filenames("bounding_box_test")
            query_filenames = self.read_filenames("query")
            self.test = self.construct_dataset(
                test_filenames, raw_test_attributes, self.test_transforms
            )
            self.query = self.construct_dataset(
                query_filenames, raw_test_attributes, self.test_transforms
            )

        super().setup(stage)

    def construct_dataset(
        self,
        filenames: Set[str],
        raw_attributes: pandas.DataFrame,
        transform: transforms.Compose,
    ):
        filenames_ = np.array(list(sorted(filenames)))
        identities = np.array(self.parse_ids(filenames_))

        # Filter out junk and distractors.
        non_junk_mask = identities.astype(np.int32) >= 1
        filenames_ = filenames_[non_junk_mask]
        identities = identities[non_junk_mask]

        # Get attributes per image.
        attribute_names = list(raw_attributes.columns)
        selected_attribute_indexs = self.get_indexes(attribute_names, self.attribute)
        attributes = torch.as_tensor(
            raw_attributes.loc[identities].values, dtype=torch.float  # type: ignore
        )

        # Change from 1-indexed categoricals to 0-indexed.
        attributes = attributes - 1
        attributes = attributes.to(torch.int64)

        # Reduce to the selected attributes.
        attributes = attributes[:, selected_attribute_indexs]
        selected_attribute_names = [
            attribute_names[i] for i in selected_attribute_indexs
        ]

        attribute_support = self.calc_attribute_support(attributes)

        return Market1501Dataset(
            filenames_.tolist(),
            torch.as_tensor(identities.astype(np.int64)),
            attributes,
            selected_attribute_names,
            attribute_support,
            self.target_type,
            transform,
        )

    def parse_ids(self, filenames: Iterable) -> List[str]:
        return [os.path.basename(fn).split("_")[0] for fn in filenames]

    def read_filenames(self, subdir: str) -> Set[str]:
        image_absdir = os.path.join(self.data_dir, subdir)
        filenames = set(
            [
                f.path
                for f in os.scandir(image_absdir)
                if f.name.split(".")[1] == self.ext
            ]
        )
        return filenames

    def read_attributes(
        self, filename="market_attribute.mat"
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        path = os.path.join(self.data_dir, filename)

        attributes_mat = spio.loadmat(
            path, struct_as_record=False, squeeze_me=True, simplify_cells=True
        )
        train_mat = attributes_mat["market_attribute"]["train"]
        test_mat = attributes_mat["market_attribute"]["test"]

        train_attributes = self._process_attributes(train_mat)
        test_attributes = self._process_attributes(test_mat)

        return train_attributes, test_attributes

    def _process_attributes(self, attributes_mat) -> pandas.DataFrame:
        attributes = pandas.DataFrame.from_dict(attributes_mat)
        attributes = attributes.set_index("image_index")

        return attributes

    def _process_train_valid(
        self, train_attributes: pandas.DataFrame, filenames: Set[str]
    ) -> Tuple[Dataset, Dataset]:

        filenames_ = list(sorted(filenames))
        identities = self.parse_ids(filenames_)

        val_classes = self.get_val_set_classes(
            set(identities), self.valid_split, self.valid_split_seed
        )

        train_filenames: Set[str] = set([])
        val_filenames: Set[str] = set([])

        assert len(identities) == len(filenames_)
        for identity, filename in zip(identities, filenames_):
            if identity in val_classes:
                val_filenames.add(filename)
            else:
                train_filenames.add(filename)

        train = self.construct_dataset(
            train_filenames, train_attributes, self.train_transforms
        )
        val = self.construct_dataset(
            val_filenames, train_attributes, self.val_transforms
        )

        return train, val


class Market1501Dataset(Dataset):
    def __init__(
        self,
        filenames: List[str],
        identities: torch.Tensor,
        attributes: torch.Tensor,
        attribute_names: List[str],
        attributes_support: torch.Tensor,
        target_type: List[str],
        transform: transforms.Compose,
    ) -> None:
        super().__init__()
        self.filenames = filenames
        self.identities = identities
        self.attributes = attributes
        self.attribute_names = attribute_names
        self.attributes_support = attributes_support
        self.target_type = target_type
        self.transform = transform

        assert len(self.filenames) == len(self.identities) == len(self.attributes)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x_path = self.filenames[index]
        x = PIL.Image.open(x_path)  # type: ignore

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attributes[index, :])
            elif t == "identity":
                # Reads identity from the separate tensor provided.
                # Messy but helps ensure everything is correctly aligned.
                identity_ = self.identities[index]
                target.append(identity_)
            else:
                raise ValueError('Target type "{}" is not recognized.'.format(t))

        if self.transform is not None:
            x = self.transform(x)

        return x, tuple(target)
