import os
from typing import List, Optional, Tuple

import pandas
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

        raw_train_attributes, raw_test_attributes = self.read_attributes()

        raise NotImplementedError
        assert isinstance(attributes, pandas.DataFrame)

        if stage is None or stage == "fit":
            self.train, self.valid = self._process_train_valid(attributes)

        if stage is None or stage == "test":
            self.test = self._process_test(attributes)

        super().setup(stage)

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
        self, train_attributes: pandas.DataFrame
    ) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def _process_test(
        self, test_attributes: pandas.DataFrame
    ) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError
