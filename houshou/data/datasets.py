from typing import List, Union

import torch
from torch.utils.data import Dataset

from .celeba import CelebA

HoushouDataset = Union[CelebA]


class AttributeDataset(Dataset):
    def __init__(
        self, base_dataset: HoushouDataset, selected_attributes: List[str], **kwargs
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.selected_attributes = selected_attributes

        self.indexs = self.get_indexes(
            self.base_dataset.attr_names, self.selected_attributes
        )

    @property
    def identity(self) -> torch.Tensor:
        return self.base_dataset.identity

    def get_indexes(self, attr_names: List[str], selected_attrs: List[str]):
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

    def __getitem__(self, index):
        image, (identity, attr) = self.base_dataset[index]
        return image, (identity, attr[self.indexs])

    def __len__(self):
        return len(self.base_dataset)
