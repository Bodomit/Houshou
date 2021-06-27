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


class RAP2(TripletsAttributeDataModule):
    def __init__(
        self,
        batch_size: int,
        attribute: List[str],
        buffer_size: Optional[int] = None,
        data_dir: str = "RAP2",
        valid_split: float = 0.05,
        valid_split_seed: int = 42,
        ext: str = "png",
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

        train, val, test = self.read_annotations()

        if stage is None or stage == "fit":
            self.train = self.construct_dataset(
                train, self.train_transforms
            )

            self.valid = self.construct_dataset(
                val, self.val_transforms
            )

        if stage is None or stage == "test":
            self.test = self.construct_dataset(
                test, self.test_transforms
            )

        super().setup(stage)

    def construct_dataset(
        self,
        annotations: pandas.DataFrame,
        transform: transforms.Compose,
    ):
        annotations = annotations.sort_values("identity")
        filenames = annotations.index
        identities = annotations["identity"]

        assert isinstance(filenames, pandas.Index)
        assert isinstance(identities, pandas.Series)

        # Get attributes per image.
        selected_attributes = annotations[self.attribute]

        attributes = torch.as_tensor(
           selected_attributes.values, dtype=torch.int64  # type: ignore
        )


        # Filter out junk and distractors.
        non_junk_mask = identities.astype(np.int32) >= 1
        filenames = filenames[non_junk_mask]
        identities = identities[non_junk_mask]
        attributes = attributes[non_junk_mask]

        # Add path to filenames.
        filenames = [os.path.join(self.data_dir, "RAP_dataset", f) for f in filenames]

        attribute_support = self.calc_attribute_support(attributes)

        return RAP2Dataset(
            filenames,
            torch.as_tensor(identities.values.astype(np.int64)),
            attributes,
            selected_attributes.columns.tolist(),
            attribute_support,
            self.target_type,
            transform,
        )

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

    def read_annotations(
        self, filename=os.path.join("RAP_annotation", "RAP_annotation.mat")
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        path = os.path.join(self.data_dir, filename)

        attributes_mat = spio.loadmat(
            path, struct_as_record=False, squeeze_me=True, simplify_cells=True
        )

        filenames = attributes_mat["RAP_annotation"]["name"]
        identities = attributes_mat["RAP_annotation"]["person_identity"]
        attributes_data = attributes_mat["RAP_annotation"]["data"]
        attribute_names = attributes_mat["RAP_annotation"]["attribute"]

        partitions = attributes_mat["RAP_annotation"]["partition_attribute"][0]
        train_idx = (partitions["train_index"] - 1).tolist()
        val_idx = (partitions["val_index"] - 1).tolist()
        test_idx = (partitions["test_index"] - 1).tolist()

        attributes = pandas.DataFrame(attributes_data, columns=attribute_names, index=filenames)
        attributes["Male"] = (attributes["Femal"] == 0).astype(np.int64)

        filenames_with_identities = pandas.DataFrame(identities, columns=["identity"], index=filenames)

        full_annotation = filenames_with_identities.join(attributes, how="inner")

        train = full_annotation.iloc[train_idx]
        val = full_annotation.iloc[val_idx]
        test = full_annotation.iloc[test_idx]

        assert isinstance(train, pandas.DataFrame)
        assert isinstance(val, pandas.DataFrame)
        assert isinstance(test, pandas.DataFrame)

        return train, val, test


class RAP2Dataset(Dataset):
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

        self.classes = self.identities.unique()

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
