import os
from functools import partial

from typing import Any, List, Optional

import pandas
import PIL

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .base import TripletsAttributeDataModule


class CelebA(TripletsAttributeDataModule):
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = None,
        buffer_size: int = None,
        selected_attributes: List[str] = None,
        target_type: List[str] = ["identity", "attr"],
        use_png=True,
    ):
        assert data_dir
        assert batch_size
        assert buffer_size
        assert selected_attributes

        super().__init__(batch_size, buffer_size)

        # Store attributes.
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.selected_attributes = selected_attributes
        self.target_type = target_type
        self.image_dir = "img_align_celeba_png" if use_png else "img_align_celeba"
        self.ext = "png" if use_png else "jpg"

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

        fn = partial(os.path.join, self.data_dir)
        image_absdir = fn(self.image_dir)
        real_image_ids = set(
            [
                os.path.basename(f.path)
                for f in os.scandir(image_absdir)
                if f.name.split(".")[1] == self.ext
            ]
        )

        split_map = {"train": 0, "valid": 1, "test": 2}

        replace_ext_ = partial(self.replace_etx, self.ext)
        identities = pandas.read_csv(  # type: ignore
            fn("identity_CelebA.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
            converters={0: replace_ext_},
        )

        sort_order = identities.sort_values(1).index
        identities = identities.loc[sort_order]

        splits = pandas.read_csv(  # type: ignore
            fn("list_eval_partition.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
            converters={0: replace_ext_},
        ).loc[sort_order]
        bboxes = pandas.read_csv(
            fn("list_bbox_celeba.txt"),
            delim_whitespace=True,
            header=1,
            index_col=0,
            converters={0: replace_ext_},
        ).loc[sort_order]
        landmarks_aligns = pandas.read_csv(
            fn("list_landmarks_align_celeba.txt"),
            delim_whitespace=True,
            header=1,
            converters={0: replace_ext_},
        ).loc[sort_order]
        attrs = pandas.read_csv(
            fn("list_attr_celeba.txt"),
            delim_whitespace=True,
            header=1,
            converters={0: replace_ext_},
        ).loc[sort_order]

        # Clip the attributes to boolean values.
        def clip(x):
            if x <= 0:
                return 0
            else:
                return 1

        assert isinstance(attrs, pandas.DataFrame)
        attrs = attrs.applymap(clip).astype("bool")

        # Remove attribute lines that have no corresponding image.
        diff = list(sorted(set(splits.index.values) - real_image_ids))

        def rm_diff(df: pandas.DataFrame) -> pandas.DataFrame:
            return df.drop(diff)

        splits = rm_diff(splits)
        assert len(splits) == len(real_image_ids)
        assert isinstance(bboxes, pandas.DataFrame)
        assert isinstance(landmarks_aligns, pandas.DataFrame)
        assert isinstance(attrs, pandas.DataFrame)

        # Get the attribute names and coresponding indexes.
        attr_names = list(attrs.columns)
        selected_attribute_indexs = self.get_indexes(
            attr_names, self.selected_attributes
        )

        # For each split, consturct a mask and create a correspondign dataset.
        for split in split_map:
            if stage == "fit" and split == "test":
                continue
            elif stage == "test" and split in ["train", "valid"]:
                continue

            mask = splits[1] == split_map[split]

            filenames = splits[mask].index.values
            identities_ = torch.as_tensor(rm_diff(identities)[mask].values)
            bboxes_ = torch.as_tensor(rm_diff(bboxes)[mask].values)
            landmarks_aligns_ = torch.as_tensor(rm_diff(landmarks_aligns)[mask].values)
            attrs_ = torch.as_tensor(rm_diff(attrs)[mask].values)
            attrs_ = (attrs_ + 1) // 2  # map from {-1, 1} to {0, 1}

            assert isinstance(attrs_, torch.Tensor)
            attrs_ = attrs_[:, selected_attribute_indexs]

            if split == "train":
                transform = self.train_transforms
            elif split == "valid":
                transform = self.val_transforms
            elif split == "test":
                transform = self.test_transforms
            else:
                raise ValueError()

            split_dataset = CelebADataset(
                os.path.join(self.data_dir, self.image_dir),
                self.target_type,
                transform,
                filenames,
                identities_,
                bboxes_,
                landmarks_aligns_,
                attrs_,
                attr_names,
            )

            if split == "train":
                self.train = split_dataset
            elif split == "valid":
                self.valid = split_dataset
            elif split == "test":
                self.test = split_dataset

    @staticmethod
    def replace_etx(ext: str, val: str) -> str:
        return val.split(".")[0] + "." + ext


class CelebADataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        target_type: List[str],
        transform: transforms.Compose,
        filenames: pandas.Series,
        identities: torch.Tensor,
        bboxes: torch.Tensor,
        landmarks_aligns: torch.Tensor,
        attributes: torch.Tensor,
        attribute_names: List[str],
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.target_type = target_type
        self.transform = transform
        self.filenames = filenames
        self.identities = identities
        self.bboxes = bboxes
        self.landmarks_aligns = landmarks_aligns
        self.attributes = attributes
        self.attribute_names = attribute_names

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        X = PIL.Image.open(  # type: ignore
            os.path.join(self.data_dir, self.filenames[index])  # type: ignore
        )

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attributes[index, :])
            elif t == "identity":
                target.append(self.identities[index, 0])
            elif t == "bbox":
                target.append(self.bboxes[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_aligns[index, :])
            else:
                raise ValueError('Target type "{}" is not recognized.'.format(t))

        if self.transform is not None:
            X = self.transform(X)

        return X, tuple(target)
