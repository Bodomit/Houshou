import os
import glob
from functools import partial
from os.path import basename

from typing import Any, Callable, List, Optional, Tuple

import pandas
import PIL

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg


class CelebA(VisionDataset):
    """
    This code is based on the standard torchvision.datasets.CelebA dataset.
    """

    def __init__(
        self,
        root: str,
        split: str,
        base_folder="CelebA_MTCNN",
        target_type: List[str] = ["identity", "attr"],
        use_png=True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super(CelebA, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.base_folder = base_folder
        self.target_type = target_type
        self.image_dir = "img_align_celeba_png" if use_png else "img_align_celeba"

        ext = "png" if use_png else "jpg"
        fn = partial(os.path.join, self.root, self.base_folder)
        pattern = fn(self.image_dir, f"*.{ext}")
        real_image_ids = set([os.path.basename(f) for f in list(glob.glob(pattern))])

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }

        split_ = split_map[
            verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))
        ]

        replace_ext_ = partial(self.replace_etx, ext)
        identity = pandas.read_csv(  # type: ignore
            fn("identity_CelebA.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
            converters={0: replace_ext_},
        )

        sort_order = identity.sort_values(1).index
        identity = identity.loc[sort_order]

        splits = pandas.read_csv(  # type: ignore
            fn("list_eval_partition.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
            converters={0: replace_ext_},
        ).loc[sort_order]
        bbox = pandas.read_csv(
            fn("list_bbox_celeba.txt"),
            delim_whitespace=True,
            header=1,
            index_col=0,
            converters={0: replace_ext_},
        ).loc[sort_order]
        landmarks_align = pandas.read_csv(
            fn("list_landmarks_align_celeba.txt"),
            delim_whitespace=True,
            header=1,
            converters={0: replace_ext_},
        ).loc[sort_order]
        attr = pandas.read_csv(
            fn("list_attr_celeba.txt"),
            delim_whitespace=True,
            header=1,
            converters={0: replace_ext_},
        ).loc[sort_order]

        diff = list(sorted(set(splits.index.values) - real_image_ids))

        def rm_diff(df: pandas.DataFrame) -> pandas.DataFrame:
            return df.drop(diff)

        splits = rm_diff(splits)
        assert len(splits) == len(real_image_ids)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        assert isinstance(bbox, pandas.DataFrame)
        assert isinstance(landmarks_align, pandas.DataFrame)
        assert isinstance(attr, pandas.DataFrame)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(rm_diff(identity)[mask].values)
        self.bbox = torch.as_tensor(rm_diff(bbox)[mask].values)
        self.landmarks_align = torch.as_tensor(rm_diff(landmarks_align)[mask].values)
        self.attr = torch.as_tensor(rm_diff(attr)[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    @staticmethod
    def replace_etx(ext: str, val: str) -> str:
        return val.split(".")[0] + "." + ext

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(  # type: ignore
            os.path.join(
                self.root, self.base_folder, self.image_dir, self.filename[index]
            )
        )

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError('Target type "{}" is not recognized.'.format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)
